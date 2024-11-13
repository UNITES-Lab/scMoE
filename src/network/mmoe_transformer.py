
from copy import deepcopy
from typing import Dict
from fmoe.layers import *
from fmoe.layers import _fmoe_general_global_forward
from fmoe.linear import FMoELinear

import tree
import os
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.nn import Dropout
import torch.nn.functional as F
import math

from fmoe.functions import prepare_forward, ensure_comm, count_by_gate
from fmoe.functions import MOEScatter, MOEGather
from fmoe.functions import AllGather, Slice
from fmoe.gates import NaiveGate, NoisyGate
from fmoe.gates.base_gate import BaseGate

from fmoe.fastermoe.config import switch_from_env


class MultiModalityConfig:
    num_experts = 16
    base_capacity = 16
    capacity_per_expert = 10
    gate = NoisyGate
    num_tasks = 1
    load_expert_count = False
    seed = 1
    attn_modality_specific = False
    modalities_name = []
    modality_remap = {}
    capacity_ratio = 1.0
    dynamic_reweight = False
    mlp_top_k = 2
    attn_top_k = 2
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
            
    def setting_modality_remap(self, mapping:Dict):
        setattr(self, 'modality_remap', mapping)
        # clear modalities_name
        if hasattr(self, 'modalities_name'):
            self.modalities_name = []
        else:
            setattr(self, 'modalities_name', [])
        for key in self.modality_remap:
            if self.modality_remap[key] not in self.modalities_name:
                self.modalities_name.append(self.modality_remap[key])


def _fmoe_limited_global_forward(inp, gate, gate_score, expert_fn, num_expert, world_size, expert_capacity, is_train, top_k = 2, **kwargs):
    r"""
    A private function that performs the following steps to complete the MoE
    computation.
    * Count the number of tokens from each worker to each expert.
    * Send the features to their target position so that input features to each
    expert are contiguous in memory.
    * Perform the forward computation of the experts using `expert_fn`
    * Gather the output features of experts back, and reorder them as sentences.
    Intermediate results like expert counts are hidden from users by this
    function.
    """
    (
        pos,
        local_expert_count,
        global_expert_count,
        fwd_expert_count,
        fwd_batch_size,
    ) = prepare_forward(gate, num_expert, world_size)
    topk = 1
    if len(gate.shape) == 2:
        topk = gate.shape[1]

    def scatter_func(tensor):
        return MOEScatter.apply(
            tensor,
            torch.div(pos, topk, rounding_mode='floor'),
            local_expert_count,
            global_expert_count,
            fwd_batch_size,
            world_size,
        )
    x = tree.map_structure(scatter_func, inp)

    end_indexes = fwd_expert_count.cumsum(dim=0)
    start_indexes = deepcopy(end_indexes)
    start_indexes[1:] = end_indexes[:-1]
    start_indexes[0] = 0
    mask_scores , _= gate_score.max(dim=-1)
    mask = torch.ones_like(pos, dtype=torch.bool)
    drop_idx = {}
    select_idx = {}

    def generate_mask(pos_start, pos_end, index):
        # get token index
        token_pos = pos[pos_start:pos_end]
        token_number = pos_end - pos_start
        if token_number <= expert_capacity:
            return None
        expert_token_scores = mask_scores[(token_pos/top_k).to(torch.long)]
        drop_token_idx = token_pos[expert_token_scores.argsort()[:-expert_capacity]]
        select_token_idx = token_pos[expert_token_scores.argsort()[-expert_capacity:]]
        drop_idx[index] = drop_token_idx
        select_idx[index] = select_token_idx
        mask[drop_token_idx] = False
        fwd_expert_count[index] = expert_capacity
    tree.map_structure(generate_mask, start_indexes.tolist(), end_indexes.tolist(), [i for i in range(len(fwd_expert_count))])


    def delete_mask_func(tensor):
        tensor = tensor[mask == True, :]
        return tensor
    exp_inp = tree.map_structure(delete_mask_func, x)
    exp_out = expert_fn(exp_inp, fwd_expert_count)

    # recover input tensor
    def recover_func(tensor):
        x[mask == True] = tensor
        return x
    x = tree.map_structure(recover_func, exp_out)

    
    out_batch_size = tree.flatten(inp)[0].shape[0]
    if len(gate.shape) == 2:
        out_batch_size *= gate.shape[1]

    def gather_func(tensor):
        return MOEGather.apply(
            tensor,
            pos,
            local_expert_count,
            global_expert_count,
            out_batch_size,
            world_size,
        )

    outp = tree.map_structure(gather_func, x)
    return outp, drop_idx, select_idx


class LimitCapacityMoE(FMoE):
    """Modify bugs while using FMoE

    Args:
        FMoE (_type_): _description_
    """
    def __init__(self, num_expert=32, d_model=1024, world_size=1,
                  mp_group=None, slice_group=None, moe_group=None, top_k=2, gate=NaiveGate,
                  expert=None, gate_hook=None, mask=None, mask_dict=None, capacity_per_expert = 10):
        super().__init__(num_expert = num_expert, 
                         d_model = d_model, world_size = world_size, 
                         mp_group = mp_group, slice_group = slice_group, 
                         moe_group = moe_group, top_k = top_k, 
                         gate = gate, expert = expert, 
                         gate_hook = gate_hook, mask = mask, mask_dict = mask_dict)
        self.capacity_per_expert = capacity_per_expert
        self.co_input_modalities_name = None
        self.drop_idx = None
        self.select_idx = None
        self.batch_size = 0
        
    def set_capacity(self, capacity):
        self.capacity_per_expert = capacity
        
    def forward(self, moe_inp):
        r"""
        The FMoE module first computes gate output, and then conduct MoE forward
        according to the gate.  The score of the selected gate given by the
        expert is multiplied to the experts' output tensors as a weight.
        """
        
        moe_inp_batch_size = tree.flatten(
            tree.map_structure(lambda tensor: tensor.shape[0], moe_inp)
        )
        assert all(
            [batch_size == moe_inp_batch_size[0] for batch_size in moe_inp_batch_size]
        ), "MoE inputs must have the same batch size"

        if self.world_size > 1:

            def ensure_comm_func(tensor):
                ensure_comm(tensor, self.moe_group)

            tree.map_structure(ensure_comm_func, moe_inp)
        if self.slice_size > 1:

            def slice_func(tensor):
                return Slice.apply(
                    tensor, self.slice_rank, self.slice_size, self.slice_group
                )

            moe_inp = tree.map_structure(slice_func, moe_inp)

        # input different modality data into different gating network for routing
        # require setting co_input_modalities_name and tasks_gates in sub-class
        # here different modality with same sequence length 
        if self.co_input_modalities_name is not None and hasattr(self, 'tasks_gates'):
            modality_step_in_inp = moe_inp.shape[0] // len(self.co_input_modalities_name)
            idx_list = []
            score_list = []
            for i in range(len(self.co_input_modalities_name)):
                if self.args.modality_gating_merge:
                    idx, score = self.tasks_gates[self.args.modality_remap[self.co_input_modalities_name[i]]](moe_inp[i * modality_step_in_inp: (i+1) * modality_step_in_inp])
                else:
                    idx, score = self.tasks_gates[self.co_input_modalities_name[i]](moe_inp[i * modality_step_in_inp: (i+1) * modality_step_in_inp])
                idx_list.append(idx)
                score_list.append(score)
            gate_top_k_idx = torch.cat(idx_list, dim=0)
            gate_score = torch.cat(score_list, dim=0)
        else:
            gate_top_k_idx, gate_score = self.gate(moe_inp)

        # gate_top_k_idx, gate_score = self.gate(moe_inp)
        gate_score = gate_score.reshape(moe_inp.shape[0], self.top_k)
        gate_top_k_idx = gate_top_k_idx.reshape(moe_inp.shape[0], self.top_k)

        if self.gate_hook is not None:
            self.gate_hook(gate_top_k_idx, gate_score, None)
        
        # delete masked tensors
        if self.mask is not None and self.mask_dict is not None:
            def delete_mask_func(tensor):
                # to: (BxL') x d_model
                tensor = tensor[mask == 0, :]
                return tensor

            mask = self.mask.view(-1)
            moe_inp = tree.map_structure(delete_mask_func, moe_inp)
            gate_top_k_idx = gate_top_k_idx[mask == 0, :]

        # imp of fmoe not support limit expert capacity, extend for Batch Prioritized Routing (VMOE) 
        fwd, drop_idx, select_idx = _fmoe_limited_global_forward(
            moe_inp, gate_top_k_idx, gate_score, self.expert_fn,
            self.num_expert, self.world_size,
            experts=self.experts, expert_capacity=self.capacity_per_expert, is_train=self.training, top_k=self.top_k
        )
        self.drop_idx = drop_idx
        self.select_idx = select_idx

        # recover deleted tensors
        if self.mask is not None and self.mask_dict is not None:

            def recover_func(tensor):
                # to: (BxL') x top_k x dim
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                # to: (BxL) x top_k x d_model
                x = torch.zeros(
                    mask.shape[0],
                    self.top_k,
                    dim,
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
                # recover
                x[mask == 0] = tensor
                for k, v in self.mask_dict.items():
                    x[mask == k] = v
                return x

            moe_outp = tree.map_structure(recover_func, fwd)
        else:

            def view_func(tensor):
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                return tensor

            moe_outp = tree.map_structure(view_func, fwd)

        gate_score = gate_score.view(-1, 1, self.top_k)

        def bmm_func(tensor):
            dim = tensor.shape[-1]
            tensor = torch.bmm(gate_score, tensor).reshape(-1, dim)
            return tensor

        moe_outp = tree.map_structure(bmm_func, moe_outp)

        if self.slice_size > 1:

            def all_gather_func(tensor):
                return AllGather.apply(
                    tensor, self.slice_rank, self.slice_size, self.slice_group
                )

            moe_outp = tree.map_structure(all_gather_func, moe_outp)

        moe_outp_batch_size = tree.flatten(
            tree.map_structure(lambda tensor: tensor.shape[0], moe_outp)
        )
        assert all(
            [batch_size == moe_outp_batch_size[0] for batch_size in moe_outp_batch_size]
        ), "MoE outputs must have the same batch size"
        return moe_outp

class QKVSeperateExpert(nn.Module):
    r"""
    An expert using 2 FMoELinear modules to speed up the computation of experts
    within one worker.
    """

    def __init__(self, num_expert, in_dim, out_dim, bias, rank=0, args = None):
        super().__init__()
        self.seperate_qkv = FMoELinear(num_expert, in_dim, out_dim, bias=bias, rank=rank)
        self.args = args
        self.expert_count = None

    def get_loss(self, clear=True):
        loss = self.loss
        if clear:
            self.loss = None
        return loss

    def get_expert_count(self):
        if self.args.load_expert_count:
            return self.expert_count
        
        return None

    def forward(self, inp, fwd_expert_count):
        r"""
        First expand input to 4h (the hidden size is variable, but is called h4
        for convenience). Then perform activation. Finally shirink back to h.
        """
        if self.args.load_expert_count:
            self.expert_count = fwd_expert_count
        x = self.seperate_qkv(inp, fwd_expert_count)
        return x

class VMoETransformerSeperateQKV(LimitCapacityMoE):
    def __init__(self, 
                 num_expert=32, 
                 d_model=1024,
                 out_dim = 1024, 
                 bias = False,
                 expert_dp_comm="none",
                 world_size=1, 
                 mp_group=None, 
                 slice_group=None, 
                 moe_group=None, 
                 top_k=2, 
                 gate=NaiveGate, 
                 expert=None, 
                 gate_hook=None, 
                 mask=None, 
                 mask_dict=None,
                 args : MultiModalityConfig = None
    ):
        super().__init__(num_expert, d_model, world_size, mp_group, slice_group, moe_group, top_k, gate, expert, gate_hook, mask, mask_dict)
        self.experts = QKVSeperateExpert(num_expert, d_model, out_dim, bias, args = args)
        self.capacity_per_expert = args.capacity_per_expert
        self.out_dim = out_dim
        self.args = args
        self.tasks_gates = nn.ModuleDict()
        if not self.args.attn_modality_specific:
            for i in range(args.num_tasks):
                self.tasks_gates[str(i)] = gate(d_model, num_expert, world_size, top_k)
        else:
            for ms in self.args.modalities_name:
                self.tasks_gates[ms] = gate(d_model, num_expert, world_size, top_k)
        self.mark_parallel_comm(expert_dp_comm)
        self.expert_local_count = None
        
        
    def gate_loss(self, task_idx = None, modality_name = None):
        if self.args.attn_modality_specific:
            if self.args.modality_gating_merge:
                return self.tasks_gates[self.args.modality_remap[modality_name]].get_loss()
            else:
                return self.tasks_gates[modality_name].get_loss()
        else:
            return self.tasks_gates[str(task_idx)].get_loss()
        
    def get_expert_count(self, num_modalities):
        if not self.args.co_input:
            return self.experts.get_expert_count()
        else:
            index_range = self.batch_size * self.top_k // num_modalities
            all_expert_count = {}
            for i in range(num_modalities):
                expert_count = [0 for _ in range(self.num_expert)]
                for e in range(self.num_expert):
                    if e in self.select_idx:
                        lower_bound = self.select_idx[e] >= i * index_range
                        higher_bound = self.select_idx[e] < (i+1) * index_range
                        expert_count[e] = (lower_bound * higher_bound).sum().cpu().item()
                all_expert_count[i] = torch.tensor(expert_count)
            return all_expert_count
        
    def forward(self, inp: torch.Tensor, task_idx = None, modality_name = None):
        if type(modality_name) is list and self.args.attn_modality_specific:
            self.co_input_modalities_name = modality_name
        else:
            if self.args.attn_modality_specific:
                if self.args.modality_gating_merge:
                    self.gate = self.tasks_gates[self.args.modality_remap[modality_name]]
                else:
                    self.gate = self.tasks_gates[modality_name]
            else:
                self.gate = self.tasks_gates[str(task_idx)]
            self.co_input_modalities_name = None
        output_shape = list(inp.shape)
        output_shape[-1] = self.out_dim

        
        inp = inp.reshape(-1, self.d_model)
        self.batch_size = inp.shape[0]
        output = super().forward(inp)
        return output.reshape(output_shape)


class AttentionMoEQKVSeperate(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., args = None, top_k=2):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # seperate for BPR (Batch Prioritized Routing) -- require: input_dim==output_dim, otherwise add additional proj layer (not imp)
        self.q = VMoETransformerSeperateQKV(args.num_experts, dim, dim, bias = qkv_bias, gate=args.gate, args = args, top_k=top_k)
        self.k = VMoETransformerSeperateQKV(args.num_experts, dim, dim, bias = qkv_bias, gate=args.gate, args = args, top_k=top_k)
        self.v = VMoETransformerSeperateQKV(args.num_experts, dim, dim, bias = qkv_bias, gate=args.gate, args = args, top_k=top_k)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def gate_loss(self, task_idx = None, modality_name = None):
        return self.q.gate_loss(task_idx, modality_name) + self.k.gate_loss(task_idx, modality_name) + self.v.gate_loss(task_idx, modality_name)
    
    def get_expert_count(self, num_modalities = 2):
        return {'q': self.q.get_expert_count(num_modalities), 'k': self.k.get_expert_count(num_modalities), 'v': self.v.get_expert_count(num_modalities)}

    def forward(self, x, task_idx = None, modality_name = None):
        B, N, C = x.shape
        q = self.q(x, task_idx, modality_name).reshape(B, N, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x, task_idx, modality_name).reshape(B, N, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x, task_idx, modality_name).reshape(B, N, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x