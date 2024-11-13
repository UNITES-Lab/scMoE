
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from src.network.constants import *
from src.network.moe_module import *

import copy


def create_encoder(config):
    encoder_type = config[str_encoder_type]
    kwargs = copy.deepcopy(config)
    kwargs.pop(str_encoder_type)

    try:
        if encoder_type == 'PatchEmbeddings':
            return eval(encoder_type)(**kwargs)
        else:
            return eval(encoder_type)(config)
    except:
        raise NotImplementedError(f"{encoder_type} is not implemented.")
    
def create_transformer(config: dict):
    transformer_type = config[str_transformer_type]
    kwargs = copy.deepcopy(config)
    kwargs.pop(str_transformer_type)
    try:
        return eval(transformer_type)(**kwargs)
    except:
        raise NotImplementedError(f"{transformer_type} is not implemented.")

def create_fuser(config: dict):
    fuser_type = config[str_fuser_type]
    kwargs = copy.deepcopy(config)
    kwargs.pop(str_fuser_type)
    try:
        return eval(fuser_type)(**kwargs)
    except:
        raise NotImplementedError(f"{fuser_type} is not implemented.")

def create_decoder(config):
    decoder_type = config[str_decoder_type]
    kwargs = copy.deepcopy(config)
    kwargs.pop(str_decoder_type)

    try:
        return eval(decoder_type)(**kwargs)
    except:
        raise NotImplementedError(f"{decoder_type} is not implemented.")

class SequenceConcat(nn.Module):
    """
    Concatenate a list of sequences into a single sequence.

    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.cat(x, dim=1)

class EncoderMLP(nn.Module):
    '''Encoder Module, the input dimension should be B x N'''
    def __init__(self, config):
        super().__init__()
        in_feat_size = config[str_input]
        out_feat_size = config[str_output]
        layers = []
        layers.append(nn.Linear(in_feat_size, out_feat_size, bias=False))
        if config[str_use_batch_norm]:
            layers.append(nn.BatchNorm1d(out_feat_size))
        if config[str_use_layer_norm]:
            layers.append(nn.LayerNorm(out_feat_size))
        activation = config[str_activation]
        if activation is not None:
            if activation == str_relu:
                layers.append(nn.ReLU())
            elif activation == str_sigmoid:
                layers.append(nn.Sigmoid())
            elif activation == str_softmax:
                layers.append(nn.Softmax())
            elif activation == str_tanh:
                layers.append(nn.Tanh())
            elif activation.startswith("leaky_relu"):
                neg_slope = float(activation.split(":")[1])
                layers.append(nn.LeakyReLU(neg_slope))
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class PatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding.

    """
    def __init__(self, feature_size, num_patches, embed_dim, dropout=0.25):
        super().__init__()
        patch_size = math.ceil(feature_size / num_patches)
        pad_size = num_patches*patch_size - feature_size
        self.pad_size = pad_size
        self.num_patches = num_patches
        self.feature_size = feature_size
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size, embed_dim)


    def forward(self, x):
        x = F.pad(x, (0, self.pad_size)).view(x.shape[0], self.num_patches, self.patch_size)
        x = self.projection(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, 
                 nhead, 
                 dropout=0.1, 
                 activation=nn.GELU, 
                 hidden_times=2, 
                 mlp_sparse = False, 
                 self_attn = True,
                 **kwargs) -> None:
        super(TransformerEncoderLayer, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation()
        self.attn = Attention(
            d_model, num_heads=nhead, qkv_bias=False, attn_drop=dropout, proj_drop=dropout)
        
        self.mlp_sparse = mlp_sparse
        self.self_attn = self_attn
        
        if self.mlp_sparse:
            self.mlp = FMoETransformerMLP(d_model=d_model, d_hidden=d_model * hidden_times, activation=nn.GELU(), **kwargs)
        else:
            self.mlp = Mlp(in_features=d_model, hidden_features=d_model * hidden_times, act_layer=nn.GELU, drop=dropout)

    def forward(self, x, attn_mask = None):
        if self.self_attn:
            chunk_size = [item.shape[1] for item in x]
            x = self.norm1(torch.cat(x, dim=1))
            kv = x
            x = self.attn(x, kv, attn_mask)
            x = x + self.dropout1(x)
            x = torch.split(x, chunk_size, dim=1)
            x = [item for item in x]
            if self.mlp_sparse:
                for i in range(len(chunk_size)):
                    x[i] = x[i] + self.dropout2(self.mlp(self.norm2(x[i]), i))
            else:
                for i in range(len(chunk_size)):
                    x[i] = x[i] + self.dropout2(self.mlp(self.norm2(x[i])))
        else:
            chunk_size = [item.shape[1] for item in x]
            x = [item for item in x]
            for i in range(len(chunk_size)):
                other_m = [x[j] for j in range(len(chunk_size)) if j != i]
                other_m = torch.cat([x[i], *other_m], dim=1)
                x[i] = self.attn(x[i], other_m, attn_mask)
            x = [x[i]+self.dropout1(x[i]) for i in range(len(chunk_size))]
            if self.mlp_sparse:
                for i in range(len(chunk_size)):
                    x[i] = x[i] + self.dropout2(self.mlp(self.norm2(x[i]), i))
            else:
                for i in range(len(chunk_size)):
                    x[i] = x[i] + self.dropout2(self.mlp(self.norm2(x[i])))

        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.head_dim = head_dim
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(head_dim * self.num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x, kv, attn_mask=None):
        # attn_mask: (B, N+1, N+1) input-dependent
        eps = 1e-6

        Bx, Nx, Cx = x.shape
        B, N, C = kv.shape
        q = self.q(x).reshape(Bx, Nx, self.num_heads, Cx//self.num_heads)
        q = q.permute(0, 2, 1, 3)
        kv = self.kv(kv)
        kv = kv.reshape(B, N, 2, self.num_heads, C // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N+1, C/H) @ (B, H, C/H, N+1) -> (B, H, N+1, N+1)

        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(Bx, Nx, -1)  # (B, H, N+1, N+1) * (B, H, N+1, C/H) -> (B, H, N+1, C/H) -> (B, N+1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DecoderMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x