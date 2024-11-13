from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from fmoe.gates.base_gate import BaseGate

from cytomoe.network.modules import *
from cytomoe.network.mmoe_transformer import MultiModalityConfig
from cytomoe.network.constants import *

import copy

def kaiming_init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight)


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_path = None
        self.label_encoder = preprocessing.LabelEncoder()
        
        self.n_modality = len(config[str_encoders])
        self.noise_level = config[str_noise]
        args = MultiModalityConfig(**config[str_multimodality])

        self.encoders = nn.ModuleList(
            [create_encoder(encoder) for encoder in config[str_encoders]]
        )

        self.latent_projector = torch.nn.Linear(config[str_hidden_dim]*self.n_modality, config[str_hidden_dim], bias=False)
        self.n_out = config[str_categories]

        if config[str_encoders][0]['type'] == 'PatchEmbeddings':
            self.pos_embed = nn.Parameter(torch.zeros(1, np.sum([encoder["num_patches"] for encoder in config[str_encoders]]), config[str_hidden_dim]))
        else:
            self.pos_embed = None

        self.transformers = nn.ModuleList(
            [create_transformer(trans) for trans in config[str_transformer]]
        )

        if config[str_categories] > 0:
            self.projector = torch.nn.Linear(config[str_hidden_dim], config[str_categories], bias=False)
        else:
            self.projector = None
        
        self.decoders = nn.ModuleList(
            [create_decoder(decoder) for decoder in config[str_decoders]]
        )

        self.prob_layer = torch.nn.Softmax(dim=-1)

        self.train()
    
    def add_noise(self, inputs, levels, device):
        noised_input = []
        for input, level in zip(inputs, levels):
            shape = input.shape
            m_ = 0
            v_ = torch.var(input).detach() * level
            if v_ > 0:
                noise = torch.normal(m_, v_, size=shape).to(device=device)
                # print(input.device, noise.device)
                noised_input.append(input + noise)
            else:
                noised_input.append(input)
        return noised_input

    def impute_check(self, orig_modality):
        '''Check if the dim of encoders match dim of modalties'''
        self.input_dims = [encoder["input"] for encoder in self.config[str_encoders]]
        if type(orig_modality) is not list:
            checked_modalities = []
            for sd in self.input_dims:
                if orig_modality.shape[1] == sd:
                    checked_modalities.append(torch.tensor(orig_modality))
                else:
                    checked_modalities.append(torch.zeros([orig_modality.shape[0], sd]))
        else:
            assert len(orig_modality) == self.n_modality, "please give either full list of all modalities or a single modality"
            checked_modalities = orig_modality
        return checked_modalities
    
    def forward(self, modalities, mask = None, labels=None, use_cluster = False):
        self.modalities = [
            modality.to(device=self.device_in_use) for modality in modalities
        ]

        if self.noise_level!=None and self.training:
            self.modalities = self.add_noise(inputs=self.modalities,levels=self.noise_level,device=self.device_in_use)

        self.latents = [
            encoder(modality)
            for (encoder, modality) in zip(self.encoders, self.modalities)
        ]
        self.chunk_size = [item.shape[1] for item in self.latents]

        self.latent = torch.cat(self.latents, dim=1)
        
        if self.pos_embed != None:
            self.latent += self.pos_embed

        self.latents = torch.split(self.latent, self.chunk_size, dim=1)
        hidden_feat = self.latents

        for i, layer in enumerate(self.transformers):
            hidden_feat = layer(hidden_feat)

        hidden_feat_1 = [item for item in hidden_feat]
        hidden_feat = [item.mean(dim=1) for item in hidden_feat]


        recons = [
            [decoder(hidden) for hidden in hidden_feat] for decoder in self.decoders
        ]

        self.latent_projection = self.latent_projector(torch.cat(hidden_feat, dim=1))

        if self.projector:
            proj_in = torch.stack(hidden_feat).mean(dim=0)
            latent_projected = self.projector(proj_in)

            self.latent_projected = latent_projected
            
            probablities = self.prob_layer(latent_projected)
            self.predictions = probablities.argmax(1)
        else:
            probablities = None

        return probablities, recons
    
    def save_model(self, filename):
        if self.save_path is not None:
            self.modalities = None
            self.labels = None
            path = f"{self.save_path}/{filename}.pt"
            torch.save({pn:p for pn, p in self.named_parameters()}, path)

    def load_model(self, filename):
        path = f"{self.save_path}/{filename}.pt"
        model = torch.load(path)
        print(model.keys())
        with torch.no_grad():
            for pn, p in self.named_parameters():
                # print(pn)
                if pn in model:
                    print(pn)
                    p.copy_(model[pn])


    def gate_loss(self):
        g_loss = []
        for mn, mm in self.named_modules():
            if hasattr(mm, 'all_gates'):
                for i in range(len(mm.all_gates)):
                    i_loss = mm.all_gates[f'{i}'].get_loss()
                    if i_loss is None:
                        print(f"[WARN] The gate loss if {mn}, modality: {i} is emtpy, check weather call <get_loss> twice.")
                    else:
                        g_loss.append(i_loss)
        return sum(g_loss)

    def reset_parameters(self, param) -> None:
        torch.nn.init.kaiming_uniform_(param.weight, a=math.sqrt(5))
        if param.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(param.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(param.bias, -bound, bound)



        