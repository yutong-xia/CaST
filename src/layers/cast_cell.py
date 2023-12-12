import math
import numpy as np
from typing import List

import torch
from torch import nn
import torch.nn.functional as F
import torch.fft as fft
from einops import reduce, rearrange

from src.layers.vq_functions import vq, vq_st
from torch_geometric.data import Dataset, Data

from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import OptTensor
from torch_geometric.data import Data

from src.layers.cell import *

class TempDisentangler(nn.Module):
    def __init__(self, input_dims, output_dims,
                 kernels: List[int],
                 length: int,
                 hidden_dims, depth, dropout):
        super().__init__()

        component_dims = output_dims // 2
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.component_dims = component_dims
        self.hidden_dims = hidden_dims

        self.kernels = kernels

        self.EnvEncoder = nn.ModuleList(
            [nn.Conv1d(input_dims, component_dims, k, padding=k-1) for k in kernels]
        )

        self.Ent_time = SelfAttention(num_heads=4, in_dim=hidden_dims, hid_dim= hidden_dims, dropout=dropout)

        #### frequency settings
        self.length = length
        self.num_freqs = (self.length // 2) + 1

        self.Ent_freq_weight = nn.Parameter(torch.empty((self.num_freqs, hidden_dims, hidden_dims), dtype=torch.cfloat))
        self.Ent_freq_bias = nn.Parameter(torch.empty((self.num_freqs, hidden_dims), dtype=torch.cfloat))
        self.reset_parameters()

        self.Ent_dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.Ent_freq_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.Ent_freq_weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.Ent_freq_bias, -bound, bound)

    def forward(self, x):  # x: B x T x input_dims
        env_rep = []
        for idx, mod in enumerate(self.EnvEncoder):
            out = mod(x)  # b d t
            if self.kernels[idx] != 1:
                out = out[..., :-(self.kernels[idx] - 1)]
            env_rep.append(out.transpose(1, 2))  # b t d
        env_rep = reduce(
            rearrange(env_rep, 'list b t d -> list b t d'),
            'list b t d -> b t d', 'mean'
        )

        x = x.transpose(1, 2) 

        entity_time = self.Ent_time(x)
        input_freq = fft.rfft(x, dim=1)
        output_freq = torch.zeros(x.size(0), x.size(1) // 2 + 1, self.hidden_dims, device=x.device, dtype=torch.cfloat)
        output_freq[:, :self.num_freqs] = torch.einsum('bti,tio->bto', input_freq[:, :self.num_freqs], self.Ent_freq_weight) + self.Ent_freq_bias
        entity_freq = fft.irfft(output_freq, n=x.size(1), dim = 1)

        entity_rep = torch.add(entity_time, entity_freq)
        entity_rep = self.Ent_dropout(entity_rep)
        return env_rep, entity_rep

class EnvEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        # codebook
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, x): # x [b, l, h_d]
        x_ = x.contiguous()
        latents = vq(x_, self.embedding.weight)
        return latents
    
    def straight_through(self, z_e_x):# x [b, h_d]
        '''
        z_e_x: the latent vectors for environments
        '''
        z_e_x_ = z_e_x.contiguous()
        # get the feature from the codebook and its index
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach()) # z_q_x_: [b, h_d]    indices:[b]
        z_q_x = z_q_x_.contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight, dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.contiguous()

        return z_q_x, z_q_x_bar, indices

    def straight_through_test(self, z_e_x):# the index is soft
        inputs = z_e_x.contiguous()
        codebook = self.embedding.weight.detach()

        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_flatten = inputs.view(-1, embedding_size)
            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)
            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr, inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0) 
            # get the index
            indices = torch.softmax(distances, dim=1)    
            # compute the env vector
            codes_flatten = torch.mm(indices, codebook)
            codes = codes_flatten.view_as(inputs)

            return codes.contiguous(), None, indices

## spatial
class HodgeLaguerreConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, K: int, 
                  bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        assert K > 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lins = nn.ModuleList([
            Linear(in_channels, out_channels, bias=False,
                    weight_initializer='glorot') for _ in range(K)
        ])

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        zeros(self.bias)

    
    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: OptTensor = None):
        norm = edge_weight
        Tx_0 = x
        Tx_1 = x 
        out = self.lins[0](Tx_0)
        xshape = x.shape
        k = 1

        if len(self.lins) > 1:
            x = x.reshape(xshape[0],-1)
            Tx_1 = x - self.propagate(edge_index, x=x, norm=norm, size=None)
            if len(xshape)>=3:
                Tx_1 = Tx_1.view(xshape[0],xshape[1],-1)
            out = out + self.lins[1](Tx_1)

        for lin in self.lins[2:]:
            inshape = Tx_1.shape
            Tx_1 = Tx_1.view(inshape[0],-1)
            Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm, size=None)
            if len(xshape)>=3:
                Tx_2 = Tx_2.view(inshape[0],inshape[1],-1)
                Tx_1 = Tx_1.view(xshape[0],xshape[1],-1)
            Tx_2 = (-Tx_2 + (2*k+1)*Tx_1 - k* Tx_0) / (k+1)
            k += 1
            out = out + lin.forward(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={len(self.lins)}')

class GCNConv(MessagePassing):
    def __init__(self, in_channels, num_nodes, out_channels, K):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.layer_norm = nn.LayerNorm([num_nodes, out_channels])
        self.bias = Parameter(torch.Tensor(out_channels))
        self.K = K
        self.num_nodes = num_nodes
        self.out_channels = out_channels

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, input, edge_index, edge_weight):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        x = self.lin(input)
        x_res = x
        edge_index = self.undir2dir(edge_index)
        edge_weight = edge_weight.reshape(-1,self.K)
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=edge_weight[...,k])
            x = F.relu(x)

        out = self.bias + x_res + x
        out = self.layer_norm(out.reshape(-1,self.num_nodes, self.out_channels))
        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]
        return norm.view(-1, 1) * x_j

    def undir2dir(self, edge_index):
        src, dst = edge_index[0], edge_index[1]
        directed_edge_index = torch.stack([src, dst], dim=0)
        reversed_edge_index = torch.stack([dst, src], dim=0)
        edge_index = torch.cat([directed_edge_index, reversed_edge_index], dim=1)
        return edge_index





