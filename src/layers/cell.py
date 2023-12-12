import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt

class LayerNorm(nn.Module):
    def __init__(self, hid_dim, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hid_dim))
        self.bias = nn.Parameter(torch.zeros(hid_dim))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfAttention(nn.Module):
    def __init__(self, num_heads, in_dim, hid_dim, dropout):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.attention_head_size = int(hid_dim / num_heads)
        self.hid_dim = hid_dim

        self.query = nn.Linear(in_dim, self.hid_dim)
        self.key = nn.Linear(in_dim, self.hid_dim)
        self.value = nn.Linear(in_dim, self.hid_dim)

        self.attn_dropout = nn.Dropout(dropout)

        self.dense = nn.Linear(hid_dim, hid_dim)
        self.LayerNorm = LayerNorm(hid_dim, eps=1e-12)
        self.out_dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / sqrt(self.attention_head_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hid_dim,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states




