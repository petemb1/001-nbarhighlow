#!/usr/bin/env python
# encoding: utf-8
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import os

class GraphConvolution(nn.Module):

    def __init__(self, input_dim, output_dim, use_bias=False):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        support = torch.mm(input_feature, self.weight)
        output = torch.sparse.mm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' -> ' + str(self.output_dim) + ')'

class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_size, hidden_size)
        self.key = nn.Linear(input_size, hidden_size)
        self.value = nn.Linear(input_size, hidden_size)
        self.scale = torch.sqrt(torch.tensor(hidden_size, dtype=torch.float32))  # Use float32 for consistency
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Explicitly set device
        self.scale = self.scale.to(self.device) #Put this to the device

    def forward(self, x):
        # Move input to the device
        x = x.to(self.device)

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Move Q, K, V to the device
        Q, K, V = Q.to(self.device), K.to(self.device), V.to(self.device)

        attention_weights = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_weights, dim=-1)

        # Move attention_weights to device
        attention_weights = attention_weights.to(self.device)

        attended_values = torch.matmul(attention_weights, V)
        return attended_values

class Layer1(nn.Module):
    def __init__(self, input_size, hidden_size, device):  # Add device parameter
        super(Layer1, self).__init__()
        self.attn1 = nn.Linear(input_size, hidden_size).to(device)  # Move to device
        self.attn2 = nn.Linear(hidden_size, hidden_size).to(device) # Move to device
        self.device = device

    def forward(self, x):
        z1 = self.attn1(x)
        a1 = F.relu(z1)
        z2 = self.attn2(a1)
        a2 = F.relu(z2)
        return a2

class Layer2(nn.Module):
    def __init__(self, input_size, hidden_size, device):  # Add device parameter
        super(Layer2, self).__init__()
        self.attn1 = nn.Linear(input_size, hidden_size).to(device)  # Move to device
        self.attn2 = nn.Linear(hidden_size, hidden_size).to(device)  # Move to device
        self.device = device

    def forward(self, x):
        z1 = self.attn1(x)
        a1 = F.relu(z1)
        z2 = self.attn2(a1)
        a2 = F.relu(z2)
        return a2

class DAS(nn.Module):
    def __init__(self, feature_size, hidden_size, time_step, drop_ratio, device):
        super(DAS, self).__init__()
        self.layer1 = Layer1(feature_size, hidden_size, device)  # Pass device
        self.layer2 = Layer2(time_step, hidden_size, device)     # Pass device
        self.drop = nn.Dropout(drop_ratio)
        self.device = device

    def forward(self, ems, var_y, cis):
        var_x = torch.cat((ems, var_y, cis.unsqueeze(-1)), dim=2)
        var_x = var_x.to(self.device) # Move to the device
        out1 = self.layer1(var_x)
        out1 = out1.transpose(-1,-2)
        out2 = self.layer2(out1)
        out2 = out2.transpose(-1,-2)
        out = self.drop(out2)
        return out

class PriceGraph(nn.Module):
    def __init__(self, feature_size, hidden_size, time_step, drop_ratio, num_features):
        super(PriceGraph, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Set the device here
        print(f"Using device in model: {self.device}")
        self.das = nn.ModuleList()
        for i in range(num_features):
            self.das.append(DAS(feature_size, hidden_size, time_step, drop_ratio, self.device)) # Pass device to DAS

    def forward(self, var):
        out = self.das[0](var[0]['ems'], var[0]['ys'], var[0]['cis'])
        for i in range(1, len(var)):
            out += self.das[i](var[i]['ems'], var[i]['ys'], var[i]['cis'])
        return out


class output_layer(nn.Module):
    def __init__(self, last_hidden_size, output_size):
        super(output_layer, self).__init__()
        self.out_layer = nn.Linear(last_hidden_size, output_size)
    def forward(self, x):
        out = self.out_layer(x)
        return out
