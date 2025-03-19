#!/usr/bin/env python
# encoding: utf-8
import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

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
        nn.init.kaiming_uniform_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)

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
        self.last_hidden_size = input_size
        self.hidden_size = hidden_size

        self.wq = nn.Linear(in_features=input_size, out_features=hidden_size, bias=False)
        self.wk = nn.Linear(in_features=input_size, out_features=hidden_size, bias=False)
        self.wv = nn.Linear(in_features=input_size, out_features=hidden_size, bias=False)
        self.scale = torch.sqrt(torch.tensor(hidden_size, dtype=torch.float32))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scale = self.scale.to(self.device) # Put this to the device.

    def forward(self, h):

        #Move input to device
        h = h.to(self.device)

        Q = self.wq(h)
        K = self.wk(h)
        V = self.wv(h)

        Q, K, V = Q.to(self.device), K.to(self.device), V.to(self.device)

        attention_weights = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = attention_weights.to(self.device) #Move to device

        attended_values = torch.matmul(attention_weights, V)
        return attended_values


class Layer1(nn.Module):
    def __init__(self, input_size, hidden_size, device):  # Add device parameter
        super(Layer1, self).__init__()
        self.attn1 = nn.Linear(input_size, hidden_size).to(device)  # Move to device
        self.attn2 = nn.Linear(hidden_size, hidden_size).to(device) # Move to device
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        z1 = self.attn1(x)
        a1 = F.relu(z1, inplace=False)  # Add inplace=False
        z2 = self.attn2(a1)
        a2 = F.relu(z2, inplace=False)  # Add inplace=False
        return a2

class Layer2(nn.Module):
    def __init__(self, input_size, hidden_size, device):  # Add device parameter
        super(Layer2, self).__init__()
        self.attn1 = nn.Linear(input_size, hidden_size).to(device)  # Move to device
        self.attn2 = nn.Linear(hidden_size, hidden_size).to(device)  # Move to device
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        z1 = self.attn1(x)
        a1 = F.relu(z1, inplace=False) # Add inplace=False
        z2 = self.attn2(a1)
        a2 = F.relu(z2, inplace=False) # Add inplace=False
        return a2

class DAS(nn.Module):
    def __init__(self, feature_size, hidden_size, time_step, drop_ratio, device): # input_size here!
        super(DAS, self).__init__()
        # Correctly calculate input_size based on whether embeddings are used.
        input_size = 1 + 1  # ys and cis
        if config['model'].get('embedding_dim') is not None:
            input_size += config['model']['embedding_dim']  # Add embedding dimension
        self.layer1 = Layer1(input_size, hidden_size, device)  # Pass correct input_size
        self.layer2 = Layer2(time_step, hidden_size, device)
        self.drop = nn.Dropout(drop_ratio)
        self.device = device

    def forward(self, ems, var_y, cis):
        # Ensure ems, var_y, cis have correct shapes.
        if ems is not None:
            ems = ems.to(self.device)
        var_y = var_y.to(self.device)
        cis = cis.to(self.device)

        if ems is not None:
          var_x = torch.cat((ems, var_y, cis.unsqueeze(-1)), dim=2)
        else:
          var_x = torch.cat((var_y, cis.unsqueeze(-1)), dim=2)
        #print(f"var_x shape: {var_x.shape}") # Debugging print statement

        var_x = var_x.to(self.device)
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
        # Correctly calculate input_size here
        input_size = 1 + 1  # ys and cis
        if config['model'].get('embedding_dim') is not None:
            input_size += config['model']['embedding_dim']  # Add embedding dimension
        for _ in range(num_features):
            self.das.append(DAS(input_size, hidden_size, time_step, drop_ratio, self.device))

    def forward(self, var):
        out = self.das[0](var[0]['ems'], var[0]['ys'], var[0]['cis'])
        for i in range(1, len(var)):
            out += self.das[i](var[i]['ems'], var[i]['ys'], var[i]['cis'])
        return out


class output_layer(nn.Module):
    def __init__(self, last_hidden_size, output_size):
        super(output_layer, self).__init__()
        self.out_layer = nn.Linear(last_hidden_size, output_size)
        # self.sigmoid = nn.Sigmoid()  # REMOVED Sigmoid

    def forward(self, x):
        out = self.out_layer(x)
        return out