# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from attentionlayer import Attention

from sklearn.manifold import TSNE
import numpy as np

class Capsule_Att(nn.Module):
    def __init__(self, dim_vector, final_dropout_rate, use_cuda=True):
        super(Capsule_Att, self).__init__()
        self.dim_vector = dim_vector
        self.add_module('linear_prob', nn.Linear(dim_vector, 1))
        self.add_module('final_dropout', nn.Dropout(final_dropout_rate))
        self.add_module('attention_layer', Attention(attention_size=dim_vector, return_attention=True, use_cuda=use_cuda))

    def forward(self, vect_instance, matrix_hidden_pad, len_hidden_pad=None):
        r_s, attention = self.attention_layer(matrix_hidden_pad, torch.LongTensor(len_hidden_pad))
        prob = F.sigmoid(self.linear_prob(self.final_dropout(r_s)))
        r_s_prob = prob * r_s
        return prob, r_s_prob

def _squash(input_tensor, dim=-1):
    squared_norm = (input_tensor ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * input_tensor / torch.sqrt(squared_norm)

def _leaky_softmax(logits):
    """
    Args:
        logits: The original logits with shape 
            [batch_size, input_num, output_num] if fully connected.
        output_num: The number of units in the second dimmension of logits.
    Returns:
        Routing probabilities for each pair of capsules. Same shape as logits.
    """
    output_num = logits.size(2)
    leak = torch.zeros_like(logits)
    leak = torch.sum(leak,dim=2,keepdim=True)
    leaky_logits = torch.cat((leak, logits),dim=2)
    leaky_softmax = F.softmax(leaky_logits, dim=2)
    return torch.split(leaky_softmax, [1, output_num], dim=2)[1]

class Capsule(nn.Module):

    def __init__(self, input_dim, output_num, output_dim, iters, leaky, use_cuda):
        """
        Args:
            input_dim: scalar, dimensions of each capsule in the input layer.
            input_num: scalar, number of capsules in the output layer.
            input_dim: scalar, dimensions of each capsule in the output layer.
            iters: scalar, number of iterations
            leaky: boolean, if set use leaky routing
            use_cuda: boolean, whether use cuda
        """

        super(Capsule, self).__init__()
        self.input_dim = input_dim
        self.output_num = output_num
        self.output_dim = output_dim
        self.iters = iters
        self.leaky = leaky
        self.weight = nn.Parameter(torch.randn(input_dim, output_num*output_dim))
        self.biases = nn.Parameter(torch.randn(1, output_num, output_dim))
        self.use_cuda = use_cuda

        self.add_module('batch_norm', nn.BatchNorm1d(output_num, momentum=0.5))
        # self.add_module('layer_norm', nn.LayerNorm(output_dim))

    def _update_routing(self, votes, verbose=False):
        """
        Args:
            votes: tensor, The transformed outputs of the layer below with shape [batch_size, input_num, output_num, output_dim]

        Returns:
            activation: The activation tensor of the output layer after num_routing iterations with shape [batch_size, output_num, output_dim]
            logits: logits of shape [batch_size, input_num, output_num]
        """
        batch_size = votes.size(0)
        input_num = votes.size(1)
        logits = torch.zeros(batch_size, input_num, self.output_num)
        if self.use_cuda:
            logits = logits.cuda()
        votes_trans = votes.permute(0, 2, 3, 1)

        for i in range(self.iters):
            if self.leaky:
                logits = _leaky_softmax(logits)
            else:
                logits = F.softmax(logits, dim=2)
            preactive = logits.unsqueeze(-1) * votes
            preactive = torch.sum(preactive, dim=1) + self.biases
            preactive = self.batch_norm(preactive)
            activation = _squash(preactive) 
            # activation => [batch_size, output_num, output_dim]
            activation_trans = activation.unsqueeze(-1) 
            dist = torch.sum(activation_trans * votes_trans, dim=2).permute(0, 2, 1) 
            logits_before = logits
            logits = dist
        
        return activation, logits_before


    def forward(self, x):
        """
        Args:
            x: tensor, input capsules with shape [batch_size, input_num, input_dim]
        
        Returns:
            The activation tensor of the output layer after num_routing iterations with shape [batch_size, output_num, output_dim]
        """
        batch_size = x.size(0)
        votes = torch.matmul(x, self.weight).view(batch_size, -1, self.output_num, self.output_dim) + self.biases
        activation, logits = self._update_routing(votes)
        return activation, logits

    def get_weights(self):
        return self.weight

if __name__ == '__main__':
    batch_size = 3
    input_num, input_dim, output_num, output_dim = 5,2,2,32
    capslayer = Capsule(input_dim, output_num, output_dim, iters=3, leaky=True, use_cuda=False)
    x = torch.randn(batch_size, input_num, input_dim)
    y = capslayer(x)
    # print(y)
    # caps_output = _squash(x)
    # print(caps_output.shape)
    print(torch.sqrt((y ** 2).sum(dim=-1)))
