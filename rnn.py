# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from capsule import Capsule

class CRUCell(nn.Module):
    """
        Capsule-based Recurrent Unit
    """

    def __init__(self, input_caps, hidden_caps, caps_dim, biases=True, iters=3, leaky=True, use_cuda=True):
        super(CRUCell, self).__init__()
        self.caps_dim = caps_dim
        self.input_caps = input_caps
        self.hidden_caps = hidden_caps

        self.add_module('capsule', Capsule(caps_dim, self.hidden_caps, caps_dim, iters=iters, leaky=leaky, use_cuda=use_cuda))

    def check_forward_input(self, input):
        if input.size(1) != self.input_caps:
            raise RuntimeError(
                "input has inconsistent input_caps: got {}, expected {}".format(
                    input.size(1), self.input_caps))
        if input.size(2) != self.caps_dim:
            raise RuntimeError(
                "input has inconsistent caps_dim: got {}, expected {}".format(
                    input.size(2), self.caps_dim))

    def check_forward_hidden(self, input, hx, hidden_label=''):
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.hidden_caps:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_caps: got {}, expected {}".format(hidden_label,
                    hx.size(1), self.hidden_caps))
        if hx.size(2) != self.caps_dim:
            raise RuntimeError(
                "hidden{} has inconsistent caps_dim: got {}, expected {}".format(hidden_label,
                    hx.size(2), self.caps_dim))
        
    def forward(self, input, hx=None):
        """
        Args:
            input: tensor, input capsules with shape [batch_size, input_caps, caps_dim]
            hx: tensor, hidden capsules with shape [batch_size, hidden_caps, caps_dim]
        
        Returns:
            hidden_new: tensor, new hidden capsules with shape [batch_size, hidden_caps, caps_dim]
        """
        self.check_forward_input(input)
        if hx is None:
            hx = input.new_zeros([input.size(0), self.hidden_caps, self.caps_dim], requires_grad=False)
        self.check_forward_hidden(input, hx)
        capsule_input = torch.cat([hx, input], 1)
        h, _ = self.capsule(capsule_input)
        # print('hx', hx[0])
        # raw_input()
        # print('input', input[0])
        # raw_input()
        # print('h', h[0])
        # raw_input()
        h = F.tanh(h)
        return h


class CRU(nn.Module):
    """
        CRU-based RNN

        Args:
            input_size: scalar, last dimension of input
            hidden_size: scalar, same as that in rnn
            caps_dim: scalar, length of each capsule in the CRU cell
            biases: boolean, whether to use biases
            bidirectional: boolean, if true then the sequence length of final output is 2 * seq_len
            use_cuda: boolean, whether to use use_cuda 
    """
    def __init__(self, input_size, hidden_size, caps_dim, biases, bidirectional, use_cuda):
        super(CRU, self).__init__()
        self.check_size_match(hidden_size, caps_dim)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.caps_dim = caps_dim

        self.input_caps = int(hidden_size / caps_dim)
        self.hidden_caps = int(hidden_size / caps_dim)
        self.use_cuda = use_cuda

        self.add_module('trans_input', nn.Linear(self.input_size, self.hidden_size))
        self.add_module('crucell', CRUCell(self.input_caps, self.hidden_caps, caps_dim, biases=biases, use_cuda=use_cuda))

    def check_size_match(self, hidden_size, caps_dim):
        if hidden_size % caps_dim != 0:
            raise RuntimeError(
                "CRU size dosn't match: hidden_size {} cannot be divided by caps_dim {}".format(
                    hidden_size, caps_dim))

    def init_hidden(self, batch_size):
        h0 = torch.zeros(batch_size, self.hidden_caps, self.caps_dim, requires_grad=False)
        if self.use_cuda:
            h0 = h0.cuda()
        return h0

    def forward(self, input, h0=None):
        """
        Args:
            input: tensor, rnn input with shape [batch_size, seq_len, input_size]
            h0: tensor, init state of shape [batch_size, hidden_caps, caps_dim]
        Return:
            output: tensor, containing the output features h_t from the last layer of the CRU, of shape [batch_size, seq_len, hidden_size]
            hn: tensor containing the hidden state for `t = seq_len`, of shape [batch_size, 1, hidden_size] 
        """
        batch_size = input.size(0)
        seq_len = input.size(1)
        if h0 is None:
            hx = self.init_hidden(batch_size)
        else:
            hx = h0
        input = self.trans_input(input)
        input = input.permute(1,0,2)
        input = input.view(seq_len, batch_size, self.input_caps, self.caps_dim)
        output = []
        for i in range(seq_len):
            cur_input = input[i].squeeze(0)
            hx = self.crucell(cur_input, hx)
            output.append(hx.unsqueeze(0))
        output = torch.cat(output, dim=0).permute(1,0,2,3)
        # output -> [batch_size, seq_len, hidden_caps, caps_dim]
        output = output.view(batch_size, seq_len, -1)
        return output, hx

if __name__ == '__main__':
    rnn = CRU(10, 20, 5, biases=True, bidirectional=False, use_cuda=False)
    input = torch.randn(5, 3, 10)
    output, hn = rnn(input)
    print(output)