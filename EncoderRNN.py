# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from capsule import Capsule, Capsule_Att
from rnn import CRU

class EncoderRNN(nn.Module):
    def __init__(self,
            dim_input,
            dim_hidden,
            dim_label,
            dim_caps,
            n_layers,
            n_label,
            n_vocab,
            embed_dropout_rate,
            cell_dropout_rate,
            final_dropout_rate,
            embed_list,
            bidirectional,
            rnn_type,
            use_cuda,
            model_name):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_caps = dim_caps
        self.n_label = n_label
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.use_cuda = use_cuda
        self.model_name = model_name

        self.add_module('embed', nn.Embedding(n_vocab, dim_input))
        self.add_module('embed_dropout', nn.Dropout(embed_dropout_rate))
        self.add_module('rnn', getattr(nn, self.rnn_type)(dim_input, dim_hidden, n_layers, batch_first=True, dropout=cell_dropout_rate, bidirectional=bidirectional,))     
        
        if self.model_name == 'Attention':
            for i in range(self.n_label):
                self.add_module('capsule_%s' % i, Capsule_Att(dim_hidden * (2 if self.bidirectional else 1), final_dropout_rate, self.use_cuda))
        else:
            dim_middle = 64
            # self.add_module('capslayer', Capsule(dim_hidden * (2 if self.bidirectional else 1), n_label, dim_label, iters=3, leaky=True, use_cuda=self.use_cuda))
            self.add_module('capslayer_0', Capsule(dim_hidden * (2 if self.bidirectional else 1), 8, dim_middle, iters=3, leaky=True, use_cuda=self.use_cuda))
            self.add_module('capslayer_1', Capsule(dim_middle, n_label, dim_label, iters=3, leaky=True, use_cuda=self.use_cuda))

        self.add_module('reconstruct', nn.Linear(dim_label, dim_hidden * (2 if self.bidirectional else 1)))
        # self.add_module('reconstruct', nn.Linear(dim_label, dim_hidden))

        self.init_weights(embed_list)
        ignored_params = list(map(id, self.embed.parameters()))
        self.base_params = filter(lambda p: id(p) not in ignored_params,
                     self.parameters())

    def init_weights(self, embed_list):
        self.embed.weight.data.copy_(torch.from_numpy(embed_list))

    def forward(self, input, hidden, lengths, tensor_label):
        embedded = self.embed(input)
        embedded = self.embed_dropout(embedded)
        input_packed = pack_padded_sequence(embedded, lengths=lengths, batch_first=True)
        output, hidden = self.rnn(input_packed, hidden)
        output_pad, output_len = pad_packed_sequence(output, batch_first=True)

        # output_pad [batch_size, input_num, input_dim]

        variable_len = Variable(torch.from_numpy(1.0/lengths.astype(np.float32))).unsqueeze(-1)
        v_s = torch.sum(output_pad, 1) * (variable_len.cuda() if self.use_cuda else variable_len)

        if self.model_name == 'Attention':
            list_prob, list_r_s = [], []
            for i in range(self.n_label):
                prob_tmp, r_s_tmp = getattr(self, 'capsule_%s' % i)(v_s, output_pad, torch.LongTensor(output_len))
                list_prob.append(prob_tmp)
                list_r_s.append(r_s_tmp)

            list_r_s = torch.stack(list_r_s)
            list_sim = torch.sum(v_s*list_r_s, 2).t()
            prob = torch.stack(list_prob).squeeze(-1).t()

            # list_sim with shape [batch_size, output_num]
            # prob with shape [batch_size, output_num]

            return list_sim, prob
        else:
            # caps_output, logits = self.capslayer(output_pad)
            caps_mid_output, logits_middle = self.capslayer_0(output_pad)
            caps_output, logits = self.capslayer_1(caps_mid_output)
            # caps_output: [batch_size, output_num, output_dim]
            # logits: [batch_size, input_num, output_num]q
            prob = F.softmax(torch.sqrt((caps_output ** 2).sum(dim=-1)), dim=-1)
            caps_rec = self.reconstruct(caps_output)
            v_s_stack = torch.stack([v_s] * self.n_label, dim=1)
            list_sim = torch.sum(caps_rec * v_s_stack, dim=-1)
            
            return list_sim, prob, {'middle': logits_middle, 'output': logits}


    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        h_0 = Variable(weight.new(self.n_layers * (2 if self.bidirectional else 1), batch_size, self.dim_hidden).zero_(), requires_grad=False)
        h_0 = h_0.cuda() if self.use_cuda else h_0
        return (h_0, h_0) if self.rnn_type == "LSTM" else h_0