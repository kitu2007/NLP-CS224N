#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Highway

Usage:
x_highway.py

"""
from docopt import docopt
import torch
import torch.nn.functional as F
import numpy as np
import sys

### YOUR CODE HERE for part 1d

class Highway(torch.nn.Module):

    def __init__(self, D_in, non_lin=F.relu):
        """
        Write a two layer net
        D_in: output of the conv
        """
        super(Highway, self).__init__()
        self.wproj = torch.nn.Linear(D_in, D_in, bias=True)
        self.wgate = torch.nn.Linear(D_in, D_in, bias=True)
        self.proj_nonlin = non_lin
        self.gate_nonlin = torch.nn.Sigmoid()

    def forward(self, x):
        """
        x is batched input from x_conv_out (batch_size, max_sentence_length, max_word_length)
        """
        x_proj = self.proj_nonlin(self.wproj(x))
        x_gate = self.gate_nonlin(self.wgate(x))
        x_highway = x_gate * x_proj + (1-x_gate)* x # is * the hammard product between tensors
        return x_highway

### END YOUR CODE



def test_forward():
    dtype = torch.float
    device = torch.device("cpu")
    batch_size = 3
    max_sentence_length = 6
    word_embedding = 5
    highway_net = Highway(word_embedding)
    input_shape = (batch_size, max_sentence_length, word_embedding)
    x = torch.randn(input_shape, device = device, dtype=dtype)
    assert(x.shape == input_shape), "shape aren't same"

    # manual define wproj and wgate and input and know expected output.
    # not doing that.
    y = highway_net.forward(x)

    assert y.shape == (batch_size, max_sentence_length, word_embedding), "shape mismatch. expected:{} output:{}".format((batch_size, max_sentence_length, word_embedding), y.shape)

def main():

    assert( sys.version_info >=(3,5)), "Please update your installation of Python to version >= 3.5"
    assert( torch.__version__ >= '1.0.0'), "Please update your installation of Pytorch. you have {} and we need 1.0.0 or greater".format(torch.__version__)

    # Seed the random number Generators
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13//7)
    test_forward()

if __name__ == '__main__':
    args = docopt(__doc__)
    main()
