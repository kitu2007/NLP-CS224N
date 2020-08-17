#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1e

import torch
import torch.nn.functional as F
import numpy as np


class CNN(torch.nn.Module):
    def __init__(self, e_char, e_word, kernel_size=5):
        """
        e_char: Size of the input
        e_word: Number of features which is also the dimension of word_embedsing.
        kernel_size: Size of the kernel that is applied to the character
        """

        #initialize the super class
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=e_char, out_channels=e_word,
                                     kernel_size=kernel_size)

    def forward(self, x):
        """
        x is x_reshaped where x (batch_size, e_char, m_word)
        """
        # x.shape == (e_char, m_word)
        x = self.conv1(x) # x output is e_word * (m_word-k+1)
        x = F.relu(x)
        if 0:
            kernel_size = x.shape[-1]
            x = F.max_pool1d(x, kernel_size)
            x = torch.squeeze(x)
        else:
            x = torch.max(x, dim=2)[0] # x is e_word
        return x

### END YOUR CODE


def test_forward():
    dtype = torch.float
    device = torch.device("cpu")

    batch_size = 2
    m_word = 7
    e_char = 5
    e_word = 11
    k = 3

    charNet = CNN(e_char, e_word)
    x = torch.randn((batch_size, e_char, m_word), dtype=dtype, device=device)
    y = charNet.forward(x)
    y_shape_expected = (batch_size, e_word)
    assert (y.shape == y_shape_expected), "shape mismatch"


def main():
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13//7)
    test_forward()




if __name__ == "__main__":
    main()
