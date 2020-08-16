#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import ipdb

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(f)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, embed_size, vocab, char_embed_size=50):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        self.embed_size = embed_size
        self.char_embed_size = char_embed_size
        self.pad_token_idx = vocab.char2id['<pad>']
        self.char_embeddings = nn.Embedding(len(vocab.char2id), self.char_embed_size, padding_idx = self.pad_token_idx)
        self.cnn = CNN(self.char_embed_size, self.embed_size)
        self.highway = Highway(self.embed_size)
        self.dropout = nn.Dropout(p=0.3)
        ### END YOUR CODE

    def forward(self, input_tensor):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input_tensor: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code
        ### YOUR CODE HERE for part 1f
        sentence_length, batch_size, max_word_length = input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2]
        x = self.char_embeddings(input_tensor) # (sentence_length, batch_size, max_word_len, char_embed_size)
        x = x.permute(0,1,3,2) # change it to sentence_length, batch_size, char_embed_size, max_word_length

        x = x.reshape(-1, self.char_embed_size, max_word_length) # sentence_len* batch_size, char_embed_size, max_word_length
        # Test x.shape == (char_embed_size, m_word)
        x_conv = self.cnn(x)
        x_highway = self.highway(x_conv)
        x_highway = x_highway.reshape(sentence_length, batch_size, -1) # undo the shape unwrapping the same way you did. x_highway is supposed to be sentence first, batch second.
        x_word_embed = self.dropout(x_highway)
        return x_word_embed
        ### END YOUR CODE
