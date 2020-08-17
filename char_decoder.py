#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
DEBUG = 0

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.

        super(CharDecoder,self).__init__()
        self.charDecoder = nn.LSTM(input_size=char_embedding_size, hidden_size = hidden_size, num_layers=1)
        self.vocab_len = len(target_vocab.char2id)
        self.pad_token = target_vocab.char2id['<pad>']
        self.hidden_size = hidden_size
        self.char_embedding_size = char_embedding_size

        self.char_output_projection = nn.Linear(hidden_size, self.vocab_len)
        self.decoderCharEmb = nn.Embedding(self.vocab_len, char_embedding_size, padding_idx = self.pad_token)
        self.target_vocab = target_vocab

        ### END YOUR CODE



    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        length, batch_size = input.size()
        x = self.decoderCharEmb(input) # do I need to do packing and unpacking?
        if dec_hidden:
            h_0, c_0 = dec_hidden
        else:
            h_0 = torch.zeros((1, batch_size, self.hidden_size))
            c_0 = torch.zeros((1,batch_size, self.hidden_size))

        o_t, (h_t, c_t) = self.charDecoder(x, (h_0, c_0))
        s_t = self.char_output_projection(o_t)

        dec_hidden = (h_t, c_t)
        return s_t, dec_hidden
        ### END YOUR CODE


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch, for every character in the sequence.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>)

        length, batch = char_sequence.size()
        #target = torch.zeros((length-1,batch))
        input_decoder = char_sequence[:-1,:]
        s_t, (dec_hidden) = self.forward(input_decoder, dec_hidden)
        s_t = s_t.reshape(-1,self.vocab_len)
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_token, reduction='sum')
        target = char_sequence[1:,:].reshape(-1)
        loss = loss_fn(s_t, target)
        if DEBUG:
            dd = torch.nn.functional.softmax(s_t, dim=1)
            dd = torch.log(dd)
            dd = torch.nn.functional.log_softmax(s_t, dim=1)
            index1  = list(enumerate(target.numpy()))
            predictions = torch.tensor([dd[i] for i in index1])
            non_pad_ind = [i for i,val in enumerate(target) if val != self.pad_token]
            non_pad_items = predictions[non_pad_ind]
            loss_debug = torch.mean(non_pad_items)
        return loss
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.


        _, batch_size, hidden_size = initialStates[0].size()
        start_token_id = self.target_vocab.char2id['{']
        end_token = '}'
        end_token_id = self.target_vocab.char2id[end_token]

        inpt = torch.empty((1, batch_size), device=device, dtype=torch.long).fill_(start_token_id)

        h_t, c_t = initialStates
        out_word = ["" for i in range(batch_size)]
        for i in range(max_length):
            s_t, (h_t1, c_t1) = self.forward(inpt, (h_t,c_t))
            p_t1 = torch.nn.functional.softmax(s_t,dim=-1)
            pred_chars = torch.argmax(p_t1,dim=-1)
            assert(len(pred_chars)==1)
            for i_b, b_val in enumerate(pred_chars[0]):
                out_word[i_b] += self.target_vocab.id2char[b_val.item()]
            h_t, c_t = h_t1, c_t1
            inpt = pred_chars #torch.tensor(pred_chars, device=device, dtype=torch.long)

        ### END YOUR CODE
        # post_process.
        def clean_words(tmp_words):
            """ for each batch purge the list once end token is seen
            @returns purged out_word
            """
            new_words = []
            for i, word in enumerate(tmp_words):
                ind = word.find(end_token)
                new_words.append(word[:ind])
            return new_words

        new_out_word = clean_words(out_word)
        return new_out_word
