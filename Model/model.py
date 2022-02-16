from vocab import EOS_token, Vocab, load_vocab, tensor_to_sentence

import time
import math

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
debug = False

# input size = sentence embedding size
# hidden size = hyper-parameter
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, input, hidden):
        # embedded = self.embedding(input).view(1, 1, -1)
        # output = embedded
        # output, hidden = self.gru(output, hidden)
        if debug: print(f'input : {input.shape}')
        if debug: print(f'hidden : {hidden.shape}')
        output, hidden = self.gru(input, hidden)
        
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# REMEMBER TO CHANGE MAX LENGTH TO APPROPRIATE VALUE (BIGGEST SUMMARY)
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=10):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_hidden):

        if debug: print(f'input: {input.shape}')
        if debug: print(f'hidden: {hidden.shape}')
        if debug: print(f'encoder_hidden {encoder_hidden.shape}')

        ####################################################
        # REPLACE WITH WORD2VEC LOOKUP
        
        embedded = self.embedding(input)


        ####################################################
        if debug: print(f'input word embedding: {embedded.shape}')

        embedded = self.dropout(embedded)

        if debug: print(f'embedded[:,0] : {embedded[:,0].shape}')
        if debug: print(f'hidden[:,0] : {hidden[0,:].shape}')
        if debug: print(f'torch.cat((embedded[:,0], hidden[:,0]), 1) : {torch.cat((embedded[:,0], hidden[0]), 1).shape}')
        if debug: print(f'self.attn : {self.attn}')

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[:,0], hidden[0]), 1)), dim=1).unsqueeze(1)
        
        if debug: print(f'attn weights: {attn_weights.shape}')
        if debug: print(f'encoder_hidden: {encoder_hidden.shape}')

        attn_applied = torch.bmm(attn_weights,
                                 encoder_hidden)



        if debug: print(f'embedded : {embedded.shape}')
        if debug: print(f'embedded[:,0] : {embedded[:,0].shape}')
        if debug: print(f'attn_applied : {attn_applied.shape}')
        if debug: print(f'attn_applied[:,0] : {attn_applied[:,0].shape}')

        output = torch.cat((embedded[:,0], attn_applied[:,0]), 1)
        if debug: print(f'output : {output.shape}')
        output = self.attn_combine(output).unsqueeze(1)
        if debug: print(f'output : {output.shape}')

        output = F.relu(output)
        if debug: print(f'output : {output.shape}')
        if debug: print(f'hidden : {hidden.shape}')

        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[:,0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

if __name__=="__main__":
    sent_embedding_size = 20

    # MAKE THIS SAME AS WORD EMBEDDING SIZE FOR NOW
    # IF WANT DIFFERENT REFACTOR DECODER INIT
    hidden_size = 30

    vocab_size = 100
    num_sents = 5

    sentence_embeddings = torch.rand((num_sents, sent_embedding_size))

    encoder = EncoderRNN(sent_embedding_size, hidden_size)

    ##################################################################################

    summ_vcb = load_vocab('DialogSum_Data/summary.vcb')
    output_size = summ_vcb.n_words

    decoder = AttnDecoderRNN(hidden_size, output_size, dropout_p=0.1).to(device)
    decoder_hidden = encoder_hidden



    hidden_size = 256
    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

    trainIters(encoder, attn_decoder, 75000, print_every=5000)


if __name__=="__main__":
    sent_embedding_size = 20

    # MAKE THIS SAME AS WORD EMBEDDING SIZE FOR NOW
    # IF WANT DIFFERENT REFACTOR DECODER INIT
    hidden_size = 30
    vocab_size = 100
    num_sents = 5

    sentence_embeddings = torch.rand((num_sents, sent_embedding_size))

    encoder = EncoderRNN(sent_embedding_size, hidden_size)

    ##################################################################################

    summ_vcb = load_vocab('DialogSum_Data/summary.vcb')
    vocab_size = len(summ_vcb.index2word)
    output_size = summ_vcb.n_words

    decoder = AttnDecoderRNN(hidden_size, output_size, dropout_p=0.1).to(device)
    decoder_hidden = encoder_hidden



    hidden_size = 256
    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

    trainIters(encoder, attn_decoder, 75000, print_every=5000)
