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
from gensim.models import FastText
import json


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
debug = False

word_embedding_size = 300

summary_vcb = load_vocab('DialogSum_Data/summary.vcb')

# FastText embedding
train_path = 'DialogSum_Data/dialogsum.train.tok.jsonl'
dialogues = []
summaries = []

with open(train_path, 'rb') as f:
    for line in f:
        d = json.loads(line)
        dialogues.append(d['dialogue'])
        summaries.append(d['summary'])

summaries_sentences = [d.split(' ') for d in summaries ]
fast_text_model = FastText(sentences=summaries_sentences, vector_size=word_embedding_size, window=3, min_count=1, epochs=10)


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
        
        #embedded = self.embedding(input)

        embedded = torch.tensor([fast_text_model.wv[summary_vcb.index2word.get(int(word), 2)] for word in input]).unsqueeze(1)
        

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

