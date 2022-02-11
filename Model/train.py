from vocab import EOS_token, Vocab, load_vocab, tensor_to_sentence
from model import EncoderRNN, AttnDecoderRNN
from summary_dataset import SummaryDataset

import time
import math
import os.path as op
import random
import json

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 100
teacher_forcing_ratio = 0.5

DATA_DIR = "DialogSum_Data"

SAMPLE_DATA = op.join(DATA_DIR, 'dialogsum.sample.jsonl')
TEST_DATA = op.join(DATA_DIR, 'dialogsum.test.jsonl')
TRAIN_DATA = op.join(DATA_DIR, 'dialogsum.train.jsonl')
DEV_DATA = op.join(DATA_DIR, 'dialogsum.dev.jsonl')

## HELPER FUNCTIONS
def load_jsonl(filepath):
    
    output_list = []
    
    with open(filepath) as sd_file:
        lines = sd_file.readlines()
    
    for line in lines:
        output_list.append(json.loads(line))

    return output_list

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

## TRAINING FUNCTIONS
def train(input_tensor, output_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, batch_size=1, max_length=MAX_LENGTH, debug=False):
    batch_encoder_hidden = encoder.initHidden()

    if debug: print('input tensor: {}'.format(input_tensor.shape))
    if debug: print('output tensor: {}'.format(output_tensor.shape))

    # ACCOUNT FOR BATCH SIZE BEING 1
    if batch_size==1:
        input_tensor = input_tensor.unsqueeze(0)

    if batch_size==1:
        output_tensor = output_tensor.unsqueeze(0)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(1)
    output_length = output_tensor.size(1)

    if debug: print('input tensor: {}'.format(input_tensor.shape))
    if debug: print('output tensor: {}'.format(output_tensor.shape))

    if debug: print('input_length: {}'.format(input_length))
    if debug: print('output_length: {}'.format(output_length))

    batch_encoder_outputs = torch.zeros(batch_size, max_length, encoder.hidden_size, device=device)

    if debug: print('encoder outputs: {}'.format(batch_encoder_outputs.shape))

    loss = 0

    for ei in range(input_length):
        batch_encoder_inputs = input_tensor[:,ei,:].view(batch_size, 1, -1)
        if debug: print(f'batch_encoder_inputs: {batch_encoder_inputs.shape}')
        batch_encoder_output, batch_encoder_hidden = encoder(
            batch_encoder_inputs, batch_encoder_hidden)
        if debug: print(f'encoder_output: {batch_encoder_output.shape}')
        if debug: print(f'encoder_hidden: {batch_encoder_hidden.shape}')
        batch_encoder_outputs[:,ei,:] = batch_encoder_output[0, 0]

    if debug: print(f'batch_encoder_hidden {batch_encoder_hidden.shape}')
    decoder_input = torch.tensor([[SOS_token]], device=device)

    batch_decoder_hidden = batch_encoder_hidden

    if debug: print('decoder hidden: {}'.format(batch_decoder_hidden.shape))
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # CHANGE BACK TO USE TEACHER FORCING
    if True:
        # Teacher forcing: Feed the target as the next input
        for di in range(output_length):
            if debug: print('decoder input: {}'.format(decoder_input.shape))
            if debug: print(f'decoder hidden {batch_decoder_hidden.shape}')
            if debug: print(f'encoder outputs {batch_encoder_outputs.shape}')
            
            decoder_output, batch_decoder_hidden, decoder_attention = decoder(
                decoder_input, batch_decoder_hidden, batch_encoder_outputs)
            loss += criterion(decoder_output, output_tensor[di])
            decoder_input = output_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(output_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, output_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / output_length


def trainIters(encoder, decoder, train_dataset, num_epochs, batch_size=1, print_every=500, plot_every=100, learning_rate=0.01, debug=False):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    order = np.arange(len(train_dataset))
    np.random.shuffle(order)
    for epoch in range(1, num_epochs+1):
        for iter, i in enumerate(order):
            training_pair = train_dataset[i]
            input_tensor = training_pair['source']
            output_tensor = training_pair['target']

            loss = train(input_tensor, output_tensor, encoder,
                        decoder, encoder_optimizer, decoder_optimizer, 
                        criterion, batch_size=batch_size, debug=debug)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, epoch*len(order) + iter / (epoch*len(order))),
                                            epoch*len(order) + iter, iter / (epoch*len(order)) * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

    showPlot(plot_losses)


## PLOTTING FUNCTIONS
def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


if __name__=="__main__":

    ##################################################################################
    dialogue_vcb = load_vocab('DialogSum_Data/dialogue.vcb')
    summary_vcb = load_vocab('DialogSum_Data/summary.vcb')
    vocab_size = len(summary_vcb.index2word)

    # Create Dataset clas
    train_data_list = load_jsonl(SAMPLE_DATA)
    train_dataset = SummaryDataset(train_data_list,
                                    sentence_transformers_model="all-MiniLM-L6-v2",
                                    debug=False)

    dev_data_list = load_jsonl(SAMPLE_DATA)
    dev_dataset = SummaryDataset(dev_data_list,
                                sentence_transformers_model="all-MiniLM-L6-v2",
                                debug=False)


    # Model parameters
    # MAKE HIDDEN SIZE SAME AS WORD EMBEDDING SIZE FOR NOW
    # IF WANT DIFFERENT REFACTOR DECODER INIT
    sent_embedding_size = train_dataset.source_embedding_dimension
    hidden_size = 30

    encoder = EncoderRNN(sent_embedding_size, hidden_size)
    attn_decoder = AttnDecoderRNN(hidden_size, summary_vcb.n_words, dropout_p=0.1).to(device)
    trainIters(encoder, attn_decoder, train_dataset, batch_size=1, num_epochs=3, print_every=500, debug=True)