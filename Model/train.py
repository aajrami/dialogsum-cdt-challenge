from vocab import EOS_token, Vocab, load_vocab, tensor_to_sentence
from model import EncoderRNN, AttnDecoderRNN
from summary_dataset import SummaryDataset

import time
import math
import os.path as op
import random

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
MAX_LENGTH = 10
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
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(encoder, decoder, train_dataset, num_epochs, print_every=500, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    order = np.random.shuffle(np.arange(len(train_dataset)))
    for epoch in range(1, num_epochs+1):
        for iter, i in enumerate(order):
            training_pair = train_dataset[i]
            input_tensor = training_pair['source']
            target_tensor = training_pair['target']

            loss = train(input_tensor, target_tensor, encoder,
                        decoder, encoder_optimizer, decoder_optimizer, criterion)
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
    trainIters(encoder, attn_decoder, train_dataset, num_epochs=3, print_every=500)