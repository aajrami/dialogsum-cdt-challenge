from vocab import EOS_token, Vocab, load_vocab, tensor_to_sentence
from model import EncoderRNN, AttnDecoderRNN
from summary_dataset import SummaryDataset

import argparse
import copy
import time
import math
import os
import os.path as op
import random
import sys
import json
import shutil

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from gensim.models import FastText

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1

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
def train(input_tensor, output_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, batch_size=1, max_length=50, debug=False):
    
    batch_encoder_hidden = torch.cat(batch_size * [encoder.initHidden()], dim=1)
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(1)
    output_length = output_tensor.size(1)

    batch_encoder_hidden_states = torch.zeros(batch_size, max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):

        batch_encoder_inputs = input_tensor[:,ei,:].reshape(batch_size, 1, -1)

        batch_encoder_output, batch_encoder_hidden = encoder(
            batch_encoder_inputs, batch_encoder_hidden)
        batch_encoder_hidden_states[:,ei,:] = batch_encoder_output[0, 0]

    batch_decoder_input = torch.tensor([[SOS_token]], device=device)
    batch_decoder_input = torch.cat(batch_size * [batch_decoder_input], dim=0)

    batch_decoder_hidden = batch_encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(output_length):
            
            batch_decoder_output, batch_decoder_hidden, decoder_attention = decoder(
                batch_decoder_input, batch_decoder_hidden, batch_encoder_hidden_states)
 
            loss += criterion(batch_decoder_output, output_tensor[:,di].to(torch.long))
            batch_decoder_input = output_tensor[:,di].to(torch.long).unsqueeze(1)  # Teacher forcing

    else:

        eos_list = [False] * batch_size

        # Without teacher forcing: use its own predictions as the next input
        for di in range(output_length):
            batch_decoder_output, batch_decoder_hidden, decoder_attention = decoder(
                batch_decoder_input, batch_decoder_hidden, batch_encoder_hidden_states)
            topv, topi = batch_decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(batch_decoder_output, output_tensor[:,di].to(torch.long))
            for i, inp in enumerate(decoder_input):
                if inp.item() == EOS_token:
                    eos_list[i] = True
            if len(eos_list) == sum(eos_list): #If we have seen EOS for every item in the batch, then we break
                break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / output_length


# Decodes a batch
def decode(input_tensor, output_tensor, encoder, decoder, criterion, vocab, batch_size=1, max_length=50, debug=False):
    
    loss = 0

    input_length = input_tensor.size(1)
    output_length = output_tensor.size(1)

    batch_encoder_hidden = torch.cat(batch_size * [encoder.initHidden()], dim=1)
    batch_encoder_hidden_states = torch.zeros(batch_size, max_length, encoder.hidden_size, device=device)

    # encoding
    for ei in range(input_length):

        batch_encoder_inputs = input_tensor[:,ei,:].reshape(batch_size, 1, -1)

        batch_encoder_output, batch_encoder_hidden = encoder(
            batch_encoder_inputs, batch_encoder_hidden)

        batch_encoder_hidden_states[:,ei,:] = batch_encoder_output[0, 0]

    # decoding
    batch_decoder_input = torch.tensor([[SOS_token]], device=device)
    batch_decoder_input = torch.cat(batch_size * [batch_decoder_input], dim=0)
    batch_decoder_hidden = batch_encoder_hidden

    eos_list = [False] * batch_size

    # Without teacher forcing: use its own predictions as the next input
    outputs = []
    for di in range(output_length):
        batch_decoder_output, batch_decoder_hidden, decoder_attention = decoder(
            batch_decoder_input, batch_decoder_hidden, batch_encoder_hidden_states)
        topv, topi = batch_decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input
        outputs.append(topi)
        #print(f'batch_decoder_output : {batch_decoder_output.shape}')
        #print(f'outputs : {outputs}')
        #print(f'output_tensor : {output_tensor}')
        #print(f'output_tensor : {output_tensor.shape}')
        #print(f'output_tensor[:,di] {output_tensor[:,di].shape} ')
        loss += criterion(batch_decoder_output, output_tensor[:,di].to(torch.long))

        for i, inp in enumerate(decoder_input):
            if inp.item() == EOS_token:
                eos_list[i] = True
        if len(eos_list) == sum(eos_list): #If we have seen EOS for every item in the batch, then we break
            break

    outputs = torch.stack(outputs).swapaxes(0, 1)
    output_sentence = [tensor_to_sentence(outputs[i], vocab) for i in range(len(outputs))]
    

    return output_sentence, loss.item() / output_length


def collate_function(batch):
    
    dataset_index = torch.tensor([s["index"] for s in batch]) #To let us find the original text of dialogue and summary
    dialogue_batch   = [s["source"] for s in batch]
    summary_batch = [s["target"] for s in batch]    

    dialogue_lengths = torch.tensor([ len(dialogue) for dialogue in dialogue_batch ]).to(device)
    summary_lengths = torch.tensor([ len(summary) for summary in summary_batch ]).to(device)    

    dialogues_padded =   [   torch.stack([torch.ones_like(dialogue_batch[0][0])]   * max(dialogue_lengths)) for dialogue in dialogue_batch]
    summaries_padded  =   [ torch.tensor([1] * (max(summary_lengths) + 2)) for summary in summary_batch] 


    dialogues_padded  = torch.stack(dialogues_padded).to(device)
    summaries_padded = torch.stack(summaries_padded).to(device)
    
    dialogues_padded *= 3   #We set the pad symbol to 3, same as in the vocab
    summaries_padded *= 3
    
    for i, dialogue in enumerate(dialogue_batch):
        dialogues_padded[i][:len(dialogue)] = dialogue
    
    for i, summary in enumerate(summary_batch):
        summaries_padded[i][0] = SOS_token                    #SOS_token
        summaries_padded[i][1:len(summary)+1] = summary
        summaries_padded[i][len(summary)+1] = EOS_token               #EOS_token
    

    return { "source": dialogues_padded, "target": summaries_padded, "dataset_index": dataset_index}



def get_all_predictions(encoder, decoder, vocab, criterion, dataset, batch_size=4, max_length=50):

    dev_loss = 0

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_function)

    predictions = []

    for batch in dataloader:
        dataset_indices = batch["dataset_index"]
        input_tensor = batch["source"]
        output_tensor = batch["target"]
        outputs, loss = decode(input_tensor, output_tensor, encoder, decoder, criterion, vocab, batch_size=batch_size, max_length=max_length)
        pred_summaries = [" ".join(decoded_sent) for decoded_sent in outputs]
        dev_loss += loss

        dialogues = [dataset[idx.item()]["dialogue_text"] for idx in dataset_indices]
        gold_summaries = [dataset[idx.item()]["summary_text"] for idx in dataset_indices]
        for i in range(len(dataset_indices)):
            pred_dict = {}
            pred_dict["dialogue"] = dialogues[i]
            pred_dict["gold_summary"] = gold_summaries[i]
            pred_dict["predicted_summary"] = pred_summaries[i]
            predictions.append(pred_dict)

    predictions_df = pd.DataFrame(predictions)

    return(predictions_df), dev_loss


def get_all_predictions_sanity_check(encoder, decoder, vocab, criterion, dataset, batch_size=4):

    dev_loss = 0

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_function)

    predictions = []

    batches = dataloader.__iter__()
    batch = next(batches)
    
    for _ in dataloader:

        dataset_indices = batch["dataset_index"]
        input_tensor = batch["source"]
        output_tensor = batch["target"]
        outputs = decode(input_tensor, output_tensor, encoder, decoder, criterion, vocab, batch_size=batch_size)
        pred_summaries = [" ".join(decoded_sent) for decoded_sent in outputs]
        dialogues = [dataset[idx.item()]["dialogue_text"] for idx in dataset_indices]
        gold_summaries = [dataset[idx.item()]["summary_text"] for idx in dataset_indices]
        for i in range(len(dataset_indices)):
            pred_dict = {}
            pred_dict["dialogue"] = dialogues[i]
            pred_dict["gold_summary"] = gold_summaries[i]
            pred_dict["predicted_summary"] = pred_summaries[i]
            predictions.append(pred_dict)

    predictions_df = pd.DataFrame(predictions)

    return(predictions_df), dev_loss




def trainIters_sanity_check(encoder, decoder, train_dataset, dev_dataset, max_epochs, vocab=None, batch_size=1, print_every=500, plot_every=100, learning_rate=0.01, debug=False):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=3)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_function)

    os.makedirs(op.join("experiments", experiment_name), exist_ok=True)

    for epoch in range(1, max_epochs+1):
        print(f"starting epoch {epoch}")
    

        training_batches = train_loader.__iter__()
        batch_0 = next(training_batches)

        for iter, _ in enumerate(train_loader):
            print(f"training batch {iter} of {len(train_loader)}")
            training_batch = batch_0
    
            input_tensor = training_batch['source']
            output_tensor = training_batch['target']

            loss = train(input_tensor, output_tensor, encoder,
                        decoder, encoder_optimizer, decoder_optimizer, 
                        criterion, batch_size=batch_size, debug=debug)
            
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, epoch*len(train_dataset) + iter / (epoch*len(train_dataset))),
                                            epoch*len(train_dataset) + iter, iter / (epoch*len(train_dataset)) * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        predictions_df, loss = get_all_predictions_sanity_check(encoder, attn_decoder, vocab, train_dataset, batch_size=batch_size)
        predictions_df.to_csv(op.join("experiments", experiment_name, f"{epoch}_epochs.csv"))

    showPlot(plot_losses)




def trainIters(encoder, decoder, train_dataset, dev_dataset, max_epochs, vocab=None, batch_size=1, print_every=500, plot_every=100, learning_rate=0.01, debug=False, patience = 5, early_stopping=True, teacher_forcing_ratio=0.5, max_length=50):
    if not early_stopping:
        patience = float("inf")

    start = time.time()
    train_losses = []
    dev_losses = []
    train_loss_total = 0  # Reset every epoch
    plot_loss_total = 0  # Reset every epoch

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=3) # ignore padding token

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_function)

    os.makedirs(op.join("experiments", experiment_name), exist_ok=True)

    current_patience = patience


    for epoch in range(1, max_epochs+1):
        print(f"starting epoch {epoch}")
    
        for iter, training_batch in enumerate(train_loader):

            input_tensor = training_batch['source']
            output_tensor = training_batch['target']

            loss = train(input_tensor, output_tensor, encoder,
                        decoder, encoder_optimizer, decoder_optimizer, 
                        criterion, batch_size=batch_size, debug=debug)
            
            train_loss_total += loss
            plot_loss_total += loss

        # Validate at the end of each epoch
        predictions_df, dev_loss_total = get_all_predictions(encoder, attn_decoder, vocab, criterion, dev_dataset, max_length=max_length)
        predictions_df.to_csv(op.join("experiments", experiment_name, f"{epoch}_epochs.csv"))

        train_loss_avg = train_loss_total / len(train_dataset)
        train_losses.append(train_loss_avg)
        dev_loss_avg = dev_loss_total / len(dev_dataset)
        dev_losses.append(dev_loss_avg)

        print('{} . Epoch {:2d}: train_loss: {:.4f} : dev_loss: {:.4f}'.format(timeSince(start, epoch/max_epochs+1), epoch, train_loss_avg, dev_loss_avg))
        train_loss_total = 0

        # save model: encoder and decoder
        if not os.path.exists('tmp/encoder'):
            os.makedirs('tmp/encoder')

        if not os.path.exists('tmp/decoder'):
            os.makedirs('tmp/decoder')

        torch.save(encoder.state_dict(), f'tmp/encoder/{epoch}')
        torch.save(decoder.state_dict(), f'tmp/decoder/{epoch}')


        if not dev_loss_avg < min(dev_losses):
            current_patience -= 1
        else:
            current_patience = patience  #reset

        if current_patience == 0:
            break

    # keep the best model and delete tmp
    if not os.path.exists('saved_model/encoder'):
            os.makedirs('saved_model/encoder')
    if not os.path.exists('saved_model/decoder'):
            os.makedirs('saved_model/decoder')
    
    # index of best model
    idx  = dev_losses.index(min(dev_losses)) + 1
    # remove old files if exist
    if os.path.exists(f'saved_model/encoder/{idx}'):
        os.remove(f'saved_model/encoder/{idx}')
    if os.path.exists(f'saved_model/decoder/{idx}'):
        os.remove(f'saved_model/decoder/{idx}')
    # move the best model from tmp to saved_model
    print(f'Saving the best model from epoch {idx} ...')
    shutil.move(f'tmp/encoder/{idx}', 'saved_model/encoder')
    shutil.move(f'tmp/decoder/{idx}', 'saved_model/decoder')
    # remove tmp folder
    shutil.rmtree('tmp')

    showPlot(train_losses, dev_losses)



## PLOTTING FUNCTIONS
def showPlot(train_points, dev_points):

    plt.figure()
    x = range(1,len(train_points)+1)
    y1 = train_points
    y2 = dev_points
    plt.ylim(0,2)
    plt.plot(x, y1, "-b", label="train")
    plt.plot(x, y2, "-r", label="dev")
    plt.legend(loc="upper right")
    plt.savefig('plot.png')


if __name__=="__main__":

    params_fp = sys.argv[1]
   
    
    with open(params_fp, "r") as params_file:
        params = json.load(params_file)
    experiment_name         = params['experiment_name']
    experiment_id           = params['experiment_id']
    debug                   = params['debug']
    sanity_check            = params['sanity_check']
    early_stopping          = params['early_stopping']
    patience                = params['patience']
    max_epochs              = params['max_epochs']
    hidden_size             = params['hidden_size']
    batch_size              = params['batch_size']
    max_length              = params['max_length']
    dropout                 = params['dropout']
    teacher_forcing_ratio   = params['teacher_forcing_ratio']
    learning_rate           = params['learning_rate']


    dialogue_vcb = load_vocab('DialogSum_Data/dialogue.vcb')
    summary_vcb = load_vocab('DialogSum_Data/summary.vcb')
    vocab_size = len(summary_vcb.index2word)

    # Create Dataset clas
    train_data_list = load_jsonl(TRAIN_DATA)
    
    if debug:
        train_data_list = train_data_list[:100]
        print("warning: DEBUG_MODE\n"*3)

    train_dataset = SummaryDataset(train_data_list,
                                    sentence_transformers_model="all-MiniLM-L6-v2",
                                    debug=debug)

    dev_data_list = load_jsonl(DEV_DATA)
    dev_dataset = SummaryDataset(dev_data_list,
                                sentence_transformers_model="all-MiniLM-L6-v2",
                                debug=debug)


    sent_embedding_size = train_dataset.source_embedding_dimension

    encoder = EncoderRNN(sent_embedding_size, hidden_size).to(device)
    attn_decoder = AttnDecoderRNN(hidden_size, summary_vcb.n_words, dropout_p=dropout, max_length=max_length).to(device)

    orig_encoder = copy.deepcopy(encoder)
    orig_decoder = copy.deepcopy(attn_decoder)

    if sanity_check:
        print("doing sanity check")
        trainIters_sanity_check(encoder, attn_decoder, train_dataset, dev_dataset, max_epochs=max_epochs, vocab=summary_vcb, batch_size=batch_size, print_every=500, debug=debug)
        exit(0)

    trainIters(encoder, attn_decoder, train_dataset, dev_dataset, max_epochs=max_epochs, 
                teacher_forcing_ratio=teacher_forcing_ratio, vocab=summary_vcb, 
                batch_size=batch_size, learning_rate=learning_rate,
                patience=patience, early_stopping=early_stopping,
                print_every=500, debug=debug)

    assert encoder != orig_encoder #sanity check
    assert attn_decoder != orig_decoder



    # import pdb; pdb.set_trace()

