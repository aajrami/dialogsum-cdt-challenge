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
import json

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

parser = argparse.ArgumentParser()
parser.add_argument("--DEBUG_ON_SAMPLE", action="store_true")
parser.add_argument("--EXPERIMENT_NAME", type=str, default="prelim")
parser.add_argument("--N_EPOCHS", type=int, default=5)

args = parser.parse_args()


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
    
    batch_encoder_hidden = torch.cat(batch_size * [encoder.initHidden()], dim=1)
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(1)
    output_length = output_tensor.size(1)

    batch_encoder_hidden_states = torch.zeros(batch_size, max_length, encoder.hidden_size, device=device)

    loss = 0

    ### NEED TO CHANGE TO IMPLEMENT SOME MAX LENGTH
    for ei in range(input_length):

        batch_encoder_inputs = input_tensor[:,ei,:].reshape(batch_size, 1, -1)

        batch_encoder_output, batch_encoder_hidden = encoder(
            batch_encoder_inputs, batch_encoder_hidden)

        # MIGHT NOT WORK WITH BATCHING
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
def decode(input_tensor, encoder, decoder, vocab, batch_size=1, max_length=MAX_LENGTH, debug=False):
    
    batch_encoder_hidden = torch.cat(batch_size * [encoder.initHidden()], dim=1)
    
    input_length = input_tensor.size(1)
    batch_encoder_hidden_states = torch.zeros(batch_size, max_length, encoder.hidden_size, device=device)


    ### NEED TO CHANGE TO IMPLEMENT SOME MAX LENGTH
    for ei in range(input_length):

        batch_encoder_inputs = input_tensor[:,ei,:].reshape(batch_size, 1, -1)

        batch_encoder_output, batch_encoder_hidden = encoder(
            batch_encoder_inputs, batch_encoder_hidden)

        # MIGHT NOT WORK WITH BATCHING
        batch_encoder_hidden_states[:,ei,:] = batch_encoder_output[0, 0]

    batch_decoder_input = torch.tensor([[SOS_token]], device=device)
    batch_decoder_input = torch.cat(batch_size * [batch_decoder_input], dim=0)

    batch_decoder_hidden = batch_encoder_hidden

    eos_list = [False] * batch_size

    # Without teacher forcing: use its own predictions as the next input

    outputs = []

    for di in range(max_length):
        batch_decoder_output, batch_decoder_hidden, decoder_attention = decoder(
            batch_decoder_input, batch_decoder_hidden, batch_encoder_hidden_states)
        topv, topi = batch_decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input

        outputs.append(topi)

        
        for i, inp in enumerate(decoder_input):
            if inp.item() == EOS_token:
                eos_list[i] = True
        if len(eos_list) == sum(eos_list): #If we have seen EOS for every item in the batch, then we break
            break

    outputs = torch.stack(outputs).swapaxes(0, 1)
    output_sentence = [tensor_to_sentence(outputs[i], vocab) for i in range(len(outputs))]
    

    return output_sentence


def collate_function(batch):
    
    dataset_index = torch.tensor([s["index"] for s in batch]) #To let us find the original text of dialogue and summary
    dialogue_batch   = [s["source"] for s in batch]
    summary_batch = [s["target"] for s in batch]    

    dialogue_lengths = torch.tensor([ len(dialogue) for dialogue in dialogue_batch ]).to(device)
    summary_lengths = torch.tensor([ len(summary) for summary in summary_batch ]).to(device)
    

    dialogues_padded =   [   torch.stack([torch.ones_like(dialogue_batch[0][0])]   * max(dialogue_lengths)) for dialogue in dialogue_batch]
    summaries_padded  =   [ torch.tensor([1] * max(summary_lengths)) for summary in summary_batch]

    dialogues_padded  = torch.stack(dialogues_padded).to(device)
    summaries_padded = torch.stack(summaries_padded).to(device)
    
    dialogues_padded *= 3   #We set the pad symbol to 3, same as in the vocab
    summaries_padded *= 3
    
    for i, dialogue in enumerate(dialogue_batch):
        dialogues_padded[i][:len(dialogue)] = dialogue
    
    for i, summary in enumerate(summary_batch):
        summaries_padded[i][:len(summary)] = summary
    
    return { "source": dialogues_padded, "target": summaries_padded, "dataset_index": dataset_index}



def get_all_predictions(encoder, decoder, vocab, dataset, batch_size=4):

    dev_loss = 0

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_function)

    predictions = []

    for batch in dataloader:
        dataset_indices = batch["dataset_index"]
        input_tensor = batch["source"]
        outputs = decode(input_tensor, encoder, decoder, vocab, batch_size=batch_size)
        pred_summaries = [" ".join(decoded_sent) for decoded_sent in outputs]
        dialogues = [dataset[idx.item()]["dialogue_text"] for idx in dataset_indices]
        gold_summaries = [dev_dataset[idx.item()]["summary_text"] for idx in dataset_indices]
        for i in range(len(dataset_indices)):
            pred_dict = {}
            pred_dict["dialogue"] = dialogues[i]
            pred_dict["gold_summary"] = gold_summaries[i]
            pred_dict["predicted_summary"] = pred_summaries[i]
            predictions.append(pred_dict)

    predictions_df = pd.DataFrame(predictions)

    return(predictions_df), dev_loss





def trainIters(encoder, decoder, train_dataset, dev_dataset, num_epochs, vocab=None, batch_size=1, print_every=500, plot_every=100, learning_rate=0.01, debug=False, experiment_name=args.EXPERIMENT_NAME):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every epoch
    plot_loss_total = 0  # Reset every epoch

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=3) # ignore padding token

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_function)

    os.makedirs(op.join("experiments", experiment_name), exist_ok=True)

    for epoch in range(1, num_epochs+1):
        print(f"starting epoch {epoch}")
    
        for iter, training_batch in enumerate(train_loader):

            input_tensor = training_batch['source']
            output_tensor = training_batch['target']

            loss = train(input_tensor, output_tensor, encoder,
                        decoder, encoder_optimizer, decoder_optimizer, 
                        criterion, batch_size=batch_size, debug=debug)
            
            print_loss_total += loss
            plot_loss_total += loss

            # Want to print every epoch not every print_every batches
            # if iter % print_every == 0:
            #     print_loss_avg = print_loss_total / print_every
            #     print_loss_total = 0
                # print('%s (%d %d%%) %.4f' % (timeSince(start, epoch*len(train_dataset) + iter / (epoch*len(train_dataset))),
                #                             epoch*len(train_dataset) + iter, iter / (epoch*len(train_dataset)) * 100, print_loss_avg))

            # if iter % plot_every == 0:
            #     plot_loss_avg = plot_loss_total / plot_every
            #     plot_losses.append(plot_loss_avg)
            #     plot_loss_total = 0

        print_loss_avg = print_loss_total / len(train_dataset)
        print_loss_total = 0

        print('{} . Epoch {:2d}: avg_loss: {:.4f}'.format(timeSince(start, epoch/num_epochs), epoch, print_loss_avg))

        plot_loss_total = plot_loss_total / len(train_dataset)
        plot_losses.append(plot_loss_total)
        plot_loss_total = 0

        # Validate at the end of each epoch
        predictions_df, dev_loss = get_all_predictions(encoder, attn_decoder, vocab, dev_dataset)
        predictions_df.to_csv(op.join("experiments", experiment_name, f"{epoch}_epochs.csv"))

    showPlot(plot_losses)



## PLOTTING FUNCTIONS
def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig('plot.png')


if __name__=="__main__":

    ##################################################################################
    dialogue_vcb = load_vocab('DialogSum_Data/dialogue.vcb')
    summary_vcb = load_vocab('DialogSum_Data/summary.vcb')
    vocab_size = len(summary_vcb.index2word)

    # Create Dataset clas
    train_data_list = load_jsonl(TRAIN_DATA)
    
    if args.DEBUG_ON_SAMPLE:
        train_data_list = train_data_list[:100]
        print("warning: DEBUG_MODE\n"*3)

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

    batch_size=4

    hidden_size = 300
    num_epochs = args.N_EPOCHS


    print(summary_vcb.n_words)
    encoder = EncoderRNN(sent_embedding_size, hidden_size).to(device)
    attn_decoder = AttnDecoderRNN(hidden_size, summary_vcb.n_words, dropout_p=0.1, max_length=MAX_LENGTH).to(device)

    orig_encoder = copy.deepcopy(encoder)
    orig_decoder = copy.deepcopy(attn_decoder)

    trainIters(encoder, attn_decoder, train_dataset, dev_dataset, num_epochs=num_epochs, vocab=summary_vcb, batch_size=batch_size, print_every=500, debug=False)

    assert encoder != orig_encoder #sanity check
    assert attn_decoder != orig_decoder



    predictions_df = get_all_predictions(encoder, attn_decoder, summary_vcb, dev_dataset)


    import pdb; pdb.set_trace()

