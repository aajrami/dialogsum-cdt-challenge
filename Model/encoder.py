from vocab import EOS_token, Vocab, load_vocab, tensor_to_sentence

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10

# input size = sentence embedding size
# hidden size = hyper-parameter
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, input, hidden):
        # embedded = self.embedding(input).view(1, 1, -1)
        # output = embedded
        # output, hidden = self.gru(output, hidden)
        input = input.view(1, 1, -1)   
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
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):

        ####################################################
        # REPLACE WITH WORD2VEC LOOKUP
        embedded = self.embedding(input).view(1, 1, -1)
        #embedded = torch.randn(hidden_size).view(1, 1, -1)
        ####################################################

        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        print(attn_weights.shape)
        print(encoder_outputs.shape)
        print(attn_weights.unsqueeze(0).shape)
        print(encoder_outputs.unsqueeze(0).shape)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
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
    encoder_hidden = encoder.initHidden()
    encoder_outputs = torch.zeros(5, hidden_size, device=device)

    for ei in range(num_sents):
        
        encoder_output, encoder_hidden = encoder(
            sentence_embeddings[ei], encoder_hidden
        )
        
        

        encoder_outputs[ei] = encoder_output[0,0]



    ##################################################################################

    summ_vcb = load_vocab('DialogSum_Data/summary.vcb')
    output_size = summ_vcb.n_words

    decoder = AttnDecoderRNN(hidden_size, output_size, dropout_p=0.1).to(device)
    decoder_hidden = encoder_hidden


    target_length = 3
    target_tensor = torch.tensor([5, 2, 36])

    use_teacher_forcing = True
    loss = 0
    criterion = nn.NLLLoss()

    decoder_input = torch.tensor([[SOS_token]], device=device)

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
        