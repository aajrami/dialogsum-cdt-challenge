#!/usr/bin/env python
# coding: utf-8



from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModel, AutoTokenizer
from torch.nn import Module
from pprint import pprint
import copy
import torch


class inspect_func(Module):
    def __call__(self, *args, **kwargs):
        pprint(args)
        pprint(kwargs)
        print(kwargs["encoder_hidden_states"].shape)



import os
import os.path as op
import torch
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer


def concat_dicts(dict_1, dict_2):
    for k in dict_1:
        dict_1[k] = torch.cat((dict_1[k], dict_2[k]),dim=1)
    return dict_1



class STOnlyEncoder(Module):
    def __init__(self, encoder, st_tokenizer=None, t5_tokenizer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.st_encoder = encoder
        self.st_tokenizer = st_tokenizer
        self.t5_tokenizer = t5_tokenizer
        self.extra_token_id = self.t5_tokenizer.encode("<extra_id_0>")[0]
        print("Extra token id: ", self.extra_token_id)


    def forward(self, input_ids, **kwargs):


        batches = []

        for b in input_ids:
            text = self.t5_tokenizer.decode(b, ignore_special_tokens=True)[5:-5]
            text = text.strip(" ")
            sents = text.split("<extra_id_0>")
            sents_tokenized = self.st_tokenizer.batch_encode_plus(sents, return_tensors="pt", padding=True)
            
            encoded = self.st_encoder.forward(**sents_tokenized)
            pooler_output = encoded["pooler_output"]
            batches.append(pooler_output)
            print("n.sents:",len(pooler_output))

        embeddings = pad_sequence(batches, batch_first=True)
        output = BaseModelOutputWithPastAndCrossAttentions(embeddings, )

        return output
        

class CustomEncoder(Module):
    def __init__(self, encoder, *args, **kwargs):    
        super().__init__(*args, **kwargs)
        self.t5_encoder = encoder

    def forward(self, input_ids, **kwargs):
        
        output = self.t5_encoder.forward(input_ids, **kwargs)
        print(output)
        print(output["last_hidden_state"].shape)
        exit()


        print("kwargs")
        print(kwargs)

        input_ids = input_ids[:input_ids.shape[0],:input_ids.shape[1]//2]
        kwargs["attention_mask"] = kwargs["attention_mask"][:input_ids.shape[0], :input_ids.shape[1]]
        encoder_1_output = self.t5_encoder.forward(input_ids, **kwargs)
        encoder_2_output = self.t5_encoder.forward(input_ids, **kwargs)
        print("input ids shape",input_ids.shape)
        print("attention mask shape", kwargs["attention_mask"].shape)
        print(encoder_1_output["last_hidden_state"].shape)

        encoder_output = concat_dicts(encoder_1_output, encoder_2_output)
        print(dir(encoder_1_output))
        #print(encoder_output)
        print(encoder_output["last_hidden_state"].shape)

        print(encoder_output)

        pprint(kwargs)
        return(encoder_output)


