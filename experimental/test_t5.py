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
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
from vocab import Vocab, load_vocab, sentence_to_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#Sentences we want sentence embeddings for
sentences = ['This framework generates embeddings for each input sentence',
             'Sentences are passed as a list of string.',
             'The quick brown fox jumps over the lazy dog.']

#Load AutoModel from huggingface model repository
st_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/msmarco-bert-base-dot-v5") #model_args={"embedding_size":512})
st_model = AutoModel.from_pretrained("sentence-transformers/msmarco-bert-base-dot-v5")#embedding_size=512)

#Tokenize sentences
encoded_input = st_tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')




t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from torch.nn.utils.rnn import pad_sequence


def concat_dicts(dict_1, dict_2):
    for k in dict_1:
        dict_1[k] = torch.cat((dict_1[k], dict_2[k]),dim=1)
    return dict_1



class STOnlyEncoder(Module):
    def __init__(self, encoder, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.st_encoder = encoder
        self.tokenizer = tokenizer
        self.extra_token_id = self.tokenizer.encode("<extra_id_0>")[1]



    def forward(self, input_ids, **kwargs):


        batches = []

        for b in input_ids:
            text = self.tokenizer.decode(b, ignore_special_tokens=True)[5:-5]
            text = text.strip(" ")
            sents = text.split("<extra_id_0>")
            sents_tokenized = self.tokenizer.batch_encode_plus(sents, return_tensors="pt", padding=True)
            
            encoded = self.st_encoder.forward(**sents_tokenized)
            pooler_output = encoded["pooler_output"]
            print(sents)
            print(pooler_output)
            print(pooler_output.shape)
            batches.append(pooler_output)
        
        embeddings = pad_sequence(batches, batch_first=True)
        print(embeddings)
        print(embeddings.size())
        output = BaseModelOutputWithPastAndCrossAttentions(embeddings, )
        pprint(kwargs)
 #       encoded = self.st_encoder.forward(input_ids, **kwargs)
#        pooler_output = encoded["pooler_output"]

        return output
        

class CustomEncoder(Module):
    def __init__(self, encoder, *args, **kwargs):    
        super().__init__(*args, **kwargs)
        self.t5_encoder = encoder

    def forward(self, input_ids, **kwargs):
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





example_dialogue = "\nhello\nhello\n\nsentence 3"




st_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/msmarco-bert-base-dot-v5", extra_ids=1)

st_tokenizer.add_tokens(['<extra_id_0>'], special_tokens=True) ##This line is updated
st_model.resize_token_embeddings(len(tokenizer))
st_only_encoder = STOnlyEncoder(st_model,st_tokenizer)

t5_model.encoder = st_only_encoder

#t5_model.encoder = CustomEncoder(t5_model.encoder)



st_tokens = st_tokenizer.encode(replace_newlines(example_dialogue), return_tensors="pt")

    
print(st_tokens)
t5_model.generate(st_tokens, attention_mask=torch.Tensor([[1]*4]))


# In[28]:


t5_tokenizer.decode(torch.Tensor([0, 3, 2, 3, 2, 3, 2, 3, 7, 3, 9, 3, 9, 3, 9, 3, 9, 3, 9, 3]))


# In[12]:


words = "here \n are \n some \n sentences"
st_tokens = st_tokenizer.encode(replace_newlines(words), return_tensors="pt")

t5_model.generate(st_tokens)


# In[ ]:


tokens = tokenizer.encode("Translate to German: This is an unreasonably long sentence", return_tensors="pt")
print(tokens)
tokens = torch.cat((tokens, tokens), dim=1)
generated = model.generate(tokens)
tokenizer.decode(generated[0], skip_special_tokens=True)



model.forward(tokens)


t5_generated = t5_model.generate(tokens)
tokenizer.decode(t5_generated[0], skip_special_tokens=True)


generated = model.generate(tokens)
tokenizer.decode(generated[0], skip_special_tokens=True)


t5_model.forward(tokens)