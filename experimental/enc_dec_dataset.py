import os
import os.path as op
import torch
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
import re
from torch.nn import Module

def replace_newlines(text):
    text = re.sub("[\n\W]*\n[\n\W]*", "<extra_id_0>", text)
    return text


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncDecDataset(Dataset):
    """
    A dataset for returning pairs of dialogue encodings and summary encodings
    """

    def __init__(
        self, source_target_list,
              tokenizer=None,
              source_sent_len = 100,
              target_sent_len = 100,    
              debug=False):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.debug = debug
        self.source_target_list = source_target_list
        self.tokenizer = tokenizer

        self.source_sent_len = source_sent_len # unused so far 
        self.target_sent_len = target_sent_len # unused so far


    def __len__(self):
        """returns the length of dataframe"""

        return len(self.source_target_list)

    def _dialogue_encode(self, dialogue_text):

        dialogue_list = dialogue_text.split("\n")

        
        encoding = self.tokenizer.batch_encode_plus(dialogue_list, return_tensors="pt", padding=True)

        input_ids = encoding["input_ids"][0]

        encoding["attention_mask"] = torch.Tensor([[1]*len(input_ids)])

        return encoding

  
    def _summary_encode(self, summary_text):
        """TODO"""
        encoding = self.tokenizer.batch_encode_plus([summary_text], return_tensors="pt", padding=True)

        return encoding

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        datapoint_dict = self.source_target_list[index]
        # 
        source_text = datapoint_dict["dialogue"]
        target_text = datapoint_dict["summary"]
# 
        source = self._dialogue_encode(source_text)
        target = self._summary_encode(target_text)
# 

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_mask": target_mask.to(dtype=torch.long),
        }


from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.modeling_utils import PreTrainedModel
from torch.nn.utils.rnn import pad_sequence
from transformers.models.bert.modeling_bert import BertPooler
import torch

class Config():
    pass

class CustomEncoder(Module):
    def __init__(self, encoder, *args, **kwargs):    
        super().__init__(*args, **kwargs)
        self.sub_encoder = encoder
        pooler_config = Config()
        pooler_config.hidden_size = 768
        self.bert_pooler = BertPooler(pooler_config)

    def forward(self, input_ids, **kwargs):

        attention_masks = kwargs.pop("attention_mask")

        print(attention_masks)

        hidden_state_outputs = []
        print(len(input_ids))
        for i in range(len(input_ids)):
            inputs = torch.unsqueeze(input_ids[i],0)
            attn_mask = torch.ones_like(inputs)
            encoder_outputs = self.sub_encoder.forward(inputs, attention_mask=attn_mask, **kwargs)
            hidden_state_outputs.append(encoder_outputs["pooler_output"])
        hidden_state_outputs = pad_sequence(hidden_state_outputs).contiguous()
       
        
        pooler_output = self.bert_pooler(hidden_state_outputs).contiguous()

        output = BaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state=hidden_state_outputs,
                                                              pooler_output = pooler_output)

        return output
