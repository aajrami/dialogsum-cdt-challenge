import os
import os.path as op
import torch
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
import re


def replace_newlines(text):
    text = re.sub("[\n\W]*\n[\n\W]*", "<extra_id_0>", text)
    return text


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class T5Dataset(Dataset):
    """
    A dataset for returning pairs of dialogue encodings and summary encodings
    """

    def __init__(
        self, source_target_list,
              dialogue_tokenizer=None,
              summary_tokenizer=None,
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
        self.dialogue_tokenizer = dialogue_tokenizer
        self.summary_tokenizer = summary_tokenizer

        self.source_sent_len = source_sent_len # unused so far 
        self.target_sent_len = target_sent_len # unused so far


    def __len__(self):
        """returns the length of dataframe"""

        return len(self.source_target_list)

    def _dialogue_encode(self, dialogue_text):

        text_with_newline_breaks = replace_newlines(dialogue_text)

        attention_mask = 
        encoding = st_tokenizer.encode(replace_newlines(dialogue_text), return_tensors="pt")

        if self.debug == True:
            print(utterances)
            print(encoding.shape)

        return encoding

        # source = self.tokenizer.batch_encode_plus(
        #     [source_text],
        #     max_length=self.source_len,
        #     pad_to_max_length=True,
        #     truncation=True,
        #     padding="max_length",
        #     return_tensors="pt",
        # )


  
    def _summary_encode(self, summary_text):
        """TODO"""
        summary_text = summary_text.strip("\n")
        output = sentence_to_tensor(summary_text, self.target_vocab)
 
        if self.debug == True:
            print(summary_text)
            print(output)
            input()
        return output  
        #         target = self.tokenizer.batch_encode_plus(
        #     [target_text],
        #     max_length=self.summ_len,
        #     pad_to_max_length=True,
        #     truncation=True,
        #     padding="max_length",
        #     return_tensors="pt",
        # )



    # def __getitem__(self, index):
        # """
        # return the input ids, attention masks and target ids"""
# 
        # datapoint_dict = self.source_target_list[index]
        # 
        # source_text = datapoint_dict["dialogue"]
        # target_text = datapoint_dict["summary"]
# 
        # source = self._dialogue_encode(source_text)
        # target = self._summary_encode(target_text)
# 
# 
        # return {
            # "dialogue_text": source_text,
            # "summary_text": target_text,
            # "source": source.to(dtype=torch.float).to(device),
            # "target": target.to(dtype=torch.float).to(device),
            # "index": index
        # }

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
            "target_ids_y": target_ids.to(dtype=torch.long),
        }
