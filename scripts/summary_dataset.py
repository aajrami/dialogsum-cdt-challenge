import torch
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer

class SummaryDataset(Dataset):
    """
    A dataset for returning pairs of dialogue encodings and 
    """

    def __init__(
        self, source_target_list,
              source_sent_len = 100,
              target_sent_len = 100,    
              sentence_transformers_model=None):
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
        self.source_target_list = source_target_list
        if sentence_transformers_model:
            self.sentence_transformer = SentenceTransformer(sentence_transformers_model)
            self.tokenizer = None,
        import pdb; pdb.set_trace()

        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def _dialogue_encode(self, index):
        """ TODO"""    
        pass
    
    def _summary_encode(self, index):
        """TODO"""
        
        print("_summary_encode not yet implemented")
        return Torch.tensor([1])        


    def __getitem__(self, index):
        """
        NOT YET IMPLEMENTED


        return the input ids, attention masks and target ids"""

        datapoint_dict = self.source_target_list[index]
        
        source_text = datapoint_dict["dialogue"]
        target_text = datapoint_dict["summary"]

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self._dialogue_encode(self, idx)
        target = self._dialogue_encode(self, idx)


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
