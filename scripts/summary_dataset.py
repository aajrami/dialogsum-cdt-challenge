import torch
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer

class SummaryDataset(Dataset):
    """
    A dataset for returning pairs of dialogue encodings and 
    """

    def __init__(
        self, source_target_list,
              dialogue_sent_tokenizer, 
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
        self.tokenizer = tokenizer
        if sentence_transformers_model:
            self.sentence_transformers_model = SentenceTransformer("all-MiniLM-L6-v2")
        import pdb; pdb.set_trace()
        self.data = dataframe
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

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self._dialogue_encode(self, idx)
        target = self._dialogue_encode(self, idx)

#        source = self.tokenizer.batch_encode_plus(
#            [source_text],
#            max_length=self.source_len,
#            pad_to_max_length=True,
#            truncation=True,
#            padding="max_length",
#            return_tensors="pt",
#        )
#        target = self.tokenizer.batch_encode_plus(
#            [target_text],
#            max_length=self.summ_len,
#            pad_to_max_length=True,
#            truncation=True,
#            padding="max_length",
#            return_tensors="pt",
#        )

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
