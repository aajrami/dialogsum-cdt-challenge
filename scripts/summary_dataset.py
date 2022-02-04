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

    def _dialogue_encode(self, dialogue_text):
        """ TODO"""
        utterances = dialogue_text.split("\n")
        print(utterances)
        input()
        return self.sentence_transformer.encode(utterances)
    
    def _summary_encode(self, summary_text):
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

        source = self._dialogue_encode(self, idx)
        target = self._summary_encode(self, idx)


        return {
            "source": source.to(dtype=torch.long),
            "target": target.to(dtype=torch.long),
        }
