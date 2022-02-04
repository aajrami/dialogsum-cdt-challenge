import torch
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
from processing.vocab import Vocab, load_vocab
from processing.data_tokenize

class SummaryDataset(Dataset):
    """
    A dataset for returning pairs of dialogue encodings and 
    """

    def __init__(
        self, source_target_list,
              source_sent_len = 100,
              target_sent_len = 100,    
              sentence_transformers_model=None,
              summary_vocab_path = 'DialogSum_Data/summary.vcb',
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
        if sentence_transformers_model:
            self.sentence_transformer = SentenceTransformer(sentence_transformers_model)
            self.tokenizer = None,

        self.source_embedding_dimension = self.sentence_transformer.get_sentence_embedding_dimension()
        self.source_sent_len = source_sent_len # unused so far 
        self.target_sent_len = target_sent_len # unused so far
        self.target_vocab = load_vocab(summary_vocab_path)

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.source_target_list)

    def _dialogue_encode(self, dialogue_text):
        """ TODO"""
        utterances = dialogue_text.split("\n")
        encoding = self.sentence_transformer.encode(utterances)
        encoding = torch.Tensor(encoding)
        if self.debug == True:
            print(utterances)
            print(encoding.shape)
        return encoding
  
    def _summary_encode(self, summary_text):
        """TODO"""
        
        print("_summary_encode not yet implemented")
        output = torch.Tensor([1])
        if self.debug == True:
            print(output)
            input()
        return output  


    def __getitem__(self, index):
        """
        NOT YET IMPLEMENTED


        return the input ids, attention masks and target ids"""

        datapoint_dict = self.source_target_list[index]
        
        source_text = datapoint_dict["dialogue"]
        target_text = datapoint_dict["summary"]

        source = self._dialogue_encode(source_text)
        target = self._summary_encode(source_text)


        return {
            "source": source.to(dtype=torch.long),
            "target": target.to(dtype=torch.long),
        }
