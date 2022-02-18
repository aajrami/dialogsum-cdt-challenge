import os
import os.path as op
import torch
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
from vocab import Vocab, load_vocab, sentence_to_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            self.sentence_transformer = self.load_sentence_transformers_model(sentence_transformers_model)
            # SentenceTransformer(sentence_transformers_model)
            self.tokenizer = None,

        self.source_embedding_dimension = self.sentence_transformer.get_sentence_embedding_dimension()
        self.source_sent_len = source_sent_len # unused so far 
        self.target_sent_len = target_sent_len # unused so far
        self.target_vocab = load_vocab(summary_vocab_path)



    def load_sentence_transformers_model(self, model_name):
    
        """If model has already been downloaded, load the model from a cache.
            Else, download it from the sentence transformers website and save it to the cache.

            Note, the latter will not work in a batch job on Jade. 
            """        

        if not op.exists("models_cache"):
            os.mkdir("models_cache")
        
        cache_path = op.join("models_cache", model_name)
        
        if op.exists(cache_path):
            model = SentenceTransformer(cache_path)
        else:
            model = SentenceTransformer(model_name) 
            model.save(cache_path)
        
        return model


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
        summary_text = summary_text.strip("\n")
        output = sentence_to_tensor(summary_text, self.target_vocab)
 
        if self.debug == True:
            print(summary_text)
            print(output)
            input()
        return output  


    def __getitem__(self, index):
        """
        return the input ids, attention masks and target ids"""

        datapoint_dict = self.source_target_list[index]
        
        source_text = datapoint_dict["dialogue"]
        target_text = datapoint_dict["summary"]

        source = self._dialogue_encode(source_text)
        target = self._summary_encode(target_text)


        return {
            "dialogue_text": source_text,
            "summary_text": target_text,
            "source": source.to(dtype=torch.float).to(device),
            "target": target.to(dtype=torch.float).to(device),
            "index": index
        }
