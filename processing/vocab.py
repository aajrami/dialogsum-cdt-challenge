import pickle
import json
if __name__ == "__main__":
    from data_tokenize import tokenize_sentence
else:
    from processing.data_tokenize import tokenize_sentence

SOS_token = 0
EOS_token = 1

class Vocab:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "UNK"}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def create_vocab(name, sentences):
    vcb = Vocab(name)
    for s in sentences:
        vcb.addSentence(s)
    return vcb

def sentence_to_tensor(sent, vocab):
    sent = tokenize_sentence(sent)
    tensor = [vocab.word2index.get(w,2) for w in sent]
    return tensor

def tensor_to_sentence(tensor, vocab):
    sent = [vocab.index2word.get(i,"UNK") for i in tensor]
    return sent

def load_vocab(filepath):
    with open(filepath, 'rb') as f:
        vocab = pickle.load(f)
    return vocab

if __name__=='__main__':
   
    train_path = 'DialogSum_Data/dialogsum.train.tok.jsonl'

    dialogues = []
    summaries = []

    with open(train_path, 'rb') as f:
        for line in f:
            d = json.loads(line)
            dialogues.append(d['dialogue'])
            summaries.append(d['summary'])
    
    summary_vcb = create_vocab('summary', summaries)
    dialogue_vcb = create_vocab('dialogue', dialogues)

    summary_vcb_path = 'DialogSum_Data/summary.vcb'
    dialogue_vcb_path = 'DialogSum_Data/dialogue.vcb'

    with open(summary_vcb_path, "wb") as summ_vcb:
        pickle.dump(summary_vcb, summ_vcb)

    with open(dialogue_vcb_path, "wb") as dia_vcb:
        pickle.dump(dialogue_vcb, dia_vcb)
