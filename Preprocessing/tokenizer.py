import nltk
from nltk import word_tokenize
import os

nltk.download('punkt')

if __name__=="__main__":
    # tok
    with open('DialogSum_Data/dialogsum.sample.jsonl', "r") as f1:
        with open('DialogSum_Data/dialogsum.sample.tok', "w+") as f2:
            for line in f1:
                words = word_tokenize(line)
                f2.write(' '.join(words) + '\n')   