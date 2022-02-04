import nltk
from nltk import word_tokenize
from nltk.tokenize import MWETokenizer
import os
import json

nltk.download('punkt')

def tokenize_sentence(text):
    # create protected mwe 
    mwe = []
    for i in range(1,9):
        mwe.append('\n#Person{}#:'.format(i))
        mwe.append('#Person{}#'.format(i))

    tok_sent = multiword_tokenize(text, mwe)

    return tok_sent

def multiword_tokenize(text, mwe):
    # Initialize the MWETokenizer
    protected_tuples = [word_tokenize(word) for word in mwe]
    protected_tuples_underscore = ['_'.join(word) for word in protected_tuples]
    tokenizer = MWETokenizer(protected_tuples)
    # Tokenize the text.
    tokenized_text = tokenizer.tokenize(word_tokenize(text))
    # Replace the underscored protected words with the original MWE
    for i, token in enumerate(tokenized_text):
        if token in protected_tuples_underscore:
            tokenized_text[i] = mwe[protected_tuples_underscore.index(token)]
    return tokenized_text

if __name__=="__main__":
    # tok
    with open('DialogSum_Data/dialogsum.sample.jsonl', "r") as f1:
        with open('DialogSum_Data/dialogsum.sample.tok.jsonl', "w+") as f2:
            for line in f1:
                d = json.loads(line)
                dialogue = d['dialogue']
                summary = d['summary']

                # create protected mwe 
                mwe = []
                for i in range(1,9):
                    mwe.append('\n#Person{}#:'.format(i))
                    mwe.append('#Person{}#'.format(i))

                tok_dialogue = ' '.join(multiword_tokenize(dialogue, mwe)).strip()
                tok_summary = ' '.join(multiword_tokenize(summary, mwe)).strip()

                d['dialogue'] = tok_dialogue
                d['summary'] = tok_summary
                json.dump(d, f2)
                f2.write('\n')