from nltk import word_tokenize

nltk.download('punkt')

if __name__=="__main__":
    # tok
    with open('DialogSum_Data/dialogsum.sample(150).jsonl', "r") as f1:
        with open('DialogSum_Data/dialogsum.sample(150).tok', "w") as f2:
            for line in f1:
                words = word_tokenize(line)
                f2.write(' '.join(words) + '\n')   