#tokenize
python Preprocessing/tokenizer.py

echo "Tokenize done"

# bpe
subword-nmt learn-bpe -s 32000 < DialogSum_Data/dialogsum.sample(150).tok > codes_file.bpe
subword-nmt apply-bpe -c codes_file.bpe < DialogSum_Data/dialogsum.sample(150).tok.bpe