#!/bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=32G 
#SBATCH --mail-user=kgreed1@sheffield.ac.uk 
#SBATCH --comment=preprocess
#SBATCH --output=output.%j.preprocess.out

#tokenize
#python Preprocessing/tokenizer.py

#echo "Tokenize done"

# bpe

module load python/anaconda3
source activate dialogsum

subword-nmt learn-bpe -s 32000 < DialogSum_Data/dialogsum.sample.jsonl > codes_file.bpe
subword-nmt apply-bpe -c codes_file.bpe < DialogSum_Data/dialogsum.sample.tok.bpe