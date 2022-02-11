#!/bin/bash 
#SBATCH --partition=gpu 
#SBATCH --qos=gpu 
#SBATCH --nodes=1 
#SBATCH --gpus-per-node=1 
#SBATCH --mem=34G 
#SBATCH --mail-user=kgreed1@sheffield.ac.uk 
#SBATCH --comment=job_name 
#SBATCH --output=output.%j.name.out

module load Anaconda3/5.3.0
source activate dialogsum
cd dialogsum-cdt-challenge
python Model/train.py