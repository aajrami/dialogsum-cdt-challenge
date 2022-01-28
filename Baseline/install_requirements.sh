#!/bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=32G 
#SBATCH --mail-user=kgreed1@sheffield.ac.uk 
#SBATCH --comment=requirements_installation 
#SBATCH --output=output.%j.requirements_installation.out

module load Anaconda3/5.3.0

source activate dialogsum

pip install -r requirements.txt