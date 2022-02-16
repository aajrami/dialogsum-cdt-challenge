#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=10:00:00

# set name of job
#SBATCH --job-name=job123

# memory
#SBATCH --mem=32

# set number of GPUs
#SBATCH --gres=gpu:4

#SBATCH -o "slurm_output/bessemer_job"

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

source ~/.bashrc

module load python/anaconda3
source activate dialogsum
python Model/train.py --DEBUG_ON_SAMPLE

