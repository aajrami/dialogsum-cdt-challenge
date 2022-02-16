#!/bin/bash
#Author: Jon
#Usage: sbatch sbatch_wrapper.sh <file_to_run>
#Optionally add amount of memory and number of gpus, e.g:
#sbatch sbatch_wrapper.sh <file_to_run> 64G 8 

file_to_run=$1
amount_of_memory=${2-32G}
n_gpus=${3-4}
sbatch <<EOT
#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=10:00:00

# set name of job
#SBATCH --job-name=job123

# memory
#SBATCH --mem=$amount_of_memory

# set number of GPUs
#SBATCH --gres=gpu:$n_gpus

#SBATCH -o "slurm_output/REPORT_"$1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

echo "amount of memory:"${amount_of_memory}
source ~/.bashrc
module load python/anaconda3
conda activate dialogsum
echo $file_to_run
source ./$file_to_run

EOT
exit 0
