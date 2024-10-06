#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=llm-embedding
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=06:00:00
#SBATCH --output=slurm_output_%A.out

module load 2022 
module load Anaconda3/2022.05 
module load torchvision/0.13.1-foss-2022a-CUDA-11.7.0 

pip install --user transformers
pip install --user accelerate
pip install --user scikit-learn
pip install --user datasets
pip install --user pandas
pip install --user 'seacrowd>=0.2.0'
pip install --user seaborn

# Run your code
srun python main.py