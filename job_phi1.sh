#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=phi1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:30:00
#SBATCH --output=slurm_output_%A.out

module load 2023
module load Anaconda3/2023.07-2
module load torchvision/0.13.1-foss-2022a-CUDA-11.7.0

pip install --user --upgrade transformers
pip install --user --upgrade accelerate
pip install --user --upgrade scikit-learn
pip install --user --upgrade datasets
pip install --user --upgrade pandas
pip install --user --upgrade 'seacrowd>=0.2.0'
pip install --user --upgrade seaborn
pip install --user --force-reinstall 'numpy>=1.23,<2.0'

# Run your code
srun python main.py \
    --model_name_or_path microsoft/phi-1 \
    --save_path results/phi1 \
    --subtract_means