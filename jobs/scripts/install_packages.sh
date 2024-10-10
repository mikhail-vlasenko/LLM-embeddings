#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=phi3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=jobs/outputs/install_packages_%A.out

module load 2023
module load Anaconda3/2023.07-2
module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1

pip install --user --upgrade transformers
pip install --user --upgrade accelerate
pip install --user --upgrade scikit-learn
pip install --user --upgrade datasets
pip install --user --upgrade pandas
pip install --user --upgrade 'seacrowd>=0.2.0'
pip install --user --upgrade seaborn
pip install --user --upgrade tiktoken
pip install --user --upgrade triton
pip install --user --upgrade einops
pip install --user --upgrade flash-attn --no-build-isolation
pip install --user --upgrade bottleneck
pip install --user --force-reinstall 'numpy>=1.23,<2.0'