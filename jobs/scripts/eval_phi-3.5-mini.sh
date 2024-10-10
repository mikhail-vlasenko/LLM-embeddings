#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=phi-3-mini
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:30:00
#SBATCH --output=jobs/outputs/eval_phi-3-mini_%A.out

module load 2023
module load Anaconda3/2023.07-2

model="Phi-3.5-mini-4k-instruct"

cd $HOME/LLM-embeddings

# look for a file in the results/${model} directory, including all subdirectories, that ends in .pkl and only take the first one.
embeddings_file_name=$(find results/${model} -type f -name "*.pkl" | head -n 1)

# If the embeddings file exists, load from that file. Otherwise, use the model_name_or_path
if [ -z "$embeddings_file_name" ]
then
    echo "No embeddings file found. Running the model."

    # Run using the right model_name_or_path
    srun python main.py \
        --model_name_or_path microsoft/${model} \
        --save_path results/${model}/no_sub_means
    
    # update the embeddings_file_name variable to the name of the .pkl file that was saved
    embeddings_file_name=$(find results/${model} -type f -name "*.pkl" | head -n 1)
else
    echo "Embeddings file found. Loading from file: ${embeddings_file_name}"
    
    srun python main.py \
        --load_from_file ${embeddings_file_name} \
        --save_path results/${model}/no_sub_means
fi

# Run with --subtract_means
srun python main.py \
    --load_from_file ${embeddings_file_name} \
    --save_path results/${model}/sub_means \
    --subtract_means