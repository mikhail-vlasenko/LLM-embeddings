# Multilingual Sentence Embeddings with LLMs

This repository contains code for generating multilingual sentence embeddings using autoregressive Large Language Models (LLMs). The project implements several enhancements, including mean shift techniques, Multi-Layer Perceptrons (MLP) for contrastive learning, and low-rank linear layers for improved efficiency.

## Table of Contents

- [Features](#features)
- [Usage](#usage)
- [Dataset](#dataset)
- [Results](#results)


## Features

- Generate sentence embeddings in multiple languages using LLMs.
- Apply mean shift techniques to align embedding distributions.
- Use a Multi-Layer Perceptron (MLP) for contrastive learning.
- Implement low-rank linear layers to enhance efficiency.
- Evaluate embeddings using cross-lingual retrieval tasks.
- Save and load embeddings from files for later use.


## Usage

1. **Run the script**: Use the following command to execute the main script:

```bash
python main.py --model_name_or_path <MODEL_PATH> --dataset_name <DATASET_NAME>
```

Replace `<MODEL_PATH>` with the desired model path (e.g., `microsoft/Phi-3.5-mini-instruct`) and `<DATASET_NAME>` with the dataset you wish to use (e.g., `facebook/flores`).

2. **Available Arguments**:
   - `--model_name_or_path`: Path to the pre-trained model.
   - `--save_path`: Directory to save results.
   - `--csv_path`: Path to the CSV file containing sentence pairs.
   - `--load_kbit`: Specify model quantization (options: 4, 8, 16).
   - `--avg`: Use average pooling for embeddings.
   - `--batch_size`: Specify the batch size for DataLoader.
   - `--max_samples`: Limit the number of samples loaded from the dataset.
   - `--self_prompts`: Use prompt templates in the same language.
   - `--subtract_means`: Subtract language-wide means from embeddings.
   - `--contrastive_learning`: Apply MLP trained with contrastive learning.

## Dataset

The project utilizes the FLORES dataset, which contains multilingual sentence pairs for evaluation. Ensure the dataset is accessible or specify the dataset name when running the script.

## Results

The performance of the embeddings can be evaluated using the `evaluate_translation_accuracy` function. Results are saved as CSV files, including average accuracy metrics across different languages.
