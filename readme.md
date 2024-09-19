# README

## Usage

### Command Line Arguments

The script can be run from the command line with the following options:

```bash
python main.py --model_name_or_path <model> --csv_path <path> --load_kbit <4|8|16> --avg --k <number> --batch_size <number> --dataset_name <dataset> --max_samples <number>
```

#### Options:

- `--model_name_or_path`: The name or path of the transformer model to use (default: `microsoft/Phi-3.5-mini-instruct`).
- `--csv_path`: Path to the CSV file containing sentence pairs (default: `sample_data.csv`).
- `--load_kbit`: Load model in kbit (choices: 4, 8, 16; default: 16).
- `--avg`: Use average pooling for embeddings (optional).
- `--k`: Number of most similar items for recall (default: 3).
- `--batch_size`: Batch size for DataLoader (default: 1).
- `--dataset_name`: The name of the dataset to load (default: `Muennighoff/flores200`).
- `--max_samples`: Maximum number of samples to load from the dataset (default: None).

### Example

To run the script with the default settings:

```bash
python main.py
```

To specify custom parameters, for example, using a different model and loading a limited number of samples:

```bash
python main.py --model_name_or_path "your_model_name" --max_samples 100
```

## Output

The script will generate the following outputs:

1. **embedding.csv**: Contains the generated embeddings for each language.
2. **average_accuracy_results.csv**: Stores the results of the translation accuracy evaluation.
3. **average_accuracy_table.csv**: A pivoted table of average recall results.

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [FLORES-200 Dataset](https://huggingface.co/datasets/Muennighoff/flores200)
- [PyTorch](https://pytorch.org/) for the deep learning framework.

---

