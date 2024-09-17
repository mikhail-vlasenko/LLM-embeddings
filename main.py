import argparse
import torch
import numpy as np
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from datasets import load_dataset
from data_utils import FloresMultiLangDataset, compare_languages, collate_fn


template = 'This sentence: "{sentence}" means in one word:'  # 100% accuracy on sample data
# template = '{sentence}'  # 10% accuracy on sample data

# Language dictionary mapping language names to their FLORES-200 codes
languages = {
    'English': 'eng_Latn',
    'Chinese_Simplified': 'zho_Hans',
    'Russian': 'rus_Cyrl',
    'Dutch': 'nld_Latn',
    'German': 'deu_Latn'
}


def load_csv(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        return list(reader)


def get_embeddings(model, tokenizer, sentences, device, args):
    embeddings = []
    for sentence in sentences:
        sentence = template.format(sentence=sentence)
        inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get raw embeddings
        with torch.no_grad():
            hidden_states = model(output_hidden_states=True, return_dict=True, **inputs).hidden_states
            if args.avg:
                last_layer = hidden_states[-1]
                attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(last_layer.shape)
                outputs = (last_layer * attention_mask).mean(1)
            else:
                outputs = hidden_states[-1][:, -1, :]

            if outputs.dtype == torch.bfloat16:
                # bfloat16 not support for .numpy()
                outputs = outputs.float()

            embeddings.append(outputs.cpu().numpy())

    return np.vstack(embeddings)


def find_most_similar(query_embedding, target_embeddings):
    similarities = cosine_similarity(query_embedding.reshape(1, -1), target_embeddings)[0]
    return np.argmax(similarities)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="microsoft/Phi-3.5-mini-instruct", help="Transformers' model name or path")
    parser.add_argument("--csv_path", type=str, default="sample_data.csv",
                        help="Path to the CSV file with English-Russian sentence pairs")
    parser.add_argument("--load_kbit", type=int, choices=[4, 8, 16], default=16, help="Load model in kbit")
    parser.add_argument('--avg', action='store_true', help="Use average pooling for embeddings")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    if args.load_kbit == 4:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            load_in_4bit=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
            ),
            torch_dtype=torch.float16,
            device_map='auto',
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                     device_map='auto',
                                                     output_hidden_states=True,
                                                     trust_remote_code=True,
                                                     load_in_8bit=args.load_kbit == 8)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    # Load FLORES-200 dataset
    dataset = load_dataset('Muennighoff/flores200', 'all', split='devtest',trust_remote_code=True)
    flores_dataset = FloresMultiLangDataset(dataset, languages, tokenizer)
    data_loader = DataLoader(flores_dataset, batch_size=64, shuffle=False)

    # Iterate through batches
    print("\nEvaluating multi-language comparison...")
    for batch in data_loader:
        embeddings_dict = {}

        # Get embeddings for each language
        for lang_name in languages.keys():
            inputs = batch[f"{lang_name}"]
            embeddings_dict[lang_name] = get_embeddings(model, inputs, tokenizer, device, args)
            
        # compare_languages(embeddings_dict, languages)
    print(embeddings_dict)


if __name__ == "__main__":
    main()
