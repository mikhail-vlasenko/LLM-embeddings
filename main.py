import argparse
import torch
import numpy as np
import pandas as pd
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from datasets import load_dataset
from data import FloresMultiLangDataset, compare_languages, collate_fn
from eval import evaluate_translation_accuracy
from utils import save_embeddings, plot_heatmap, load_embeddings
from tqdm import tqdm

# Define the sentence template for each language
template = {
    'English': 'This sentence: "{sentence}" means in one word:',
    'Chinese_Simplified': '这句话: "{sentence}" 用一个词来表示是:',
    'Russian': 'Это предложение: "{sentence}" означает одним словом:',
    'German': 'Dieser Satz: "{sentence}" bedeutet mit einem Wort:',
    'Korean': '이 문장: "{sentence}"는 한 단어로 다음과 같은 의미를 갖습니다:',
    'Dutch': 'Deze zin: "{sentence}" betekent in één woord:',
    'French': 'Cette phrase: "{sentence}" signifie en un mot:',
    'Spanish': 'Esta frase: "{sentence}" significa en una palabra:',
    'Italian': 'Questa frase: "{sentence}" significa in una parola:',
    'Polish': 'To zdanie: "{sentence}" oznacza jednym słowem:',
}

languages = {
    'English': 'eng_Latn',
    'Chinese_Simplified': 'zho_Hans',
    'Russian': 'rus_Cyrl',
    'German': 'deu_Latn',
    'Korean': 'kor_Hang',
    'Dutch': 'nld_Latn',
    'French': 'fra_Latn',
    'Spanish': 'spa_Latn',
    'Italian': 'ita_Latn',
    'Polish': 'pol_Latn',
}

def load_csv(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        return list(reader)

def get_embeddings(model, tokenizer, sentences, device, language, args):
    embeddings = []
    for sentence in sentences:
        if args.self_prompts:
            sentence = template[language].format(sentence=sentence)
        else:
            sentence = template["English"].format(sentence=sentence)
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
                outputs = outputs.float()

            embeddings.append(outputs.cpu().numpy())

    return np.vstack(embeddings)


def find_most_similar(query_embedding, target_embeddings):
    similarities = cosine_similarity(query_embedding.reshape(1, -1), target_embeddings)[0]
    return np.argmax(similarities)


def make_embeddings_dict(args):
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

    # Load dataset based on argument
    dataset = load_dataset(args.dataset_name, 'all', split='devtest')

    # Select only max_samples if specified
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    flores_dataset = FloresMultiLangDataset(dataset, languages)
    data_loader = DataLoader(flores_dataset, batch_size=args.batch_size, shuffle=False)

    embeddings_dict = {target_language: np.array([]) for target_language in languages.keys()}

    # Stage 1: Save embeddings for each language and target language
    print("Generating embeddings for all languages")

    # Iterate through batches and generate embeddings
    for batch in tqdm(data_loader, desc="Embedding Progress", leave=True):
        for lang_name in languages.keys():

            inputs = batch[f"{lang_name}"]
            embeddings = get_embeddings(model, tokenizer, inputs, device, lang_name, args)

            if len(embeddings_dict[lang_name]) == 0:
                # If it's the first batch, initialize the array
                embeddings_dict[lang_name] = embeddings

            else:
                # Concatenate the new embeddings with the existing array
                embeddings_dict[lang_name] = np.concatenate(
                    (embeddings_dict[lang_name], embeddings),
                    axis=0
                )
    return embeddings_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="microsoft/Phi-3.5-mini-instruct", help="Transformers' model name or path")
    parser.add_argument("--csv_path", type=str, default="sample_data.csv", help="Path to the CSV file with sentence pairs")
    parser.add_argument("--load_kbit", type=int, choices=[4, 8, 16], default=16, help="Load model in kbit")
    parser.add_argument('--avg', action='store_true', help="Use average pooling for embeddings")
    parser.add_argument("--k", type=int, default=3, help="The number of most similar items for recall (k)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for DataLoader")
    parser.add_argument("--dataset_name", type=str, default='facebook/flores', help="Name of the dataset to load")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to load from the dataset")
    parser.add_argument("--self_prompts", default=False, action="store_true", help="Use prompt template in the same language")
    parser.add_argument("--load_from_file", type=str, default="", help="Load embeddings from file")

    args = parser.parse_args()
    # args.self_prompts = True
    args.load_from_file = "embedding_english_prompts.pkl"

    name_suffix = 'self_prompts' if args.self_prompts else 'english_prompts'

    if args.load_from_file:
        embeddings_dict = load_embeddings(args.load_from_file)
    else:
        embeddings_dict = make_embeddings_dict(args)
        save_embeddings(embeddings_dict, f"embedding_{name_suffix}.pkl")

    # Stage 2: Evaluate translation accuracy using the stored embeddings
    # Initialize list to store results
    all_results = []

    for target_language in languages.keys():
        print(f"\nEvaluating target language: {target_language}")

        # Evaluate using the embeddings in the dictionary for current target language vs. other languages
        results_table = evaluate_translation_accuracy(embeddings_dict, target_language, k=args.k)

        # Store the results for each comparison
        all_results += results_table

    # Convert the list to a DataFrame
    df = pd.DataFrame(all_results)

    df.to_csv(f'average_accuracy_results_{name_suffix}.csv', index=False)

    # Dynamically pivot the DataFrame based on target and compared languages
    pivot_df = df.pivot(index='Target Language', columns='Compared Language', values='Avg Recall@k')

    # Save the DataFrame to a CSV file
    pivot_df.to_csv(f'average_accuracy_table_{name_suffix}.csv', index=True)

    print("\nFinal Results:")
    # Show the result
    print(pivot_df)

    plot_heatmap(pivot_df, f'Average Recall@{args.k} for {name_suffix}')

if __name__ == "__main__":
    main()
