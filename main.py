import argparse
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from datasets import load_dataset
from data import FloresMultiLangDataset, compare_languages, collate_fn
from eval import evaluate_translation_accuracy
from improvements.distribution_shift import subtract_mean
from improvements.contrastive_learning import contrastive_learning, apply_mlp
from utils import save_embeddings, plot_heatmap, load_embeddings, plot_pca_means_and_variances
from tqdm import tqdm
from pathlib import Path

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


def get_embeddings(model, sentences, language, args):
    if args.self_prompts:
        sentences = [template[language].format(sentence=sentence) for sentence in sentences]
    else:
        sentences = [template["English"].format(sentence=sentence) for sentence in sentences]

    embeddings = model.encode(sentences, convert_to_tensor=True)
    return embeddings.cpu().numpy()


def make_embeddings_dict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device)

    # Load dataset based on argument
    dataset = load_dataset(args.dataset_name, 'all', split='devtest',trust_remote_code=True)

    # Select only max_samples if specified
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    flores_dataset = FloresMultiLangDataset(dataset, languages)
    data_loader = DataLoader(flores_dataset, batch_size=256, shuffle=False)

    embeddings_dict = {target_language: np.array([]) for target_language in languages.keys()}

    # Generate embeddings for each language
    print("Generating embeddings for all languages")

    # Iterate through batches and generate embeddings
    for batch in tqdm(data_loader, desc="Embedding Progress", leave=True):
        for lang_name in languages.keys():

            inputs = batch[f"{lang_name}"]
            embeddings = get_embeddings(model, inputs, lang_name, args)

            if len(embeddings_dict[lang_name]) == 0:
                # If it's the first batch, initialize the array
                embeddings_dict[lang_name] = embeddings

    return embeddings_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="", help="Path to where to save the results to")
    parser.add_argument("--dataset_name", type=str, default='facebook/flores', help="Name of the dataset to load")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to load from the dataset")
    parser.add_argument("--self_prompts", default=False, action="store_true",
                        help="Use prompt template in the same language")
    parser.add_argument("--k", type=int, default=1, help="The number of most similar items for recall (k)")

    args = parser.parse_args()
    args.self_prompts = True

    name_suffix = 'self_prompts' if args.self_prompts else 'english_prompts'

    embeddings_dict = make_embeddings_dict(args)

    subtract_mean(embeddings_dict, embeddings_dict)

    all_results = []

    for target_language in languages.keys():
        print(f"\nEvaluating target language: {target_language}")
        results_table = evaluate_translation_accuracy(embeddings_dict, target_language, k=args.k)
        all_results += results_table

    df = pd.DataFrame(all_results)
    df.to_csv(f'average_accuracy_results_{name_suffix}.csv', index=False)

    pivot_df = df.pivot(index='Target Language', columns='Compared Language', values=f'Avg Recall@{args.k}')
    pivot_df.to_csv(f'average_accuracy_table_{name_suffix}.csv', index=True)

    print("\nFinal Results:")
    print(pivot_df)

    print(f"Overall mean over the primary metric: {np.nanmean(pivot_df.to_numpy())}")

    args_dict = vars(args)
    args_dict['metric_mean'] = np.nanmean(pivot_df.to_numpy())
    if args.save_path != "":
        Path(args.save_path).mkdir(parents=True, exist_ok=True)
        with open(args.save_path + 'metrics.txt', 'a') as file:
            file.write(str(args_dict) + '\n')


if __name__ == "__main__":
    main()
