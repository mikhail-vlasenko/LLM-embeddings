import argparse
import torch
import numpy as np
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics.pairwise import cosine_similarity


template = 'This sentence: "{sentence}" means in one word:'  # 100% accuracy on sample data
# template = '{sentence}'  # 10% accuracy on sample data


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

    # Load sentences from CSV
    sentence_pairs = load_csv(args.csv_path)
    english_sentences = [pair[0] for pair in sentence_pairs]
    russian_sentences = [pair[1] for pair in sentence_pairs]

    # Get embeddings
    print("Generating embeddings for English sentences...")
    english_embeddings = get_embeddings(model, tokenizer, english_sentences, device, args)
    print("Generating embeddings for Russian sentences...")
    russian_embeddings = get_embeddings(model, tokenizer, russian_sentences, device, args)

    # Evaluate translation accuracy
    correct_matches = 0
    total_pairs = len(sentence_pairs)

    print("\nEvaluating translation accuracy...")
    for i, (eng_sentence, true_rus_sentence) in enumerate(sentence_pairs):
        most_similar_idx = find_most_similar(english_embeddings[i], russian_embeddings)
        predicted_rus_sentence = russian_sentences[most_similar_idx]

        if predicted_rus_sentence == true_rus_sentence:
            correct_matches += 1
        else:
            print(f"\nMismatch for English sentence: {eng_sentence}")
            print(f"True Russian translation: {true_rus_sentence}")
            print(f"Predicted Russian translation: {predicted_rus_sentence}")

    accuracy = correct_matches / total_pairs
    print(f"\nAccuracy: {accuracy:.2%} ({correct_matches}/{total_pairs})")


if __name__ == "__main__":
    main()
