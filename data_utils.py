import torch
from torch.utils.data import Dataset
from sklearn.metrics.pairwise import cosine_similarity
# from main import languages
languages = {
    'English': 'eng_Latn',
    'Chinese_Simplified': 'zho_Hans',
    'Russian': 'rus_Cyrl',
    'Dutch': 'nld_Latn',
    'German': 'deu_Latn'
}

class FloresMultiLangDataset(Dataset):
    def __init__(self, dataset, languages, tokenizer, max_length=512):
        """
        Custom PyTorch Dataset to load multiple languages from FLORES-200.
        Args:
            dataset (datasets.Dataset): The loaded dataset from FLORES-200.
            languages (dict): Dictionary of language names and their codes (e.g., {'English': 'eng_Latn', ...}).
            tokenizer (AutoTokenizer): The tokenizer for encoding text.
            max_length (int): Maximum sequence length for tokenization.
        """
        self.dataset = dataset
        self.languages = languages
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sentences = {}
        for lang_name, lang_code in self.languages.items():
            sentence = self.dataset[idx][f"sentence_{lang_code}"]
            sentences[lang_name] = sentence

            # # Tokenize the sentence
            # tokenized_input = self.tokenizer(
            #     template.format(sentence=sentence),
            #     return_tensors='pt',
            #     padding=True,
            #     truncation=True,
            #     max_length=self.max_length
            # )
            # tokenized_input = {k: v.to(device) for k, v in tokenized_input.items()}
            # sentences[f"{lang_name}_inputs"] = tokenized_input

        return sentences



def compare_languages(embeddings_dict, languages):
    """
    Compare each language embedding with every other language embedding and print cosine similarity scores.
    Args:
        embeddings_dict (dict): Dictionary of embeddings for each language.
        languages (dict): Dictionary of language names and codes.
    """
    lang_names = list(languages.keys())
    for i, lang1 in enumerate(lang_names):
        for lang2 in lang_names[i + 1:]:
            sim_score = cosine_similarity(embeddings_dict[lang1], embeddings_dict[lang2])
            print(f"Similarity between {lang1} and {lang2}: {sim_score[0][0]:.4f}")


def collate_fn(batch):
    """
    Custom collate function for padding the inputs in a batch and including the actual sentences.
    """
    batch_dict = {}

    for lang_name in languages.keys():
        # Extract all the tokenized inputs, attention masks, and actual sentences for this language in the batch
        inputs = [item[f"{lang_name}_inputs"]['input_ids'].squeeze(0) for item in batch]
        attention_masks = [item[f"{lang_name}_inputs"]['attention_mask'].squeeze(0) for item in batch]
        sentences = [item[lang_name] for item in batch]  # Actual sentences

        # Pad the sequences for each language
        padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
        padded_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)

        # Store padded inputs, masks, and sentences
        batch_dict[f"{lang_name}_inputs"] = {
            'input_ids': padded_inputs,
            'attention_mask': padded_masks
        }
        # batch_dict[f"{lang_name}_attention_masks"] = padded_masks
        batch_dict[lang_name] = sentences

    return batch_dict
