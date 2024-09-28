import numpy as np
import pandas as pd

def normalize_embeddings(embeddings):
    return (embeddings.T / np.linalg.norm(embeddings, axis=1)).T

def compute_similarity(embedding, embeddings_to_compare):
    return embedding @ embeddings_to_compare.T

def recall_at_1(similarity, targets):
    similarity = np.argsort(similarity, axis=1)[:,::-1]

    correct = 0

    for i in range(similarity.shape[0]):
        if targets[i] == similarity[i,0]:
            correct += 1
    
    recall_at_1 = correct / similarity.shape[0]
    return recall_at_1

def recall_at_k(similarity, targets, k=3):
    similarity = np.argsort(similarity, axis=1)[:,::-1]

    correct = 0

    for i in range(similarity.shape[0]):
        if targets[i] in similarity[i, :k]:
            correct += 1
    
    recall_at_k = correct / similarity.shape[0]
    return recall_at_k


def evaluate_translation_accuracy(embeddings_dict, target_language, k=3):
    total_recall_1_per_language = {lang: 0 for lang in embeddings_dict if lang != target_language}
    total_recall_k_per_language = {lang: 0 for lang in embeddings_dict if lang != target_language}
    total_pairs = len(embeddings_dict[target_language])  # Assuming target language is the reference

    # Loop through each sentence embedding in the target language
    for i, target_embedding in enumerate(embeddings_dict[target_language]):
        for lang_name, lang_embeddings in embeddings_dict.items():
            if lang_name == target_language:
                continue  # Skip comparing the target language with itself

            # Compute similarity between the current target sentence embedding and all sentence embeddings in the other language
            similarity = compute_similarity(target_embedding.reshape(1, -1), lang_embeddings)

            # Calculate recall@1 and recall@k for this specific sentence
            recall_1 = recall_at_1(similarity, [i])  # i is the target index for the corresponding sentence
            recall_k = recall_at_k(similarity, [i], k=k)

            # Accumulate total recall values for each language
            total_recall_1_per_language[lang_name] += recall_1
            total_recall_k_per_language[lang_name] += recall_k

    # Calculate average recall@1 and recall@k for each language
    avg_recall_1_per_language = {lang: total_recall_1_per_language[lang] / total_pairs for lang in total_recall_1_per_language}
    avg_recall_k_per_language = {lang: total_recall_k_per_language[lang] / total_pairs for lang in total_recall_k_per_language}

    # Save the average recall scores into a table
    results_table = []
    for lang_name in avg_recall_1_per_language:
        results_table.append({
            'Target Language': target_language,
            'Compared Language': lang_name,
            'Avg Recall@1': avg_recall_1_per_language[lang_name],
            'Avg Recall@k': avg_recall_k_per_language[lang_name]
        })

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results_table)
    print(results_df)

    return results_table


embs = np.array([[1.,2.,3.], 
                [3.,4.,5.]])
embs_to_compare = np.array([[1.,2.,3.], 
                            [0.,1.,0.], 
                            [2.,1.,3.]])

targets = ([0, 2]) #embs[0]'s target is embs_to_compare[0], embs[1]'s is embs_to_compare[2]

# normalize
embs = normalize_embeddings(embs)
embs_to_compare = normalize_embeddings(embs_to_compare)

similarity = compute_similarity(embs, embs_to_compare)
print(recall_at_1(similarity, targets))
print(recall_at_k(similarity, targets, k=2))

