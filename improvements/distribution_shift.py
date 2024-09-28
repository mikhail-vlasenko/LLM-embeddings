def subtract_mean(embeddings_dict):
    for lang in embeddings_dict:
        embeddings_dict[lang] -= embeddings_dict[lang].mean(axis=0)
