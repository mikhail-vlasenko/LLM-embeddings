def subtract_mean(train_embeddings_dict, test_embeddings_dict):
    for lang in test_embeddings_dict:
        test_embeddings_dict[lang] -= train_embeddings_dict[lang].mean(axis=0)
