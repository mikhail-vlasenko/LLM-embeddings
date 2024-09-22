import pickle


def save_embeddings(embeddings_dict, filename):
    with open(filename, 'wb') as f:
        pickle.dump(embeddings_dict, f)


def load_embeddings(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
