import pickle

import matplotlib.pyplot as plt
import seaborn as sns

def save_embeddings(embeddings_dict, filename):
    with open(filename, 'wb') as f:
        pickle.dump(embeddings_dict, f)


def load_embeddings(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def plot_heatmap(data, title):
    plt.figure(figsize=(12, 10), dpi=160)
    sns.heatmap(data, cmap='Blues', annot=True, fmt=".2f")
    plt.title(title)
    plt.show()
