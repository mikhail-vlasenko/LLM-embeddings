import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA


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


def plot_pca_means(embeddings_dict):
    means = {lang: embeddings.mean(axis=0) for lang, embeddings in embeddings_dict.items()}
    mean_matrix = np.stack(list(means.values()))

    pca = PCA(n_components=2)
    reduced_means = pca.fit_transform(mean_matrix)

    plt.figure(figsize=(12, 8))
    for i, (lang, _) in enumerate(means.items()):
        plt.scatter(reduced_means[i, 0], reduced_means[i, 1], label=lang)
        plt.annotate(lang, (reduced_means[i, 0], reduced_means[i, 1]))

    plt.title("PCA of Language Embedding Means")
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.legend()
    plt.tight_layout()
    plt.show()
