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


def plot_pca_means_and_variances(embeddings_dict):
    means = {lang: embeddings.mean(axis=0) for lang, embeddings in embeddings_dict.items()}
    variances = {lang: embeddings.var(axis=0) for lang, embeddings in embeddings_dict.items()}

    mean_matrix = np.stack(list(means.values()))
    var_matrix = np.stack(list(variances.values()))

    pca = PCA(n_components=2)
    reduced_means = pca.fit_transform(mean_matrix)
    reduced_vars = pca.transform(var_matrix)

    plt.figure(figsize=(12, 8))
    for i, (lang, _) in enumerate(means.items()):
        plt.scatter(reduced_means[i, 0], reduced_means[i, 1], label=f"{lang} (Mean)")
        plt.annotate(lang, (reduced_means[i, 0], reduced_means[i, 1]))

        # Plot variance as error bars
        plt.errorbar(reduced_means[i, 0], reduced_means[i, 1],
                     xerr=np.sqrt(abs(reduced_vars[i, 0])),
                     yerr=np.sqrt(abs(reduced_vars[i, 1])),
                     fmt='none', alpha=0.3, capsize=5)

    plt.title("PCA of Language Embedding Means with Variances")
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.legend()
    plt.tight_layout()
    plt.show()
