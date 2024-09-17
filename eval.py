import numpy as np

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

