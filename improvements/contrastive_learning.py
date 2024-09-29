import torch
from torch import nn


@torch.no_grad()
def apply_model(embedding_dict, normalize=True):
    num_sentences, embedding_dim = embedding_dict[list(embedding_dict.keys())[0]].shape

    train_size = int(num_sentences * 0.7)
    model = torch.load("result/mlp_english_prompt.pt", weights_only=False, map_location="cuda")
    model.eval()
    model.to("cuda")
    def normalize_tensor(tensor):
        if normalize:
            return tensor / torch.norm(tensor, dim=-1, keepdim=True)
        else:
            return tensor

    for lang in embedding_dict:
        # embedding_dict[lang] = normalize_tensor(model(torch.tensor(embedding_dict[lang][train_size:], device="cuda"))).cpu().numpy()
        embedding_dict[lang] = embedding_dict[lang][train_size:]
