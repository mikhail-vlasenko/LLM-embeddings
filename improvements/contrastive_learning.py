import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


@torch.no_grad()
def apply_model(embedding_dict, model_path, normalize=True):
    num_sentences, embedding_dim = embedding_dict[list(embedding_dict.keys())[0]].shape

    train_size = int(num_sentences * 0.7)  # fixme: magic number
    model = torch.load(model_path, weights_only=False, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    def normalize_tensor(tensor):
        if normalize:
            return tensor / torch.norm(tensor, dim=-1, keepdim=True)
        else:
            return tensor

    for lang in embedding_dict:
        embedding_dict[lang] = normalize_tensor(model(torch.tensor(embedding_dict[lang][train_size:], device="cuda"))).cpu().numpy()
        # baseline
        # embedding_dict[lang] = embedding_dict[lang][train_size:]
