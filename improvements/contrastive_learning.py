# enable importing from parent directory
import math
import os
import sys
parent_dir = os.path.split(os.getcwd())[0]
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# import other modules
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import itertools
from utils import load_embeddings
from tqdm import tqdm
from functools import partial


class EmbeddingsDataset(Dataset):
    def __init__(self, embedding_dict, prompt_type="eng"):
        """
        embedding_dict: the dict with all the embeddings
        eng_prompt: if true, uses English prompting, otherwise self-prompting
        """
        self.embedding_dict = embedding_dict
        self.prompt_type = prompt_type
        self.languages = list(embedding_dict.keys())
        self.num_sencences, self.embedding_dim = embedding_dict[self.languages[0]].shape
        
    def __len__(self):
        return self.num_sencences
    
    def __getitem__(self, idx):
        """
        returns a dataset example of size (num_langs, emb_dim) which represents the embeddings of 
        the same sentence in each of the languages
        """
        return torch.stack([torch.from_numpy(self.embedding_dict[lang][idx]) for lang in self.languages])

def get_random_idx_pairs(batch_size, num_languages, num_pos_pairs, num_neg_pairs):
    """
    returns a dict with positive and negative pairs of indices (a single index is a tuple (batch_idx, lang_idx)):
    {
        "pos": [(idx1, idx2), ...],
        "neg": [(idx1, idx2), ...],
    }
    where e.g. idx1 is (batch_idx_1, lang_idx_1)
    """
    all_pos_idx_pairs = [
        ((b1, l1), (b2, l2))
        for ((b1, l1), (b2, l2))
        in list(itertools.combinations([(b, l) for b in range(batch_size) for l in range(num_languages)], 2))
        if b1 == b2
    ]
    all_neg_idx_pairs = [
        ((b1, l1), (b2, l2))
        for ((b1, l1), (b2, l2))
        in list(itertools.combinations([(b, l) for b in range(batch_size) for l in range(num_languages)], 2))
        if b1 != b2
    ]

    return {
        "pos": random.sample(all_pos_idx_pairs, num_pos_pairs),
        "neg": random.sample(all_neg_idx_pairs, num_neg_pairs),
    }

def collate_fn(embeddings, args):
    """
    embeddings: a tensor of size (batch_size, num_languages, embedding_dim)

    returns a tuple of:
    - a tensor of size (batch_size, embedding_dim)
    - a list of positive index pairs
    - a list of negative index pairs
    """
    embeddings = torch.stack(embeddings)
    batch_size, num_languages, _ = embeddings.shape
    positive_coef = args.contrastive_loss_positive_coef  # increasing this to 8 helps for the norm-based loss but not for the cosine-based loss
    idx_pairs = get_random_idx_pairs(batch_size, num_languages, num_pos_pairs=batch_size // positive_coef, num_neg_pairs=batch_size * (positive_coef - 1) // positive_coef)
    # idx_pairs = get_random_idx_pairs(batch_size, num_languages, num_pos_pairs=batch_size // 2, num_neg_pairs=batch_size // 2)
    pos_idx_pairs, neg_idx_pairs = idx_pairs["pos"], idx_pairs["neg"]

    output_embeddings = {}
    for ((b1, l1), (b2, l2)) in pos_idx_pairs + neg_idx_pairs:
        output_embeddings[(b1, l1)] = embeddings[b1, l1]
        output_embeddings[(b2, l2)] = embeddings[b2, l2]
    indices = list(output_embeddings.keys())
    
    # Convert index pairs
    pos_idx_pairs = [(indices.index(idx1), indices.index(idx2)) for (idx1, idx2) in pos_idx_pairs]
    neg_idx_pairs = [(indices.index(idx1), indices.index(idx2)) for (idx1, idx2) in neg_idx_pairs]
    
    return torch.stack(list(output_embeddings.values())), pos_idx_pairs, neg_idx_pairs


class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super(LowRankLinear, self).__init__()
        self.u = nn.Parameter(torch.Tensor(in_features, rank))
        self.v = nn.Parameter(torch.Tensor(rank, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.u, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.u)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return input @ self.u @ self.v + self.bias


class MLP(nn.Module):
    def __init__(self, input_dim, n_hidden, hidden_dim, output_dim, rank=-1, do_skip_connections=False):
        # rank = -1 is full rank
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()
        self.activation = nn.ReLU
        self.do_skip_connections = do_skip_connections
        if do_skip_connections:
            assert input_dim == output_dim == hidden_dim, "Skip connections require input_dim == output_dim == hidden_dim"

        def linear_layer(dim_a, dim_b):
            if rank != -1:
                return LowRankLinear(dim_a, dim_b, rank=rank)
            else:
                return nn.Linear(dim_a, dim_b)

        dim_a = input_dim
        for k in range(n_hidden):
            self.layers.append(nn.Sequential(linear_layer(dim_a, hidden_dim), self.activation()))
            dim_a = hidden_dim

        self.layers.append(linear_layer(dim_a, output_dim))
    
    def forward(self, x):
        for layer in self.layers:
            if self.do_skip_connections:
                x = x + layer(x)
            else:
                x = layer(x)
        return x

# Define the contrastive loss
# def contrastive_loss(embeddings, pos_idx_pairs, neg_idx_pairs, margin=1.0):
#     pos_loss = sum(
#         torch.norm(embeddings[idx1] - embeddings[idx2], p=2) ** 2
#         for idx1, idx2 in pos_idx_pairs
#     )

#     neg_loss = sum(
#         max(0, margin - torch.norm(embeddings[idx1] - embeddings[idx2], p=2)) ** 2
#         for idx1, idx2 in neg_idx_pairs
#     )

#     return (pos_loss + neg_loss) / (len(pos_idx_pairs) + len(neg_idx_pairs))

def contrastive_loss(embeddings, pos_idx_pairs, neg_idx_pairs, args):
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    margin = args.contrastive_loss_margin
    
    pos_loss = sum(
        (1 - cos(embeddings[idx1], embeddings[idx2])) ** 2
        for idx1, idx2 in pos_idx_pairs
    )

    neg_loss = sum(
        max(0, cos(embeddings[idx1], embeddings[idx2]) - margin) ** 2
        for idx1, idx2 in neg_idx_pairs
    )
    
    C = args.contrastive_loss_C
    return 2*(C*pos_loss + (1-C)*neg_loss) / (len(pos_idx_pairs) + len(neg_idx_pairs))
    
def normalize_tensor(tensor):
    return tensor / torch.norm(tensor, dim=-1, keepdim=True)
 
def contrastive_learning(embedding_dict, prompt_type, args):
    save_model_path = f"result/mlp_{prompt_type}_prompt.pt"
    num_sentences, embedding_dim = embedding_dict[list(embedding_dict.keys())[0]].shape
    batch_size = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mlp = MLP(
        input_dim=embedding_dim,
        n_hidden=args.mlp_n_hidden,
        hidden_dim=args.mlp_hidden_dim,
        output_dim=args.mlp_output_dim,
        rank=args.rank,
        do_skip_connections=args.skip_connections
    ).to(device)

    try: # try to load the mlp
        if not args.reuse_mlp: 
            raise Exception() # go to the except case

        mlp.load_state_dict(torch.load(save_model_path, weights_only=True))
    
    except: 
        # define train,val,test sizes (70%, 30%)
        train_size, val_size = int(num_sentences*1.), int(num_sentences*0.)

        def get_split(embedding_dict, start_idx, end_idx):
            return {k: v[start_idx:end_idx] for k, v in embedding_dict.items()}

        # make the datasets
        train_set = EmbeddingsDataset(get_split(embedding_dict, start_idx=0, end_idx=train_size), prompt_type=prompt_type)
        # val_set = EmbeddingsDataset(get_split(embedding_dict, start_idx=train_size, end_idx=train_size + val_size), prompt_type=prompt_type)

        # do the training
        epochs = args.mlp_train_epochs
        
        # make the dataloaders
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=partial(collate_fn, args=args), drop_last=True)
        # val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=partial(collate_fn, args=args), drop_last=True)

        optimizer = torch.optim.SGD(mlp.parameters(), lr=0.001)

        # training loop: 
        training_losses, val_losses = [], []
        for epoch in range(1, epochs + 1):

            # train one epoch
            mlp.train()
            train_loss = 0.0
            for embeddings, pos_idx_pairs, neg_idx_pairs in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
                optimizer.zero_grad()
                embeddings = mlp(embeddings.to(device))
                loss = contrastive_loss(embeddings, pos_idx_pairs, neg_idx_pairs, args)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # save training loss
            training_losses.append(train_loss / len(train_loader))

            # eval one epoch
            mlp.eval()
#             val_loss = 0.0
#             for (embeddings, pos_idx_pairs, neg_idx_pairs) in tqdm(val_loader, desc=f"Validation Epoch {epoch}"):
#                 embeddings = mlp(embeddings.to(device))
#                 loss = contrastive_loss(embeddings, pos_idx_pairs, neg_idx_pairs)
#                 val_loss += loss.item()

#             # save val loss
#             val_losses.append(val_loss / len(val_loader))

        # save model
        torch.save(mlp.state_dict(), save_model_path)
    return mlp
        
def apply_mlp(embedding_dict, mlp):
    # pass the embeddings through the mlp
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for lang in embedding_dict:
        embedding_dict[lang] = normalize_tensor(mlp(torch.tensor(embedding_dict[lang]).to(device))).cpu().detach().numpy()