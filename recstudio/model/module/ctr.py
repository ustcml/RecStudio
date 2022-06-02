import torch
import torch.nn as nn
import torch.nn.functional as F
from recstudio.model.module import SeqPoolingLayer
from typing import Set



# def create_embeddings(feat_cols, feat2type, feat2vocab_size, embed_dims):
#     token_feat_cols = list(
#         filter(lambda x: feat2type[x]=='token', feat_cols)) if len(feat_cols) else []

#     token_seq_feat_cols = list(
#         filter(lambda x: feat2type[x]=='token_seq', feat_cols)) if len(feat_cols) else []

#     assert len(embed_dims) == 1 or len(embed_dims) == len(token_feat_cols+token_seq_feat_cols), \
#         "expecting embed_dims to be an interger or a dict with the same length as sparse features."\
#         f"but get length {len(embed_dims)} while sparse feature length is {len(token_feat_cols+token_seq_feat_cols)}."
#     if len(embed_dims) == 1:
#         embed_dims = {feat: embed_dims for feat in token_feat_cols+token_seq_feat_cols}
    
#     embed_dict = nn.ModuleDict(
#         {feat: nn.Embedding(feat2vocab_size(feat), embed_dims[feat], 0) for feat in token_feat_cols+token_seq_feat_cols}
#     )

#     return embed_dict

class DenseEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.norm = torch.nn.BatchNorm1d(num_embeddings)
        self.W = torch.nn.Embedding(1, embedding_dim)


    def forward(self, input):
        input = self.norm(input)
        emb = input.unsqueeze(-1) @ self.W.weight
        return emb



class Embeddings(torch.nn.Module):

    def __init__(self, fields: Set, field2types, field2vocab_sizes, embed_dim, rating_field, reduction='mean'):
        super(Embeddings, self).__init__()
        self.embed_dim = embed_dim
        self.reduction = reduction
        self.dense_field_list = []
        self.embeddings = torch.nn.ModuleDict()

        for f in fields:
            t = field2types[f]
            if f == rating_field:
                continue
            if (t=="token" or t=='token_seq'):
                self.embeddings[f] = torch.nn.Embedding(
                    field2vocab_sizes[f], self.embed_dim, 0)
            elif (t=="float"):
                self.dense_field_list.append(f)
        self.num_features = len(self.embeddings)

        if len(self.dense_field_list) > 0:
            # TODO: deepctr, other dense embedding methods
            # TODO: whether to do minmax on the whole dataset
            self.dense_embedding= DenseEmbedding(len(self.dense_field_list), self.embed_dim)
            self.num_features += len(self.dense_field_list)

        self.seq_pooling_layer = SeqPoolingLayer(reduction, keepdim=True)


    def forward(self, batch):
        embs = []
        for f in self.embeddings:
            d = batch[f]
            if d.dim() > 1:
                length = (d>0).float().sum(dim=-1, keepdim=False) 
                embs.append(self.seq_pooling_layer(self.embeddings[f](d), length))
            else:
                embs.append(self.embeddings[f](d.view(-1, 1)))
        
        if len(self.dense_field_list) > 0:
            dense_field = []
            for f in self.dense_field_list:
                dense_field.append(batch[f])
            dense_field = torch.stack(dense_field, dim=1)   # BxN
            dense_emb = self.dense_embedding(dense_field) # BxNxD
            embs.append(dense_emb)

        emb = torch.cat(embs, dim=1)    # BxNxnum_fieldsxD
        return emb


class LinearLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1,), requires_grad=True)

    def forward(self, inputs):
        sum_of_embs = torch.sum(inputs, dim=1).squeeze()
        return sum_of_embs + self.bias



class FMLayer(nn.Module):

    def forward(self, inputs):
        square_of_sum = torch.sum(inputs, dim=1) ** 2
        sum_of_square = torch.sum(inputs ** 2, dim=1)
        cross_term = 0.5 * (square_of_sum - sum_of_square)
        return cross_term


