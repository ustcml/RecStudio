import torch
from recstudio.ann import sampler
from recstudio.data import dataset
from recstudio.model import basemodel, loss_func, scorer

r"""
HGN
########

Paper Reference:
    Chen ma, et al. "HGN: Hierarchical Gating Networks for Sequential Recommendation" in KDD2019.
    https://dl.acm.org/doi/abs/10.1145/3292500.3330984
"""

class HGNQueryEncoder(torch.nn.Module):
    
    def __init__(self, fuid, fiid, num_users, embed_dim, max_seq_len, item_encoder, pooling_type='mean') -> None:
        super().__init__()
        self.fuid = fuid
        self.fiid = fiid
        self.item_encoder = item_encoder
        self.pooling_type = pooling_type
        self.user_embedding = torch.nn.Embedding(num_users, embed_dim, 0)
        self.W_g_1 = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_g_2 = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.b_g = torch.nn.Parameter(torch.empty(embed_dim), requires_grad=True)
        self.w_g_3 = torch.nn.Linear(embed_dim, 1, bias=False)
        self.W_g_4 = torch.nn.Linear(embed_dim, max_seq_len)


    def forward(self, batch):
        U = self.user_embedding(batch[self.fuid])
        S = self.item_encoder(batch['in_'+self.fiid])
        S_F = S * torch.sigmoid(self.W_g_1(S) + self.W_g_2(U).view(U.size(0), 1, -1) + self.b_g)
        weight = torch.sigmoid(self.w_g_3(S_F) +  (U@self.W_g_4.weight[:S.size(1)].T).view(U.size(0), -1, 1))    # BxLx1
        S_I = S_F * weight
        if self.pooling_type == 'mean':
            s = S_I.sum(1) / weight.sum(1)
        elif self.pooling_type == 'max':
            s = torch.max(S_I, dim=1).values
        else:
            raise ValueError("`pooling_type` only support `avg` and `max`")
        query = U + s + S.sum(1)
        return query



class HGN(basemodel.BaseRetriever):
    r"""HGN proposes a hierarchical gating network, integrated with the Bayesian Personalized Ranking
    (BPR) to capture both the long-term and short-term user interests. HGN consists of a feature
    gating module, an instance gating module, and an item-item product module."""

    def _get_dataset_class(self):
        r"""The dataset is SeqDataset."""
        return dataset.SeqDataset

        
    def _get_query_encoder(self, train_data):
        return HGNQueryEncoder(self.fuid, self.fiid, train_data.num_users, self.embed_dim, \
            train_data.config['max_seq_len'], self.item_encoder, self.config['pooling_type'])


    def _get_scorer_func(self):
        return scorer.InnerProductScorer()


    def _get_loss_func(self):
        r"""BPR loss is used."""
        return loss_func.BPRLoss()


    def _get_sampler(self, train_data):
        return sampler.UniformSampler(train_data.num_items-1)
