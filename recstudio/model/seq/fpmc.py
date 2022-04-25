from recstudio.model import basemodel, loss_func, scorer
from recstudio.data import dataset
from recstudio.ann import sampler
import torch

r"""
FPMC
#########

Paper Reference:
    Steffen Rendle, et al. "Factorizing personalized Markov chains for next-basket recommendation" in WWW2010.
    https://dl.acm.org/doi/10.1145/1772690.1772773
"""

class FPMC(basemodel.TwoTowerRecommender):
    r"""
    | FPMC is based on personalized transition graphs over underlying Markov chains. It 
      factorizes the transition cube with a pairwise interaction model which is a special case of
      the Tucker Decomposition.
    """
    def init_model(self, train_data):
        r"""FPMC add a feature matrix for the last transition."""
        super().init_model(train_data)
        self.VL_emb = torch.nn.Embedding(train_data.num_items, self.embed_dim, padding_idx=0)

    def get_dataset_class(self):
        r"""The dataset FPMC used is SeqDataset."""
        return dataset.SeqDataset

    def build_user_encoder(self, train_data):
        r"""The user encoder is just an Embedding layer."""
        VU_emb = torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)
        return VU_emb

    def build_item_encoder(self, train_data):
        r""" The item encoder is just an Embedding layer.
        
        | The dimension of item embedding is twice as user embedding, because the query is
          concatencated with the feature of user and the last transition.
        """
        return torch.nn.Embedding(train_data.num_items, self.embed_dim*2, padding_idx=0)

    def construct_query(self, batch_data):
        r"""The query is concatencated with the feature of user and the last transition."""
        user_hist = batch_data['in_item_id']
        seq_len = batch_data['seqlen'] - 1
        user_emb = self.user_encoder(self.get_user_feat(batch_data))    # B x D
        last_item_id = torch.gather(user_hist, dim=-1, index=seq_len.unsqueeze(1))
        last_item_emb = self.VL_emb(last_item_id).squeeze(1)   # B x D
        return torch.cat([user_emb, last_item_emb], dim=-1) # B x D

    def config_loss(self):
        r"""The loss function is BPR loss."""
        return loss_func.BPRLoss()

    def config_scorer(self):
        r"""Inner Product is used as the score function."""
        return scorer.InnerProductScorer()

    def build_sampler(self, train_data):
        r"""Uniform sampler is used to generate negative samples."""
        return sampler.UniformSampler(train_data.num_items-1, self.score_func)