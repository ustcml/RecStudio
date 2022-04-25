from recstudio.model import basemodel, loss_func, scorer
from recstudio.ann import sampler
import torch
class BPR(basemodel.TwoTowerRecommender):
    r"""
    | BPR is a kind of matrix factorization method, which optimizes the scores order between
      interacted and uninteracted items.

    Model hyper parameters:
        - ``embed_dim(int)``: The dimension of embedding layers. Default: ``64``.
    """
    def build_user_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)
    
    def config_loss(self):
        return loss_func.BPRLoss()
    
    def config_scorer(self):
        return scorer.InnerProductScorer()

    def build_sampler(self, train_data):
        return sampler.UniformSampler(train_data.num_items-1, self.score_func)