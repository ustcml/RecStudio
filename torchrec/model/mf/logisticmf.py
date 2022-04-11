from torchrec.model import basemodel, loss_func, scorer, module
from torchrec.ann import sampler
import torch

class LogisticMF(basemodel.TwoTowerRecommender):
    def build_user_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)

    def config_loss(self):
        def LogitLoss(label, pos_score, log_pos_prob, neg_score, log_neg_prob):
            l1 = self.config['alpha']*pos_score - (1+self.config['alpha'])*torch.nn.functional.softplus(pos_score)
            l2 = torch.nn.functional.softplus(neg_score).mean(dim=-1)
            loss = (l1 - l2).mean()
            return -loss
        return LogitLoss

    def config_scorer(self):
        return scorer.InnerProductScorer()

    def build_sampler(self, train_data):
        return sampler.UniformSampler(train_data.num_items-1, self.score_func)