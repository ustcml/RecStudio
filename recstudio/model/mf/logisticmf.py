import torch
from recstudio.ann import sampler
from recstudio.data import dataset
from recstudio.model import basemodel, scorer, loss_func

class LogisticMF(basemodel.BaseRetriever):

    def _get_dataset_class(self):
        return dataset.MFDataset

    def _get_item_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_items, self.embed_dim, padding_idx=0)

    def _get_query_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)

    def _get_score_func(self):
        return scorer.InnerProductScorer()

    def _get_loss_func(self):
        class LogitLoss(loss_func.PairwiseLoss):
            def __init__(self, alpha) -> None:
                super().__init__()
                self.alpha = alpha

            def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
                l1 = self.alpha * pos_score - (1+self.alpha) * torch.nn.functional.softplus(pos_score)
                l2 = torch.nn.functional.softplus(neg_score).mean(dim=-1)
                loss = (l1 - l2).mean()
                return - loss

        return LogitLoss(self.config['alpha'])


    def _get_sampler(self, train_data):
        return sampler.UniformSampler(train_data.num_items-1)
