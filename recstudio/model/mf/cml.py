import torch
from recstudio.ann import sampler
from recstudio.data import advance_dataset
from recstudio.model import basemodel, loss_func, scorer


class CML(basemodel.BaseRetriever):

    def add_model_specific_args(parent_parser):
        parent_parser = basemodel.Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('CML')
        parent_parser.add_argument("--negative_count", type=int, default=1, help='negative sampling numbers')
        parent_parser.add_argument("--margin", type=int, default=1, help='margin for CML loss')
        parent_parser.add_argument("--use_rank_weight", action='store_true', help='whether to use rank weight in CML loss')
        return parent_parser

    def _get_dataset_class():
        return advance_dataset.ALSDataset

    def _get_item_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_items, self.embed_dim, padding_idx=0)

    def _get_query_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)

    def _get_score_func(self):
        return scorer.EuclideanScorer()

    def _get_loss_func(self, train_data):
        class CMLoss(loss_func.PairwiseLoss):
            def __init__(self, margin=2, use_rank_weight=False, n_items: int=None):
                super().__init__()
                self.margin = margin
                self.use_rank_weight = use_rank_weight
                self.n_items = n_items - 1  # remove padding

            def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
                pos_score[pos_score == -float("inf")] = float("inf")
                loss = torch.max(torch.max(neg_score, dim=-1).values.unsqueeze(-1) \
                    - pos_score + self.margin, pos_score.new_zeros(pos_score.size(1)))
                if self.use_rank_weight is not None:
                    impostors = neg_score.unsqueeze(1) - pos_score.unsqueeze(-1) + self.margin > 0
                    rank = torch.mean(impostors.to(torch.float32), -1) * self.n_items
                    return torch.mean(loss * torch.log(rank + 1))
                else:
                    return torch.mean(loss)
        return CMLoss(self.config['margin'], self.config['use_rank_weight'], train_data.num_items)


    def _get_sampler(self, train_data):
        return sampler.UniformSampler(train_data.num_items)
