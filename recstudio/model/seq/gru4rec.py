from operator import mod
import torch
from recstudio.ann import sampler
from recstudio.data import dataset
from recstudio.model import basemodel, loss_func, module, scorer

r"""
GRU4Rec
############

Paper Reference:
    Balazs Hidasi, et al. "Session-Based Recommendations with Recurrent Neural Networks" in ICLR2016.
    https://arxiv.org/abs/1511.06939
"""


class GRU4Rec(basemodel.BaseRetriever):
    r"""
    GRU4Rec apply RNN in Recommendation System, where sequential behavior of user is regarded as input
    of the RNN.
    """

    def add_model_specific_args(parent_parser):
        parent_parser = basemodel.Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('GRU4Rec')
        parent_parser.add_argument("--hidden_size", type=int, default=128, help='hidden size of feedforward')
        parent_parser.add_argument("--layer_num", type=int, default=1, help='layer num of transformers')
        parent_parser.add_argument("--dropout_rate", type=float, default=0.2, help='dropout rate')
        parent_parser.add_argument("--negative_count", type=int, default=1, help='negative sampling numbers')
        return parent_parser

    def _get_dataset_class():
        r"""The dataset is SeqDataset."""
        return dataset.SeqDataset

    def _get_query_encoder(self, train_data):
        return (
            module.VStackLayer(
                module.HStackLayer(
                    torch.nn.Sequential(
                        module.LambdaLayer(lambda x: x['in_'+self.fiid]),
                        self.item_encoder,
                        torch.nn.Dropout(self.config['dropout_rate']),
                        module.GRULayer(self.embed_dim, self.config['hidden_size'], self.config['layer_num']),
                    ),
                    module.LambdaLayer(lambda_func=lambda x: x['seqlen']),
                ),
                module.SeqPoolingLayer(pooling_type='last'),
                torch.nn.Linear(self.config['hidden_size'], self.embed_dim)
            )
        )

    def _get_score_func(self):
        return scorer.InnerProductScorer()

    def _get_loss_func(self):
        r"""SoftmaxLoss is used as the loss function."""
        return loss_func.BPRLoss()

    def _get_sampler(self, train_data):
        return sampler.UniformSampler(train_data.num_items)
