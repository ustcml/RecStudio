import torch
from recstudio.ann import sampler
from recstudio.data import dataset
from recstudio.model import basemodel, loss_func, module, scorer
from recstudio.model.module.layers import LambdaLayer, SeqPoolingLayer

r"""
FPMC
#########

Paper Reference:
    Steffen Rendle, et al. "Factorizing personalized Markov chains for next-basket recommendation" in WWW2010.
    https://dl.acm.org/doi/10.1145/1772690.1772773
"""


class FPMC(basemodel.BaseRetriever):
    r"""
    | FPMC is based on personalized transition graphs over underlying Markov chains. It 
      factorizes the transition cube with a pairwise interaction model which is a special case of
      the Tucker Decomposition.
    """

    def _get_dataset_class(self):
        r"""The dataset FPMC used is SeqDataset."""
        return dataset.SeqDataset

    def _get_item_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_items, 2*self.embed_dim, padding_idx=0)


    def _get_query_encoder(self, train_data):
        return module.VStackLayer(
            module.HStackLayer(
                module.VStackLayer(
                    module.LambdaLayer(lambda x: x[self.fuid]),
                    torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0),
                ),
                module.VStackLayer(
                    module.HStackLayer(
                        module.VStackLayer(
                            module.LambdaLayer(lambda x: x['in_'+self.fiid]),
                            torch.nn.Embedding(train_data.num_items, self.embed_dim, padding_idx=0),
                        ),
                        module.LambdaLayer(lambda x: x['seqlen'])
                    ),
                    module.SeqPoolingLayer(pooling_type='last'),
                )
            ),
            module.LambdaLayer(lambda x: torch.cat(x, dim=-1))
        )


    def _get_score_func(self):
        r"""Inner Product is used as the score function."""
        return scorer.InnerProductScorer()


    def _get_loss_func(self):
        r"""The loss function is BPR loss."""
        return loss_func.BPRLoss()

    
    def _get_sampler(self, train_data):
        return sampler.UniformSampler(train_data.num_items-1)
