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

    def _get_dataset_class(self):
        r"""The dataset is SeqDataset."""
        return dataset.SeqDataset

    
    def _get_query_encoder(self, train_data):
        return module.VStackLayer(
            module.HStackLayer(
                module.VStackLayer(
                    module.LambdaLayer(lambda x: x['in_'+self.fiid]),
                    self.item_encoder,
                    torch.nn.Dropout(self.config['dropout_rate']),
                    module.GRULayer(self.embed_dim, self.config['hidden_size'], self.config['layer_num'] ),
                ),
                module.LambdaLayer(lambda_func=lambda x: x['seqlen']),
            ),
            module.SeqPoolingLayer(pooling_type='last'),
            torch.nn.Linear(self.config['hidden_size'], self.embed_dim)
        )


    def _get_score_func(self):
        return scorer.InnerProductScorer()


    def _get_loss_func(self):
        r"""SoftmaxLoss is used as the loss function."""
        return loss_func.BPRLoss()


    def _get_sampler(self, train_data):
        return sampler.UniformSampler(train_data.num_items-1)
