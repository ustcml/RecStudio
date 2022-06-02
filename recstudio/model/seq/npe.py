from email.mime import base
import torch
import torch.nn.functional as F
from recstudio.ann import sampler
from recstudio.data import dataset
from recstudio.model import basemodel, loss_func, module, scorer

r"""
NPE
#######################

Paper Reference: 
    ThaiBinh Nguyen, et al. "NPE: Neural Personalized Embedding for Collaborative Filtering" in IJCAI2018.
    https://www.ijcai.org/proceedings/2018/0219.pdf
"""
        


class NPE(basemodel.BaseRetriever):
    r"""
        NPE models a user’s click to an item in two terms: the personal preference of the user for the item, 
        and the relationships between this item and other items clicked by the user.
    """

    def _get_dataset_class(self):
        r"""SeqDataset is used for NPE."""
        return dataset.SeqDataset

    
    def _get_query_encoder(self, train_data):
        return torch.nn.Sequential(
            module.HStackLayer(
                torch.nn.Sequential(
                    module.LambdaLayer(lambda x: x['in_'+self.fiid]),
                    self.item_encoder[0],
                    module.LambdaLayer(lambda x: torch.sum(x, dim=1)),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(p=self.config['dropout_rate'])
                ),
                torch.nn.Sequential(
                    module.LambdaLayer(lambda x: x[self.fuid]),
                    torch.nn.Embedding(train_data.num_users, self.embed_dim, 0),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(p=self.config['dropout_rate'])
                )
            ),
            module.LambdaLayer(lambda x: x[0]+x[1])
        )


    def _get_item_encoder(self, train_data):
        r"""NPE combine an Embedding layer with a ReLU layer as item encoder."""
        return torch.nn.Sequential(
            super()._get_item_encoder(train_data),
            torch.nn.ReLU())


    def _get_item_vector(self):
        """Get all item vectors, simply apply ReLU operation on the weight of Embedding layer."""
        return self.item_encoder[1](self.item_encoder[0].weight[1:])


    def _get_score_func(self):
        r"""Innerproduct operation is applied to calculate scores between query and item."""
        return scorer.InnerProductScorer()


    def _get_loss_func(self):
        r"""According to the original paper, BCE loss is applied.
            Also, other loss functions like softmax loss and BPR loss can be used too.
        """
        return loss_func.BinaryCrossEntropyLoss()


    def _get_sampler(self, train_data):
        return sampler.UniformSampler(train_data.num_items-1)
