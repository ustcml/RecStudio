import torch
from recstudio.ann import sampler
from recstudio.data import dataset
from recstudio.model import basemodel, loss_func, scorer

r"""
NPE
#######################

Paper Reference: 
    ThaiBinh Nguyen, et al. "NPE: Neural Personalized Embedding for Collaborative Filtering" in IJCAI2018.
    https://www.ijcai.org/proceedings/2018/0219.pdf
"""

class NPE(basemodel.TwoTowerRecommender):
    r"""
        NPE models a user’s click to an item in two terms: the personal preference of the user for the item, 
        and the relationships between this item and other items clicked by the user.
    """

    def init_model(self, train_data):
        r""" Init NPE with a dropout and a ReLU layer.
        """
        super().init_model(train_data)
        self.dropout = torch.nn.Dropout(self.config['dropout'])
        self.relu = torch.nn.ReLU()

    def build_user_encoder(self, train_data):
        r"""A simple user embedding is used as user encoder in NPE"""
        return torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)

    def build_item_encoder(self, train_data):
        r"""NPE combine an Embedding layer with a ReLU layer as item encoder."""
        return torch.nn.Sequential(
            torch.nn.Embedding(train_data.num_items, self.embed_dim, padding_idx=0),
            torch.nn.ReLU()
        )

    def get_item_vector(self):
        r"""Get all item vectors, simply apply ReLU operation on the weight of Embedding layer."""
        return self.item_encoder[1](self.item_encoder[0].weight[1:])

    def construct_query(self, batch_data):
        r"""Contruct query with addition user embedding and sequence embedding."""
        item_seq = batch_data['in_item_id']
        user_emb = self.user_encoder(batch_data[self.fuid])
        seq_embs = self.item_encoder[0](item_seq).sum(dim=1)
        h_u = self.dropout(self.relu(user_emb))
        v_c = self.dropout(self.relu(seq_embs))
        return h_u + v_c 

    def get_dataset_class(self):
        r"""SeqDataset is used for NPE."""
        return dataset.SeqDataset

    def config_loss(self):
        r"""According to the original paper, BCE loss is applied.
            Also, other loss functions like softmax loss and BPR loss can be used too.
        """
        return loss_func.BinaryCrossEntropyLoss()

    def config_scorer(self):
        r"""Innerproduct operation is applied to calculate scores between query and item."""
        return scorer.InnerProductScorer()

    def build_sampler(self, train_data):
        r"""Negative items are sampled uniformly."""
        return sampler.UniformSampler(train_data.num_items-1, self.score_func)
