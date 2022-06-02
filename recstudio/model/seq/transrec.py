from recstudio.model import basemodel, loss_func, scorer
from recstudio.data import dataset
from recstudio.ann import sampler
import torch


class TransRecQueryEncoder(torch.nn.Module):
    def __init__(self, fuid, fiid, num_users, embed_dim, item_encoder):
        super().__init__()
        self.user_embedding = torch.nn.Embedding(num_users, embed_dim, 0)
        self.global_user_emb = torch.nn.parameter.Parameter()




class TransRec(basemodel.BaseRetriever):
    r"""
    TransRec embeds items into a ‘transition space’ where users are modeled as translation vectors operating on item sequences.

    Model hyper parameters:
        - ``embed_dim(int)``: The dimension of embedding layers. Default: ``64``.
    """
    # def init_model(self, train_data):
    #     self.global_user_emb = torch.nn.Parameter(torch.zeros(self.embed_dim))
    #     # self.bias = nn.Embedding(train_data.num_items, 1, padding_idx=0)
    #     # TODO: bias here is not easy to construct query, abandoned now
    #     super().init_model(train_data)

    def _get_dataset_class(self):
        r"""SeqDataset is used for TransRec."""
        return dataset.SeqDataset


    def _get_item_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_items, self.embed_dim, 0)


    def _get_query_encoder(self, train_data):
        return super()._get_query_encoder(train_data)

    def build_user_encoder(self, train_data):
        r"""User encoder is bulit with an embedding layer."""
        return torch.nn.Embedding(train_data.num_users, embedding_dim=self.embed_dim, padding_idx=0)


    def construct_query(self, batch_data):
        user_hist = batch_data['in_item_id']
        seq_len = batch_data['seqlen'] - 1
        local_user_emb = self.user_encoder(batch_data[self.fuid])
        user_emb = local_user_emb + self.global_user_emb.expand_as(local_user_emb) # B x D
        last_item_id = torch.gather(user_hist, dim=-1, index=seq_len.unsqueeze(1))
        last_item_emb = self.item_encoder(last_item_id).squeeze(1) # B x D
        query = user_emb + last_item_emb
        return query

    def build_sampler(self, train_data):
        return sampler.UniformSampler(train_data.num_items-1, self.score_func)

    def config_scorer(self):
        r"""InnerProduct is used as the score function."""
        return scorer.EuclideanScorer()

    def config_loss(self):
        r"""BPRLoss is used as the loss function."""
        return loss_func.BPRLoss()
