import torch
from recstudio.ann import sampler
from recstudio.data import dataset
from recstudio.model import basemodel, loss_func, scorer


class TransRecQueryEncoder(torch.nn.Module):
    def __init__(self, fuid, fiid, num_users, embed_dim, item_encoder):
        super().__init__()
        self.fuid = fuid
        self.fiid = fiid
        self.item_encoder = item_encoder
        self.user_embedding = torch.nn.Embedding(num_users, embed_dim, 0)
        self.global_user_emb = torch.nn.Parameter(torch.zeros(embed_dim))

    def forward(self, batch):
        user_hist = batch['in_'+self.fiid]
        seq_len = batch['seqlen'] - 1
        local_user_emb = self.user_embedding(batch[self.fuid])
        user_emb = local_user_emb + self.global_user_emb.expand_as(local_user_emb)  # B x D
        last_item_id = torch.gather(user_hist, dim=-1, index=seq_len.unsqueeze(1))
        last_item_emb = self.item_encoder(last_item_id).squeeze(1)  # B x D
        query = user_emb + last_item_emb
        return query


class TransRec(basemodel.BaseRetriever):
    r"""
    TransRec embeds items into a ‘transition space’ where users are modeled as translation vectors operating on item sequences.

    Model hyper parameters:
        - ``embed_dim(int)``: The dimension of embedding layers. Default: ``64``.
    """

    # TODO(@AngusHuang17): bias here is not easy to construct query, abandoned now

    def _get_dataset_class():
        r"""SeqDataset is used for TransRec."""
        return dataset.SeqDataset

    def _get_item_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_items, self.embed_dim, 0)

    def _get_query_encoder(self, train_data):
        return TransRecQueryEncoder(
            self.fuid, self.fiid, train_data.num_users, self.embed_dim, self.item_encoder
        )

    def _get_sampler(self, train_data):
        return sampler.UniformSampler(train_data.num_items, self.score_func)

    def _get_scorer(self):
        r"""InnerProduct is used as the score function."""
        return scorer.EuclideanScorer()

    def _get_loss_func(self):
        r"""BPRLoss is used as the loss function."""
        return loss_func.BPRLoss()
