from recstudio.model import basemodel, loss_func, scorer
from recstudio.data import dataset
import torch

r"""
GRU4Rec
############

Paper Reference:
    Balazs Hidasi, et al. "Session-Based Recommendations with Recurrent Neural Networks" in ICLR2016.
    https://arxiv.org/abs/1511.06939
"""

class GRU4Rec(basemodel.ItemTowerRecommender):
    r"""
    GRU4Rec apply RNN in Recommendation System, where sequential behavior of user is regarded as input
    of the RNN.
    """
    def init_model(self, train_data):
        r"""The kernel module of GRU4Rec is GRU layer."""
        super().init_model(train_data)
        self.hidden_size = self.config['hidden_size']
        self.num_layers = self.config['layer_num']
        self.dropout_rate = self.config['dropout_rate']
        self.emb_dropout = torch.nn.Dropout(self.dropout_rate)
        self.GRU = torch.nn.GRU(
            input_size = self.embed_dim,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            bias = False,
            batch_first = True,
            bidirectional = False
        )
        self.dense = torch.nn.Linear(self.hidden_size, self.embed_dim)

    def get_dataset_class(self):
        r"""The dataset is SeqDataset."""
        return dataset.SeqDataset

    def construct_query(self, batch_data):
        r"""The last output of the last item will be used as query. """
        user_hist = batch_data['in_item_id']
        emb_hist = self.item_encoder(user_hist)
        emb_hist_dropout = self.emb_dropout(emb_hist)   # B x L x H_in
        gru_vec, _ = self.GRU(emb_hist_dropout)    # B x L x H_out
        query = self.dense(gru_vec)
        gather_index = (batch_data['seqlen']-1).view(-1, 1, 1).expand(-1, -1, query.shape[-1]) # B x 1 x H_out
        query_output = query.gather(dim=1, index=gather_index).squeeze(1)  # B x H_out
        return query_output

    def config_loss(self):
        r"""SoftmaxLoss is used as the loss function."""
        return loss_func.SoftmaxLoss()

    def config_scorer(self):
        r"""InnerProduct is used as the score function."""
        return scorer.InnerProductScorer()