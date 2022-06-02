import torch
from recstudio.ann import sampler
from recstudio.data import dataset
from recstudio.model import basemodel, loss_func, module, scorer


class STAMPQueryEncoder(torch.nn.Module):

    def __init__(self, fiid, embed_dim, item_encoder) -> None:
        super().__init__()
        self.fiid = fiid
        self.item_encoder = item_encoder
        self.gather_layer = module.SeqPoolingLayer(pooling_type='last')
        self.attention_layer = module.AttentionLayer(
            q_dim = 2 * embed_dim,
            k_dim = embed_dim,
            mlp_layers = [embed_dim],
        )
        self.mlp = module.MLPModule( [2 * embed_dim, 2 * embed_dim], torch.nn.Tanh())

    
    def forward(self, batch):
        user_hist = batch['in_'+self.fiid]
        seq_emb = self.item_encoder(user_hist)
        m_t = self.gather_layer(seq_emb, batch['seqlen'])
        m_s = seq_emb.sum(dim=1) / batch['seqlen'].unsqueeze(1).float() # B x D

        query = torch.cat((m_t, m_s), dim=1)    # Bx2D
        m_a = self.attention_layer(query.unsqueeze(1), seq_emb, seq_emb, \
            key_padding_mask=(user_hist==0)).squeeze(1)
        h_cat = self.mlp(torch.cat((m_s, m_a), dim=1))
        h_s, h_t = h_cat.tensor_split(2, dim=1)
        return h_s * h_t



class STAMP(basemodel.BaseRetriever):
    r"""
    STAMP is capable of capturing users’ general interests from the long-term memory of a session
    context, while taking into account users’ current interests from the short-term memory of the
    last-clicks. 

    Model hyper parameters:
        - ``embed_dim(int)``: The dimension of embedding layers. Default: ``64``.
    """

    def _get_dataset_class(self):
        r"""SeqDataset is used for STAMP."""
        return dataset.SeqDataset
    

    def _get_query_encoder(self, train_data):
        return STAMPQueryEncoder(self.fiid, self.embed_dim, self.item_encoder)


    def _get_score_func(self):
        r"""InnerProduct is used as the score function."""
        return scorer.InnerProductScorer()


    def _get_loss_func(self):
        r"""SoftmaxLoss is used as the loss function."""
        return loss_func.SoftmaxLoss()


    def _get_sampler(self, train_data):
        return None
        