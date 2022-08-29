import torch
from recstudio.ann import sampler
from recstudio.data import dataset
from recstudio.model import basemodel, loss_func, module, scorer


class SASRecQueryEncoder(torch.nn.Module):
    def __init__(
            self, fiid, embed_dim, max_seq_len, n_head, hidden_size, dropout, activation, n_layer, item_encoder,
            bidirectional=False, return_full_seq_training=False) -> None:
        super().__init__()
        self.fiid = fiid
        self.item_encoder = item_encoder
        self.bidirectional = bidirectional
        self._return_full_seq_training = return_full_seq_training
        self.position_emb = torch.nn.Embedding(max_seq_len, embed_dim)
        transformer_encoder = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_head,
            dim_feedforward=hidden_size,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=False
        )
        self.transformer_layer = torch.nn.TransformerEncoder(
            encoder_layer=transformer_encoder,
            num_layers=n_layer,
        )
        self.dropout = torch.nn.Dropout(p=dropout)
        self.gather_layer = module.SeqPoolingLayer(pooling_type='last')

    def forward(self, batch):
        user_hist = batch['in_'+self.fiid]
        positions = torch.arange(user_hist.size(1), dtype=torch.long, device=user_hist.device)
        positions = positions.unsqueeze(0).expand_as(user_hist)
        position_embs = self.position_emb(positions)
        seq_embs = self.item_encoder(user_hist)

        mask4padding = user_hist == 0  # BxL
        L = user_hist.size(-1)
        if not self.bidirectional:
            attention_mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=user_hist.device), 1)
        else:
            attention_mask = torch.zeros((L, L), dtype=torch.bool, device=user_hist.device)
        transformer_out = self.transformer_layer(
            src=self.dropout(seq_embs+position_embs),
            mask=attention_mask,
            src_key_padding_mask=mask4padding)  # BxLxD

        if self.training and self._return_full_seq_training:
            if batch['seqlen'].dim() == 2:  # masked token
                mask_token = batch['seqlen']
                transformer_out = transformer_out[mask_token]
            return transformer_out  # BxLxD or NxD
        else:
            return self.gather_layer(transformer_out, batch['seqlen'])


class SASRec(basemodel.BaseRetriever):
    r"""
    SASRec models user's sequence with a Transformer.

    Model hyper parameters:
        - ``embed_dim(int)``: The dimension of embedding layers. Default: ``64``.
        - ``hidden_size(int)``: The output size of Transformer layer. Default: ``128``.
        - ``layer_num(int)``: The number of layers for the Transformer. Default: ``2``.
        - ``dropout_rate(float)``:  The dropout probablity for dropout layers after item embedding
         | and in Transformer layer. Default: ``0.5``.
        - ``head_num(int)``: The number of heads for MultiHeadAttention in Transformer. Default: ``2``.
        - ``activation(str)``: The activation function in transformer. Default: ``"gelu"``.
        - ``layer_norm_eps``: The layer norm epsilon in transformer. Default: ``1e-12``.
    """

    def _get_dataset_class():
        r"""SeqDataset is used for SASRec."""
        return dataset.SeqDataset

    def _get_query_encoder(self, train_data):
        return SASRecQueryEncoder(
            fiid=self.fiid, embed_dim=self.embed_dim,
            max_seq_len=train_data.config['max_seq_len'], n_head=self.config['head_num'],
            hidden_size=self.config['hidden_size'], dropout=self.config['dropout_rate'],
            activation=self.config['activation'], n_layer=self.config['layer_num'],
            item_encoder=self.item_encoder
        )

    def _get_item_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_items, self.embed_dim, padding_idx=0)

    def _get_score_func(self):
        r"""InnerProduct is used as the score function."""
        return scorer.InnerProductScorer()

    def _get_loss_func(self):
        r"""Binary Cross Entropy is used as the loss function."""
        return loss_func.BinaryCrossEntropyLoss()

    def _get_sampler(self, train_data):
        r"""Uniform sampler is used as negative sampler."""
        return sampler.UniformSampler(train_data.num_items)
