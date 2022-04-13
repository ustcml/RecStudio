from recstudio.model import basemodel, loss_func, scorer
from recstudio.data import dataset
import torch

class SASRec(basemodel.ItemTowerRecommender):
    def init_model(self, train_data):
        self.n_layers = self.config['layer_num']
        self.n_head = self.config['head_num']
        self.hidden_size = self.config['hidden_size']
        self.dropout_rate = self.config['dropout_rate']
        self.activation = self.config['activation'] # relu, gelu
        self.layer_norm_eps = self.config['layer_norm_eps']
        self.max_seq_len = train_data.config['max_seq_len']
        super().init_model(train_data)
        self.position_emb = torch.nn.Embedding(self.max_seq_len, self.embed_dim)
        transformer_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.n_head,
            dim_feedforward=self.hidden_size,
            dropout=self.dropout_rate,
            activation=self.activation,
            layer_norm_eps=self.layer_norm_eps,
            batch_first=True,
            norm_first=False
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer=transformer_layer,
            num_layers=self.n_layers,
        )
        self.layer_norm = torch.nn.LayerNorm(self.embed_dim, eps=self.layer_norm_eps)
        self.dropout = torch.nn.Dropout(p=self.dropout_rate)

    def get_dataset_class(self):
            return dataset.SeqDataset

    def construct_query(self, batch_data):
        user_hist = batch_data['in_item_id']
        seq_len = batch_data['seqlen']
        positions = torch.arange(user_hist.size(1), dtype=torch.long, device=user_hist.device)
        positions = positions.unsqueeze(0).expand_as(user_hist)
        position_embs = self.position_emb(positions)

        seq_embs = self.item_encoder(user_hist)
        input_embs = position_embs + seq_embs
        input_embs = self.layer_norm(input_embs)    # BxLxD
        input_embs = self.dropout(input_embs)

        mask4padding = user_hist==0 # BxL
        L = user_hist.size(-1)
        attention_mask = ~torch.tril(torch.ones((L, L), dtype=torch.bool, device=user_hist.device))
        attn_output = self.transformer_encoder(src=input_embs, mask=attention_mask, src_key_padding_mask=mask4padding)  # BxLxD

        gather_index = (seq_len-1).view(-1, 1, 1).expand(-1, -1, attn_output.shape[-1]) # Bx1xD
        query_output = attn_output.gather(dim=1, index=gather_index).squeeze(1) # BxD
        return query_output

    def config_loss(self):
        return loss_func.SoftmaxLoss()

    def config_scorer(self):
        return scorer.InnerProductScorer()