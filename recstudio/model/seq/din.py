import torch
from recstudio.data import dataset
from recstudio.model import basemodel, loss_func
from recstudio.model.module.layers import AttentionLayer, MLPModule


r"""
DIN
######################

Paper Reference:
    Guorui Zhou, et al. "Deep Interest Network for Click-Through Rate Prediction" in KDD2018.
    https://dl.acm.org/doi/10.1145/3219819.3219823

"""


class DIN(basemodel.BaseRanker):
    r"""
        | Deep Interest Network (DIN) designs a local activation unit to adaptively learn the representation
          of user interests from historical behaviors with respect to a certain ad.

        | DIN calculate the relevance between the target item and items in the sequence by adapting an
          attention machnism. But the method could not be applied to recall on all items in prediction
          due to the huge time cost.
    """
    def add_model_specific_args(parent_parser):
        parent_parser = basemodel.Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('DIN')
        parent_parser.add_argument("--activation", type=str, default='dice', help='activation for MLP')
        parent_parser.add_argument("--attention_mlp", type=int, nargs='+', default=[128, 64], help='MLP layer size for attention calculation')
        parent_parser.add_argument("--fc_mlp", type=int, nargs='+', default=[128, 64, 64], help='MLP layer size for the MLP before prediction')
        parent_parser.add_argument("--negative_count", type=int, default=1, help='negative sampling numbers')
        parent_parser.add_argument("--dropout", type=float, default=0.3, help='dropout rate for MLP')
        parent_parser.add_argument("--batch_norm", action='store_true', default=False, help='whether to use batch_norm')
        return parent_parser

    def _set_data_field(self, data):
        pass

    def _get_dataset_class():
        r"""The dataset is SeqDataset."""
        return dataset.SeqDataset


    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        d = self.embed_dim
        act_f = self.config['activation']
        fc_mlp = self.config['fc_mlp']
        dropout_p = self.config['dropout']
        self.item_embedding = torch.nn.Embedding(train_data.num_items, d, 0)
        self.item_bias = torch.nn.Embedding(train_data.num_items, 1, padding_idx=0)
        self.activation_unit = AttentionLayer(
            3*d, d, mlp_layers=self.config['attention_mlp'], activation=act_f)
        norm = [torch.nn.BatchNorm1d(d)] if self.config['batch_norm'] else []
        norm.append(torch.nn.Linear(d, d))
        self.norm = torch.nn.Sequential(*norm)
        self.dense_mlp = MLPModule(
            [3*d]+fc_mlp, activation_func=act_f, dropout=dropout_p, batch_norm=self.config['batch_norm'])
        self.fc = torch.nn.Linear(fc_mlp[-1], 1)

    def score(self, batch):
        seq_emb = self.item_embedding(batch['in_'+self.fiid])
        target_emb = self.item_embedding(batch[self.fiid])
        item_bias = self.item_bias(batch[self.fiid]).squeeze(-1)

        target_emb_ = target_emb.unsqueeze(1).repeat(
            1, seq_emb.size(1), 1)   # BxLxD
        attn_seq = self.activation_unit(
            query=target_emb.unsqueeze(1),
            key=torch.cat((target_emb_, target_emb_*seq_emb,
                          target_emb_-seq_emb), dim=-1),
            value=seq_emb,
            key_padding_mask=(batch['in_'+self.fiid] == 0),
            softmax=False
        ).squeeze(1)
        attn_seq = self.norm(attn_seq)
        cat_emb = torch.cat(
            (attn_seq, target_emb, target_emb*attn_seq), dim=-1)
        score = self.fc(self.dense_mlp(cat_emb)).squeeze(-1)
        return score + item_bias

    def _get_loss_func(self):
        return loss_func.BCEWithLogitLoss(self.rating_threshold)
