from typing import Set

import torch
import torch.nn as nn
from .layers import MLPModule, AttentionLayer, SeqPoolingLayer, get_act


class DenseEmbedding(torch.nn.Module):

    def __init__(self, embedding_dim, bias=False, batch_norm=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.bias = bias
        self.batch_norm = batch_norm
        if batch_norm:
            self.batch_norm_layer = torch.nn.BatchNorm1d(1)
        self.weight = torch.nn.Linear(1, embedding_dim, bias=bias)

    def forward(self, input):
        input = input.view(-1, 1)

        if self.batch_norm:
            input = self.batch_norm_layer(input)
        emb = self.weight(input)
        return emb

    def extra_repr(self):
        return f"embedding_dim={self.embedding_dim}, bias={self.bias}, batch_norm={self.batch_norm}"

class DenseKernel(torch.nn.Module):
    
    def __init__(self, num_dense_feat):
        super().__init__()
        self.num_dense_feat= num_dense_feat
        self.kernel = nn.Parameter(torch.Tensor(num_dense_feat, 1))
    
    def forward(self, input):
        return input.matmul(self.kernel)
    
    def __repr__(self):
        return f"Parameter(Tensor({self.num_dense_feat}, 1))"
        
class Embeddings(torch.nn.Module):

    def __init__(self, fields: Set, embed_dim, data, reduction='mean',
                 share_dense_embedding=True, dense_emb_bias=False, dense_emb_norm=True,
                 with_dense_kernel=False):
        r"""
        Args:
            dense_kernel (bool): if `True`, concat all float feats together and inner product
        """
        super(Embeddings, self).__init__()
        self.embed_dim = embed_dim

        self.field2types = {f: data.field2type[f] for f in fields if f != data.frating}
        self.reduction = reduction
        self.share_dense_embedding = share_dense_embedding
        self.dense_emb_bias = dense_emb_bias
        self.dense_emb_norm = dense_emb_norm
        self.with_dense_kernel = with_dense_kernel

        self.embeddings = torch.nn.ModuleDict()
        self.num_features = len(self.field2types)

        _num_token_seq_feat = 0
        _num_dense_feat = 0
        _dense_feat = []
        for f, t in self.field2types.items():
            if (t == "token" or t == 'token_seq'):
                if t == 'token_seq':
                    _num_token_seq_feat += 1
                self.embeddings[f] = torch.nn.Embedding(
                    data.num_values(f), embed_dim, 0)
            elif (t == "float"):
                if share_dense_embedding or with_dense_kernel:
                    _num_dense_feat += 1
                    _dense_feat.append(f)
                else:
                    self.embeddings[f] = DenseEmbedding(
                        embed_dim, dense_emb_bias, dense_emb_norm)

        if _num_dense_feat > 0:
            if with_dense_kernel:
                self.embeddings['dense_kernel'] = DenseKernel(_num_dense_feat)
            else:
                dense_emb = DenseEmbedding(
                    embed_dim, dense_emb_bias, dense_emb_norm)
                for f in _dense_feat:
                    self.embeddings[f] = dense_emb

        if _num_token_seq_feat > 0:
            self.seq_pooling_layer = SeqPoolingLayer(reduction, keepdim=False)

    def forward(self, batch):
        embs = []
        dense_value_list = []
        for f, t in self.field2types.items():
            d = batch[f]
            if t == 'token' or (t == 'float' and not self.with_dense_kernel):
                # shape: [B,] or [B,N]
                e = self.embeddings[f](d)
                embs.append(e)
            elif t == 'float' and self.with_dense_kernel:
                dense_value_list.append(d)
            else:
                # shape: [B, L] or [B,N,L]
                length = (d > 0).float().sum(dim=-1, keepdim=False)
                seq_emb = self.embeddings[f](d)
                e = self.seq_pooling_layer(seq_emb, length)
                embs.append(e)

        if 'dense_kernel' in self.embeddings:
            dense_emb = self.embeddings['dense_kernel'](
                            torch.vstack(dense_value_list).transpose(0, 1)
                        ).expand(-1, self.embed_dim)
            embs.append(dense_emb)
            
        # [B,num_features,D] or [B,N,num_features,D]
        emb = torch.stack(embs, dim=-2)
        return emb

    def extra_repr(self):
        s = "num_features={num_features}, embed_dim={embed_dim}, reduction={reduction}"
        if self.share_dense_embedding:
            s += ", share_dense_embedding={share_dense_embedding}"
        if self.dense_emb_bias:
            s += ", dense_emb_bias={dense_emb_bias}"
        if not self.dense_emb_norm:
            s += ", dense_emb_norm={dense_emb_norm}"
        return s.format(**self.__dict__)


class LinearLayer(Embeddings):
    def __init__(self, fields, data, bias=True):
        super(LinearLayer, self).__init__(fields, 1, data)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(1,))
        else:
            self.bias = None

    def forward(self, batch):
        # input: [B, num_fields, 1]
        embs = super().forward(batch).squeeze(-1)
        sum_of_embs = torch.sum(embs, dim=-1)
        return sum_of_embs + self.bias

    def extra_repr(self):
        if self.bias is None:
            bias = False
        else:
            bias = True
        return f"bias={bias}"


class FMLayer(nn.Module):

    def __init__(self, first_order=True, reduction=None):
        super(FMLayer, self).__init__()
        self.reduction = reduction

        if reduction is not None:
            if reduction not in {'sum', 'mean'}:
                raise ValueError(f"reduction only support `mean`|`sum`, but get {reduction}")

    def forward(self, inputs):
        square_of_sum = torch.sum(inputs, dim=1) ** 2
        sum_of_square = torch.sum(inputs ** 2, dim=1)
        output = 0.5 * (square_of_sum - sum_of_square)
        if self.reduction is None:
            pass
        elif self.reduction == 'sum':
            output = output.sum(-1)
        else:
            output = output.mean(-1)
        return output

    def extra_repr(self):
        if self.reduction is None:
            reduction_repr = 'None'
        else:
            reduction_repr = self.reduction
        return f"reduction={reduction_repr}"


class CrossNetwork(torch.nn.Module):
    def __init__(self, embed_dim, num_layers):
        super(CrossNetwork, self).__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.weight = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(self.embed_dim))
            for _ in range(num_layers)
        ])
        self.bias = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(self.embed_dim))
            for _ in range(num_layers)
        ])

    def forward(self, input):
        x = input
        for i in range(self.num_layers):
            x_1 = torch.tensordot(x, self.weight[i], dims=([1], [0]))
            x_2 = (input.transpose(0, 1) * x_1).transpose(0, 1)
            x = x_2 + x + self.bias[i]
        return x

    def extra_repr(self) -> str:
        return f'embed_dim={self.embed_dim}, num_layers={self.num_layers}'


class DINScorer(torch.nn.Module):
    def __init__(self, fuid, fiid, num_users, num_items, embed_dim, attention_mlp, dense_mlp, dropout=0.0,
                 activation='sigmoid', batch_norm=False):
        super().__init__()
        self.fuid = fuid
        self.fiid = fiid
        self.item_embedding = torch.nn.Embedding(num_items, embed_dim, 0)
        self.item_bias = torch.nn.Embedding(num_items, 1, padding_idx=0)
        self.activation_unit = AttentionLayer(
            3*embed_dim, embed_dim, mlp_layers=attention_mlp, activation=activation)
        norm = [torch.nn.BatchNorm1d(embed_dim)] if batch_norm else []
        norm.append(torch.nn.Linear(embed_dim, embed_dim))
        self.norm = torch.nn.Sequential(*norm)
        self.dense_mlp = MLPModule(
            [3*embed_dim]+dense_mlp, activation_func=activation, dropout=dropout, batch_norm=batch_norm)
        self.fc = torch.nn.Linear(dense_mlp[-1], 1)

    def forward(self, batch):
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


class BehaviorSequenceTransformer(nn.Module):
    def __init__(self, fuid, fiid, num_users, num_items, max_len,
                 embed_dim, hidden_size, n_layer, n_head, dropout,
                 mlp_layers=[1024, 512, 256], activation='leakyrelu',
                 batch_first=True, norm_first=False):
        super().__init__()
        self.fuid = fuid
        self.fiid = fiid
        self.max_len = max_len
        self.item_embedding = torch.nn.Embedding(num_items, embed_dim, 0)
        self.position_embedding = torch.nn.Embedding(max_len+2, embed_dim, 0)
        tfm_encoder = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_head, dim_feedforward=hidden_size,
            dropout=dropout, activation='relu', batch_first=batch_first,
            norm_first=norm_first)
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer=tfm_encoder, num_layers=n_layer)
        self.mlp = MLPModule([(max_len+1)*embed_dim, ] + mlp_layers,
                             activation_func = activation,
                             dropout = dropout)
        self.predict = torch.nn.Linear(mlp_layers[-1], 1)

    def forward(self, batch):
        hist = batch['in_'+self.fiid]
        target = batch[self.fiid]
        seq_len = batch['seqlen']
        hist = torch.cat((hist, torch.zeros_like(target.view(-1, 1))), dim=1)
        B, L = hist.shape
        idx_ = torch.arange(0, B, dtype=torch.long)
        hist[idx_, seq_len] = target.long()
        seq_emb = self.item_embedding(hist)
        positions = torch.arange(1, L+1, device=seq_emb.device)
        positions = torch.tile(positions, (B,)).view(B, -1)
        padding_mask = hist == 0
        positions[padding_mask] = 0
        position_emb = self.position_embedding(positions)
        attention_mask = torch.triu(torch.ones(
            (L, L), dtype=torch.bool, device=hist.device), 1)

        tfm_out = self.transformer(
            src=seq_emb+position_emb, mask=attention_mask, src_key_padding_mask=padding_mask)

        padding_emb = tfm_out.new_zeros(
            (B, self.max_len+1-L, tfm_out.size(-1)))
        tfm_out = torch.cat((tfm_out, padding_emb), dim=1)
        flatten_tfm_out = tfm_out.view(B, -1)
        logits = self.predict(self.mlp(flatten_tfm_out))
        return logits.squeeze(-1)


class DIENScorer(torch.nn.Module):
    def __init__(self, fuid, fiid, num_users, num_items, embed_dim, attention_mlp, fc_mlp,
                 activation='sigmoid', batch_norm=False) -> None:
        super().__init__()
        self.fuid = fuid
        self.fiid = fiid
        self.user_embedding = torch.nn.Embedding(num_users, embed_dim, 0)
        self.item_embedding = torch.nn.Embedding(num_items, embed_dim, 0)
        self.item_bias = torch.nn.Embedding(num_items, 1, padding_idx=0)
        self.activation_unit = AttentionLayer(3*embed_dim, embed_dim, mlp_layers=attention_mlp, activation=activation)
        self.norm = torch.nn.Sequential(
            torch.nn.BatchNorm1d(embed_dim),
            torch.nn.Linear(embed_dim, embed_dim),
        ) if batch_norm else torch.nn.Linear(embed_dim, embed_dim)

        self.fc = torch.nn.Sequential(
            torch.nn.BatchNorm1d(3 * embed_dim),
            MLPModule([3*embed_dim]+fc_mlp, activation_func=activation),
            torch.nn.Linear(fc_mlp[-1], 1),
        ) if batch_norm else \
            torch.nn.Sequential(
                MLPModule([3*embed_dim]+fc_mlp, activation_func=activation),
                torch.nn.Linear(fc_mlp[-1], 1),
        )

    def forward(self, batch):
        pass


class CIN(torch.nn.Module):
    def __init__(self, embed_dim, num_features, cin_layer_size, activation='relu', direct=True):
        super(CIN, self).__init__()
        self.embed_dim = embed_dim
        self.num_features = num_features
        self.cin_layer_size = _temp = cin_layer_size
        self.activation = get_act(activation)
        self.direct = direct
        self.weight = torch.nn.ModuleList()
        # Check whether the size of the CIN layer is legal.
        if not self.direct:
            self.cin_layer_size = list(map(lambda x: int(x // 2 * 2), _temp))
            if self.cin_layer_size[:-1] != _temp[:-1]:
                self.logger.warning(
                    "Layer size of CIN should be even except for the last layer when direct is True."
                    "It is changed to {}".format(self.cin_layer_size)
                )
        #
        # Convolutional layer for each CIN layer
        self.weight_list = nn.ModuleList()
        self.field_num_list = [self.num_features]
        for i, layer_size in enumerate(self.cin_layer_size):
            conv1d = nn.Conv1d(self.field_num_list[-1] * self.field_num_list[0], layer_size, 1)
            self.weight_list.append(conv1d)
            if self.direct:
                self.field_num_list.append(layer_size)
            else:
                self.field_num_list.append(layer_size // 2)
        #
        # Get the output size of CIN
        if self.direct:
            output_dim = sum(self.cin_layer_size)
        else:
            output_dim = (
                sum(self.cin_layer_size[:-1]) // 2 + self.cin_layer_size[-1]
            )
        self.linear = torch.nn.Linear(output_dim, 1)

    def forward(self, input):
        B, _, D = input.shape
        hidden_nn_layers = [input]
        final_result = []
        for i, layer_size in enumerate(self.cin_layer_size):
            z_i = torch.einsum("bhd,bmd->bhmd", hidden_nn_layers[-1], hidden_nn_layers[0])
            z_i = z_i.view(B, self.field_num_list[0] * self.field_num_list[i], D)
            z_i = self.weight_list[i](z_i)
            output = self.activation(z_i)
            # Get the output of the hidden layer.
            if self.direct:
                direct_connect = output
                next_hidden = output
            else:
                if i != len(self.cin_layer_size) - 1:
                    next_hidden, direct_connect = torch.split(output, 2 * [layer_size // 2], 1)
                else:
                    direct_connect = output
                    next_hidden = 0

            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)
        result = torch.cat(final_result, dim=1)
        result = torch.sum(result, -1)
        score = self.linear(result).squeeze(-1)
        return score
