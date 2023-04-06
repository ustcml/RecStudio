from typing import Set
import math
import torch
import torch.nn as nn
from .layers import MLPModule, AttentionLayer, SeqPoolingLayer, get_act, KMaxPoolingLayer, VStackLayer, HStackLayer


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


class Embeddings(torch.nn.Module):

    def __init__(self, fields: Set, embed_dim, data, reduction='mean',
                 share_dense_embedding=False, dense_emb_bias=False, dense_emb_norm=True):
        super(Embeddings, self).__init__()
        self.embed_dim = embed_dim

        self.field2types = {f: data.field2type[f] for f in fields if f != data.frating}
        self.reduction = reduction
        self.share_dense_embedding = share_dense_embedding
        self.dense_emb_bias = dense_emb_bias
        self.dense_emb_norm = dense_emb_norm

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
                if share_dense_embedding:
                    _num_dense_feat += 1
                    _dense_feat.append(f)
                else:
                    self.embeddings[f] = DenseEmbedding(
                        embed_dim, dense_emb_bias, dense_emb_norm)

        if _num_dense_feat > 0:
            dense_emb = DenseEmbedding(
                embed_dim, dense_emb_bias, dense_emb_norm)
            for f in _dense_feat:
                self.embeddings[f] = dense_emb

        if _num_token_seq_feat > 0:
            self.seq_pooling_layer = SeqPoolingLayer(reduction, keepdim=False)

    def forward(self, batch):
        embs = []
        for f in self.embeddings:
            d = batch[f]
            t = self.field2types[f]
            if t == 'token' or t == 'float':
                # shape: [B,] or [B,N]
                e = self.embeddings[f](d)
            else:
                # shape: [B, L] or [B,N,L]
                length = (d > 0).float().sum(dim=-1, keepdim=False)
                seq_emb = self.embeddings[f](d)
                e = self.seq_pooling_layer(seq_emb, length)
            embs.append(e)

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
            if reduction not in {'sum', 'mean', 'none'}:
                raise ValueError(f"reduction only support `mean`|`sum`|`none`|, but get {reduction}")

    def forward(self, inputs):
        square_of_sum = torch.sum(inputs, dim=1) ** 2
        sum_of_square = torch.sum(inputs ** 2, dim=1)
        output = 0.5 * (square_of_sum - sum_of_square)
        if self.reduction is None or self.reduction == 'none':
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


class AFMLayer(nn.Module):

    def __init__(self, embed_dim, attention_dim, num_fields, dropout=0):
        super(AFMLayer, self).__init__()
        self.attention_dim = attention_dim
        self.dropout = dropout
        self.attention = nn.Sequential(
                            nn.Linear(embed_dim, attention_dim),
                            nn.ReLU(),
                            nn.Linear(attention_dim, 1, bias=False),
                            nn.Softmax(dim=1))
        self.p = nn.Linear(embed_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.triu_index = nn.Parameter(
                            torch.triu_indices(num_fields, num_fields, offset=1), 
                            requires_grad=False)

    def forward(self, inputs):
        # inputs: B x F x D
        emb0 = torch.index_select(inputs, 1, self.triu_index[0])
        emb1 = torch.index_select(inputs, 1, self.triu_index[1])
        prod = emb0 * emb1
        attn = self.attention(prod)
        attn_sum = (attn * prod).sum(1)
        output = self.dropout(attn_sum)
        output = self.p(output)
        output = output.squeeze(-1)
        return output

    def extra_repr(self):
        return f"attention dim={self.attention_dim}, dropout={self.dropout}"
   
    
class LogTransformLayer(nn.Module):

    def __init__(self, num_fields, hidden_size, clamp_min=1e-5):
        super(LogTransformLayer, self).__init__()
        self.hidden_size = hidden_size
        self.clamp_min = clamp_min
        self.log_bn = nn.BatchNorm1d(num_fields)
        self.exp_bn = nn.BatchNorm1d(hidden_size)
        self.linear = nn.Linear(num_fields, hidden_size, bias=True)

    def forward(self, inputs):
        emb = torch.clamp(torch.abs(inputs), min=self.clamp_min)
        log_emb = torch.log(emb)
        log_emb = self.log_bn(log_emb)
        log_out = self.linear(log_emb.transpose(1, 2)).transpose(1, 2)
        exp_out = torch.exp(log_out)
        exp_out = self.exp_bn(exp_out)
        output = exp_out.view(exp_out.size(0), -1)
        return output
    
    def extra_repr(self):
        return f"hidden_size={self.hidden_size}, clamp_min={self.clamp_min}"
    
    
class InteractingLayer(nn.Module):
    
    def __init__(self, embed_dim, n_head=1, dropout=0, residual=True, residual_project=True, layer_norm=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.dropout = dropout
        self.residual = residual
        self.residual_project = residual_project
        self.layer_norm = layer_norm
        self.attn = AttentionLayer(
                        q_dim=embed_dim,
                        n_head=n_head,
                        dropout=dropout,
                        attention_type='multi-head',
                        batch_first=True)       
        if residual and residual_project:
            self.res = nn.Linear(embed_dim, embed_dim)
        if layer_norm:
            self.ln = nn.LayerNorm(embed_dim)
            
    def forward(self, inputs):
        output = self.attn(inputs, inputs, inputs)
        if self.residual:
            if self.residual_project:
                res_out = self.res(inputs)
            else:
                res_out = inputs
            output += res_out
        if self.layer_norm:
            output = self.ln(output)   
        return output.relu()
    
    def extra_repr(self):
        s = f"attention_dim={self.embed_dim}, n_head={self.n_head}, dropout={self.dropout}"
        if self.residual:
            s += f", residual={self.residual}"
        return s
 
    
class ConvLayer(nn.Module):
    
    def __init__(self, num_fields, channels, heights, activation):
        super().__init__()
        
        if len(heights) != len(channels):
            raise ValueError("channels and widths should have the same length.")
        
        self.channels = [1] + channels
        self.heights = heights
        self.activation = activation
        layers = len(heights)
        
        module_list = []
        for i in range(1, len(self.channels)):
            in_channels = self.channels[i - 1]
            out_channels = self.channels[i]
            height = heights[i - 1]
            module_list.append(nn.ZeroPad2d((0, 0, math.floor(height - 1), math.ceil(height - 1))))
            module_list.append(nn.Conv2d(in_channels, out_channels, kernel_size=(height, 1)))
            if i < layers:
                k = max(3, int((1 - pow(float(i) / layers, layers - i)) * num_fields))
            else:
                k = 3
            module_list.append(get_act(activation))
            module_list.append(KMaxPoolingLayer(k, dim=2))
        self.conv = nn.Sequential(*module_list)

    def forward(self, inputs):
        return self.conv(inputs.unsqueeze(1)) # -> N(Batch size) x C(Channels) x H(Height) x W(Embed dim)
                
    def extra_repr(self):
        return f"channels={self.channels}, heights={self.heights}, activation={self.activation}"
        

class SqueezeExcitation(nn.Module):
    
    def __init__(self, num_fields, reduction_ratio, activation, pool='avg') -> None:
        super().__init__()
        
        if pool.lower() not in ['avg', 'max']:
            raise ValueError(f'Expect pool to be `avg` or `max`, but got {pool}')
        
        if not isinstance(activation, list):
            self.activation = [activation]
        if len(self.activation) == 1:
            self.activation = 2 * self.activation    
        elif len(self.activation) > 2 or len(self.activation) < 1:
            raise ValueError(f'Expect activation to be one or two, but got {len([activation])}')
        
            
        self.pool = pool.lower()
        self.reduced_size = max(1, int(num_fields // reduction_ratio))
        self.excitation = nn.Sequential(
            nn.Linear(num_fields, self.reduced_size, bias=False),
            get_act(self.activation[0]),
            nn.Linear(self.reduced_size, num_fields, bias=False),
            get_act(self.activation[1])
        )
        
    def forward(self, inputs):
        if self.pool == 'avg':
            Z = torch.mean(inputs, dim=-1)
        else:
            Z = torch.max(inputs, dim=-1).values
        A = self.excitation(Z)
        V = inputs * A.unsqueeze(-1)    # reweight
        return V
    
    def extra_repr(self):
        return f"pool={self.pool}, reduced_size={self.reduced_size}, activation={self.activation}"
    
    
class BilinearInteraction(nn.Module):
    def __init__(self, num_fields, embed_dim, bilinear_type='interaction'):
        super().__init__()
        if bilinear_type.lower() not in ['all', 'each', 'interaction']:
            raise ValueError(f'Expect bilinear_type to be `all`|`each`|`interaction`, '
                             f'but got {bilinear_type}.')
        self.bilinear_type = bilinear_type.lower()
        if self.bilinear_type == 'all':
            self.weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        elif self.bilinear_type == 'each':
            self.weight = nn.Parameter(torch.Tensor(num_fields, embed_dim, embed_dim))
        else:
            self.weight = nn.Parameter(torch.Tensor(
                            num_fields * (num_fields - 1) // 2, embed_dim, embed_dim))

        self.triu_index = nn.Parameter(
                            torch.triu_indices(num_fields, num_fields, offset=1), 
                            requires_grad=False)

    def forward(self, inputs):
        if self.bilinear_type == 'all':
            hidden_emb = inputs @ self.weight
            emb0 =  torch.index_select(hidden_emb, 1, self.triu_index[0])
            emb1 = torch.index_select(inputs, 1, self.triu_index[1])
            bilinear_out = emb0 * emb1
        elif self.bilinear_type =='each':
            hidden_emb = (emb0.unsqueeze(2) @ self.weight).squeeze(2)
            emb0 =  torch.index_select(hidden_emb, 1, self.triu_index[0])
            emb1 = torch.index_select(inputs, 1, self.triu_index[1])
            bilinear_out = emb0 * emb1
        else:
            emb0 =  torch.index_select(inputs, 1, self.triu_index[0])
            emb1 = torch.index_select(inputs, 1, self.triu_index[1])
            bilinear_out = (emb0.unsqueeze(2) @ self.weight).squeeze(2) * emb1
        return bilinear_out
    
    def extra_repr(self):
        return f'biliniear_type={self.bilinear_type}'

     
class MaskBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, 
                 reduction_ratio=1, hidden_activation='relu', 
                 dropout=0, layer_norm=True):
        super().__init__()
        if reduction_ratio < 1:
            raise ValueError(f'Expect reduction_ratio > 1, but got {reduction_ratio}')
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.reduction_ratio = reduction_ratio
        self.layer_norm = layer_norm
        self.mask_layer = nn.Sequential(
                            nn.Linear(input_dim, int(hidden_dim * reduction_ratio)),
                            nn.ReLU(),
                            nn.Linear(int(hidden_dim * reduction_ratio), hidden_dim)
                        )
        
        hidden_layer = [nn.Linear(hidden_dim, output_dim, bias=False),
                        get_act(hidden_activation),
                        nn.Dropout(dropout)]
        if layer_norm:
            hidden_layer.insert(1, nn.LayerNorm(output_dim))
        self.hidden_layer = nn.Sequential(*hidden_layer)

    def forward(self, V_emb, V):
        V_mask = self.mask_layer(V_emb)
        V_maskemb = V_mask * V
        V_out = self.hidden_layer(V_maskemb)
        return V_out    
    
    def extra_repr(self):
        return f'hidden_dim={self.hidden_dim}, output_dim={self.output_dim}, reduction_ratio={self.reduction_ratio}, layer_norm={self.layer_norm}'
    
    
class ParallelMaskNet(nn.Module):
    def __init__(self, num_fields, embed_dim, num_blocks=1, blockout_dim=64, reduction_ratio=1, 
                 mlp_layer=[], activation='relu', dropout=0, hidden_layer_norm=True):
        super().__init__()
        self.num_blocks = num_blocks
        self.block_dim = blockout_dim
        self.mask_blocks = nn.ModuleList([MaskBlock(
                                num_fields * embed_dim, 
                                num_fields * embed_dim, 
                                blockout_dim, 
                                reduction_ratio, 
                                activation, 
                                dropout,
                                hidden_layer_norm) 
                            for _ in range(num_blocks)])
        self.mlp = MLPModule(mlp_layers=[num_blocks * blockout_dim] + mlp_layer + [1],
                             activation_func=activation,
                             dropout=dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
    def forward(self, inputs):
        bs = inputs.size(0)
        ln_emb = self.layer_norm(inputs)
        block_out = [mb(inputs.view(bs, -1), ln_emb.view(bs, -1)) for mb in self.mask_blocks]
        concat_out = torch.cat(block_out, dim=-1)
        V_out = self.mlp(concat_out)
        return V_out
    
    def extra_repr(self):
        return f'num_blocks={self.num_blocks}, block_dim={self.block_dim}'
    
    
class SerialMaskNet(nn.Module):
    def __init__(self, num_fields, embed_dim, block_dim, reduction_ratio=1, 
                 activation='relu', dropout=0, hidden_layer_norm=True):
        super().__init__()
        if not isinstance(block_dim, list):
            self.block_dim = [block_dim]
        self.block_dim = [num_fields * embed_dim] + self.block_dim
        self.mask_blocks = VStackLayer(
                               HStackLayer(*[
                                    MaskBlock(
                                        num_fields * embed_dim, 
                                        self.block_dim[i], 
                                        self.block_dim[i + 1], 
                                        reduction_ratio, 
                                        activation, 
                                        dropout,
                                        hidden_layer_norm)
                                for i in range(len(self.block_dim) - 1)]))
        self.fc = nn.Linear(self.block_dim[-1], 1)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, inputs):
        ln_emb = self.layer_norm(inputs)
        block_out = self.mask_blocks(inputs, ln_emb)
        V_out = self.fc(block_out)
        return V_out
    
    def extra_repr(self):
        return f'block_dim={self.block_dim}'
    

class InnerProductLayer(nn.Module):
    def __init__(self, num_fields):
        super().__init__()
        self.triu_index = nn.Parameter(
                            torch.triu_indices(num_fields, num_fields, offset=1), 
                            requires_grad=False)

    def forward(self, inputs):
        # inputs: B x F x D
        emb0 = torch.index_select(inputs, 1, self.triu_index[0])
        emb1 = torch.index_select(inputs, 1, self.triu_index[1])
        outer_prod_mat = torch.einsum('bnd,bnd->bn', [emb0, emb1])
        return outer_prod_mat.view(inputs.size(0), -1)
    
    
class OuterProductLayer(nn.Module):
    def __init__(self, num_fields):
        super().__init__()
        self.triu_index = nn.Parameter(
                            torch.triu_indices(num_fields, num_fields, offset=1), 
                            requires_grad=False)

    def forward(self, inputs):
        # inputs: B x F x D
        emb0 = torch.index_select(inputs, 1, self.triu_index[0])
        emb1 = torch.index_select(inputs, 1, self.triu_index[1])
        outer_prod_mat = torch.einsum('bni,bnj->bnij', [emb0, emb1])
        return outer_prod_mat.view(inputs.size(0), -1)
    
    
class OperationAwareFMLayer(nn.Module):
    def __init__(self, num_fields):
        super().__init__()
        self.num_fields = num_fields
        self.triu_mask = nn.Parameter(
                            torch.triu(
                                torch.ones(num_fields, num_fields), diagonal=1
                            ).bool(),                                       # F x F; mask_{ij}
                            requires_grad=False)
        self.diag_mask = nn.Parameter(
                            torch.eye(num_fields).bool().unsqueeze(-1),     # F x 1; mask_{ii}
                            requires_grad=False)
        
    def forward(self, inputs):
        bs = inputs.size(0)
        field_wise_emb = inputs.view(bs, self.num_fields, self.num_fields, -1)      # B x F x F x D
        emb_copy = torch.masked_select(field_wise_emb, self.diag_mask)              # B x F x D; copy i-th of emb_i
        emb_copy = emb_copy.view(bs, -1)
        
        inner_prod = (field_wise_emb.transpose(1, 2) * field_wise_emb).sum(dim=-1)  # B x F x F; <j-th of emb_i, i-th of emb_j> 
        ffm_out = torch.masked_select(inner_prod, self.triu_mask)
        ffm_out = ffm_out.view(bs, -1)
        
        output = torch.cat([emb_copy, ffm_out], dim=1)
        return output
        
        
class FieldAwareFMLayer(nn.Module):
    def __init__(self, num_fields):
        super().__init__()
        self.num_fields = num_fields
        self.triu_mask = nn.Parameter(
                            torch.triu(
                                torch.ones(num_fields, num_fields - 1), 
                                diagonal=0
                            ).bool().unsqueeze(-1),     # F x F-1 x 1; mask_{ij}
                            requires_grad=False)
        self.tril_mask = nn.Parameter(
                            torch.tril(
                                torch.ones(num_fields, num_fields - 1), 
                                diagonal=-1
                            ).bool().t().unsqueeze(-1), # F-1 x F x 1; mask_{ji}
                            requires_grad=False)
    
    def forward(self, inputs):
        bs = inputs.size(0)
        field_wise_emb = inputs.view(bs, self.num_fields, self.num_fields - 1, -1)      # B x F x F-1 x D
        emb0 = torch.masked_select(field_wise_emb, self.triu_mask)                      # B*num_pairs*D, 1D tensor
        emb1 = torch.masked_select(field_wise_emb.transpose(1, 2), self.tril_mask)      # B*num_pairs*D, 1D tensor
        output = (emb0 * emb1).view(bs, -1).sum(-1)                                     # w_{ij} * w_{ji}
        return output
        

class GeneralizedInteractionFusion(nn.Module):
    def __init__(self, num_fields, embed_dim, in_subspaces, out_subspaces):
        super().__init__()
        self.W = nn.Parameter(torch.eye(embed_dim, embed_dim).unsqueeze(0).repeat(out_subspaces, 1, 1))
        self.alpha = nn.Parameter(torch.ones(num_fields, in_subspaces, out_subspaces))
        self.h = nn.Parameter(torch.ones(out_subspaces, embed_dim, 1))

    def forward(self, B0, Bi):
        outer_prod = torch.einsum('bfi,bnj->bfnij', [B0, Bi])                           # B x F x IN x D x D 
        fusion= torch.einsum('bfiDd,fin->bnDd', [outer_prod, self.alpha])               # B x N x D x D
        fusion= fusion * self.W                                                         # B x N x D x D
        output = torch.matmul(fusion, self.h).squeeze(-1)   # B x N x D
        return output  
    
    
class GeneralizedInteractionNet(nn.Module):
    def __init__(self, num_fields, embed_dim, num_layers, num_subspaces):
        super().__init__()
        self.layers = nn.ModuleList([
                    GeneralizedInteractionFusion( 
                        num_fields, 
                        embed_dim,
                        num_fields if i == 0 else num_subspaces, 
                        num_subspaces)
                    for i in range(num_layers)])
    
    def forward(self, inputs):
        B_i = inputs
        for layer in self.layers:
            B_i = layer(inputs, B_i)
        return B_i  