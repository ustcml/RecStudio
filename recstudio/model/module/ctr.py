from typing import Set
import math
import torch
import torch.nn as nn
from itertools import product
from .layers import MLPModule, AttentionLayer, SeqPoolingLayer, get_act, KMaxPoolingLayer, HStackLayer, LambdaLayer

__all__ = [
    'DenseEmbedding',
    'Embeddings',
    'LinearLayer',
    'FMLayer',
    'CrossInteraction',
    'CrossNetwork',
    'CrossNetworkV2',
    'CrossNetworkMix',
    'DINScorer',
    'BehaviorSequenceTransformer',
    'DIENScorer',
    'CIN',
    'AFMLayer',
    'LogTransformLayer',
    'SelfAttentionInteractingLayer',
    'DisentangledSelfAttentionInteractingLayer',
    'ConvLayer',
    'FGCNNLayer',
    'SqueezeExcitation',
    'BilinearInteraction',
    'MaskBlock',
    'ParallelMaskNet',
    'SerialMaskNet',
    'InnerProductLayer',
    'OuterProductLayer',
    'OperationAwareFMLayer',
    'FieldAwareFMLayer',
    'GeneralizedInteractionFusion',
    'InteractionMachine',
    'BridgeLayer',
    'RegulationLayer',
    'FeatureSelection',
    'MultiHeadBilinearFusion',
    'FieldWiseBiInteraction',
    'TrianglePoolingLayer',
    'HolographicFMLayer',
    'AttentionalAggregation',
    'GateNN',
    'PPLayer',
    'SAMFeatureInteraction',
    'GraphAggregationLayer',
    'FiGNNLayer',
    'ExtractionLayer'
]


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
                 share_dense_embedding=False, dense_emb_bias=False, dense_emb_norm=True,
                 with_dense_kernel=False):
        r"""
        Args:
            dense_kernel (bool): if `True`, concat all float feats together and inner product
        """
        super(Embeddings, self).__init__()
        self.embed_dim = embed_dim

        if not isinstance(data.frating, list):
            self.field2types = {f: data.field2type[f] for f in fields if f != data.frating}
        else:
            self.field2types = {f: data.field2type[f] for f in fields if f not in data.frating}
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


class CrossInteraction(nn.Module):
    def __init__(self, embed_dim):
        super(CrossInteraction, self).__init__()
        self.weight = nn.Parameter(torch.randn(embed_dim))
        self.bias = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, X0, Xi):
        return (X0.t() * (torch.tensordot(Xi, self.weight, dims=([1], [0])))).t() + self.bias
    

class CrossNetwork(nn.Module):
    def __init__(self, embed_dim, num_layers):
        super(CrossNetwork, self).__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.cross = nn.ModuleList([CrossInteraction(embed_dim) for _ in range(num_layers)])

    def forward(self, input):
        x = input
        for cross in self.cross:
            x = x + cross(input, x)
        return x

    def extra_repr(self) -> str:
        return f'embed_dim={self.embed_dim}, num_layers={self.num_layers}'


class CrossNetworkV2(torch.nn.Module):
    def __init__(self, embed_dim, num_layers):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.linear = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim, bias=True)
            for _ in range(num_layers)
        ])

    def forward(self, input):
        x0 = input
        xl = x0
        for i in range(self.num_layers):
            xl = x0 * self.linear[i](xl) + xl
        return xl

    def extra_repr(self) -> str:
        return f'embed_dim={self.embed_dim}, num_layers={self.num_layers}'
    
    
class CrossNetworkMix(torch.nn.Module):
    def __init__(self, embed_dim, num_layers, low_rank, num_experts, activation):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.low_rank = low_rank
        self.num_experts = num_experts
        self.activation = activation
        self.U = nn.ParameterList([
                    nn.Parameter(torch.randn(num_experts, embed_dim, low_rank))
                    for _ in range(num_layers)
                ])
        self.V = nn.ParameterList([
                    nn.Parameter(torch.randn(num_experts, embed_dim, low_rank))
                    for _ in range(num_layers)
                ])
        self.C = nn.ParameterList([
                    nn.Parameter(torch.randn(num_experts, low_rank, low_rank))
                    for _ in range(num_layers)
                ])
        self.bias = nn.ParameterList([
                    nn.Parameter(torch.randn(embed_dim))
                    for _ in range(num_layers)
                ])
        self.gate = nn.Linear(embed_dim, num_experts, bias=False)
        self.act = get_act(activation)

    def forward(self, input):
        x0 = input                                          # B x F*D
        xl = x0
        for i in range(self.num_layers):
            gate_score = self.gate(xl).softmax(dim=-1)
            Vx = torch.einsum('edr,bd->ber', [self.V[i], xl])
            Vx = self.act(Vx)
            CVx = torch.einsum('eRr,beR->ber', [self.C[i], Vx])
            CVx = self.act(CVx)
            UCVx = torch.einsum('edr,ber->ebd', [self.U[i], CVx])
            expert_out = x0 * (UCVx + self.bias[i])
            xl = torch.einsum('be,ebd->bd', [gate_score, expert_out]) + xl
        return xl

    def extra_repr(self) -> str:
        return f'embed_dim={self.embed_dim}, num_layers={self.num_layers}, low_rank={self.low_rank}, num_experts={self.num_experts}, activation={self.activation}'


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
        self.attention = nn.Sequential(
                            nn.Linear(embed_dim, attention_dim),
                            nn.ReLU(),
                            nn.Linear(attention_dim, 1, bias=False),
                            nn.Softmax(dim=1))
        self.p = nn.Linear(embed_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.inner_prod = InnerProductLayer(num_fields, reduction=False)

    def forward(self, inputs):
        # inputs: B x F x D
        prod = self.inner_prod(inputs)
        attn = self.attention(prod)
        attn_sum = (attn * prod).sum(1)
        output = self.dropout(attn_sum)
        output = self.p(output)
        output = output.squeeze(-1)
        return output

    def extra_repr(self):
        return f"attention dim={self.attention_dim}, dropout={self.dropout.p}"
   
    
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
        output = exp_out.flatten(1)
        return output
    
    def extra_repr(self):
        return f"hidden_size={self.hidden_size}, clamp_min={self.clamp_min}"
    
    
class SelfAttentionInteractingLayer(nn.Module):
    
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
        return f"attention_dim={self.embed_dim}, n_head={self.n_head}, dropout={self.dropout}, residual={self.residual}, layer_norm={self.layer_norm}"
    

class DisentangledSelfAttentionInteractingLayer(nn.Module):
    
    def __init__(self, embed_dim, attention_dim, n_head=1, dropout=0, residual=True, scale=True, relu_before_att=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.attention_dim = attention_dim
        self.n_head = n_head
        self.residual = residual
        self.scale = scale
        self.relu_before_att = relu_before_att
        self.unary = nn.Linear(embed_dim, n_head)
        self.Wq = nn.Linear(embed_dim, attention_dim)
        self.Wk = nn.Linear(embed_dim, attention_dim)
        self.Wv = nn.Linear(embed_dim, attention_dim)  
        self.dropout = nn.Dropout(dropout) 
        if residual:
            self.res = nn.Linear(embed_dim, embed_dim)
            
    def forward(self, inputs):
        bs = inputs.size(0)
        
        unary = self.unary(inputs)                                      # B x F x n_head
        unary_weight = unary.softmax(dim=1)
        unary_weight = unary_weight.view(bs * self.n_head, -1, 1)       # B*n_head x F x 1
        
        query = self.Wq(inputs)
        key = self.Wk(inputs)
        value = self.Wv(inputs)
        
        if self.relu_before_att:
            query = query.relu()
            key = key.relu()
            value = value.relu()
            
        dim_per_head = self.attention_dim // self.n_head
        query = query.view(bs * self.n_head, -1, dim_per_head)
        key = key.view(bs * self.n_head, -1, dim_per_head)
        value = value.view(bs * self.n_head, -1, dim_per_head)
            
        mu_query = query - query.mean(dim=1, keepdim=True)
        mu_key = key - key.mean(dim=1, keepdim=True)
        
        pair_weight = mu_query @ mu_key.transpose(1, 2)
        if self.scale:
            pair_weight /=  dim_per_head ** 0.5
        pair_weight = pair_weight.softmax(dim=2)                        # B*n_head x F x F
        
        attn = unary_weight + pair_weight 
        attn = self.dropout(attn)
        output = attn @ value
        output = output.view(bs, -1, self.attention_dim)       
        
        if self.residual:
            res_out = self.res(inputs)
            output += res_out
        return output
    
    def extra_repr(self):
        return f"attention_dim={self.embed_dim}, n_head={self.n_head}, dropout={self.dropout.p}, " + \
                f"residual={self.residual}, scale={self.scale}, relu_before_att={self.relu_before_att}"
 
    
class ConvLayer(nn.Module):
    
    def __init__(self, num_fields, channels, heights):
        super().__init__()
        
        if len(heights) != len(channels):
            raise ValueError("channels and widths should have the same length.")
        
        self.channels = [1] + channels
        self.heights = heights
        layers = len(heights)
        
        module_list = []
        for i in range(1, len(self.channels)):
            in_channel = self.channels[i - 1]
            out_channel = self.channels[i]
            height = heights[i - 1]
            module_list.append(nn.Conv2d(in_channel, out_channel, kernel_size=(height, 1), padding='same'))
            if i < layers:
                k = max(3, int((1 - pow(float(i) / layers, layers - i)) * num_fields))
            else:
                k = 3
            module_list.append(nn.Tanh())
            module_list.append(KMaxPoolingLayer(k, dim=2))
        self.conv = nn.Sequential(*module_list)

    def forward(self, inputs):
        return self.conv(inputs.unsqueeze(1)) # -> N(Batch size) x C(Channels) x H(Height) x W(Embed dim)
                
    def extra_repr(self):
        return f"channels={self.channels}, heights={self.heights}"
        

class FGCNNLayer(nn.Module):
    def __init__(self, num_raw_fields, embed_dim, channels, heights, 
                 pooling_sizes, recombine_channels, batch_norm):
        super().__init__()
        
        if len(heights) != len(channels):
            raise ValueError("channels and widths should have the same length.")
        
        self.channels = [1] + channels
        self.heights = heights
        self.pooling_sizes = pooling_sizes
        self.recombine_channels = recombine_channels
        
        conv_layers = []
        recomb_layers = []
        self.out_height = [num_raw_fields]
        for i in range(1, len(self.channels)):
            module_list = []
            in_channel = self.channels[i - 1]
            out_channel = self.channels[i]
            height = heights[i - 1]
            pooling_size = pooling_sizes[i - 1]
            recomb_channel = recombine_channels[i - 1]
            module_list += [nn.Conv2d(in_channel, out_channel, kernel_size=(height, 1), padding='same')]
            if batch_norm:
                module_list += [nn.BatchNorm2d(out_channel)]
            module_list += [nn.Tanh(), nn.MaxPool2d((pooling_size, 1))]
            conv_layers.append(nn.Sequential(*module_list))
            self.out_height.append(self.out_height[-1] // pooling_size)
            if self.out_height[-1] == 0:
                raise ValueError(f'pooling_sizes[{i-1}:] is too large, please change them to 1.')
            recomb_layers.append(nn.Sequential(
                                    nn.Linear(out_channel*self.out_height[-1]*embed_dim, 
                                              recomb_channel*self.out_height[-1]*embed_dim), 
                                    nn.Tanh()))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.recomb_layers = nn.ModuleList(recomb_layers)

    def forward(self, inputs):
        conv_out = inputs.unsqueeze(1)
        new_emb = []
        for conv, recomb in zip(self.conv_layers, self.recomb_layers):
            conv_out = conv(conv_out)                   # N(Batch size) x C(Channels) x H(Height) x W(Embed dim)
            recomb_out = recomb(conv_out.flatten(1))
            new_emb.append(recomb_out.reshape(inputs.size(0), -1, inputs.size(-1)))
        new_emb = torch.cat(new_emb, dim=1)
        return new_emb
                
    def extra_repr(self):
        return f"channels={self.channels}, heights={self.heights}, " + \
                f"pooling_sizes={self.pooling_sizes}, recombine_channels={self.recombine_channels}, " + \
                f"out_height={self.out_height}"
        
        
class SqueezeExcitation(nn.Module):
    
    def __init__(self, num_fields, reduction_ratio, activation, pool='avg') -> None:
        super().__init__()
        
        if pool.lower() not in ['avg', 'max']:
            raise ValueError(f'Expect pool to be `avg` or `max`, but got {pool}.')
        
        if not isinstance(activation, list):
            self.activation = [activation]
        if len(self.activation) == 1:
            self.activation = 2 * self.activation    
        elif len(self.activation) > 2 or len(self.activation) < 1:
            raise ValueError(f'Expect activation to be one or two, but got {len([activation])}.')
        
            
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
            self.weight = nn.Parameter(torch.randn(embed_dim, embed_dim))
        elif self.bilinear_type == 'each':
            self.weight = nn.Parameter(torch.randn(num_fields, embed_dim, embed_dim))
        else:
            self.weight = nn.Parameter(torch.randn(
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
            raise ValueError(f'Expect reduction_ratio > 1, but got {reduction_ratio}.')
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
        self.mlp = MLPModule([num_blocks * blockout_dim] + mlp_layer + [1],
                             activation,
                             dropout,
                             last_activation=False,
                             last_bn=False)
        self.layer_norm = nn.LayerNorm(embed_dim)
    def forward(self, inputs):
        ln_emb = self.layer_norm(inputs)
        block_out = [mb(inputs.flatten(1), ln_emb.flatten(1)) for mb in self.mask_blocks]
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
        self.mask_blocks = nn.ModuleList([
                                MaskBlock(
                                    num_fields * embed_dim, 
                                    self.block_dim[i], 
                                    self.block_dim[i + 1], 
                                    reduction_ratio, 
                                    activation, 
                                    dropout,
                                    hidden_layer_norm)
                                for i in range(len(self.block_dim) - 1)])
        self.fc = nn.Linear(self.block_dim[-1], 1)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, inputs):
        bs = inputs.size(0)
        ln_emb = self.layer_norm(inputs)
        for mb in self.mask_blocks:
            block_out = mb(inputs.flatten(1), ln_emb.flatten(1))
        V_out = self.fc(block_out)
        return V_out
    
    def extra_repr(self):
        return f'block_dim={self.block_dim}'
    

class InnerProductLayer(nn.Module):
    def __init__(self, num_fields, reduction : bool = True):
        super().__init__()
        self.triu_index = nn.Parameter(
                            torch.triu_indices(num_fields, num_fields, offset=1), 
                            requires_grad=False)  
        self.reduction = reduction

    def forward(self, inputs):
        # inputs: B x F x D
        emb0 = torch.index_select(inputs, 1, self.triu_index[0])
        emb1 = torch.index_select(inputs, 1, self.triu_index[1])
        inner_prod_mat = emb0 * emb1                # B x N x D
        if self.reduction:
            return inner_prod_mat.sum(-1)
        else:
            return inner_prod_mat
    
    
class OuterProductLayer(nn.Module):
    def __init__(self, num_fields, reduction : bool = True):
        super().__init__()
        self.triu_index = nn.Parameter(
                            torch.triu_indices(num_fields, num_fields, offset=1), 
                            requires_grad=False)
        self.reduction = reduction

    def forward(self, inputs):
        # inputs: B x F x D
        emb0 = torch.index_select(inputs, 1, self.triu_index[0])
        emb1 = torch.index_select(inputs, 1, self.triu_index[1])
        outer_prod_mat = emb0.unsqueeze(-1) @ emb1.unsqueeze(-2)    # B x N x D x D
        if self.reduction:
            return outer_prod_mat.flatten(1)
        else:
            return outer_prod_mat
    
    
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
        emb_copy = emb_copy.flatten(1)
        
        inner_prod = (field_wise_emb.transpose(1, 2) * field_wise_emb).sum(dim=-1)  # B x F x F; <j-th of emb_i, i-th of emb_j> 
        ffm_out = torch.masked_select(inner_prod, self.triu_mask)
        ffm_out = ffm_out.flatten(1)
        
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
        output = (emb0 * emb1).flatten(1).sum(-1)                                       # w_{ij} * w_{ji}
        return output
        

class GeneralizedInteractionFusion(nn.Module):
    def __init__(self, num_fields, embed_dim, in_subspaces, out_subspaces):
        super().__init__()
        self.in_subspaces = in_subspaces
        self.out_subspaces = out_subspaces
        self.W = nn.Parameter(torch.eye(embed_dim, embed_dim).unsqueeze(0).repeat(out_subspaces, 1, 1))
        self.alpha = nn.Parameter(torch.ones(num_fields, in_subspaces, out_subspaces))
        self.h = nn.Parameter(torch.ones(out_subspaces, embed_dim, 1))

    def forward(self, B0, Bi):
        outer_prod = torch.einsum('bfi,bnj->bfnij', [B0, Bi])                           # B x F x IN x D x D 
        fusion= torch.einsum('bfiDd,fin->bnDd', [outer_prod, self.alpha])               # B x N x D x D
        fusion= fusion * self.W                                                         # B x N x D x D
        output = torch.matmul(fusion, self.h).squeeze(-1)   # B x N x D
        return output  
    
    def extra_repr(self):
        return f'in_subspaces={self.in_subspaces}, out_subspaces={self.out_subspaces}'
    
    
class GeneralizedInteractionNet(nn.Module):
    def __init__(self, num_fields, embed_dim, num_layers, num_subspaces):
        super().__init__()
        self.num_layers = num_layers
        self.num_subspaces = num_subspaces
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
    
    def extra_repr(self):
        return f'num_layers={self.num_layers}, num_subspaces={self.num_subspaces}'
    
    
class InteractionMachine(nn.Module):
    def __init__(self, embed_dim, order):
        super().__init__()
        self.order = order
        self.fc = nn.Linear(order * embed_dim, 1)
        
    def _2nd_order(self, p1, p2):
        return (p1.pow(2) - p2) / 2

    def _3rd_order(self, p1, p2, p3):
        return (p1.pow(3) - 3 * p1 * p2 + 2 * p3) / 6

    def _4th_order(self, p1, p2, p3, p4):
        return (p1.pow(4) - 6 * p1.pow(2) * p2 + 3 * p2.pow(2)
                + 8 * p1 * p3 - 6 * p4) / 24

    def _5th_order(self, p1, p2, p3, p4, p5):
        return (p1.pow(5) - 10 * p1.pow(3) * p2 + 20 * p1.pow(2) * p3 - 30 * p1 * p4
                - 20 * p2 * p3 + 15 * p1 * p2.pow(2) + 24 * p5) / 120
        
    def _kth_order(self, k, *args):
        sum_ = 0
        C = k * [0]                 # c1, ..., ck
        while True:
            C[0] += 1
            for i in range(0, k):
                if C[i] > k / (i+1):
                    C[i] = 0
                    C[i+1] += 1
                else:
                    break
            if sum([a*b for a, b in zip(C, range(1, k+1))]) == k:
                prod = 1
                for j, c in enumerate(C):
                    prod *= (-args[j] / (j+1)).pow(c) / math.factorial(c)
                sum_ += prod
            if C[-1] == 1:
                break
        return (-1)**k * sum_

    def forward(self, X):
        Q = X
        p1 = Q.sum(dim=1)
        P = [p1]
        interaction = [p1]
        if self.order >= 2:
            Q = Q * X
            P.append(Q.sum(dim=1))
            interaction.append(self._2nd_order(*P))
        if self.order >= 3:
            Q = Q * X
            P.append(Q.sum(dim=1))
            interaction.append(self._3rd_order(*P))
        if self.order >= 4:
            Q = Q * X
            P.append(Q.sum(dim=1))
            interaction.append(self._4th_order(*P))
        if self.order >= 5:
            Q = Q * X
            P.append(Q.sum(dim=1))
            interaction.append(self._5th_order(*P))
        if self.order >= 6:
            for k in range(6, self.order + 1):
                Q = Q * X
                P.append(Q.sum(dim=1))
                interaction.append(self._kth_order(k, *P))
        output = self.fc(torch.cat(interaction, dim=-1))
        return output
        
    def extra_repr(self):
        return f'order={self.order}'


class BridgeLayer(nn.Module):
    def __init__(self, embed_dim, bridge_type):
        super().__init__()
        if bridge_type.lower() not in ['pointwise_addition', 'hadamard_product', 
                                       'concatenation', 'attention_pooling']:
            raise ValueError('Expect bridge_type to be '
                             '`pointwise_addition`|`hadamard_product`|'
                             '`concatenation`|`attention_pooling`, '
                             f'but got {bridge_type}.')
        self.bridge_type = bridge_type.lower()
        if self.bridge_type == 'concatenation':
            self.bridge = nn.Sequential(
                            nn.Linear(2 * embed_dim, embed_dim),
                            nn.ReLU()
                        )
        else:
            self.bridge = HStackLayer(
                            nn.Sequential(
                                nn.Linear(embed_dim, embed_dim),
                                nn.ReLU(),
                                nn.Linear(embed_dim, embed_dim, bias=False),
                                nn.Softmax(dim=-1)),
                            nn.Sequential(
                                nn.Linear(embed_dim, embed_dim),
                                nn.ReLU(),
                                nn.Linear(embed_dim, embed_dim, bias=False),
                                nn.Softmax(dim=-1))
                        )
    
    def forward(self, input0, input1):
        if self.bridge_type == 'pointwise_addition':
            return input0 + input1
        elif self.bridge_type == 'hadamard_product':
            return input0 * input1
        elif self.bridge_type == 'concatenation':
            return self.bridge(torch.cat([input0, input1], dim=-1))
        else:
            a0, a1 = self.bridge(input0, input1)
            return a0 * input0 + a1 * input1
    
    def extra_repr(self):
        return f'bridge_type={self.bridge_type}'
    
    
class RegulationLayer(nn.Module):
    def __init__(self, num_fields, embed_dim, temperature=1.0, batch_norm=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.batch_norm = batch_norm
        self.cross_gate = nn.Parameter(torch.ones(num_fields))
        self.deep_gate = nn.Parameter(torch.ones(num_fields))
        if batch_norm:
            self.cross_bn = nn.BatchNorm1d(num_fields * embed_dim)
            self.deep_bn = nn.BatchNorm1d(num_fields * embed_dim)
            
    def forward(self, inputs):
        cross_g = (self.cross_gate / self.temperature).softmax(dim=0)
        deep_g = (self.deep_gate / self.temperature).softmax(dim=0)
        cross_out = cross_g.tile(self.embed_dim) * inputs
        deep_out = deep_g.tile(self.embed_dim) * inputs
        if self.batch_norm:
            cross_out = self.cross_bn(cross_out)
            deep_out = self.deep_bn(deep_out)
        return cross_out, deep_out
    
    def extra_repr(self):
        return f'temperature={self.temperature}'
    

class FeatureSelection(nn.Module):
    def __init__(self, stream1_fields, stream2_fields, embed_dim, num_fields, data, mlp_layer, dropout=0):
        super().__init__()
        self.fields1 = stream1_fields
        self.fields2 = stream2_fields
        
        num_fields1 = len(stream1_fields)
        num_fields2 = len(stream2_fields)
        
        self.embedding1 = Embeddings(stream1_fields, embed_dim, data)
        self.embedding2 = Embeddings(stream2_fields, embed_dim, data)
            
        self.gate1 = MLPModule(
                        [embed_dim * num_fields1] + mlp_layer + [embed_dim * num_fields], 
                        'relu', dropout, last_activation=False)
        self.gate2 = MLPModule(
                        [embed_dim * num_fields2] + mlp_layer + [embed_dim * num_fields], 
                        'relu', dropout, last_activation=False)


    def forward(self, batch, inputs):
        emb1 = self.embedding1(batch)
        g1 = 2 * self.gate1(emb1.flatten(1)).sigmoid()
        
        emb2 = self.embedding2(batch)
        g2 = 2 * self.gate2(emb2.flatten(1)).sigmoid()    
        
        return g1 * inputs, g2 * inputs
    
    def extra_repr(self):
        return f'stream1_fields={self.fields1}, stream2_fields={self.fields2}'


class MultiHeadBilinearFusion(nn.Module):
    def __init__(self, n_head, embed_dim1, embed_dim2, output_dim=1):
        super().__init__()
        self.n_head = n_head
        self.output_dim = output_dim
        self.dim1_per_head = embed_dim1 // n_head
        self.dim2_per_head = embed_dim2 // n_head
        self.blrs = nn.ModuleList([
                        nn.Bilinear(self.dim1_per_head, self.dim2_per_head, output_dim)
                        for _ in range(n_head)
                    ])
        self.lr1 = nn.Linear(embed_dim1, output_dim, bias=False)
        self.lr2 = nn.Linear(embed_dim2, output_dim, bias=False)

    def forward(self, input1, input2):
        lr_out = self.lr1(input1) + self.lr2(input2)
        blr_out = []
        input1 = input1.view(-1, self.n_head, self.dim1_per_head)
        input2 = input2.view(-1, self.n_head, self.dim2_per_head)  
        for i, blr in enumerate(self.blrs):
            blr_out.append(blr(input1[:, i, :], input2[:, i, :]))
        blr_out = torch.cat(blr_out, dim=-1).sum(-1, keepdim=True)
        output = lr_out + blr_out
        return output
    
    
class FieldWiseBiInteraction(nn.Module):
    def __init__(self, embed_dim, data, activation, dropout, fields):
        super().__init__()
        all_fields = set()
        for f in fields:
            all_fields = all_fields.union(set(f))
        self.fields = fields
        self.linear = LinearLayer(all_fields, data)
        self.mf = InnerProductLayer(len(fields), reduction=False)
        self.fm = FMLayer(reduction='none')
        self.fc = MLPModule(2 * [embed_dim + 1], 
                            activation, dropout,
                            bias=False, batch_norm=True,
                            last_activation=True, last_bn=True)
        self.r_mf = nn.Linear(math.comb(len(fields), 2), 1, bias=False)
        self.r_fm = nn.Linear(len(fields), 1, bias=False)
        
    def forward(self, batch, field_embs):
        lr_out = self.linear(batch)
        mf_in = torch.cat([_.sum(1, keepdim=True) for _ in field_embs], dim=1)          # B x M x D
        mf_out = self.r_mf(self.mf(mf_in).transpose(1, 2))                              # B x N x D -> B x D x N -> B x D x 1
        fm_out = torch.stack([self.fm(_) for _ in field_embs], dim=1)                   # B x M x D
        fm_out = self.r_fm(fm_out.transpose(1, 2))                                      # B x D x 1                   
        fwbi_out = self.fc(torch.cat([lr_out.unsqueeze(-1), (fm_out + mf_out).squeeze(-1)], dim=-1))  # B x (D+1)
        return fwbi_out

    def extra_repr(self):
        return f'fields={self.fields}'
 
     
class TrianglePoolingLayer(nn.Module):
    def __init__(self, num_fields):
        super().__init__()
        self.inner_product = InnerProductLayer(num_fields)
        
    def forward(self, inputs):
        '''
        gamma(u,v)  = (1 - <u, v>L - u0 - v0) / (u0 * v0)
                    = (1 + u0 * v0 - sum_1^d u_i*vi - u0 - v0) / (u0 * v0)
                    = 1 + (1 - sum_1^d u_i*vi - u0 - v0) / (u0 * v0)
        '''
        inner_prod = self.inner_product(inputs)                                         # B x N
        zero_component = torch.sqrt(1 + (inputs**2).sum(-1))                    # B x num_fields
        u0 = torch.index_select(zero_component, -1, self.inner_product.triu_index[0])   # B x N
        v0 = torch.index_select(zero_component, -1, self.inner_product.triu_index[1])   # B x N
        gamma = 1 + (1 - inner_prod - u0 - v0) / (u0 * v0)
        output = gamma.sum(-1)
        return output
    
    
class HolographicFMLayer(nn.Module):
    def __init__(self, num_fields, op):
        super().__init__()
        if op.lower() not in ['circular_convolution', 'circular_correlation']:
            raise ValueError(f'Expect op to be `circular_convolution`|'
                             f'`circular_correlation`, but got {op}.')
        self.op = op.lower()
        self.triu_index = nn.Parameter(
                            torch.triu_indices(num_fields, num_fields, offset=1), 
                            requires_grad=False)  
        
    def forward(self, inputs):
        emb0 = torch.index_select(inputs, 1, self.triu_index[0])
        emb1 = torch.index_select(inputs, 1, self.triu_index[1])
        fft0 = torch.fft.fft(emb0)
        fft1 = torch.fft.fft(emb1)
        if self.op == 'circular_correlation':
            fft0 = fft0.conj()
        output = torch.view_as_real(torch.fft.ifft(fft0 * fft1))[..., 0]
        return output
    
    def extra_repr(self):
        return f'op={self.op}'
    
    
class AttentionalAggregation(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.agg = nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim, bias=False), 
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1, bias=False),           # query layer
                    nn.Softmax(dim=1))

    def forward(self, key, value):
        attn_weight = self.agg(key)
        attn_out = (attn_weight * value).sum(dim=1)
        return attn_out
    

class GateNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout, activation, batch_norm):
        super().__init__()
        self.gate = MLPModule(
                        [in_dim, hidden_dim, out_dim],
                        activation,
                        dropout,
                        batch_norm=batch_norm,
                        last_activation=False,
                        last_bn=False
                    )
        self.gate.add_modules(nn.Sigmoid())
        
    def forward(self, inputs):
        return 2 * self.gate(inputs)
    
    
class PPLayer(nn.Module):
    def __init__(self, mlp_layer, gate_in_dim, gate_hidden_dim, activation, dropout, batch_norm):
        super().__init__()
        self.gate = GateNN(
                        gate_in_dim, 
                        gate_hidden_dim, 
                        mlp_layer[0],
                        dropout, 
                        activation, 
                        batch_norm)
        self.mlp = MLPModule(
                    mlp_layer, 
                    activation, 
                    dropout, 
                    batch_norm=batch_norm, 
                    last_activation=False)
    
    def forward(self, gate_in, mlp_in):
        gate_out = self.gate(gate_in)
        output = self.mlp(gate_out * mlp_in)
        return output


class SAMFeatureInteraction(nn.Module):
    '''
    According to table 2 in SAM, there is no sum on SAM2A and SAM2E.
    '''
    def __init__(self, interaction_type, embed_dim, num_fields, dropout):
        super().__init__()
        if interaction_type not in ['sam1', 'sam2a', 'sam2e', 'sam3a', 'sam3e']:
            raise ValueError('Expect interaction_type to be `sam1`|`sam2a`|`sam2e`|`sam3a`|`sam3e`, '
                             f'but got {interaction_type}.')
        self.interaction_type = interaction_type   
        if interaction_type in ['sam2a', 'sam3a']:
            self.W = nn.Parameter(torch.ones(num_fields, num_fields, embed_dim))
        if interaction_type == 'sam3a':
            self.K = nn.Linear(embed_dim, embed_dim, bias=False)
            self.res = nn.Linear(embed_dim, embed_dim, bias=False)
        elif interaction_type == 'sam3e':
            self.K = nn.Linear(embed_dim, embed_dim, bias=False)
            self.res = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
            
    def forward(self, inputs):
        if self.interaction_type == 'sam1':
            output = inputs                                                     # B x F x D
        elif self.interaction_type == 'sam2a':
            inner_prod = inputs @ inputs.transpose(1, 2)                        # B x F x F
            output = (inner_prod.unsqueeze(-1) * self.W)                        # B x F x F x D
        elif self.interaction_type == 'sam2e':
            inner_prod = torch.einsum('bFd,bfd->bFfd', 2 * [inputs])            # B x F x F x D
            output = (inner_prod.sum(-1, keepdim=True) * inner_prod)            # B x F x F x D
        else:
            inner_prod = inputs @ self.K(inputs).transpose(1, 2)                # B x F x F
            if self.interaction_type == 'sam3a':
                output = (inner_prod.unsqueeze(-1) * self.W).sum(2)             # B x F x F x D -> B x F x D
            else:
                output = (inner_prod.unsqueeze(-1) * (torch.einsum('bFd,bfd->bFfd', 2 * [inputs]))).sum(2)  # B x F x F x D -> B x F x D
            output = output + self.res(inputs) 
        output = self.dropout(output)
        return output   
    
    def extra_repr(self):
        return f'interaction_type={self.interaction_type}, dropout={self.dropout.p}'
    
    
class GraphAggregationLayer(nn.Module):
    def __init__(self, num_fields, embed_dim):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(num_fields, embed_dim, embed_dim))
        self.W_out = nn.Parameter(torch.randn(num_fields, embed_dim, embed_dim))
        self.bias = nn.Parameter(torch.zeros(embed_dim))
        
    def forward(self, h, w):
        '''w: F x F'''
        h_out = (self.W_out @ h.unsqueeze(-1)).squeeze(-1)          # F x D x D, B x F x D -> B x F x D
        agg = w @ h_out                                             # B x F x F, B x F x D -> B x F x D
        a = (self.W_in @ agg.unsqueeze(-1)).squeeze(-1) + self.bias # F x D x D, B x F x D -> B x F x D
        return a
        
    
class FiGNNLayer(nn.Module):
    def __init__(self, num_fields, embed_dim, num_layers):
        super().__init__()
        self.idx_i, self.idx_j = zip(*list(product(range(num_fields), repeat=2)))  
        self.w = nn.Sequential(
                    nn.Linear(2 * embed_dim, 1, bias=False),
                    nn.LeakyReLU(),
                    LambdaLayer(lambda x: x.reshape(-1, num_fields, num_fields)),
                    nn.Softmax(-1),
                    LambdaLayer(lambda x: x - x[0].diag().diag_embed()))
        self.gnn = nn.ModuleList([
                        GraphAggregationLayer(num_fields, embed_dim)
                        for _ in range(num_layers)
                    ])
        self.gru = nn.GRUCell(embed_dim, embed_dim)
        
    def forward(self, inputs):
        emb0 = inputs[:, self.idx_i, :]
        emb1 = inputs[:, self.idx_j, :]
        w = self.w(torch.cat([emb0, emb1], dim=-1))
        h = inputs
        for gnn in self.gnn:
            a = gnn(h, w)
            h = self.gru(a.flatten(end_dim=1), h.flatten(end_dim=1)).view_as(inputs)
            h = h + inputs
        return h
    
    def extra_repr(self):
        return f'num_layers={len(self.gnn)}'
    
    
class ExtractionLayer(nn.Module):
    r"""Extraction Layer of PLE.
    
    Args:
        in_dim(int): dimension of layer input
        specific_experts_per_task(int): number of task-specific experts per task
        num_task(int): number of tasks
        num_shared_experts(int): number of shared experts
        share_gate(bool): whether to set share_gate, `False` for final ExtractionLayer
        expert_mlp_layer(list): list of hidden layers of each expert
        expert_activation(str): activation function for each expert
        expert_dropout(float): dropout rate for each expert
        gate_mlp_layer(list): list of hidden layers of each gate
        gate_activation(str): activation function for each gate
        gate_dropout(float): dropout rate for each gate
        
    Returns:
        list: each element is a torch.Tensor with shape of (batch_size, expert_mlp_layer[-1])
    
    """
    def __init__(self, in_dim, specific_experts_per_task, num_task, num_shared_experts, share_gate,
                 expert_mlp_layer, expert_activation, expert_dropout, 
                 gate_mlp_layer, gate_activation, gate_dropout):
        super().__init__()
        self.specific_experts_per_task = specific_experts_per_task
        self.num_task = num_task
        self.num_shared_experts = num_shared_experts
        self.share_gate = share_gate
        self.specific_experts = nn.ModuleList([
                                    nn.ModuleList([
                                        MLPModule(
                                            [in_dim] + expert_mlp_layer,
                                            expert_activation, 
                                            expert_dropout)
                                        for _ in range(specific_experts_per_task)
                                    ])
                                    for _ in range(num_task)
                                ])
        self.shared_experts = nn.ModuleList([
                                MLPModule(
                                    [in_dim] + expert_mlp_layer,
                                    expert_activation, 
                                    expert_dropout)
                                for _ in range(num_shared_experts)
                            ])
        self.gates = nn.ModuleList([
                        MLPModule(
                            [in_dim] + gate_mlp_layer + [specific_experts_per_task + num_shared_experts],
                            gate_activation, 
                            gate_dropout)
                        for _ in range(num_task)
                    ])
        for g in self.gates:
            g.add_modules(nn.Softmax(-1))
            
        if share_gate:
            self.shared_gates = MLPModule(
                                    [in_dim] + gate_mlp_layer + [num_task * specific_experts_per_task + num_shared_experts],
                                    gate_activation, 
                                    gate_dropout)
            self.shared_gates.add_modules(nn.Softmax(-1))
    
    def forward(self, inputs):
        experts_out = []
        for i, experts_per_task in enumerate(self.specific_experts):
            experts_out.append(torch.stack([e(inputs[i]) for e in experts_per_task], dim=1))    # B x SpecificPerTask x De
                
        shared_e_out = torch.stack(
                        [shared_e(inputs[-1]) for shared_e in self.shared_experts], dim=1)      # B x Shared x De
        
        outputs = []
        for i, (g, e_out) in enumerate(zip(self.gates, experts_out)):
            gate_out = g(inputs[i])                                                             # B x (SpecificPerTask + Shared)
            outputs.append((gate_out.unsqueeze(-1) * torch.cat([e_out, shared_e_out], dim=1)).sum(1))    # B x De
        
        if self.share_gate:
            shared_gate_out = self.shared_gates(inputs[-1])
            e_out = torch.cat(experts_out, dim=1)                                               # B x num_task*SpecificPerTask x De
            outputs.append((shared_gate_out.unsqueeze(-1) * torch.cat([e_out, shared_e_out], dim=1)).sum(1))
        return outputs
    
    def extra_repr(self):
        return f'specific_experts_per_task={self.specific_experts_per_task}, ' + \
                f'num_task={self.num_task}, num_shared_experts={self.num_shared_experts}'

