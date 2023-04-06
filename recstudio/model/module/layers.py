from typing import Tuple

import torch
from torch.nn.parameter import Parameter


def get_act(activation: str, dim=None):
    if activation == None or isinstance(activation, torch.nn.Module):
        return activation
    elif type(activation) == str:
        if activation.lower() == 'relu':
            return torch.nn.ReLU()
        elif activation.lower() == 'sigmoid':
            return torch.nn.Sigmoid()
        elif activation.lower() == 'tanh':
            return torch.nn.Tanh()
        elif activation.lower() == 'leakyrelu':
            return torch.nn.LeakyReLU()
        elif activation.lower() == 'identity':
            return lambda x: x
        elif activation.lower() == 'dice':
            return Dice(dim)
        elif activation.lower() == 'gelu':
            return torch.nn.GELU()
        elif activation.lower() == 'leakyrelu':
            return torch.nn.LeakyReLU()
        else:
            raise ValueError(
                f'activation function type "{activation}"  is not supported, check spelling or pass in a instance of torch.nn.Module.')
    else:
        raise ValueError(
            '"activation_func" must be a str or a instance of torch.nn.Module. ')


class CrossCompressUnit(torch.nn.Module):
    """
    Cross & Compress unit.
    Performs feature interaction as below:
        .. math::
            C_{l}=v_{l}e_{l}^\top=\begin{bmatrix}
            v_{l}^{(1)}e_{l}^{(1)} & ...  & v_{l}^{(1)}e_{l}^{(d)} \\
            ... &  & ... \\
            v_{l}^{(d)}e_{l}^{(1)} & ... & v_{l}^{(d)}e_{l}^{(d)}
            \end{bmatrix}
            \\
            v_{l+1}=C_{l}W_{l}^{VV}+C_{l}^\top W_{l}^{EV}+b_{l}^{V}
            \\
            e_{l+1}=C_{l}W_{l}^{VE}+C_{l}^\top W_{l}^{EE}+b_{l}^{E}

    Parameters:
        embed_dim(int): dimensions of embeddings.
        weight_vv(torch.nn.Linear): transformation weights.
        weight_ev(torch.nn.Linear): transformation weights.
        weight_ve(torch.nn.Linear): transformation weights.
        weight_ee(torch.nn.Linear): transformation weights.
        bias_v(Parameter): bias on v.
        bias_e(Parameter): bias on e.

    Returns:
        v_output(torch.Tensor): the first embeddings after feature interaction.
        e_output(torch.Tensor): the second embeddings after feature interaction.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.weight_vv = torch.nn.Linear(self.embed_dim, 1, False)
        self.weight_ev = torch.nn.Linear(self.embed_dim, 1, False)
        self.weight_ve = torch.nn.Linear(self.embed_dim, 1, False)
        self.weight_ee = torch.nn.Linear(self.embed_dim, 1, False)
        self.bias_v = Parameter(data=torch.zeros(
            self.embed_dim), requires_grad=True)
        self.bias_e = Parameter(data=torch.zeros(
            self.embed_dim), requires_grad=True)

    def forward(self, inputs):
        # [batch_size, dim, 1] or [batch_size, neg, dim, 1]
        v_input = inputs[0].unsqueeze(-1)
        # [batch_size, 1, dim] or [batch_size, neg, 1, dim]
        e_input = inputs[1].unsqueeze(-2)

        # [batch_size, dim, dim] or [batch_size, neg, dim, dim]
        c_matrix = torch.matmul(v_input, e_input)
        c_matrix_transpose = c_matrix.transpose(-1, -2)

        # [batch_size, dim, 1] -> [batch_size, dim]
        v_output = (self.weight_vv(c_matrix) +
                    self.weight_ev(c_matrix_transpose)).squeeze(-1)
        v_output = v_output + self.bias_v
        e_output = (self.weight_ve(c_matrix) +
                    self.weight_ee(c_matrix_transpose)).squeeze(-1)
        e_output = e_output + self.bias_e

        return (v_output, e_output)


class FeatInterLayers(torch.nn.Module):
    """
    Feature interaction layers with varied feature interaction units.

    Args:
        dim(int): the dimensions of the feature.
        num_layers(int): the number of stacked units in the layers.
        unit(torch.nn.Module): the feature interaction used in the layer.

    Examples:
    >>> featInter = FeatInterLayers(64, 2, CrossCompressUnit)
    >>> featInter.model
    Sequential(
        (unit[0]): CrossCompressUnit(
            (weight_vv): Linear(in_features=64, out_features=1, bias=False)
            (weight_ev): Linear(in_features=64, out_features=1, bias=False)
            (weight_ve): Linear(in_features=64, out_features=1, bias=False)
            (weight_ee): Linear(in_features=64, out_features=1, bias=False)
        )
        (unit[1]): CrossCompressUnit(
            (weight_vv): Linear(in_features=64, out_features=1, bias=False)
            (weight_ev): Linear(in_features=64, out_features=1, bias=False)
            (weight_ve): Linear(in_features=64, out_features=1, bias=False)
            (weight_ee): Linear(in_features=64, out_features=1, bias=False)
        )
    )
    """

    def __init__(self, dim, num_units, unit) -> None:
        super().__init__()
        self.model = torch.nn.Sequential()
        for id in range(num_units):
            self.model.add_module(f'unit[{id}]', unit(dim))

    def forward(self, v_input, e_input):
        return self.model((v_input, e_input))


class MLPModule(torch.nn.Module):
    """
    MLPModule
    Gets a MLP easily and quickly.

    Args:
        mlp_layers(list): the dimensions of every layer in the MLP.
        activation_func(torch.nn.Module,str,None): the activation function in each layer.
        dropout(float): the probability to be set in dropout module. Default: ``0.0``.
        bias(bool): whether to add batch normalization between layers. Default: ``False``.
        last_activation(bool): whether to add activation in the last layer. Default: ``True``.
        last_bn(bool): whether to add batch normalization in the last layer. Default: ``True``.

    Examples:
    >>> MLP = MLPModule([64, 64, 64], 'ReLU', 0.2)
    >>> MLP.model
    Sequential(
        (0): Dropout(p=0.2, inplace=False)
        (1): Linear(in_features=64, out_features=64, bias=True)
        (2): ReLU()
        (3): Dropout(p=0.2, inplace=False)
        (4): Linear(in_features=64, out_features=64, bias=True)
        (5): ReLU()
    )
    >>> MLP.add_modules(torch.nn.Linear(64, 10, True), torch.nn.ReLU())
    >>> MLP.model
    Sequential(
        (0): Dropout(p=0.2, inplace=False)
        (1): Linear(in_features=64, out_features=64, bias=True)
        (2): ReLU()
        (3): Dropout(p=0.2, inplace=False)
        (4): Linear(in_features=64, out_features=64, bias=True)
        (5): ReLU()
        (6): Linear(in_features=64, out_features=10, bias=True)
        (7): ReLU()
    )
    """

    def __init__(self, mlp_layers, activation_func='ReLU', dropout=0.0, bias=True, batch_norm=False, last_activation=True, last_bn=True):
        super().__init__()
        self.mlp_layers = mlp_layers
        self.batch_norm = batch_norm
        self.bias = bias
        self.dropout = dropout
        self.activation_func = activation_func
        self.model = []
        last_bn = self.batch_norm and last_bn
        for idx, layer in enumerate((zip(self.mlp_layers[: -1], self.mlp_layers[1:]))):
            self.model.append(torch.nn.Dropout(dropout))
            self.model.append(torch.nn.Linear(*layer, bias=bias))
            if (idx == len(mlp_layers)-2 and last_bn) or (idx < len(mlp_layers)-2 and batch_norm):
                self.model.append(torch.nn.BatchNorm1d(layer[-1]))
            if ( (idx == len(mlp_layers)-2 and last_activation and activation_func is not None)
                or (idx < len(mlp_layers)-2 and activation_func is not None) ):
                activation = get_act(activation_func, dim=layer[-1])
                self.model.append(activation)
        self.model = torch.nn.Sequential(*self.model)

    def add_modules(self, *args):
        """
        Adds modules into the MLP model after obtaining the instance.

        Args:
            args(variadic argument): the modules to be added into MLP model.
        """
        for block in args:
            assert isinstance(block, torch.nn.Module)

        for block in args:
            self.model.add_module(str(len(self.model._modules)), block)

    def forward(self, input):
        return self.model(input)


class GRULayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_layer=1, bias=False, batch_first=True,
                 bidirectional=False, return_hidden=False) -> None:
        super().__init__()
        self.gru = torch.nn.GRU(
            input_size=input_dim,
            hidden_size=output_dim,
            num_layers=num_layer,
            bias=bias,
            batch_first=batch_first,
            bidirectional=bidirectional
        )
        self.return_hidden = return_hidden

    def forward(self, input):
        out, hidden = self.gru(input)
        if self.return_hidden:
            return out, hidden
        else:
            return out


class SeqPoolingLayer(torch.nn.Module):
    def __init__(self, pooling_type='mean', keepdim=False) -> None:
        super().__init__()
        if not pooling_type in ['origin', 'mask', 'concat', 'sum', 'mean', 'max', 'last']:
            raise ValueError("pooling_type can only be one of ['origin', 'mask', 'concat', 'sum', 'mean', 'max', 'last']"
                             f"but {pooling_type} is given.")
        self.pooling_type = pooling_type
        self.keepdim = keepdim

    def forward(self, batch_seq_embeddings, seq_len, weight=None, mask_token=None):
        # batch_seq_embeddings: [B, L, D] or [B, Neg, L, D]
        # seq_len: [B] or [B,Neg], weight: [B,L] or [B,Neg,L]
        B = batch_seq_embeddings.size(0)
        _need_reshape = False
        if batch_seq_embeddings.dim() == 4:
            _need_reshape = True
            batch_seq_embeddings = batch_seq_embeddings.view(
                -1, *batch_seq_embeddings.shape[2:])
            seq_len = seq_len.view(-1)
            if weight is not None:
                weight = weight.view(-1, weight.size(-1))

        N, L, D = batch_seq_embeddings.shape

        if weight is not None:
            batch_seq_embeddings = weight.unsqueeze(-1) * batch_seq_embeddings

        if self.pooling_type == 'mask':
            # Data type of mask_token should be bool and 
            # the shape of mask_token should be [B, L]
            assert mask_token != None, "mask_token can be None when pooling_type is 'mask'."
            result = batch_seq_embeddings[mask_token]
        elif self.pooling_type in ['origin', 'concat', 'mean', 'sum', 'max']:
            mask = torch.arange(L).unsqueeze(0).unsqueeze(2).to(batch_seq_embeddings.device)
            mask = mask.expand(N, -1,  D)
            seq_len = seq_len.unsqueeze(1).unsqueeze(2)
            seq_len_ = seq_len.expand(-1, mask.size(1), -1)
            mask = mask >= seq_len_
            batch_seq_embeddings = batch_seq_embeddings.masked_fill(mask, 0.0)

            if self.pooling_type == 'origin':
                return batch_seq_embeddings
            elif self.pooling_type in ['concat', 'max']:
                if not self.keepdim:
                    if self.pooling_type == 'concat':
                        result = batch_seq_embeddings.reshape(N, -1)
                    else:
                        result = batch_seq_embeddings.max(dim=1)
                else:
                    if self.pooling_type == 'concat':
                        result = batch_seq_embeddings.reshape(N, -1).unsqueeze(1)
                    else:
                        result = batch_seq_embeddings.max(dim=1).unsqueeze(1)
            elif self.pooling_type in ['mean', 'sum']:
                batch_seq_embeddings_sum = batch_seq_embeddings.sum(dim=1, keepdim=self.keepdim)
                if self.pooling_type == 'sum':
                    result = batch_seq_embeddings_sum
                else:
                    result = batch_seq_embeddings_sum / (seq_len + torch.finfo(torch.float32).eps if self.keepdim else seq_len.squeeze(2))

        elif self.pooling_type == 'last':
            gather_index = (seq_len-1).view(-1, 1, 1).expand(-1, -1, D)  # B x 1 x D
            output = batch_seq_embeddings.gather(
                dim=1, index=gather_index).squeeze(1)  # B x D
            result = output if not self.keepdim else output.unsqueeze(1)

        if _need_reshape:
            return result.reshape(B, N//B, *result.shape[1:])
        else:
            return result

    def extra_repr(self):
        return f"pooling_type={self.pooling_type}, keepdim={self.keepdim}"


class AttentionLayer(torch.nn.Module):
    def __init__(
            self,
            q_dim,
            k_dim=None,
            v_dim=None,
            mlp_layers=[],
            activation='sigmoid',
            n_head=1,
            dropout=0.0,
            bias=True,
            attention_type='feedforward',
            batch_first=True) -> None:

        super().__init__()
        assert attention_type in set(['feedforward', 'multi-head', 'scaled-dot-product']),\
            f"expecting attention_type to be one of [feedforeard, multi-head, scaled-dot-product]"
        self.attention_type = attention_type
        if k_dim is None:
            k_dim = q_dim
        if v_dim is None:
            v_dim = k_dim

        if attention_type == 'feedforward':
            mlp_layers = [q_dim+k_dim] + mlp_layers + [1]
            self.mlp = torch.nn.Sequential(
                MLPModule(
                    mlp_layers=mlp_layers[:-1],
                    activation_func=activation,
                    bias=bias
                ),
                torch.nn.Linear(mlp_layers[-2], mlp_layers[-1])
            )
            pass
        elif attention_type == 'multi-head':
            self.attn_layer = torch.nn.MultiheadAttention(
                embed_dim=q_dim, num_heads=n_head, dropout=dropout, bias=bias, kdim=k_dim, vdim=v_dim,
                batch_first=batch_first)
            pass
        elif attention_type == 'scaled-dot-product':
            assert q_dim == k_dim, 'expecting q_dim is equal to k_dim in scaled-dot-product attention'
            pass

    def forward(self, query, key, value, key_padding_mask=None,
                need_weight=False, attn_mask=None, softmax=False,
                average_attn_weights=True):
        # query: BxLxD1; key: BxSxD2; value: BxSxD; key_padding_mask: BxS
        if self.attention_type in ['feedforward', 'scaled-dot-product']:
            if self.attention_type == 'feedforward':
                query = query.unsqueeze(2).expand(-1, -1, key.size(1), -1)
                key = key.unsqueeze(1).expand(-1, query.size(1), -1, -1)
                attn_output_weight = self.mlp(
                    torch.cat((query, key), dim=-1)).squeeze(-1)   # BxLxS
            else:
                attn_output_weight = query @ key.transpose(1, 2)

            attn_output_weight = attn_output_weight / (query.size(-1) ** 0.5)

            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.unsqueeze(
                    1).expand(-1, query.size(1), -1)
                filled_value = -torch.inf if softmax else 0.0
                attn_output_weight = attn_output_weight.masked_fill(key_padding_mask, filled_value)

            if softmax:
                attn_output_weight = torch.softmax(attn_output_weight, dim=-1)

            attn_output = attn_output_weight @ value    # BxLxD

        elif self.attention_type == 'multi-head':
            attn_output, attn_output_weight = \
                self.attn_layer(query, key, value, key_padding_mask,
                                True, attn_mask, average_attn_weights)

        if need_weight:
            return attn_output, attn_output_weight
        else:
            return attn_output


class Dice(torch.nn.Module):
    __constants__ = ['num_parameters']
    num_features: int

    def __init__(self, num_parameters, init: float = 0.25, epsilon: float = 1e-08):
        super().__init__()
        self.num_parameters = num_parameters
        self.weight = torch.nn.parameter.Parameter(
            torch.empty(num_parameters).fill_(init))
        self.epsilon = epsilon

    def forward(self, x):
        mean_x = torch.mean(x, dim=-1, keepdim=True)
        var_x = torch.var(x, dim=-1, keepdim=True)
        x_std = (x - mean_x) / (torch.sqrt(var_x + self.epsilon))
        p_x = torch.sigmoid(x_std)
        f_x = p_x * x + (1-p_x) * x * self.weight.expand_as(x)
        return f_x

    def extra_repr(self) -> str:
        return 'num_parameters={}'.format(self.num_parameters)


class LambdaLayer(torch.nn.Module):
    def __init__(self, lambda_func) -> None:
        super().__init__()
        self.lambda_func = lambda_func

    def forward(self, *args):
        # attention: all data input into LambdaLayer will be tuple
        # even if there is only one input, the args will be the tuple of length 1
        if len(args) == 1:  # only one input
            return self.lambda_func(args[0])
        else:
            return self.lambda_func(args)


class HStackLayer(torch.nn.Sequential):

    def forward(self, *input):
        output = []
        assert (len(input) == 1) or (len(input) == len(self.module_list))
        for i, module in enumerate(self):
            if len(input) == 1:
                output.append(module(input[0]))
            else:
                output.append(module(input[i]))
        return tuple(output)


class VStackLayer(torch.nn.Sequential):

    def forward(self, input):
        for module in self:
            if isinstance(input, Tuple):
                input = module(*input)
            else:
                input = module(input)
        return input


class KMaxPoolingLayer(torch.nn.Module):
    def __init__(self, k, dim):
        super().__init__()
        self.k = k
        self.dim = dim

    def forward(self, input):
        output = torch.topk(input, self.k, self.dim, sorted=True)[0]
        return output