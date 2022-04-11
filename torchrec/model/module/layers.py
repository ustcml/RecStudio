import torch
from torch.nn.parameter import Parameter

def get_act(activation:str):
      if activation == None or isinstance(activation, torch.nn.Module):
          return activation
      elif type(activation) == str:
          if activation.lower()=='relu':
              return torch.nn.ReLU()
          elif activation.lower()=='sigmoid':
              return torch.nn.Sigmoid()
          elif activation.lower()=='tanh':
              return torch.nn.Tanh()
          elif activation.lower()=='leakyrelu':
              return torch.nn.LeakyReLU()
          else:
              raise ValueError(f'activation function type "{activation}"  is not supported, check spelling or pass in a instance of torch.nn.Module.')
      else:
          raise ValueError('"activation_func" must be a str or a instance of torch.nn.Module. ')

          
class CrossCompressUnit(nn.Module):
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
        weight_vv(nn.Linear): transformation weights. 
        weight_ev(nn.Linear): transformation weights.
        weight_ve(nn.Linear): transformation weights.
        weight_ee(nn.Linear): transformation weights.
        bias_v(Parameter): bias on v.
        bias_e(Parameter): bias on e.
    
    Returns:
        v_output(torch.Tensor): the first embeddings after feature interaction.
        e_output(torch.Tensor): the second embeddings after feature interaction.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.weight_vv = nn.Linear(self.embed_dim, 1, False)
        self.weight_ev = nn.Linear(self.embed_dim, 1, False)
        self.weight_ve = nn.Linear(self.embed_dim, 1, False)
        self.weight_ee = nn.Linear(self.embed_dim, 1, False)
        self.bias_v = Parameter(data=torch.zeros(self.embed_dim), requires_grad=True)
        self.bias_e = Parameter(data=torch.zeros(self.embed_dim), requires_grad=True)
    
    def forward(self, inputs):
        v_input = inputs[0].unsqueeze(-1) #[batch_size, dim, 1] or [batch_size, neg, dim, 1]
        e_input = inputs[1].unsqueeze(-2) #[batch_size, 1, dim] or [batch_size, neg, 1, dim]

        c_matrix = torch.matmul(v_input, e_input) # [batch_size, dim, dim] or [batch_size, neg, dim, dim]
        c_matrix_transpose = c_matrix.transpose(-1, -2)

        v_output = (self.weight_vv(c_matrix) + self.weight_ev(c_matrix_transpose)).squeeze(-1) # [batch_size, dim, 1] -> [batch_size, dim] 
        v_output = v_output + self.bias_v
        e_output = (self.weight_ve(c_matrix) + self.weight_ee(c_matrix_transpose)).squeeze(-1)
        e_output = e_output + self.bias_e 

        return (v_output, e_output)

      
class FeatInterLayers(nn.Module):
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
        self.model = nn.Sequential()
        for id in range(num_units):
            self.model.add_module(f'unit[{id}]', unit(dim))
    
    def forward(self, v_input, e_input):
        return self.model((v_input, e_input))

      
class MLPModule(nn.Module):
    """
    MLPModule 
    Gets a MLP easily and quickly.

    Args:
        mlp_layers(list): the dimensions of every layer in the MLP. 
        activation_func(torch.nn.Module,str,None): the activation function in each layer.
        dropout(float): the probability to be set in dropout module. Default: ``0.0``.
    
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
    def __init__(self, mlp_layers, activation_func='ReLU', dropout=0.0):
        super().__init__()
        activation_func = get_act(activation_func)
        self.mlp_layers = mlp_layers
        self.model = []
        for idx, layer in enumerate((zip(self.mlp_layers[ : -1], self.mlp_layers[1 : ]))):
            self.model.append(nn.Dropout(dropout))
            self.model.append(nn.Linear(*layer))
            if activation_func is not None:
                self.model.append(activation_func)
        self.model = nn.Sequential(*self.model)
    
    def add_modules(self, *args):
        """
        Adds modules into the MLP model after obtaining the instance.

        Args:
            args(variadic argument): the modules to be added into MLP model. 
        """
        for block in args:
            assert isinstance(block, nn.Module)
        
        for block in args:
            self.model.add_module(str(len(self.model._modules)), block)

    def forward(self, input):
        return self.model(input)


class GMFScorer(torch.nn.Module):
    def __init__(self, emb_dim, bias=False, activation='relu') -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.W = torch.nn.Linear(self.emb_dim, 1, bias=False)
        self.activation = get_act(activation)

    def forward(self, query, key):
        assert query.dim() <= key.dim(), 'query dim must be smaller than or euqal to key dim'
        if query.dim() < key.dim():
            query = query.unsqueeze(1)
        else:
            if query.size(0) != key.size(0):
                query = query.unsqueeze(1).repeat(1, key.size(0), 1)
                key = key.unsqueeze(0).repeat(query.size(0), 1, 1)
        h = query * key
        return self.activation(self.W(h)).squeeze(-1)


class FusionMFMLPScorer(torch.nn.Module):
    def __init__(self, emb_dim, hidden_size, mlp, bias=False, activation='relu') -> None:
        super().__init__()
        self.emb_dim=emb_dim
        self.hidden_size = hidden_size
        self.bias = bias
        self.activation = activation
        self.W = torch.nn.Linear(self.emb_dim+self.hidden_size, 1, bias=False)
        self.activation = get_act(activation)
        self.mlp = mlp

    def forward(self, query, key):
        assert query.dim() <= key.dim(), 'query dim must be smaller than or euqal to key dim'
        if query.dim() < key.dim():
            query = query.unsqueeze(1).repeat(1, key.shape[1], 1)
        else:
            if query.size(0) != key.size(0):
                query = query.unsqueeze(1).repeat(1, key.size(0), 1)
                key = key.unsqueeze(0).repeat(query.size(0), 1, 1)
        h_mf = query * key 
        h_mlp = self.mlp(torch.cat([query, key], dim=-1))
        h = self.W(torch.cat([h_mf, h_mlp], dim=-1))
        return self.activation(h).squeeze(-1)
