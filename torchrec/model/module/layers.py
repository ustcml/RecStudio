from turtle import forward
import torch


def get_act(name:str):
        if name=='relu':
            return torch.nn.ReLU()
        elif name=='sigmoid':
            return torch.nn.Sigmoid()
        elif name=='tanh':
            return torch.nn.Tanh()
        else:
            return torch.nn.ReLU()
        
class MLP(torch.nn.Module):
    def __init__(self, hidden_size:list, activation:str='relu', dropout:float=0.0, bn:bool=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.activation = get_act(activation) if isinstance(activation, str) else activation
        self.dropout_rate = dropout
        self.bn = bn
        dropout_layer = torch.nn.Dropout(p=self.dropout_rate)
        mlp_list = []
        for (input, output) in zip(self.hidden_size[:-1], self.hidden_size[1:]):
            mlp_list.append(dropout_layer)
            mlp_list.append(torch.nn.Linear(input, output))
            if self.bn:
                mlp_list.append(torch.nn.BatchNorm1d(num_features=output))
            mlp_list.append(self.activation)
        self.mlp = torch.nn.Sequential(*mlp_list)

    def forward(self, input):
        return self.mlp(input)


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
        self.activation = get_act(activation) if isinstance(activation, str) else activation
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