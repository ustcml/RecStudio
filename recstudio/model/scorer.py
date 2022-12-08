import torch
from recstudio.model import module


class InnerProductScorer(torch.nn.Module):
    def forward(self, query, items):
        # query, item : ([B, D], [B, D]), ([B, D], [B, neg, D]), ([B, D], [N, D]),
        # ([B, L, D], [B, L, D]), ([B, L, D], [B, L, neg, D])
        if query.size(0) == items.size(0):
            if query.dim() < items.dim(): #([B,D], [B,N,D])
                output = torch.matmul(items, query.view(*query.shape, 1))
                output = output.view(output.shape[:-1])
            else:   #([B,D], [B,D]), ([B,L,D], [B,L,D])
                output = torch.sum(query * items, dim=-1)
        else:   # ([B,D], [N,D])
            output = torch.matmul(query, items.T)
        return output

class CosineScorer(InnerProductScorer):
    def forward(self, query, items):
        output = super().forward(query, items)
        output /= torch.norm(items, dim=-1)
        output /= torch.norm(query, dim=-1,
            keepdim=(query.dim()!=items.dim() or query.size(0)!=items.size(0)))
        return output


class EuclideanScorer(InnerProductScorer):
    def forward(self, query, items):
        output = -2 * super().forward(query, items)
        output += torch.sum(torch.square(items), dim=-1)
        output += torch.sum(torch.square(query), dim=-1,
            keepdim=(query.dim()!=items.dim() or query.size(0)!=items.size(0)))
        return -output


class MLPScorer(InnerProductScorer):
    def __init__(self, transform):
        super().__init__()
        self.trans = transform

    def forward(self, query:torch.Tensor, items:torch.Tensor):
        # query, item : ([B, D], [B, D]), ([B, D], [B, neg, D]), ([B, D], [N, D]),
        # ([B, L, D], [B, L, D]), ([B, L, D], [B, L, neg, D])
        if query.size(0) == items.size(0):
            if query.dim() < items.dim():
                # [B, L, D] -> [B, L, neg, D]
                input = torch.cat(
                    (query.unsqueeze(-2).expand_as(items), items), dim=-1)
            else:
                input = torch.cat((query, items), dim=-1)
        else:
            query = query.unsqueeze(1).repeat(1, items.size(0), 1)
            items = items.expand(query.size(0), -1, -1)
            input = torch.cat((query, items), dim=-1)
        return self.trans(input).squeeze(-1)


class NormScorer(InnerProductScorer):
    def __init__(self, p=2):
        super().__init__()
        self.p = p

    def forward(self, query, items):
        # query, item : ([B, D], [B, D]), ([B, D], [B, neg, D]), ([B, D], [N, D]),
        # ([B, L, D], [B, L, D]), ([B, L, D], [B, L, neg, D])
        if query.dim() < items.dim() or query.size(0) != items.size(0):
            query.unsqueeze_(-2)
        output = torch.norm(query - items, p=self.p, dim=-1)
        return -output


class GMFScorer(InnerProductScorer):
    def __init__(self, emb_dim, bias=False, activation='relu') -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.W = torch.nn.Linear(self.emb_dim, 1, bias=bias)
        self.activation = module.get_act(activation)

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


class FusionMFMLPScorer(InnerProductScorer):
    def __init__(self, emb_dim, hidden_size, mlp, bias=False, activation='relu') -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.bias = bias
        self.W = torch.nn.Linear(self.emb_dim+self.hidden_size, 1, bias=False)
        self.activation = module.get_act(activation)
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
        h_mlp = (self.mlp(torch.cat([query, key], dim=-1)))
        h = self.activation(self.W(torch.cat([h_mf, h_mlp], dim=-1)).squeeze(-1))
        return h
