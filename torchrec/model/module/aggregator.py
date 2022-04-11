import torch
from torch import nn

class Aggregator(nn.Module):
    """
    The base class for aggregators in GNN. 

    Args:
        input_size(int): size of input representations
        output_size(int): size of output representations
        dropout(float): the probability to be set in dropout module.
        act(torch.nn.Module): the activation function.
    """
    def __init__(self, input_size, output_size, dropout=0.0, act=nn.ReLU()):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.mess_out = dropout
        self.act = act
        self.dropout = nn.Dropout(dropout)

class GCNAggregator(Aggregator):
    def __init__(self, input_size, output_size, dropout=0.0, act=nn.ReLU()):
        super().__init__(input_size, output_size, dropout, act)
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, embeddings, side_embeddings):
        """
        Applies nonlinear transformation on the summation of two representation vectors
        """
        embeddings = self.act(self.linear(embeddings + side_embeddings))
        embeddings = self.dropout(embeddings)
        return embeddings

class GraphSageAggregator(Aggregator):
    def __init__(self, input_size, output_size, dropout=0.0, act=nn.ReLU()):
        super().__init__(input_size, output_size, dropout, act)
        self.linear = nn.Linear(input_size * 2, output_size)

    def forward(self, embeddings, side_embeddings):
        """
        Concatenates the two representation vectors and the applies nonlinear transformation
        """
        embeddings = self.act(self.linear(torch.cat([embeddings, side_embeddings], dim=-1)))
        embeddings = self.dropout(embeddings)
        return embeddings

class NeighborAggregator(Aggregator):
    def __init__(self, input_size, output_size, dropout=0.0, act=nn.ReLU()):
        super().__init__(input_size, output_size, dropout, act)
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, embeddings, side_embeddings):
        """
        Applies nonlinear transformation on neighborhood representation.
        """
        embeddings = self.act(self.linear(side_embeddings))
        embeddings = self.dropout(embeddings)
        return embeddings

torch.nn.Sigmoid

class BiAggregator(Aggregator):
    def __init__(self, input_size, output_size, dropout=0.0, act=nn.ReLU()):
        super().__init__(input_size, output_size, dropout, act)
        self.linear_sum = nn.Linear(input_size, output_size)
        self.linear_product = nn.Linear(input_size, output_size)

    def forward(self, embeddings, side_embeddings):
        """
        Applies the following transformation on two representations.
        .. math::
            \text{output} = act(W_{1}(V + V_{side})+b) + act(W_{2}(V \odot V_{side})+b)
        """
        sum_embeddings = self.act(self.linear_sum(embeddings + side_embeddings))
        bi_embeddings = self.act(self.linear_product(embeddings * side_embeddings))
        embeddings = sum_embeddings + bi_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings 