from typing import Union, Optional
import torch
from torch import nn
import copy 
import torch.nn.functional as F


class Combiner(nn.Module):
    """
    The base class for combiner in GNN. 

    Args:
        input_size(int): size of input representations
        output_size(int): size of output representations
        dropout(float): the probability to be set in dropout module.
        act(torch.nn.Module): the activation function.
    """
    def __init__(self, input_size: Optional[float], output_size: Optional[float], dropout:Optional[float]=0.0, act=nn.ReLU()):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.mess_out = dropout
        self.act = act
        if dropout != None:
            self.dropout = nn.Dropout(dropout)


class GCNCombiner(Combiner):
    def __init__(self, input_size:int, output_size:int, dropout:float=0.0, act=nn.ReLU()):
        super().__init__(input_size, output_size, dropout, act)
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, embeddings, side_embeddings):
        """
        Applies nonlinear transformation on the summation of two representation vectors
        """
        embeddings = self.act(self.linear(embeddings + side_embeddings))
        embeddings = self.dropout(embeddings)
        return embeddings


class GraphSageCombiner(Combiner):
    def __init__(self, input_size:int, output_size:int, dropout:float=0.0, act=nn.ReLU()):
        super().__init__(input_size, output_size, dropout, act)
        self.linear = nn.Linear(input_size * 2, output_size)

    def forward(self, embeddings, side_embeddings):
        """
        Concatenates the two representation vectors and the applies nonlinear transformation
        """
        embeddings = self.act(self.linear(torch.cat([embeddings, side_embeddings], dim=-1)))
        embeddings = self.dropout(embeddings)
        return embeddings


class NeighborCombiner(Combiner):
    def __init__(self, input_size:int, output_size:int, dropout:float=0.0, act=nn.ReLU()):
        super().__init__(input_size, output_size, dropout, act)
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, embeddings, side_embeddings):
        """
        Applies nonlinear transformation on neighborhood representation.
        """
        embeddings = self.act(self.linear(side_embeddings))
        embeddings = self.dropout(embeddings)
        return embeddings


class BiCombiner(Combiner):
    def __init__(self, input_size:int, output_size:int, dropout:float=0.0, act=nn.ReLU()):
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


class LightGCNCombiner(Combiner):
    def __init__(self, input_size:int, output_size:int):
        super().__init__(input_size, output_size, None, None)
    
    def forward(self, embeddings, side_embeddings):
        return side_embeddings


class GraphItemEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.item_embeddings = None
    
    def forward(self, batch_data):
        return self.item_embeddings[batch_data]

class GraphUserEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.user_embeddings = None
    
    def forward(self, batch_data):
        return self.user_embeddings[batch_data]


class LightGCNNet(torch.nn.Module):
    """
    Parameters:
    combiners (torch.nn.ModuleList): the combiners used to combine central node representation and neighborhood message.
    normalize (int or None): how to normalize embeddings got in every layer. 
    If it is int, :math:`p` means performing :math:`L_p` normalization on embeddings. 
    If it is `None`, no normalization will be done on embddings. Default: `None`.
    mess_norm (str): how to normalize the adjacency matrix. 
    `left` means normalizing the adjacency matrix as 
    .. math::
        `norm_adj = D^{-1} A`
    where :math:`D` is the degree matrix of the adjacency matrix.
    `right` means normalizing the adjacency matrix as 
    .. math::
        `norm_adj = A D^{-1}`
    `both` means normalizing the adjacency matrix as
    .. math::
        `norm_adj = D^{-\frac{1}{2}}A D^{-\frac{1}{2}}`
    """
    

    def __init__(self, combiners:torch.nn.ModuleList, normalize:Optional[int]=None, mess_norm:str='both') -> None:
        super().__init__()
        try:
            import dgl
            import dgl.nn.pytorch.conv as dglnn
            import dgl.function as fn
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Package 'dgl' not found, please "
                "refer to https://www.dgl.ai/pages/start.html to install.")

        self.n_layers = len(combiners)
        self.normalize = normalize
        self.mess_norm = mess_norm
        self.combiners = combiners

    def _get_message_func(self):
        try:
            import dgl.function as fn
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Package 'dgl' not found, please "
                "refer to https://www.dgl.ai/pages/start.html to install.")

        return fn.copy_src('h', 'msg')
    
    def _get_reduce_func(self):
        try:
            import dgl.function as fn
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Package 'dgl' not found, please "
                "refer to https://www.dgl.ai/pages/start.html to install.")
                
        return fn.sum('msg', 'neigh')

    def conv_layer(self, layer_index: int, graph, feat: torch.tensor):
        with graph.local_scope():
            graph.ndata['h'] = (feat * self.l_norm) if self.mess_norm in ['left', 'both'] else feat
            graph.update_all(self._get_message_func(), self._get_reduce_func())
            output = (graph.ndata['neigh'] * self.r_norm) if self.mess_norm in ['right', 'both'] \
                else graph.ndata['neigh']            
        return output

    def forward(self, graph, feat:torch.Tensor):
        if self.mess_norm in ['left', 'both']:
            if self.mess_norm == 'both':
                self.l_norm = torch.pow(graph.out_degrees(), -0.5).unsqueeze(-1) # [num_nodes, 1]
            else:
                self.l_norm = torch.pow(graph.out_degrees(), -1.0).unsqueeze(-1) # [num_nodes, 1]
            self.l_norm[torch.isinf(self.l_norm)] = 0.
        if self.mess_norm in ['right', 'both']:
            if self.mess_norm == 'both':
                self.r_norm = torch.pow(graph.in_degrees(), -0.5).unsqueeze(-1) # [num_nodes, 1]
            else:
                self.r_norm = torch.pow(graph.in_degrees(), -1.0).unsqueeze(-1) # [num_nodes, 1]
            self.r_norm[torch.isinf(self.r_norm)] = 0.
        
        all_embeddings = [feat]
        for i in range(self.n_layers):
            neigh_feat = self.conv_layer(i, graph, feat)
            feat = self.combiners[i](feat, neigh_feat)
            if self.normalize != None:
                all_embeddings.append(F.normalize(feat, p=self.normalize))
            else:
                all_embeddings.append(feat)
        return all_embeddings


class LightGCNNet_dglnn(LightGCNNet):
    def __init__(self, combiners:torch.nn.ModuleList, normalize:int=None, mess_norm:str='both') -> None:
        super().__init__(combiners, normalize, mess_norm)
        try:
            import dgl
            import dgl.nn.pytorch.conv as dglnn
            import dgl.function as fn
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Package 'dgl' not found, please "
                "refer to https://www.dgl.ai/pages/start.html to install.")
        
        self.convs = torch.nn.ModuleList()
        for i in range(self.n_layers):
            self.convs.append(dglnn.GraphConv(self.combiners[i].input_size, self.combiners[i].output_size, \
                norm=mess_norm, weight=False, bias=False, allow_zero_in_degree=True))
        
    def conv_layer(self, layer_index: int, graph, feat: torch.tensor):
        return self.convs[layer_index](graph, feat)

    def forward(self, graph, feat: torch.Tensor):
        all_embeddings = [feat]
        for i in range(self.n_layers):
            neigh_feat = self.conv_layer(i, graph, feat)
            feat = self.combiners[i](feat, neigh_feat)
            if self.normalize != None:
                all_embeddings.append(F.normalize(feat, p=self.normalize))
            else:
                all_embeddings.append(feat)
        return all_embeddings


class EdgeDropout(torch.nn.Module):
    """
    Out-place operation. 
    Dropout some edges in the graph in sparse COO or dgl format. It is used in GNN-based models.
    Parameters:
        dropout_prob(float): probability of a node to be zeroed.
    """
    def __init__(self, dropout_prob) -> None:
        super().__init__()
        self.keep_prob = 1.0 - dropout_prob
        self.edge_dropout_dgl = None 
    
    def forward(self, X):
        """
        Returns:
            (torch.Tensor or dgl.DGLGraph): the graph after dropout in sparse COO or dgl.DGLGraph format.
        """
        try:
            import dgl
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Package 'dgl' not found, please "
                "refer to https://www.dgl.ai/pages/start.html to install.")

        if not self.training:
            return X
        if isinstance(X, torch.Tensor) and X.is_sparse and (not X.is_sparse_csr):
            X = X.coalesce()
            random_tensor = torch.rand(X._nnz(), device=X.device) + self.keep_prob
            random_tensor = torch.floor(random_tensor).type(torch.bool)
            indices = X.indices()[:, random_tensor]
            values = X.values()[random_tensor] * (1.0 / self.keep_prob)
            return torch.sparse_coo_tensor(indices, values, X.shape, dtype=X.dtype)
        elif isinstance(X, dgl.DGLGraph):
            if self.edge_dropout_dgl == None:
                self.edge_dropout_dgl = dgl.DropEdge(p=1.0 - self.keep_prob)
            new_X = copy.deepcopy(X)
            new_X = self.edge_dropout_dgl(new_X)
            return new_X