from recstudio.model import basemodel, loss_func, scorer, module
from recstudio.data import dataset
from recstudio.ann import sampler
import torch

# TODO: out-of-memory problem when predict for large dataset

r"""
DIN
######################

Paper Reference:
    Guorui Zhou, et al. "Deep Interest Network for Click-Through Rate Prediction" in KDD2018.
    https://dl.acm.org/doi/10.1145/3219819.3219823

"""
class DIN(basemodel.TwoTowerRecommender):
    r"""
        | Deep Interest Network (DIN) designs a local activation unit to adaptively learn the representation
          of user interests from historical behaviors with respect to a certain ad.

        | DIN calculate the relevance between the target item and items in the sequence by adapting an
          attention machnism. But the method could not be applied to recall on all items in prediction
          due to the huge time cost.
    """
    def init_model(self, train_data):
        r"""DIN contains attention module (default activate unit in the paper), item bias, norm and
         | full-connected layer(with Dice as activation function)."""
        super().init_model(train_data)
        self.dropout_rate = self.config['dropout']
        self.attention_type = self.config['attention_type']
        if  self.attention_type == 'multiheadattention':
            self.n_head = self.config['n_head']
            self.attention = torch.nn.MultiheadAttention(self.embed_dim, self.n_head, self.dropout_rate, batch_first=True)
            # self.attention = ActivationUnit(self.embed_dim, activation='sigmoid')
        else:
            self.attention = ActivationUnit(self.embed_dim, activation='sigmoid')
        self.item_bias = torch.nn.Embedding(train_data.num_items, 1, padding_idx=0)
        self.norm = torch.nn.Sequential(
            torch.nn.Linear(self.embed_dim, self.embed_dim)
        )
        self.fc = torch.nn.Sequential(module.MLPModule([4*self.embed_dim, 80, 2], activation_func='sigmoid'), torch.nn.Softmax(dim=-1))
        self.dropout = torch.nn.Dropout(p=self.dropout_rate)

    def build_user_encoder(self, train_data):
        r"""User encoder is an Embedding layer."""
        return torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)

    def get_dataset_class(self):
        r"""The dataset is SeqDataset."""
        return dataset.SeqDataset

    def config_scorer(self): # useless
        return scorer.InnerProductScorer()

    def build_sampler(self, train_data):
        r"""Uniform sampler is used to generate negative samples."""
        return sampler.UniformSampler(train_data.num_items-1)

    def config_loss(self):
        r"""BinaryCrossEntropy is used as the loss function."""
        return loss_func.BinaryCrossEntropyLoss()

    def score(self, uid, seq, target_ids=None):
        r"""The score function is used to score a user and the sequence to the target item.

        Args:
            uid: user id. shape: [B,]
            seq: user behavior sequence. shape: [B, L]
            target_ids(optional): the target item (candidate item in the paper). shape: [B,] or [BxN].
             | If ``None``,  all item will be regarded as target item, which is usually in evaluation to
            calculate scores on all items. (Default: ``None``)
        
        Returns:
            torch.Tensor: shape: [B,] or [BxN]. If ``target_ids`` is ``None``, the shape will be [B, num_items].
        """
        u_emb = self.user_encoder(uid).unsqueeze(1)
        if target_ids is None:
            target_ids = seq.new_zeros(seq.size(0), self.item_encoder.num_embeddings-1)
            item_emb = self.item_encoder.weight[1:].repeat(seq.size(0), 1, 1)
            item_b = self.item_bias.weight[1:].repeat(seq.size(0), 1, 1).squeeze(-1)
        else: 
            if target_ids.dim() == 1:
                target_ids_ = target_ids.view(-1,1)
            else:
                target_ids_ = target_ids
            item_emb = self.item_encoder(target_ids_)   # BxNxD
            item_b = self.item_bias(target_ids_).squeeze(-1) # BxN

        seq_emb = self.item_encoder(seq)    # BxLxD
        seq_emb = self.dropout(seq_emb)
        seq_mask = seq==0   # BxL
        if  self.attention_type == 'multiheadattention':
            item_hist, _ = self.attention(query=item_emb, key=seq_emb, value=seq_emb, key_padding_mask=seq_mask, need_weights=False)    # BxNxD
        else:
            item_hist = self.attention(queries=item_emb, keys=seq_emb, key_masks=seq_mask)    # BxNxD
        item_hist = self.norm(item_hist)   # BxN
        din_item = torch.cat((u_emb.repeat(1, item_emb.size(1), 1), item_hist, item_emb, item_hist*item_emb), dim=-1) # Bx N x3D
        # din_item = self.fc(din_item).squeeze(-1)  # BxN
        din_item = self.fc(din_item)[:, :, 0]  # BxN
        logits = (din_item + item_b).view_as(target_ids)  # BxN || B
        return logits

    def forward(self, batch_data, fullscore):
        ''' Compute score with a seqence and postive/negative items

        Returns:
            torch.Tensor: [B,], scores on positive items.
            torch.Tensor: [B,], sampling probablity for positive items.
            torch.Tensor: [B, neg], scores on negative items.
            torch.Tensor: [B, neg], sampling probablity for negative items.
        '''
        seq, pos_id = batch_data['in_item_id'], batch_data[self.fiid]
        log_pos_prob, neg_id, log_neg_prob = self.sampler(seq, self.neg_count, pos_id.view(-1,1))  # BxN
        p_n_id = torch.cat((pos_id.view(-1,1), neg_id), dim=-1)
        logits = self.score(batch_data[self.fuid], seq, p_n_id)
        p_logits, n_logits = logits.split([1, self.neg_count], dim=-1)
        return p_logits, log_pos_prob, n_logits, log_neg_prob

    def construct_query(self, batch_data):
        r"""Return user id and items in the sequence, which provide data for topk."""
        return batch_data[self.fuid], batch_data['in_item_id']

    def topk(self, query, k, user_h):
        r"""Get topk items with a query on all items.

        Args:
            query(tuple): (uid, seq), the user id and items sequence.
            k(int): top k.
            user_h(torch.Tensor): user history.
        
        Returns:
            torch.Tensor: [B, k], scores for the topk items.
            torch.Tensor: [B, k], item index fot the topk items.
        """
        more = user_h.size(1) if user_h is not None else 0
        all_scores = self.score(*query)
        score, topk_items = torch.topk(all_scores, k + more)
        topk_items += 1
        if user_h is not None:
            existing, _ = user_h.sort()
            idx_ = torch.searchsorted(existing, topk_items)
            idx_[idx_ == existing.size(1)] = existing.size(1) - 1
            score[torch.gather(existing, 1, idx_) == topk_items] = -float('inf')
            score1, idx = score.topk(k)
            return score1, torch.gather(topk_items, 1, idx)
        else:
            return score, topk_items


class Dice(torch.nn.Module):
    def __init__(self, n_dim):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.zeros(n_dim))
        self.beta = torch.nn.Parameter(torch.zeros(n_dim))
        self.norm = torch.nn.BatchNorm1d(n_dim, affine=False)

    def forward(self, x):
        x_ = x.view(-1,x.size(-1))
        x_normed = self.norm(x_)
        x_p = torch.sigmoid(self.beta * x_normed)
        print(x_p.shape)
        output = (1.0-x_p) * x_ * self.alpha + x_p * x_
        output = output.view(*x.shape)
        return output

class ActivationUnit(torch.nn.Module):
    r""" Attention machnism used in DIN."""
    def __init__(self, n_dim, activation):
        super().__init__()
        self.mlp = module.MLPModule([4*n_dim, 80, 40, 1], activation_func=Dice(n_dim) if activation=='dice' else "sigmoid")

    def forward(self, queries, keys, key_masks):
        # queries: BxD | BxNxD, keys: BxLxD, key_masks: BxL
        if len(queries.shape) == 2:
            _queries = queries.view(queries.size(0), 1, -1)  # Bx1xD
        else:
            _queries = queries
        B, N, D = _queries.shape
        L = keys.size(1)
        queries_ = torch.tile(_queries, (1,1,L)).view(B, N, L, D)
        keys_ = torch.tile(keys, (1,N,1)).view(B, N, L, D)
        din_all = torch.cat((queries_, keys_, queries_-keys_, queries_*keys_), dim=3)
        outputs = self.mlp(din_all).squeeze(3)    # BxNxL
        # Mask
        mask = torch.tile(key_masks, (1, N)).view(B, N, L)  # BxNxL
        outputs = outputs.masked_fill(mask, -torch.inf)
        # Scale
        outputs = outputs / (D**0.5)
        # Activation
        outputs = torch.softmax(outputs, dim=2)    # BxNxL
        # Weighted sum
        outputs = torch.matmul(outputs, keys)   # BxNxD
        return outputs.view_as(queries)
