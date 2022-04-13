from recstudio.model import basemodel, loss_func, scorer
from recstudio.data import dataset
from recstudio.ann import sampler
import torch

# TODO: problems when predict for large dataset
class DIN(basemodel.ItemTowerRecommender):
    def init_model(self, train_data):
        super().init_model(train_data)
        if self.config['attention_type'] == 'multiheadattention':
            self.n_head = self.config['n_head']
            self.dropout = self.config['dropout']
            self.attention = torch.nn.MultiheadAttention(self.embed_dim, self.n_head, self.dropout, batch_first=True)

        else:
            raise NotImplementedError('Not implement.')
        self.item_bias = torch.nn.Embedding(train_data.num_items, 1, padding_idx=0)
        self.norm = torch.nn.Sequential(
            torch.nn.Linear(self.embed_dim, self.embed_dim)
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(3*self.embed_dim, 80),
            torch.nn.Sigmoid(),
            torch.nn.Linear(80, 40),
            torch.nn.Sigmoid(),
            torch.nn.Linear(40, 1),
        )
        self.dropout = torch.nn.Dropout(p=self.dropout)

    def get_dataset_class(self):
        return dataset.SeqDataset

    def config_scorer(self):
        return scorer.InnerProductScorer()

    def build_sampler(self, train_data):
        return sampler.UniformSampler(train_data.num_items-1)

    def config_loss(self):
        return loss_func.SampledSoftmaxLoss()

    def score(self, seq, target_ids=None):
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
        item_hist, _ = self.attention(query=item_emb, key=seq_emb, value=seq_emb, key_padding_mask=seq_mask, need_weights=False)    # BxNxD
        item_hist = self.norm(item_hist)   # BxN
        din_item = torch.cat((item_hist, item_emb, item_hist*item_emb), dim=-1) # BxN+1x3D
        din_item = self.fc(din_item).squeeze(-1)  # BxN
        logits = (din_item + item_b).view_as(target_ids)  # BxN || B
        return logits

    def forward(self, batch_data, fullscore):
        '''
        compute score with a seqence and N target items
        uid: B, seq: BxT, seqlen: B, iid: BxN
        '''
        seq, pos_id = batch_data['in_item_id'], batch_data[self.fiid]
        log_pos_prob, neg_id, log_neg_prob = self.sampler(seq, self.neg_count, pos_id.view(-1,1))  # BxN
        p_n_id = torch.cat((pos_id.view(-1,1), neg_id), dim=-1)
        logits = self.score(seq, p_n_id)
        p_logits, n_logits = logits.split([1, self.neg_count], dim=-1)
        return p_logits, log_pos_prob, n_logits, log_neg_prob

    def construct_query(self, batch_data):
        return batch_data['in_item_id']

    def topk(self, query, k, user_h):
        more = user_h.size(1) if user_h is not None else 0
        all_scores = self.score(query)
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
        
