from typing import List, Tuple
from recstudio.model.kg.KGLearning import TransHTower
from recstudio.model import basemodel, loss_func, scorer, init, module
from recstudio.data import dataset
from recstudio.ann import sampler 
import torch
from torch import nn
import torch.nn.functional as F


class KTUPItemEncoder(nn.Module):
    def __init__(self, item_embeddings, ent_embeddings):
        super().__init__()
        self.item_embeddings = item_embeddings
        self.ent_embeddings = ent_embeddings 
    
    def forward(self, batch_data):
        return self.item_embeddings(batch_data) + self.ent_embeddings(batch_data)

"""
KTUP
    Unifying Knowledge Graph Learning and Recommendation: Towards a Better Understanding of User Preferences(WWW'19)
    Reference:
        https://doi.org/10.1145/3308558.3313705
"""

class KTUP(basemodel.BaseRetriever):
    """
    KTUP is a translation-based recommendation model. KTUP adopts the TransH to learn the entity and relation embedding. In the recommendation module, KTUP induces user preferences and 
    the preferences are enhanced by corresponding relations in KG, while the item embeddings are enhanced by related entities. Then, the user and item embeddings are projected to the preference
    and a translation-based scorer is used to calculate the score between a user-item pair.
    """
    def __init__(self, config):
        self.kg_index = config['data']['kg_network_index']
        self.train_rec_step = config['model']['train_rec_step']
        self.train_kg_step = config['model']['train_kg_step']
        self.L1_flag = config['model']['L1_flag']
        self.use_st_gumbel = config['model']['use_st_gumbel']
        super().__init__(config)

    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.fhid = train_data.get_network_field(self.kg_index, 0, 0)
        self.frid = train_data.get_network_field(self.kg_index, 0, 1)
        self.ftid = train_data.get_network_field(self.kg_index, 0, 2)
        self.num_items = train_data.num_items
        # rec
        self.user_emb = nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)
        self.pref_emb = nn.Embedding(train_data.num_values(self.frid), self.embed_dim, padding_idx=0)
        self.pref_norm_emb = nn.Embedding(train_data.num_values(self.frid), self.embed_dim, padding_idx=0)

        self.query_fields = self.query_fields | set(['user_embeddings', 'pref_embeddings'])


    def _init_parameter(self):
        self.apply(init.xavier_normal_initialization)
        self.item_emb.weight.data = F.normalize(self.item_emb.weight.data, p=2, dim=-1)
        self.pref_emb.weight.data = F.normalize(self.pref_emb.weight.data, p=2, dim=-1)
        self.pref_norm_emb.weight.data = F.normalize(self.pref_norm_emb.weight.data, p=2, dim=-1)
        self.TransHTower.item_encoder.weight.data = F.normalize(self.TransHTower.item_encoder.weight.data, p=2, dim=-1)
        self.TransHTower.rel_emb.weight.data = F.normalize(self.TransHTower.rel_emb.weight.data, p=2, dim=-1)
        self.TransHTower.norm_emb.weight.data = F.normalize(self.TransHTower.norm_emb.weight.data, p=2, dim=-1)

    def _get_dataset_class():
        return dataset.TripletDataset
    
    def _set_data_field(self, data):
        fhid = data.get_network_field(self.kg_index, 0, 0)
        frid = data.get_network_field(self.kg_index, 0, 1)
        ftid = data.get_network_field(self.kg_index, 0, 2)
        data.use_field = set([data.fuid, data.fiid, data.frating, fhid, frid, ftid])

    def _get_train_loaders(self, train_data, ddp=False) -> List:
        rec_loader = train_data.train_loader(batch_size = self.config['train']['batch_size'], shuffle = True, drop_last = False)
        kg_loader = train_data.network_feat[self.kg_index].loader(batch_size = self.config['train']['batch_size'], shuffle = True, drop_last = False)
        return [rec_loader, kg_loader]

    def current_epoch_trainloaders(self, nepoch) -> Tuple:
        if nepoch % (self.train_rec_step + self.train_kg_step) < self.train_rec_step:
            return [self.trainloaders[0]], False
        else:
            return [self.trainloaders[1]], False 

    def _get_score_func(self):
        if self.L1_flag:
            return scorer.NormScorer(p=1)
        else:
            return scorer.EuclideanScorer()
        
    def _get_loss_func(self):
        return loss_func.BPRLoss()
    
    def _get_query_encoder(self, train_data):
        return module.LambdaLayer(lambda batch_data: batch_data['user_embeddings'] + batch_data['pref_embeddings'])
    
    def _get_item_encoder(self, train_data):
        self.item_emb = nn.Embedding(train_data.num_items, self.embed_dim, padding_idx=0)
        # KG TransH
        self.TransHTower = TransHTower(self.config)
        self.TransHTower._init_model(train_data)
        return KTUPItemEncoder(self.item_emb, self.TransHTower.item_encoder)

    def getPreferences(self, user_embeddings, item_embeddings, use_st_gumbel=False):
        # [batch_size, dim] * [dim, num_relations] || [batch_size, num_items, dim] * [dim, num_relations]
        pre_probs = torch.matmul(
            user_embeddings + item_embeddings, (self.pref_emb.weight[1:] + self.TransHTower.rel_emb.weight[1:]).T
        ) / 2
        if use_st_gumbel:
            pre_probs = self._st_gumbel_softmax(pre_probs)
        
        r_e = torch.matmul(pre_probs, self.pref_emb.weight[1:] + self.TransHTower.rel_emb.weight[1:]) / 2
        norm = torch.matmul(pre_probs, self.pref_norm_emb.weight[1:] + self.TransHTower.norm_emb.weight[1:]) / 2
        
        return pre_probs, r_e, norm

    def _st_gumbel_softmax(self, logits, temperature=1.0):
        """
        Return the result of Straight-Through Gumbel-Softmax Estimation.
        It approximates the discrete sampling via Gumbel-Softmax trick
        and applies the biased ST estimator.
        In the forward propagation, it emits the discrete one-hot result,
        and in the backward propagation it approximates the categorical
        distribution via smooth Gumbel-Softmax distribution.
        Args:
            logits(torch.Tensor): A un-normalized probability values,
                which has the size (batch_size, num_classes)
            temperature(float): A temperature parameter. The higher
                the value is, the smoother the distribution is.
        Returns:
            y: The sampled output, which has the property explained above.
        """
        eps = 1e-20
        u = logits.new_empty(logits.shape).uniform_()
        gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
        # [batch_size, num_pref]
        y = logits + gumbel_noise
        y = F.softmax(y / temperature, dim=-1)
        # [batch_size]
        y_argmax = y.max(dim=-1)[1]
        y_hard = F.one_hot(y_argmax, y.size(-1))
        return (y_hard - y).detach() + y

    def _projection_transH(self, original, norm):
        # [batch_size, dim] * [batch_size, dim] 
        # || [batch_size, neg, dim] * [batch_size, neg, dim] -> [batch_size, neg, 1], [batch_size, neg, 1] * [batch_size, neg, dim]
        return original - torch.sum(original * norm, dim=-1, keepdim=True) * norm

    def forward(self, batch_data):
        pos_items = self._get_item_feat(batch_data)
        query = self.user_emb(batch_data[self.fuid]) # [batch_size, dim]
        log_pos_prob, neg_item_idx, log_neg_prob = self.sampler(query, self.neg_count, pos_items)
        
        pos_item_embeddings = self.item_encoder(pos_items)
        neg_item_embeddings = self.item_encoder(neg_item_idx)

        _, pos_r_e, pos_norm  = self.getPreferences(query, pos_item_embeddings, use_st_gumbel=self.use_st_gumbel)
        _, neg_r_e, neg_norm  = self.getPreferences(query.unsqueeze(-2), neg_item_embeddings, use_st_gumbel=self.use_st_gumbel)

        pos_proj_u_e = self._projection_transH(query, pos_norm)
        neg_proj_u_e = self._projection_transH(query.unsqueeze(-2), neg_norm)
        pos_proj_i_e = self._projection_transH(pos_item_embeddings, pos_norm)
        neg_proj_i_e = self._projection_transH(neg_item_embeddings, neg_norm)

        pos_score = self.score_func(pos_proj_u_e + pos_r_e, pos_proj_i_e)
        neg_score = self.score_func(neg_proj_u_e + neg_r_e, neg_proj_i_e)

        return {'pos_score' : pos_score, 'log_pos_prob' : log_pos_prob,
                'neg_score' : neg_score, 'log_neg_prob' : log_neg_prob}

    def training_step(self, batch):
        if self.fhid in batch:
            output = self.TransHTower.forward(batch)
            return self.TransHTower.loss_fn(None, **output['tail_score']) + output['orthogonal_loss']
        else:
            y_h = self.forward(batch)
            return self.loss_fn(None, **y_h) + self.TransHTower.orthogonalLoss(self.pref_emb.weight[1:], self.pref_norm_emb.weight[1:])
    
    def _update_item_vector(self):
        item_vector = self._get_item_vector()
        if not hasattr(self, "item_vector"):
            if item_vector == None:
                self.register_buffer('item_vector', item_vector)
            else:
                self.register_buffer('item_vector', item_vector.detach().clone() if isinstance(item_vector, torch.Tensor) else item_vector.copy())
        else:
            self.item_vector = item_vector

        if self.use_index:
            self.ann_index = self.build_ann_index()

    def _get_item_vector(self):
        return None 

    def _test_step(self, batch, metric, cutoffs):
        # [batch_size, 1] -> [batch_size, num_items]
        users = batch[self.fuid].unsqueeze(-1).expand(-1, self.num_items - 1)
        # [num_items]
        items = torch.arange(1, self.num_items).to(self.device)
        items = items.unsqueeze(0).expand(users.size()) # [num_items] -> [batch_size, num_items] 
        user_embeddings = self.user_emb(users) # [batch_size, num_items, dim]
        item_embeddings = self.item_encoder(items) 
        # [batch_size, num_items, dim]
        _, r_e, norm = self.getPreferences(user_embeddings, item_embeddings, use_st_gumbel=self.use_st_gumbel)
        # [batch_size, num_items, dim]
        user_embeddings = self._projection_transH(user_embeddings, norm)
        item_embeddings = self._projection_transH(item_embeddings, norm)

        self.item_vector = item_embeddings
        batch['user_embeddings'] = user_embeddings
        batch['pref_embeddings'] = r_e 
        
        return super()._test_step(batch, metric, cutoffs)