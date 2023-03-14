import torch
import torch.nn.functional as F
from recstudio.ann import sampler
from recstudio.model import basemodel, loss_func, scorer

class TransModel(basemodel.Recommender):
    """
    A base class for several commonly used translational distance KG embedding models in Knowledge based recommenders. (TransE, TransH, TransR, TransD)

    Parameters:
        kg_index(int): the index of the knowledge graph network in the dataset configuration file.
        neg_count(int): the number of negative samples. 
        margin(int): the margin parameter used in HingeLoss.
        normalize(bool): whether to normalize the entity and relation embedding.
        corrupt_head(bool): whether to corrput head entity to construct a broken triplet.
    """
    def __init__(self, config):
        super().__init__(config)
        self.kg_index = config['data']['kg_network_index']
        self.margin = config['model'].setdefault('margin', 2)
        self.normalize = config['model'].setdefault('normalize', False)
        self.corrupt_head = False
        self.score_func = self._get_score_func()
        
    def _init_model(self, train_data):
        self.fhid = train_data.get_network_field(self.kg_index, 0, 0)
        self.frid = train_data.get_network_field(self.kg_index, 0, 1)
        self.ftid = train_data.get_network_field(self.kg_index, 0, 2) 
        self.item_encoder = self._get_item_encoder(train_data)
        self.sampler = self._get_sampler(train_data)
        super()._init_model(train_data)
        
    def _get_dataset_class(self):
        return None
    
    def _set_data_field(self, data):
        data.use_field = set([data.fuid, data.fiid, data.frating, self.fhid, self.frid, self.ftid])

    def _get_item_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_values(self.fhid), self.embed_dim, padding_idx=0)

    def _get_sampler(self, train_data):
        return sampler.UniformSampler(train_data.num_values(self.fhid) - 1, scorer_fn=self.score_func)

    def _get_score_func(self):
        return scorer.NormScorer(2)
    
    def _get_loss_func(self):
        return loss_func.HingeLoss(self.config['model'].setdefault('margin', 2))

    def projection_trans(self, entities, relations):
        """
        TransE and its extensions implement a projection on head and tail representations. 
        Different ``projection_trans`` should be implemented in different subclasses, and the rest of the forward process will be done in ``forward`` in the base class. 

        Args:
            entities(torch.Tensor): the id of entities to be projected.
            relations(torch.Tensor): the relation with the entities in their triplets.
        """
        pass

    def forward(self, batch_data):
        output = {}
        r_e = self.rel_emb(batch_data[self.frid])
        h_e = self.projection_trans(batch_data[self.fhid], batch_data[self.frid])
        t_e = self.projection_trans(batch_data[self.ftid], batch_data[self.frid])
        if self.normalize:
            h_e = F.normalize(h_e, p=2, dim=1)
            r_e = F.normalize(r_e, p=2, dim=1)
            t_e = F.normalize(t_e, p=2, dim=1)
        pos_score = self.score_func(h_e + r_e, t_e)

        log_pos_tail_prob, neg_t_idx, log_neg_tail_prob = self.sampler(h_e + r_e, self.neg_count, batch_data[self.ftid])
        neg_t_e = self.projection_trans(neg_t_idx, batch_data[self.frid])
        if self.normalize:
            neg_t_e = F.normalize(neg_t_e, p=2, dim=1)
        neg_tail_score = self.score_func(h_e + r_e, neg_t_e)
        tail_score = {'pos_score' : pos_score, 'log_pos_prob' : log_pos_tail_prob, 
                      'neg_score' : neg_tail_score, 'log_neg_prob' : log_neg_tail_prob}
        if self.corrupt_head == True:
            log_pos_head_prob, neg_h_idx, log_neg_head_prob = self.sampler(t_e, self.neg_count, batch_data[self.fhid])
            neg_h_e = self.projection_trans(neg_h_idx, batch_data[self.frid])
            if self.normalize:
                neg_h_e = F.normalize(neg_h_e, p=2, dim=1)
            neg_head_score = self.score_func(t_e, neg_h_e + r_e.unsqueeze_(-2))
            head_score = {'pos_score' : pos_score, 'log_pos_prob' : log_pos_head_prob, 
                          'neg_score' : neg_head_score, 'log_neg_prob' : log_neg_head_prob}
            output['tail_score'] = tail_score
            output['head_score'] = head_score
            return output
        else:
            output['tail_score'] = tail_score
            return output

    def build_index(self):
        pass

    def training_step(self, batch):
        pass 

    def validation_step(self, batch):
        pass

    def test_step(self, batch):
        pass


class TransETower(TransModel):
    def __init__(self, config):
        self.p = config['model'].setdefault('p', 2)
        super().__init__(config)

    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.rel_emb = torch.nn.Embedding(train_data.num_values(self.frid), self.embed_dim, padding_idx=0)

    def _get_score_func(self):
        return scorer.NormScorer(self.p)
    
    def projection_trans(self, entities, relations):
        return self.item_encoder(entities)


class TransHTower(TransModel):
        
    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.norm_emb = torch.nn.Embedding(train_data.num_values(self.frid), self.embed_dim, padding_idx=0)
        self.rel_emb = torch.nn.Embedding(train_data.num_values(self.frid), self.embed_dim, padding_idx=0)

    def _get_score_func(self):
        return scorer.EuclideanScorer()

    def orthogonalLoss(self, rel_embeddings, norm_embeddings):
        return torch.sum(torch.sum(norm_embeddings * rel_embeddings, dim=1) ** 2 / torch.sum(rel_embeddings ** 2, dim=1))

    def projection_trans(self, entities, relations):
        entity_e = self.item_encoder(entities)
        norm_e = self.norm_emb(relations)
        if entity_e.dim() > norm_e.dim():
            norm_e = norm_e.unsqueeze(-2)
        return entity_e - torch.sum(entity_e * norm_e, dim=-1, keepdim=True) * norm_e

    def forward(self, batch_data):
        output = super().forward(batch_data)
        norm_embeddings = self.norm_emb(batch_data[self.frid])
        rel_embeddings = self.rel_emb(batch_data[self.frid])
        orthogonal_loss = self.orthogonalLoss(rel_embeddings, norm_embeddings)
        output['orthogonal_loss'] = orthogonal_loss
        return output
    

class TransRTower(TransModel):
    def __init__(self, config):
        super().__init__(config)
        self.pro_embed_dim = self.config['model']['pro_embed_dim']
        
    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.pro_matrix_emb = torch.nn.Embedding(train_data.num_values(self.frid), self.embed_dim * self.pro_embed_dim, padding_idx=0)
        self.rel_emb = torch.nn.Embedding(train_data.num_values(self.frid), self.pro_embed_dim, padding_idx=0)

    def _get_score_func(self):
        return scorer.EuclideanScorer()

    def projection_trans(self, entities, relations):
        entity_e = self.item_encoder(entities) # [batch_size, 1, dim] or [batch_size, neg, dim]
        if entities.dim() == 1:
            entity_e = entity_e.unsqueeze(-2)
        pro_e = self.pro_matrix_emb(relations).reshape(-1, self.embed_dim, self.pro_embed_dim) # [batch_size, dim, pro_dim]
        entity_e = torch.bmm(entity_e, pro_e) # [batch_size, 1, pro_dim] or [batch_size, neg, pro_dim]
        return entity_e.squeeze(-2) if entities.dim() == 1 else entity_e # [batch_size, pro_dim] or [batch_size, neg, pro_dim]

class TransDTower(TransModel):
    def __init__(self, config):
        super().__init__(config)
        self.pro_embed_dim = self.config['model']['pro_embed_dim']

    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.ent_pro_emb = torch.nn.Embedding(train_data.num_values(self.fhid), self.embed_dim, padding_idx=0)
        self.rel_emb = torch.nn.Embedding(train_data.num_values(self.frid), self.pro_embed_dim, padding_idx=0)
        self.rel_pro_emb = torch.nn.Embedding(train_data.num_values(self.frid), self.pro_embed_dim, padding_idx=0)

    def _get_score_func(self):
        return scorer.EuclideanScorer()

    def projection_trans(self, entities, relations):
        if entities.dim() > relations.dim():
            relations = relations.unsqueeze(-1)
        ent_pro_e = self.ent_pro_emb(entities) # [batch_size, emb_dim] or [batch_size, neg, emb_dim]
        rel_pro_e = self.rel_pro_emb(relations) # [batch_size, pro_dim] or [batch_size, 1, pro_dim]
        # [batch_size, pro_dim, emb_dim] or [batch_size, neg, pro_dim, emb_dim]
        pro_matrix = torch.matmul(rel_pro_e.unsqueeze(-1), ent_pro_e.unsqueeze(-2)) 
        pro_matrix = pro_matrix + torch.eye(self.pro_embed_dim, self.embed_dim).to(self.device) 
        ent_e = self.item_encoder(entities).unsqueeze(-1) # [batch_size, emb_dim, 1] or [batch_size, neg, emb_dim, 1]
        return torch.matmul(pro_matrix, ent_e).squeeze(-1) # [batch_size, pro_dim] or [batch_size, neg, pro_dim]



    

    
        


    