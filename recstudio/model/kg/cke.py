from typing import List
from recstudio.model import basemodel, loss_func, scorer, module
from recstudio.data import dataset
from recstudio.ann import sampler 
import torch
import torch.nn.functional as F
from recstudio.model.kg import KGLearning

class CKE_TransRTower(KGLearning.TransRTower):
    def _get_loss_func(self):
        return loss_func.BPRLoss()
        
r"""
CKE
#############
    Collaborative knowledge base embedding for recommender systems(KDD'16)
    Reference: 
        https://dl.acm.org/doi/10.1145/2939672.2939673
"""
class CKE(basemodel.BaseRetriever):
    r"""
    CKE jointly learns the latent representations in collaborative filtering as well as items' semantic representations from the knowledge base, 
    using three elaborate components which can extract items' semantic representations from structural content, textual content and visual content, respectively. 
    In this implementation, we only use TransR to extract items' structural representations. 
    """
    def __init__(self, config):
        super().__init__(config)
        self.kg_index = config['data']['kg_network_index']

    def _init_model(self, train_data):
        self.item_emb = torch.nn.Embedding(train_data.num_items, self.embed_dim, padding_idx=0)
        # kg tower
        self.TransRTower = CKE_TransRTower(self.config)
        self.TransRTower._init_model(train_data)
        super()._init_model(train_data)

    def _get_dataset_class():
        return dataset.TripletDataset

    def _set_data_field(self, data):
        self.fhid = data.get_network_field(self.kg_index, 0, 0)
        self.frid = data.get_network_field(self.kg_index, 0, 1)
        self.ftid = data.get_network_field(self.kg_index, 0, 2)
        data.use_field = set([data.fuid, data.fiid, data.frating, self.fhid, self.frid, self.ftid])

    def _get_train_loaders(self, train_data: dataset.TripletDataset) -> List:
        rec_loader = train_data.train_loader(batch_size = self.config['train']['batch_size'], shuffle = True, drop_last = False)
        kg_loader = train_data.network_feat[self.kg_index].loader(batch_size = self.config['train']['batch_size'], shuffle = True, drop_last = False)
        return [rec_loader, kg_loader]
    
    def current_epoch_trainloaders(self, nepoch) -> List:
        return self.trainloaders, True 

    def _get_sampler(self, train_data):
        return sampler.UniformSampler(train_data.num_items-1)

    def _get_score_func(self):
        return scorer.InnerProductScorer()

    def _get_loss_func(self):
        return loss_func.BPRLoss()

    def _get_query_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)

    def _get_item_encoder(self, train_data):
        return torch.nn.Sequential(
            module.HStackLayer(
                self.item_emb, 
                self.TransRTower.item_encoder
            ), 
            module.LambdaLayer(lambda embs: embs[0] + embs[1])
        )

    def training_step(self, batch):
        rec_score = self.forward(batch, False)['score']
        kg_score = self.TransRTower.forward(batch)['tail_score']
        loss = self.loss_fn(batch[self.frating], **rec_score) + self.TransRTower.loss_fn(None, **kg_score)
        return loss

    def _get_item_vector(self):
        return self.item_emb.weight[1 : ] + self.TransRTower.item_encoder.weight[1 : self.item_emb.num_embeddings]
