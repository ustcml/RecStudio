import copy
import torch
from .debiasedretriever import DebiasedRetriever
from ..loss_func import BCEWithLogitLoss
from recstudio.data.dataset import ConcatedLoaders
from recstudio.model import basemodel
from recstudio.model.debias.debiasedretriever import DebiasedQueryEncoder, DebiasedItemEncoder

r"""
CausE
#########

Paper Reference:
    Causal Embeddings for Recommendation (RecSys'18)
    https://dl.acm.org/doi/10.1145/3240323.3240360
"""

class CausE(DebiasedRetriever):               
        
    def add_model_specific_args(parent_parser):
        parent_parser = basemodel.Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('CausE')
        parent_parser.add_argument("--method", type=float, default='control', help='eval method')
        return parent_parser     

    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.backbone['control'].loss_fn = BCEWithLogitLoss() 
        self.backbone['treatment'].loss_fn = BCEWithLogitLoss()
        self.query_encoder.query_encoders['treatment'] = \
            self.backbone['treatment'].query_encoder = self.backbone['control'].query_encoder
        

    def _get_query_encoder(self, train_data):
        return DebiasedQueryEncoder(self.backbone, lambda d: d[self.config['eval']['method']])

    def _get_item_encoder(self, train_data):
        return DebiasedItemEncoder(self.backbone, lambda d: d[self.config['eval']['method']])
    
    def _get_item_vector(self):
        return self.backbone[self.config['eval']['method']]._get_item_vector()

    def _get_final_loss(self, loss : dict, output : dict, batch : dict):
        item_c, item_t = self.item_encoder(self._get_item_feat(batch)).chunk(2, 1)
        loss_dis = self.discrepancy(item_c, item_t)
        return loss['control'] + loss['treatment'] + self.config['train']['dis_penalty'] * loss_dis
    
    def _get_masked_batch(self, backbone_name, batch):
        masked_batch = copy.deepcopy(batch)
        control = (masked_batch['Loader'] == 0)
        if backbone_name == 'control':
            for k, v in masked_batch.items():
                masked_batch[k] = v[control]
        elif backbone_name == 'treatment':
            for k, v in masked_batch.items():
                masked_batch[k] = v[~control]
        return masked_batch
        
    def _get_train_loaders(self, train_data):
        c_loader = train_data.train_loader(
                    batch_size=self.config['backbone']['control']['train']['batch_size'],
                    shuffle=True, drop_last=False)
        t_loader = train_data.mcar_feat_for_train.loader(
                    batch_size = self.config['backbone']['treatment']['train']['batch_size'],
                    shuffle=True, drop_last=False)
        return ConcatedLoaders([c_loader, t_loader])