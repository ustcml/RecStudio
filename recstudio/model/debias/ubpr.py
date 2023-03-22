import torch
from .debiasedretriever import DebiasedRetriever
from recstudio.data import UBPRDataset
from recstudio.model import basemodel
from recstudio.model.module.propensity import Popularity

r"""
UBPR
######

Paper Reference:
    Unbiased Pairwise Learning from Biased Implicit Feedback (ICTIR'20)
    https://doi.org/10.1145/3409256.3409812
"""

class UBPR(DebiasedRetriever):

    def add_model_specific_args(parent_parser):
        parent_parser = basemodel.Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('UBPR')
        parent_parser.add_argument("--eta", type=float, default=0.5, help='adjust propensities')
        return parent_parser
    
    def _get_dataset_class():
        return UBPRDataset

    def _get_propensity(self, train_data):
        propensity = Popularity(self.config['train']['eta'], 
                                self.config['train']['truncation'],
                                self.config['train']['eps'])
        propensity.fit(train_data)
        return propensity
    
    def forward(self, batch):
        query = self.query_encoder(self._get_query_feat(batch))
        query = self.query_encoder.split(query)
        neg_item_idx = batch['sampled_items']
        output = {}
        for name, backbone in self.backbone.items():
            output[name] = backbone.forward(
                batch, 
                False,
                return_query=True, 
                return_item=True,
                return_neg_item=True,
                return_neg_id=True,
                query=query[name], 
                neg_item_idx=neg_item_idx)
        return output 
    
    def _get_final_loss(self, loss: dict, output: dict, batch: dict):
        weight_i = 1 / self.propensity(batch[self.fiid]).unsqueeze(-1)
        weight_j = 1 - batch['sampled_labels'] / self.propensity(batch['sampled_items'])
        loss_ = torch.mean(weight_i * weight_j * loss['UBPR'])
        return loss_