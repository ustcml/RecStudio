import torch
from .debiasedretriever import DebiasedRetriever
from ..loss_func import PointwiseLoss
from recstudio.model import basemodel
from recstudio.model.module.propensity import Popularity

r"""
IPS
######

Paper Reference:
    Recommendations as treatments: debiasing learning and evaluation (ICML'16)
    https://dl.acm.org/doi/10.5555/3045390.3045567
"""

class IPS(DebiasedRetriever):

    def add_model_specific_args(parent_parser):
        parent_parser = basemodel.Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('IPS')
        parent_parser.add_argument("--eta", type=float, default=1, help='adjust propensities')
        return parent_parser
    
    def _init_model(self, train_data):
        super()._init_model(train_data)
        for name, backbone in self.backbone.items():
            if not isinstance(backbone.loss_fn, PointwiseLoss):
                raise ValueError('IPS asks for PointwiseLoss '
                                 f'rather than {backbone.loss_fn}.')
            

    def _get_propensity(self, train_data):
        propensity = Popularity(self.config['train']['eta'])
        propensity.fit(train_data)
        return propensity
    
    def _get_final_loss(self, loss : dict, output : dict, batch : dict):
        unreweighted_loss = loss['IPS']
        weight = 1 / self.propensity(batch[self.fiid])
        reweighted_loss = torch.mean(weight * unreweighted_loss)
        return reweighted_loss