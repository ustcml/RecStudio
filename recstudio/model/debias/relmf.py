from .debiasedretriever import DebiasedRetriever
from ..loss_func import BCEWithLogitLoss
from recstudio.model import basemodel
from recstudio.model.module.propensity import Popularity

r"""
RelMF
######

Paper Reference:
    Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback (WSDM'20)
    https://doi.org/10.1145/3336191.3371783
"""

class RelMF(DebiasedRetriever):

    def add_model_specific_args(parent_parser):
        parent_parser = basemodel.Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('RelMF')
        parent_parser.add_argument("--eta", type=float, default=0.5, help='adjust propensities')
        return parent_parser
    
    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.backbone['RelMF'].loss_fn = None

    def _get_propensity(self, train_data):
        propensity = Popularity(self.config['train']['eta'], 
                                self.config['train']['truncation'],
                                self.config['train']['eps'])
        propensity.fit(train_data)
        return propensity

    def _get_loss_func(self):
        return BCEWithLogitLoss()
    
    def _get_final_loss(self, loss : dict, output : dict, batch : dict):
        pop = self.propensity(batch[self.fiid])
        score = output['RelMF']['score']
        label = batch[self.frating]
        score['label'] = label / pop + (1 - label) * (1 - label / pop)
        loss = self.loss_fn(**score)
        return loss