import torch.nn.functional as F
from .debiasedretriever import DebiasedRetriever
from recstudio.model import basemodel, scorer
from recstudio.model.module.propensity import Popularity

r"""
PDA
#########

Paper Reference:
    Causal Intervention for Leveraging Popularity Bias in Recommendation (SIGIR'21)
    https://doi.org/10.1145/3404835.3462875
"""
class PDAEvalScorer(scorer.InnerProductScorer):
    """
    For full score evaluation.
    """
    def __init__(self, eval_method, pop):
        super().__init__()
        self.eval_method = eval_method
        self.register_buffer('pop', pop)
    def forward(self, query, items):
        f = super().forward(query, items)
        elu_ = F.elu(f) + 1
        if self.eval_method == 'PD':
            return elu_
        elif self.eval_method == 'PDA':
            return self.pop * elu_
                
class PDA(DebiasedRetriever):               
        
    def add_model_specific_args(parent_parser):
        parent_parser = basemodel.Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('PDA')
        parent_parser.add_argument("--eta", type=float, default=0.02, help='gamma for PDA')
        parent_parser.add_argument("--method", type=str, default='PD', help='evaluation way of PDA')
        parent_parser.add_argument("--popularity", type=str, default='global', help='global or local')
        return parent_parser        

    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.score_func = self._get_score_func()
        
    def _get_propensity(self, train_data):
        if self.config['train']['popularity'].lower() == 'global':
            propensity = Popularity(self.config['train']['eta'], 
                                    self.config['train']['truncation'],
                                    self.config['train']['eps'])
            propensity.fit(train_data)
            return propensity
        elif self.config['train']['popularity'].lower() == 'local':
            raise NotImplementedError(f"Local popularity is not implemented.")

    def _get_score_func(self):         
        if not hasattr(self, 'propensity'):
            return None
        else:
            return PDAEvalScorer(self.config['eval']['method'], self.propensity.pop[1:]) 

    def _get_score(self, name, output, batch):
        score = super()._get_score(name, output, batch)
        pos_weight = self.propensity(batch[self.fiid])
        neg_weight = self.propensity(output['PDA']['neg_id'])
        score['pos_score'] = pos_weight * score['pos_score']
        score['neg_score'] = neg_weight * score['neg_score']
        return score