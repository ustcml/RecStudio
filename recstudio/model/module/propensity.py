import torch
from torch.utils.data import Dataset
# from recstudio.model.fm.lr import LR
# from recstudio.utils import get_model
# from recstudio.utils import get_model, color_dict_normal, set_color, get_logger

class Popularity(torch.nn.Module):
    """
    get propensity by popularity.
    """
    def __init__(self, eta=1.0, truncation=0.1, eps=1e-7):
        super().__init__()
        self.eta = eta
        self.trucation = truncation
        self.eps = eps
        
    def fit(self, train_data : Dataset):
        pop = (train_data.item_freq + 1) / (torch.sum(train_data.item_freq) + train_data.num_items)
        pop = (pop - pop.min()) / (pop.max() - pop.min())
        pop = pop ** self.eta
        if self.trucation is None or self.trucation == 0:
            pop = pop + self.eps
        else:
            pop = torch.max(
                    torch.vstack([pop, self.trucation * torch.ones_like(pop)]),
                    dim=0
                ).values
        self.register_buffer('pop', pop)
        
    def forward(self, batch):
        """batch (torch.tensor): item id"""
        return self.pop[batch]
    
class FromCoatFile(torch.nn.Module):
    """
    get propensity from the file in Coat dataset.
    
    Args:
        pop (torch.Tensor): a U x I matrix
    """
    def __init__(self, prop):
        super().__init__()
        self.prop = prop
    def fit(self, train_data):
        self.fuid = train_data.fuid
        self.fiid = train_data.fiid
    def forward(self, batch):
        """batch (dict)"""
        return self.prop[batch[self.fuid], batch[self.fiid]]
    
# def get_propensity(config) -> torch.nn.Module:
#     if config['propensity_estimation'].lower() == "naive_bayes":
#         return NaiveBayes()
#     elif config['propensity_estimation'].lower() == "logistic_regression":
#         _, model_conf = get_model('LR')
#         for k, v in config.items():
#             if k.startswith('p_'):
#                 model_conf.update({k[2:]: v})
#         model = LR(model_conf)
#         model.logger.info(f"\n{set_color('P-Model Config', 'green')}: \n\n" + color_dict_normal(model_conf, False))
#         return model
#     elif config['propensity_estimation'].lower() == "popularity":
#         return Popularity()
#     elif config['propensity_estimation'].lower() == "poisson_factorization":
#         return PoissonFactorization()
#     else:
#         raise ValueError(f"{config['propensity_estimation']} is not supportable.")          
              
class NaiveBayes(torch.nn.Module):
    """get propensity by naive bayes method.

    Args:
        train_data (Dataset): missing not at random data; for training recommender and propensity
        unif_data (Dataset): missing completely at random data; for training propensity only
            
    """
    def fit(self, train_data : Dataset, unif_data : Dataset):        
        y, y_cnt_given_o = torch.unique(train_data.inter_feat.get_col[train_data.frating], return_counts=True)
        y = y.tolist()
        P_y_given_o = y_cnt_given_o / torch.sum(y_cnt_given_o)
        P_o = train_data.num_inters / train_data.num_users * train_data.num_items
        
        y_, y_cnt = torch.unique(unif_data.inter_feat.get_col[unif_data.frating], return_counts=True)
        y_ = y_.tolist()
        P_y = y_cnt / torch.sum(y_cnt)
        
        y_dict = {}
        for k, v in zip(y, P_y_given_o):
            y_dict[k] = v * P_o / P_y[y_.index(k)]
        
        self.register_buffer('y_dict', y_dict)
        
    def forward(self, batch):
        p = torch.zeros_like(batch)
        for i, y in enumerate(batch):
            p[i] = self.y_dict[y]    
        return p


        
class PoissonFactorization(torch.nn.Module):
    """
    For Poisson factorization exposure model aui ~ Poi(\pi^T_u * \eta_i)
    with conjugae gamma prior on the latent embeddings pi_u and \eta_i, 
    we perform standard variational inference  on the exposure data a_{ui}. 
    After obtaining the optimal approximating variational distribution q 
    on pi_u and \eta_i at convergence, we compute the propensity score.

    Used by Dawen Liang et.al. Causal Inference for Recommendation
    """
    def fit(self, train_data : Dataset):
        #TODO(@pepsi2222)
        self.fuid = train_data.fuid
        self.fiid = train_data.fiid
        pass
    def forward(self, batch):
        lambda_ui = (self.pi[batch[self.fuid]] * self.eta[batch[self.fiid]]).sum(1)
        return 1 - torch.exp(-lambda_ui)