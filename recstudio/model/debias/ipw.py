import torch
from ..loss_func import l2_reg_loss_fn
from recstudio.model import basemodel, scorer
from recstudio.data.advance_dataset import ALSDataset
from recstudio.model.module.propensity import Popularity

r"""
IPW
#########

Paper Reference:
    Causal Inference for Recommendation
    http://www.its.caltech.edu/~fehardt/UAI2016WS/papers/Liang.pdf
"""
class IPWScorer(scorer.InnerProductScorer):
    def __init__(self, eval_method, pop):
        super().__init__()
        self.is_eval = False
        self.eval_method = eval_method
        self.register_buffer('pop', pop)
    def forward(self, query, items):
        score = super().forward(query, items)
        if self.eval_method == 'marginal' and self.is_eval:
            score = self.pop * score
        return score
                        
class IPW(basemodel.BaseRetriever):
    
    def add_model_specific_args(parent_parser):
        parent_parser = basemodel.Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('IPW')
        parent_parser.add_argument("--lambda_theta", type=float, default=1e-5, help='lambda_theta for IPW')
        parent_parser.add_argument("--lambda_beta", type=float, default=1e-5, help='lambda_beta for IPW')
        parent_parser.add_argument("--method", type=str, default='conditional', help='prediction method')
        return parent_parser                         
        
    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.propensity = self._get_propensity(train_data)
        self.score_func = self._get_score_func()
    
    def _init_parameter(self):
        super()._init_parameter()
        self.query_encoder.weight.requires_grad = False
        self.item_encoder.weight.requires_grad = False

    def _get_dataset_class():
        return ALSDataset
    
    def _get_propensity(self, train_data):
        propensity = Popularity(self.config['train']['eta'], 
                                self.config['train']['truncation'],
                                self.config['train']['eps'])
        propensity.fit(train_data)
        return propensity

    def _get_score_func(self):
        if not hasattr(self, 'propensity'):
            return None
        else:
            return IPWScorer(self.config['eval']['method'], self.propensity.pop[1:])
        
    def _get_query_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)

    def _get_item_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_items, self.embed_dim, padding_idx=0)
   
    def _get_train_loaders(self, train_data):
        loader = train_data.train_loader(batch_size = self.config['train']['batch_size'], shuffle = True, drop_last = False)
        loader_T = train_data.transpose().train_loader(
                    batch_size = self.config['train']['batch_size'], shuffle = True, drop_last = False)
        return [loader, loader_T]     

    def current_epoch_trainloaders(self, nepoch):
        return self.trainloaders[nepoch % len(self.trainloaders)], False      

    def training_step(self, batch):
        """
        Update latent user/item factors
        """
        label = (batch[self.frating] > 0).float()
        self.score_func.is_eval = False
        query_emb = self.query_encoder(self._get_query_feat(batch))
        item_emb = self.item_encoder(self._get_item_feat(batch))
        weight = 1 / self.propensity(batch[self.fiid])

        if batch[self.fuid].dim() == 1:   
            for i, uid in enumerate(batch[self.fuid]):
                weight_o = weight[i]
                A = (weight_o * self.item_encoder.weight[batch[self.fiid][i]].transpose(0, 1)) @ \
                    self.item_encoder.weight[batch[self.fiid][i]] + \
                    self.config['train']['lambda_theta'] * torch.eye(self.embed_dim, device=self.device)                # D x D                              
                B = ((weight_o * batch[self.frating][i]) @ self.item_encoder.weight[batch[self.fiid][i]]).unsqueeze(-1) # D x 1
                self.query_encoder.weight[uid] = torch.linalg.solve(A, B).squeeze(-1)
            pos_score = self.score_func(query_emb, item_emb)
            mse_loss = (weight * (label - pos_score)**2).sum(-1) 
        else:
            for i, iid in enumerate(batch[self.fiid]):
                weight_o = weight[i]
                A = (weight_o * self.query_encoder.weight[batch[self.fuid][i]].transpose(0, 1)) @ \
                    self.query_encoder.weight[batch[self.fuid][i]] + \
                    self.config['train']['lambda_beta'] * torch.eye(self.embed_dim, device=self.device)                     # D x D
                B = ((weight_o * batch[self.frating][i]) @ self.query_encoder.weight[batch[self.fuid][i]]).unsqueeze(-1)    # D x 1
                self.item_encoder.weight[iid] = torch.linalg.solve(A, B).squeeze(-1)    
            pos_score = self.score_func(item_emb, query_emb)
            mse_loss = (weight.unsqueeze(-1) * (label - pos_score)**2).sum(-1) 
            
        reg_loss = self.config['train']['lambda_theta'] * l2_reg_loss_fn(self.query_encoder.weight) + \
                self.config['train']['lambda_beta'] * l2_reg_loss_fn(self.item_encoder.weight)
        loss = mse_loss + reg_loss
        return {'loss': loss} 

    def _test_step(self, batch, metric, cutoffs):
        self.score_func.is_eval = True
        return super()._test_step(batch, metric, cutoffs) 
    
    def _get_sampler(self, train_data):
        return None