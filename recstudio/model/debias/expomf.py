import torch
import numpy as np
from recstudio.model import basemodel, scorer
from recstudio.data.advance_dataset import ALSDataset

r"""
ExpoMF
#########

Paper Reference:
    Modeling User Exposure in Recommendation (WWW'16)
    https://dl.acm.org/doi/10.1145/2872427.2883090
"""

class ExpoMF(basemodel.BaseRetriever):
    
    def add_model_specific_args(parent_parser):
        parent_parser = basemodel.Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('ExpoMF')
        parent_parser.add_argument("--lambda_y", type=float, default=1.0, help='lambda_y for ExpoMF')
        parent_parser.add_argument("--lambda_theta", type=float, default=1e-5, help='lambda_theta for ExpoMF')
        parent_parser.add_argument("--lambda_beta", type=float, default=1e-5, help='lambda_beta for ExpoMF')
        parent_parser.add_argument("--init_mu", type=float, default=0.01, help='init mu for ExpoMF')
        parent_parser.add_argument("--alpha1", type=float, default=1.0, help='alpha1 for ExpoMF')
        parent_parser.add_argument("--alpha2", type=float, default=1.0, help='alpha2 for ExpoMF')
        return parent_parser                         
        
    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.register_buffer('a', torch.ones(train_data.num_users, train_data.num_items))
        self.register_buffer('mu', self.config['train']['init_mu'] * torch.ones(train_data.num_items))
    
    def _init_parameter(self):
        super()._init_parameter()
        self.query_encoder.weight.requires_grad = False
        self.item_encoder.weight.requires_grad = False

    def _get_dataset_class():
        return ALSDataset

    def _get_query_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)

    def _get_item_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_items, self.embed_dim, padding_idx=0)

    def _get_score_func(self):       
        return scorer.InnerProductScorer()
        
    def _get_train_loaders(self, train_data):
        loader = train_data.train_loader(batch_size = self.config['train']['batch_size'], shuffle = True, drop_last = False)
        loader_T = train_data.transpose().train_loader(
                    batch_size = self.config['train']['batch_size'], shuffle = True, drop_last = False)
        return [loader, loader_T]     

    def current_epoch_trainloaders(self, nepoch):
        return self.trainloaders[nepoch % len(self.trainloaders)], False 

    def training_epoch(self, nepoch):
        super().training_epoch(nepoch)       
        self.mu = (self.config['train']['alpha1'] + torch.sum(self.a, dim=0) - 1) / \
                (self.config['train']['alpha1'] + self.config['train']['alpha2'] + self.a.shape[0] - 2)                  
        return torch.tensor(0.)

    def training_step(self, batch):
        a = self._expectation(batch)
        self._maximization(batch, a)
    
    def _expectation(self, batch):
        """
        Compute the posterior of exposure latent variables a_{ui}
        """
        if batch[self.fuid].dim() == 1:
            mu = self.mu
            P_y0_given_a1 = np.sqrt(self.config['train']['lambda_y'] / 2 * torch.pi) * \
                            torch.exp(-self.config['train']['lambda_y'] * 
                                (self.query_encoder(self._get_query_feat(batch)) @  # B x D
                                self.item_encoder.weight.transpose(0, 1))           # D x num_items
                                **2 / 2)                                            # -> B x num_items
        else: 
            mu = self.mu[batch[self.fiid]].unsqueeze(-1)
            P_y0_given_a1 = np.sqrt(self.config['train']['lambda_y'] / 2 * torch.pi) * \
                            torch.exp(-self.config['train']['lambda_y'] * 
                                (self.item_encoder(self._get_item_feat(batch)) @  # B x D
                                self.query_encoder.weight.transpose(0, 1))          # D x num_users
                                **2 / 2)                                            # -> B x num_users
                              
        a = (P_y0_given_a1 + 1e-8) / (P_y0_given_a1 + 1e-8 + (1 - mu) / mu)        
        for i, j in batch[self.frating].nonzero():
            a[i, j] = torch.tensor(1.)

        # update self.a
        if batch[self.fuid].dim() == 1:    
            for i, uid in enumerate(batch[self.fuid]):
                for j, iid  in enumerate(batch[self.fiid][i]):
                    self.a[uid, iid] = a[i, j]
        return a     
    
    def _maximization(self, batch, a):
        """
        Update latent factors theta and beta
        """
        if batch[self.fuid].dim() == 1:
            for i, uid in enumerate(batch[self.fuid]):
                A = self.config['train']['lambda_y'] * \
                    (a[i] * self.item_encoder.weight.transpose(0, 1)) @ \
                    self.item_encoder.weight + \
                    self.config['train']['lambda_theta'] * torch.eye(self.embed_dim, device=self.device)     # D x D
                B = self.config['train']['lambda_y'] * ((a[i][batch[self.fiid][i]] * batch[self.frating][i]) @ \
                    self.item_encoder.weight[batch[self.fiid][i]]).unsqueeze(-1)                             # D x 1
                self.query_encoder.weight[uid] = torch.linalg.solve(A, B).squeeze(-1)
        else:
            for i, iid in enumerate(batch[self.fiid]):
                A = self.config['train']['lambda_y'] * \
                    (a[i] * self.query_encoder.weight.transpose(0, 1)) @ \
                    self.query_encoder.weight + \
                    self.config['train']['lambda_beta'] * torch.eye(self.embed_dim, device=self.device)      # D x D
                B = self.config['train']['lambda_y'] * ((a[i][batch[self.fuid][i]] * batch[self.frating][i]) @ \
                    self.query_encoder.weight[batch[self.fuid][i]]).unsqueeze(-1)                            # D x 1
                self.item_encoder.weight[iid] = torch.linalg.solve(A, B).squeeze(-1)
                
    def _get_sampler(self, train_data):
        return None           