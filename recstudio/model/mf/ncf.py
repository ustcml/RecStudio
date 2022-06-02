import torch
from recstudio.ann import sampler
from recstudio.data import dataset
from recstudio.model import basemodel, loss_func, scorer, module

class NCF(basemodel.BaseRetriever):

    def _get_dataset_class(self):
        return dataset.MFDataset

    def _get_item_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_items, self.embed_dim, padding_idx=0)

    def _get_query_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)

    def _get_score_func(self):
        assert self.config['mode'] in set(['mlp', 'mf', 'fusion']), \
            "Only 3 modes are supported for NCF: ['mlp', 'mf', 'fusion']"
        if self.config['mode']== 'mlp':
            return scorer.MLPScorer(module.MLPModule(
                mlp_layers = [self.embed_dim*2]+self.config['mlp_hidden_size']+[1], 
                activation_func = self.config['activation'], 
                dropout = self.config['dropout_rate']))
        elif self.config['mode'] == 'mf':
            return scorer.GMFScorer(self.embed_dim, activation=self.config['activation'])
        else:
            mlp = module.MLPModule(
                mlp_layers = [self.embed_dim*2]+self.config['mlp_hidden_size'], 
                activation_func = self.config['activation'], 
                dropout = self.config['dropout_rate'])
            return scorer.FusionMFMLPScorer(
                emb_dim = self.embed_dim, 
                hidden_size = self.config['mlp_hidden_size'][-1], 
                mlp = mlp, 
                activation = self.config['activation'])

    
    def _get_loss_func(self):
        return loss_func.BinaryCrossEntropyLoss()

    def _get_sampler(self, train_data):
        return sampler.UniformSampler(train_data.num_items-1)
