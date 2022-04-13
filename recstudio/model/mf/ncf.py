from recstudio.model import basemodel, loss_func, scorer, module
from recstudio.ann import sampler
import torch

class NCF(basemodel.TwoTowerRecommender):
    def init_model(self, train_data):
        super().init_model(train_data)
        self.mlp_hidden_size = self.config['mlp_hidden_size']
        self.activation = self.config['activation']
        self.dropout = self.config['dropout_rate']
        self.mode = self.config['mode']
        assert self.mode in set(['mlp', 'mf', 'fusion']), "Only 3 modes are supported for NCF: ['mlp', 'mf', 'fusion']"
    
    def build_user_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)

    def config_loss(self):
        return loss_func.BinaryCrossEntropyLoss()
    
    def build_sampler(self, train_data):
        return sampler.UniformSampler(train_data.num_items-1, self.score_func)

    def config_scorer(self):
        if self.config['mode']== 'mlp':
            return scorer.MLPScorer(module.MLP(hidden_size=[self.embed_dim*2]+self.config['mlp_hidden_size']+[1], activation=self.config['activation'], dropout=self.config['dropout_rate']))
        elif self.config['mode'] == 'mf':
            return module.GMFScorer(self.embed_dim, activation=self.config['activation'])
        else:
            mlp = module.MLP(hidden_size=[self.embed_dim*2]+self.config['mlp_hidden_size'], activation=self.config['activation'], dropout=self.config['dropout_rate'])
            return module.FusionMFMLPScorer(self.embed_dim, hidden_size=self.config['mlp_hidden_size'][-1], mlp=mlp, activation=self.config['activation'])

    def construct_query(self, batch_data):
        return self.user_encoder(batch_data[self.fuid])

