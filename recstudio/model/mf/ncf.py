import torch
from recstudio.ann import sampler
from recstudio.data import dataset
from recstudio.model import basemodel, loss_func, scorer, module

class NCF(basemodel.BaseRetriever):

    def _get_dataset_class():
        return dataset.TripletDataset

    def _get_item_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_items, self.embed_dim, padding_idx=0)

    def _get_query_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)

    def _get_score_func(self):
        model_config = self.config['model']
        score_mode = model_config['score_mode']
        assert score_mode in set(['mlp', 'mf', 'fusion']), \
            "Only 3 score modes are supported for NCF: ['mlp', 'mf', 'fusion']"
        if score_mode == 'mlp':
            return scorer.MLPScorer(module.MLPModule(
                mlp_layers = [self.embed_dim*2]+model_config['mlp_hidden_size']+[1],
                activation_func = model_config['activation'],
                dropout = model_config['dropout']))
        elif score_mode == 'mf':
            return scorer.GMFScorer(self.embed_dim, activation=model_config['activation'])
        else:
            mlp = module.MLPModule(
                mlp_layers = [self.embed_dim*2]+model_config['mlp_hidden_size'],
                activation_func = model_config['activation'],
                dropout = model_config['dropout'])
            return scorer.FusionMFMLPScorer(
                emb_dim = self.embed_dim,
                hidden_size = model_config['mlp_hidden_size'][-1],
                mlp = mlp,
                activation = model_config['activation'])


    def _get_loss_func(self):
        return loss_func.BinaryCrossEntropyLoss()

    def _get_sampler(self, train_data):
        return sampler.UniformSampler(train_data.num_items)
