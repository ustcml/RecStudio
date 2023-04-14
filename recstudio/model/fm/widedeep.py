from recstudio.data.dataset import TripletDataset
from ..basemodel import BaseRanker
from ..loss_func import BCEWithLogitLoss
from ..module import ctr, MLPModule


class WideDeep(BaseRanker):

    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.linear = ctr.LinearLayer(self.fields, train_data)
        self.embedding = ctr.Embeddings(self.fields, self.embed_dim, train_data)
        model_config = self.config['model']
        self.mlp = MLPModule(
                        [self.embedding.num_features*self.embed_dim]+model_config['mlp_layer']+[1],
                        activation_func = model_config['activation'],
                        dropout = model_config['dropout'],
                        batch_norm = model_config['batch_norm'],
                        last_activation = False, last_bn=False)

    def score(self, batch):
        wide_score = self.linear(batch)
        emb = self.embedding(batch)
        deep_score = self.mlp(emb.flatten(1)).squeeze(-1)
        return {'score' : wide_score + deep_score}

    def _get_loss_func(self):
        return BCEWithLogitLoss()
