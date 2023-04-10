from recstudio.data.dataset import TripletDataset
from ..basemodel import BaseRanker
from ..loss_func import BCEWithLogitLoss
from ..module import ctr, MLPModule

r"""
DeepIM
######################

Paper Reference:
    Deep Interaction Machine: A Simple but Effective Model for High-order Feature Interactions (CIKM'20)
    https://doi.org/10.1145/3340531.3412077
"""

class DeepIM(BaseRanker):

    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        model_config = self.config['model']
        self.embedding = ctr.Embeddings(self.fields, self.embed_dim, train_data)
        self.im = ctr.InteractionMachine(self.embed_dim, model_config['order'])
        if model_config['deep']:
            self.mlp = MLPModule(
                    [self.embedding.num_features * self.embed_dim] + model_config['mlp_layer'] + [1],
                    model_config['activation'], 
                    model_config['dropout'],
                    last_activation=False, 
                    last_bn=False)
            
    def score(self, batch):
        emb = self.embedding(batch)
        im_score = self.im(emb).squeeze(-1)
        if self.config['model']['deep']:
            mlp_score = self.mlp(emb.view(emb.size(0), -1)).squeeze(-1)
            return {'score' : im_score + mlp_score}
        else:
            return{'score': im_score}

    def _get_loss_func(self):
        return BCEWithLogitLoss()
