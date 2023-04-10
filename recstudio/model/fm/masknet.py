from recstudio.data.dataset import TripletDataset
from .. import loss_func
from ..basemodel import BaseRanker
from ..module import ctr

r"""
MaskNet
######################

Paper Reference:
    MaskNet: Introducing Feature-Wise Multiplication to CTR Ranking Models by Instance-Guided Mask (DLP KDD'21)
    https://arxiv.org/abs/2102.07619
"""

class MaskNet(BaseRanker):

    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        self.embedding = ctr.Embeddings(self.fields, self.embed_dim, train_data)
        model_config = self.config['model']
        if model_config['parallel']:
            self.masknet = ctr.ParallelMaskNet(
                            self.embedding.num_features, 
                            self.embed_dim, 
                            model_config['num_blocks'], 
                            model_config['block_dim'], 
                            model_config['reduction_ratio'],
                            model_config['mlp_layer'],
                            model_config['activation'],
                            model_config['dropout'],
                            model_config['hidden_layer_norm'])
        else:
            self.masknet = ctr.SerialMaskNet(
                            self.embedding.num_features, 
                            self.embed_dim, 
                            model_config['block_dim'], 
                            model_config['reduction_ratio'],
                            model_config['activation'],
                            model_config['dropout'],
                            model_config['hidden_layer_norm'])

    def score(self, batch):
        emb = self.embedding(batch)
        score = self.masknet(emb).squeeze(-1)
        return {'score' : score}

    def _get_loss_func(self):
        return loss_func.BCEWithLogitLoss()
