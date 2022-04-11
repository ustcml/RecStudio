from pytorch_lightning import loops
from torchrec.data.dataset import MFDataset
from torchrec.model import basemodel
import torch.nn.functional as F
import torch
import numpy as np
import scipy as sp
class EASE(basemodel.TwoTowerRecommender):
    def __init__(self, config):
        super().__init__(config)
    
    def set_train_loaders(self, train_data):
        train_data.loaders = [[{'user_item_matrix': train_data.get_graph(0, 'csr')[0]}]]
        train_data.nepoch = None
        return False ## use combine loader or concate loaders

    def training_step(self, data, batch_idx):
        R = data['user_item_matrix']
        G = R.T @ R
        diagIndices = np.diag_indices_from(G)
        G[diagIndices] += self.config['lambda']
        P = np.linalg.inv(G.todense())
        B = P / (-np.diag(P))
        B[diagIndices] = 0
        self.item_vector = B[:, 1:]
        self.user = R
        return torch.tensor(np.linalg.norm(R-R*B, 'fro') ** 2)

    def get_dataset_class(self):
        return MFDataset
    
    def construct_query(self, batch_data):
        uf = self.get_user_feat(batch_data)
        return self.user[uf,:]

    def config_scorer(self):
        def scorer(query, items):
            return torch.from_numpy((query @ items).A)
        return scorer

    def config_fitloop(self, trainer):
        trainer.fit_loop = basemodel.AllHistTrainLoop(self.config['epochs'])

    def config_loss(self):
        pass

    def build_user_encoder(self, train_data):
        pass

    def build_item_encoder(self, train_data):
        pass
    
    def configure_optimizers(self):
        pass

    def on_test_start(self):
        pass

    def on_validation_start(self) -> None:
        pass
    


