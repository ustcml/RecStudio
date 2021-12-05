from torch.nn.modules.loss import KLDivLoss
from torchrec.model import basemodel, loss_func, scorer
from torchrec.ann import sampler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchrec.data import dataset

class Multi_DAE(basemodel.ItemTowerRecommender):
    """
    Reference : https://dl.acm.org/doi/pdf/10.1145/3178876.3186150
    Code Reference : https://github.com/younggyoseo/vae-cf-pytorch
    """
    def __init__(self, config):
        config.update({"embed_dim":config['decoder_dims'][-1]})
        super(Multi_DAE, self).__init__(config)

        self.encoder_dims = config['encoder_dims']
        self.decoder_dims = config['decoder_dims']

        self.dropout = nn.Dropout(config['dropout'])
        if 'activate_function' in config:
            self.activate = Multi_DAE._activate_set(config['activate_function'])
        else:
            self.activate = F.tanh


    @staticmethod
    def _activate_set(activate):
        at = activate.lower()
        if 'relu' == at:
            return F.relu
        elif 'tanh' == at:
            return F.tanh
        elif 'sigmoid' == at:
            return F.sigmoid
        elif 'prelu' == at:
            return F.prelu
        else:
            return F.tanh
        # Todo more activate functions

    def init_model(self, train_data):
        super().init_model(train_data)
        # self.neg_count = None
        assert self.encoder_dims[-1] == self.decoder_dims[0], "In and Out dimensions must equal to each other"
        assert self.encoder_dims[0] == self.decoder_dims[-1], "Latent dimension for encoder and decoder network mismatches."
        
        #The encoder has two outputs: mean and variance
        self.encoder_layer_0 = nn.Embedding(train_data.num_items, self.encoder_dims[0], padding_idx=0)
        en_dim = self.encoder_dims + self.decoder_dims[1:]
        self.layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(en_dim[:-1], en_dim[1:])])
    
    def config_loss(self):
        return loss_func.SoftmaxLoss()
    
    def config_scorer(self):
        return scorer.InnerProductScorer()
    
    def get_dataset_class(self):
        return dataset.AEDataset
    
    def construct_query(self, batch_data):
        assert len(self.user_fields) == 1
        data = batch_data['in_item_id']
        return self._encode_user(data)
    
    def _encode_user(self, batch_data):
        count_nonzero = batch_data.count_nonzero(dim=1).unsqueeze(-1) # batch_user * 1
        user_embs = self.encoder_layer_0(batch_data) # batch_user * dims
        user_embs = torch.sum(user_embs, dim=1) / count_nonzero.pow(0.5)
        h = self.dropout(user_embs)
        
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != len(self.layers) - 1:
                h = self.activate(h)
        return h
    

        