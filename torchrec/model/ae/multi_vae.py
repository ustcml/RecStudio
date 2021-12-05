from torch.nn.modules.loss import KLDivLoss
from torchrec.model import basemodel, loss_func, scorer
from torchrec.ann import sampler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchrec.data import dataset


class Multi_VAE(basemodel.ItemTowerRecommender):
    """
    Reference : https://dl.acm.org/doi/pdf/10.1145/3178876.3186150
    Code Reference : https://github.com/younggyoseo/vae-cf-pytorch
    """
    def __init__(self, config):
        # config['embed_dim'] = config['decoder_dims'][-1]
        config.update({"embed_dim":config['decoder_dims'][-1]})
        super(Multi_VAE, self).__init__(config)
        
        self.encoder_dims = config['encoder_dims']
        self.decoder_dims = config['decoder_dims']
        self.dropout = nn.Dropout(config['dropout'])
        # self.neg_count = None
        
        # IF there is a need for activate function
        if 'activate_function' in config:
            self.activate = Multi_VAE._activate_set(config['activate_function'])
        else:
            self.activate = F.tanh
        
        self.anneal = config['anneal']
        self.total_anneal_steps = config['total_anneal_steps']
        self.bz = -1e4


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
        en_dim = self.encoder_dims[:-1] + [self.encoder_dims[-1] * 2]        
        self.encoders = torch.nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(en_dim[:-1], en_dim[1:])]
        )
        de_dim = self.decoder_dims
        self.decoders = torch.nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(de_dim[:-1], de_dim[1:])]
        )

    
    def config_loss(self):
        return loss_func.SoftmaxLoss()
    
    def config_scorer(self):
        return scorer.InnerProductScorer()



    def construct_query(self, batch_data):
        assert len(self.user_fields) == 1
        # batch_data : B * D
        # Each row denotes the history of one sampled user
        data = batch_data['in_item_id']
        user_emb_mu, user_emb_logvar = self._encode_user(data)
        self.kld_loss = self.kl_loss_func(user_emb_mu, user_emb_logvar)
        z = self.reparameterize(user_emb_mu, user_emb_logvar)
        return self._decode_user(z)


    def get_dataset_class(self):
        return dataset.AEDataset

    def kl_loss_func(self, mu, logvar):
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return  KLD

    def _encode_user(self, batch_data):
        count_nonzero = batch_data.count_nonzero(dim=1).unsqueeze(-1) # batch_user * 1
        user_embs = self.encoder_layer_0(batch_data) # batch_user * dims
        user_embs = torch.sum(user_embs, dim=1) / count_nonzero.pow(0.5)
        h = self.dropout(user_embs)

        for i, layer in enumerate(self.encoders):
            h = layer(h)
            if i != len(self.encoders) - 1:
                h = self.activate(h)
            else:
                mu = h[:, :self.encoder_dims[-1]]
                logvar = h[:, self.encoder_dims[-1]:]
        return mu, logvar
    
    def _decode_user(self, z):
        # without the last layer
        h = z
        for i, layer in enumerate(self.decoders):
            h = layer(h)
            if i != len(self.decoders) - 1:
                h = self.activate(h)
        return h

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        self.bz = max(self.bz, batch_idx)
        if self.total_anneal_steps is not None and self.total_anneal_steps > 0:
            anneal = min(self.anneal, 1. * (self.current_epoch * self.bz + batch_idx) / self.total_anneal_steps)
        else:
            anneal = self.anneal
        return loss + anneal * self.kld_loss

    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu


    

