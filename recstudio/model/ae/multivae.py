import torch
from recstudio.data.dataset import UserDataset
from recstudio.model.basemodel import BaseRetriever, Recommender
from recstudio.model.loss_func import SoftmaxLoss
from recstudio.model.module import MLPModule
from recstudio.model.scorer import InnerProductScorer


class MultiVAEQueryEncoder(torch.nn.Module):
    def __init__(self, fiid, num_items, embed_dim, dropout_rate,
                 encoder_dims, decoder_dims, activation='relu'):
        super().__init__()
        assert encoder_dims[-1] == decoder_dims[0], 'expecting the output size of'\
            'encoder is equal to the input size of decoder.'
        assert encoder_dims[0] == decoder_dims[-1], 'expecting the output size of'\
            'decoder is equal to the input size of encoder.'

        self.fiid = fiid
        self.item_embedding = torch.nn.Embedding(num_items, embed_dim, 0)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.encoders = torch.nn.Sequential(
            MLPModule([embed_dim]+encoder_dims[:-1], activation),
            torch.nn.Linear(([embed_dim]+encoder_dims[:-1])[-1], encoder_dims[-1]*2)
        )
        self.decoders = torch.nn.Sequential(
            MLPModule(decoder_dims, activation),
            torch.nn.Linear(decoder_dims[-1], embed_dim)
        )
        self.kl_loss = 0.0

    def forward(self, batch):
        # encode
        seq_emb = self.item_embedding(batch["in_"+self.fiid])
        non_zero_num = batch["in_"+self.fiid].count_nonzero(dim=1).unsqueeze(-1)
        seq_emb = seq_emb.sum(1) / non_zero_num.pow(0.5)
        h = self.dropout(seq_emb)

        encoder_h = self.encoders(h)
        mu, logvar = encoder_h.tensor_split(2, dim=-1)

        # decode
        z = self.reparameterize(mu, logvar)
        decoder_z = self.decoders(z)

        if self.training:
            self.kl_loss = self.kl_loss_func(mu, logvar)

        return decoder_z

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def kl_loss_func(self, mu, logvar):
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return KLD


class MultiVAE(BaseRetriever):

    # def add_model_specific_args(parent_parser):
    #     parent_parser = Recommender.add_model_specific_args(parent_parser)
    #     parent_parser.add_argument_group('MultiVAE')
    #     parent_parser.add_argument("--dropout", type=int, default=0.5, help='dropout rate for MLP layers')
    #     parent_parser.add_argument("--encoder_dims", type=int, nargs='+', default=64, help='MLP layer size for encoder')
    #     parent_parser.add_argument("--decoder_dims", type=int, nargs='+', default=64, help='MLP layer size for decocer')
    #     parent_parser.add_argument("--activation", type=str, default='relu', help='activation function for MLP layers')
    #     parent_parser.add_argument("--anneal_max", type=float, default=1.0, help="max anneal coef for KL loss")
    #     parent_parser.add_argument("--anneal_total_step", type=int, default=2000, help="total anneal steps")
    #     return parent_parser

    def _get_dataset_class():
        return UserDataset

    def _get_item_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_items, self.embed_dim, 0)

    def _get_query_encoder(self, train_data):
        model_config = self.config['model']
        return MultiVAEQueryEncoder(train_data.fiid, train_data.num_items,
                                    self.embed_dim, model_config['dropout_rate'], model_config['encoder_dims'],
                                    model_config['decoder_dims'], model_config['activation'])

    def _get_score_func(self):
        return InnerProductScorer()

    def _get_sampler(self, train_data):
        return None

    def _get_loss_func(self):
        return SoftmaxLoss()

    def training_step(self, batch):
        loss = super().training_step(batch)

        if not hasattr(self, 'anneal'):
            setattr(self, 'anneal', 0)
        anneal = min(self.config['train']['anneal_max'], self.anneal)
        self.anneal = min(self.config['train']['anneal_max'],
                          self.anneal + (1.0 / self.config['train']['anneal_total_step']))
        return loss + anneal * self.query_encoder.kl_loss
