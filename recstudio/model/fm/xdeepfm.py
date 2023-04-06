from recstudio.data.dataset import TripletDataset

from ..basemodel import BaseRanker
from ..loss_func import BCEWithLogitLoss
from ..module import ctr, MLPModule


class xDeepFM(BaseRanker):
    
    def add_model_specific_args(parent_parser):
        parent_parser.add_argument_group("xDeepFM")
        parent_parser.add_argument("--cin_layer_size", type=int, nargs='+', default=[100,100,100], help="the CIN layer size")
        parent_parser.add_argument("--mlp_layer", type=int, nargs='+', default=[128,128,128], help="the MLP layer size")
        parent_parser.add_argument("--activation", type=str, default='relu', help="activation function")
        parent_parser.add_argument("--dropout", type=float, default=0.2, help="dropout probablity")
        parent_parser.add_argument("--direct", type=bool, default=False, help="")
        return parent_parser

    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        self.linear = ctr.LinearLayer(self.fields, train_data)
        self.fm = ctr.FMLayer(reduction='sum')
        self.embedding = ctr.Embeddings(self.fields, self.embed_dim, train_data)
        model_config = self.config['model']
        self.cin = ctr.CIN(self.embed_dim, self.embedding.num_features,
                           model_config['cin_layer_size'], model_config['activation'],
                           direct=model_config['direct'])
        self.mlp = MLPModule([self.embedding.num_features*self.embed_dim]+model_config['mlp_layer']+[1],
                             model_config['activation'], model_config['dropout'],
                             last_activation=False, last_bn=False)

    def score(self, batch):
        lr_score = self.linear(batch)
        emb = self.embedding(batch)
        cin_score = self.cin(emb).squeeze(-1)
        mlp_score = self.mlp(emb.view(emb.size(0), -1)).squeeze(-1)
        return {'score' : lr_score + cin_score + mlp_score}

    def _get_loss_func(self):
        return BCEWithLogitLoss()
