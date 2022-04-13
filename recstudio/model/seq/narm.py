from recstudio.model import basemodel, loss_func, scorer
from recstudio.data import dataset
import torch

class NARM(basemodel.ItemTowerRecommender):
    def init_model(self, train_data):
        super().init_model(train_data)
        self.hidden_size = self.config['hidden_size']
        self.num_layers = self.config['layer_num']
        self.dropout_rate = self.config['dropout_rate']

        self.emb_dropout = torch.nn.Dropout(self.dropout_rate[0])
        self.GRU = torch.nn.GRU(
            input_size = self.embed_dim,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            bias = False,
            batch_first = True,
            bidirectional = False
        )
        self.A_1 = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.A_2 = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.activation = torch.nn.Sigmoid()
        self.v = torch.nn.Linear(self.hidden_size, 1, bias=False)

        # decoder
        self.sim_dropout = torch.nn.Dropout(self.dropout_rate[1])
        self.B = torch.nn.Linear(self.hidden_size*2, self.embed_dim, bias=False)

    def get_dataset_class(self):
        return dataset.SeqDataset

    def construct_query(self, batch_data):
        user_hist = batch_data['in_item_id']
        emb_hist = self.item_encoder(user_hist)
        emb_hist_dropout = self.emb_dropout(emb_hist)   # B x L x D
        gru_vec, _ = self.GRU(emb_hist_dropout)    # B x L x H
        gather_index = (batch_data['seqlen']-1).view(-1, 1, 1).expand(-1, -1, self.hidden_size) # B x 1 x H
        c_global = h_t = gru_vec.gather(dim=1, index=gather_index).squeeze(1)  # B x H

        # get attention weight
        alpha = self.v(self.activation(self.A_1(h_t).unsqueeze(1) + self.A_2(gru_vec))) # B x L x 1
        c_local = torch.sum(alpha.expand_as(gru_vec) * gru_vec, dim=1)  # B x H

        c = torch.cat((c_global, c_local), dim=1)   # B x 2H
        query = self.B(self.sim_dropout(c))   # B x D
        return query

    def config_loss(self):
        return loss_func.SoftmaxLoss()

    def config_scorer(self):
        return scorer.InnerProductScorer()
