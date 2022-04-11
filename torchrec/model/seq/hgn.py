from torchrec.model import basemodel, loss_func, scorer
from torchrec.data import dataset
from torchrec.ann import sampler
import torch

class HGN(basemodel.TwoTowerRecommender):
    def init_model(self, train_data):
        super().init_model(train_data)
        self.max_seq_len = train_data.config['max_seq_len']
        self.item_embedding = torch.nn.Embedding(train_data.num_items, self.embed_dim, padding_idx=0)
        # Personalized Feature Gating
        self.W_g_1 = torch.nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.W_g_2 = torch.nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.b_g = torch.nn.Parameter(torch.empty(self.embed_dim), requires_grad=True)
        self.sigma = torch.nn.Sigmoid()

        # Personalized Instance Gating
        self.w_g_3 = torch.nn.Linear(self.embed_dim, 1, bias=False)
        self.W_g_4 = torch.nn.Linear(self.embed_dim, self.max_seq_len)
        self.pooling_type = self.config['pooling_type']     # avg or max

    def build_user_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)

    def construct_query(self, batch_data):
        item_seq = batch_data['in_item_id']
        max_seq_len = item_seq.size(1)
        seq_len = batch_data['seqlen']
        user_id = batch_data[self.fuid]
        U = self.user_encoder(user_id)
        S = self.item_embedding(item_seq)
        S_F = S * self.sigma(self.W_g_1(S) + self.W_g_2(U).view(U.size(0), 1, -1) + self.b_g)

        weight = self.sigma(self.w_g_3(S_F) +  (U@self.W_g_4.weight[:max_seq_len].T).view(U.size(0), -1, 1))    # BxLx1
        S_I = S_F * weight
        if self.pooling_type == 'avg':
            s = S_I.sum(1) / weight.sum(1)
        elif self.pooling_type == 'max':
            s = torch.max(S_I, dim=1).values
        else:
            raise ValueError("`pooling_type` only support `avg` and `max`")

        query = U + s + S.sum(1)
        return query

    def get_dataset_class(self):
        return dataset.SeqDataset

    def config_loss(self):
        return loss_func.BPRLoss()

    def config_scorer(self):
        return scorer.InnerProductScorer()

    def build_sampler(self, train_data):
        return sampler.UniformSampler(train_data.num_items-1, self.score_func)


