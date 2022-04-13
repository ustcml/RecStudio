from recstudio.model import basemodel, loss_func, scorer
from recstudio.data import dataset
from recstudio.ann import sampler
import torch

class TransRec(basemodel.TwoTowerRecommender):
    def init_model(self, train_data):
        # TODO: bias here is not easy to construct query, abandoned now
        # self.bias = nn.Embedding(train_data.num_items, 1, padding_idx=0)
        self.global_user_emb = torch.nn.Parameter(torch.zeros(self.embed_dim))
        super().init_model(train_data)

    def get_dataset_class(self):
        return dataset.SeqDataset

    def build_user_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_users, embedding_dim=self.embed_dim, padding_idx=0)

    def construct_query(self, batch_data):
        user_hist = batch_data['in_item_id']
        seq_len = batch_data['seqlen'] - 1
        local_user_emb = self.user_encoder(batch_data[self.fuid])
        user_emb = local_user_emb + self.global_user_emb.expand_as(local_user_emb) # B x D
        last_item_id = torch.gather(user_hist, dim=-1, index=seq_len.unsqueeze(1))
        last_item_emb = self.item_encoder(last_item_id).squeeze(1) # B x D
        query = user_emb + last_item_emb
        return query

    def build_sampler(self, train_data):
        return sampler.UniformSampler(train_data.num_items-1, self.score_func)

    def config_scorer(self):
        return scorer.EuclideanScorer()

    def config_loss(self):
        return loss_func.BPRLoss()
