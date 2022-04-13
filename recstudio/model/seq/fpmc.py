from recstudio.model import basemodel, loss_func, scorer
from recstudio.data import dataset
from recstudio.ann import sampler
import torch

class FPMC(basemodel.TwoTowerRecommender):
    def init_model(self, train_data):
        super().init_model(train_data)
        self.VL_emb = torch.nn.Embedding(train_data.num_items, self.embed_dim, padding_idx=0)

    def get_dataset_class(self):
        return dataset.SeqDataset

    def build_user_encoder(self, train_data):
        VU_emb = torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)
        return VU_emb

    def build_item_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_items, self.embed_dim*2, padding_idx=0)

    def construct_query(self, batch_data):
        user_hist = batch_data['in_item_id']
        seq_len = batch_data['seqlen'] - 1
        user_emb = self.user_encoder(self.get_user_feat(batch_data))    # B x D
        last_item_id = torch.gather(user_hist, dim=-1, index=seq_len.unsqueeze(1))
        last_item_emb = self.VL_emb(last_item_id).squeeze(1)   # B x D
        return torch.cat([user_emb, last_item_emb], dim=-1) # B x D

    def config_loss(self):
        return loss_func.BPRLoss()

    def config_scorer(self):
        return scorer.InnerProductScorer()

    def build_sampler(self, train_data):
        return sampler.UniformSampler(train_data.num_items-1, self.score_func)