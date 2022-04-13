from recstudio.model import basemodel, loss_func, scorer
from recstudio.data import dataset
import torch

class STAMP(basemodel.ItemTowerRecommender):
    def init_model(self, train_data):
        super().init_model(train_data)
        self.W_0 = torch.nn.Linear(self.embed_dim, 1, bias=False)
        self.W_1 = torch.nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.W_2 = torch.nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.W_3 = torch.nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.b_a = torch.nn.Parameter(torch.zeros(self.embed_dim), requires_grad=True)
        self.sigma = torch.nn.Sigmoid()
        self.MLP_A = torch.nn.Sequential(
            torch.nn.Linear(self.embed_dim, self.embed_dim, bias=True),
            torch.nn.Tanh()
        )
        self.MLP_B = torch.nn.Sequential(
            torch.nn.Linear(self.embed_dim, self.embed_dim, bias=True),
            torch.nn.Tanh()
        )

    def get_dataset_class(self):
        return dataset.SeqDataset
    
    def construct_query(self, batch_data):
        user_hist = batch_data['in_item_id']
        seq_len = batch_data['seqlen']
        seq_emb = self.item_encoder(user_hist)
        gather_index = (seq_len-1).view(-1, 1, 1).expand(-1, -1, self.embed_dim) # B x 1 x D
        last_item = seq_emb.gather(dim=1, index=gather_index).squeeze(1)  # B x D

        m_s = seq_emb.sum(dim=1) / seq_len.unsqueeze(1).float() # B x D
        alpha = self.W_0(self.sigma(self.W_1(seq_emb) + self.W_2(last_item).unsqueeze(1) + self.W_3(m_s).unsqueeze(1) + self.b_a.view(1,1,-1))) # B x L x 1
        m_a = (alpha*seq_emb).sum(dim=1)    # B x D

        h_s = self.MLP_A(m_a)
        h_t = self.MLP_B(last_item)
        query = h_s * h_t
        return query

    def config_loss(self):
        return loss_func.SoftmaxLoss()

    def config_scorer(self):
        return scorer.InnerProductScorer()
        