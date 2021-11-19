from torchrec.model import basemodel
import torch
from torchrec.model.fm import layers
class FM(basemodel.TowerFreeRecommender):
    
    def init_model(self, train_data):
        self.embeddings = torch.nn.ModuleDict()
        for f, t in train_data.field2type.items():
            if f != self.frating and 'time' not in f:
                self.embeddings[f] = torch.nn.Embedding(train_data.num_values(f), self.embed_dim, padding_idx=0)
        self.inter = layers.Order2Inter()

    def config_loss(self):
        return torch.nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, batch):
        embs = []
        for f in self.embeddings:
            d = batch[f]
            if d.dim() > 1:
                len = (d>0).sum(dim=-1, keepdim=True).sqrt() + torch.finfo(torch.float32).eps
                embs.append(self.embeddings[f](d) / len)
            else:
                embs.append(self.embeddings[f](d.view(-1, 1)))
        emb = torch.cat(embs, dim=1)
        rep = self.inter(emb)
        return rep.sum(dim=-1)



