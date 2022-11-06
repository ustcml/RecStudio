import torch
from recstudio.ann import sampler
from recstudio.data import dataset
from recstudio.model import basemodel, loss_func, scorer
from recstudio.model.module import functional as recfn 
from .sasrec import SASRecQueryEncoder


class BERT4Rec(basemodel.BaseRetriever):

    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.mask_token = train_data.num_items
        self.query_fields = self.query_fields | set(["mask_token"])

    def _get_dataset_class():
        return dataset.SeqDataset

    def _get_query_encoder(self, train_data):
        return SASRecQueryEncoder(
            fiid=self.fiid, embed_dim=self.embed_dim,
            max_seq_len=train_data.config['max_seq_len'], n_head=self.config['head_num'],
            hidden_size=self.config['hidden_size'], dropout=self.config['dropout'],
            activation=self.config['activation'], layer_norm_eps=self.config['layer_norm_eps'],
            n_layer=self.config['layer_num'],
            training_pooling_type='mask',
            item_encoder=self.item_encoder,
            bidirectional=True,
        )

    def _get_item_encoder(self, train_data):
        # id num_items is used for mask token
        return torch.nn.Embedding(train_data.num_items+1, self.embed_dim, padding_idx=0)

    def _get_score_func(self):
        return scorer.InnerProductScorer()

    def _get_loss_func(self):
        r"""SoftmaxLoss is used as the loss function."""
        return loss_func.SoftmaxLoss()

    def _get_sampler(self, train_data):
        return None

    def _reconstruct_train_data(self, batch):
        item_seq = batch['in_'+self.fiid]

        padding_mask = item_seq == 0
        rand_prob = torch.rand_like(item_seq, dtype=torch.float)
        rand_prob.masked_fill_(padding_mask, 1.0)
        masked_mask = rand_prob < self.config['mask_ratio']
        masked_token = item_seq[masked_mask]

        item_seq[masked_mask] = self.mask_token
        batch['in_'+self.fiid] = item_seq

        batch[self.fiid] = masked_token     # N
        batch['mask_token'] = masked_mask
        return batch

    def training_step(self, batch):
        batch = self._reconstruct_train_data(batch)
        return super().training_step(batch)
