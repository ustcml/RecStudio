import torch
from recstudio.ann import sampler
from recstudio.data import dataset
from recstudio.model import basemodel, loss_func, scorer

r"""
Caser
######################

Paper Reference:
     Jiaxi Tang, et al. "Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding" in WSDM 2018.
     https://dl.acm.org/doi/abs/10.1145/3159652.3159656
"""


class CaserQueryEncoder(torch.nn.Module):

    def __init__(self, fiid, fuid, num_users, num_items, embed_dim, max_seq_len, n_v, n_h, dropout_rate, ) -> None:
        super().__init__()
        self.fiid = fiid
        self.fuid = fuid
        self.max_seq_len = max_seq_len
        self.user_embedding = torch.nn.Embedding(num_users, embed_dim, 0)
        self.item_embedding = torch.nn.Embedding(num_items, embed_dim, 0)
        self.vertical_filter = torch.nn.parameter.Parameter(
            torch.zeros(size=(max_seq_len, n_v), dtype=torch.float32),
            requires_grad=True
        )
        height = range(1, max_seq_len + 1)
        self.horizontal_filter = torch.nn.ModuleList([
            torch.nn.Conv2d(in_channels=1, out_channels=n_h, kernel_size=(h, embed_dim))
            for h in height
        ])
        self.fc_w = torch.nn.parameter.Parameter(
            torch.zeros((n_v*embed_dim+n_h*max_seq_len, embed_dim), dtype=torch.float32),
            requires_grad=True
        )
        self.fc_b = torch.nn.parameter.Parameter(
            torch.zeros(embed_dim, dtype=torch.float32),
            requires_grad=True
        )

    def forward(self, batch):
        P_u = self.user_embedding(batch[self.fuid])
        E_ut = self.item_embedding(batch['in_' + self.fiid])

        o_v = (E_ut.transpose(1, 2) @ self.vertical_filter[: E_ut.size(1), :]).transpose(1, 2)
        o_v = o_v.reshape(o_v.size(0), -1)

        o_h = []
        E_ut_ = E_ut.view(-1, 1, *E_ut.shape[1:])
        for i in range(E_ut.size(1)):
            conv_out = torch.relu(self.horizontal_filter[i](E_ut_).squeeze(3))
            pool_out = torch.nn.functional.max_pool1d(conv_out, conv_out.size(2))
            o_h.append(pool_out.squeeze(2))
        o_h = torch.cat(o_h, dim=1)

        o = torch.cat((o_v, o_h), dim=1)
        z = torch.relu(o @ self.fc_w[: o.size(1), :] + self.fc_b)
        return torch.cat((z, P_u), dim=1)


class Caser(basemodel.BaseRetriever):
    r"""
        | Caser models user's behavior with multi CNN layers of different kernel size, which aim
          to capture different length of behavior sequences.The idea is to embed a sequence of
          recent itemsinto an “image” in the time and latent spaces and learn sequential patterns
          as local features of the image using convolutional filters.
    """

    def add_model_specific_args(parent_parser):
        parent_parser = basemodel.Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('Caser')
        parent_parser.add_argument("--n_v", type=int, default=8, help='number of vertival CNN cells')
        parent_parser.add_argument("--n_h", type=int, default=16, help='number of horizontal CNN cells')
        parent_parser.add_argument("--dropout", type=float, default=0.5, help='dropout rate')
        parent_parser.add_argument("--negative_count", type=int, default=3, help='negative sampling numbers')
        return parent_parser

    def _get_dataset_class():
        r"""SeqDataset is used for Caser."""
        return dataset.SeqDataset

    def _get_item_encoder(self, train_data):
        r"""A simple item embedding is used as item encoder in Caser.

        The item embedding output dimension is twice as the dimension of user embedding.
        """
        return torch.nn.Embedding(train_data.num_items, self.embed_dim*2, padding_idx=0)

    def _get_query_encoder(self, train_data):
        return CaserQueryEncoder(
            self.fiid, self.fuid, train_data.num_users, train_data.num_items,
            self.embed_dim, train_data.config['max_seq_len'], self.config['n_v'],
            self.config['n_h'], self.config['dropout']
        )

    def _get_score_func(self):
        r"""Innerproduct operation is applied to calculate scores between query and item."""
        return scorer.InnerProductScorer()

    def _get_loss_func(self):
        r"""According to the original paper, BPR loss is applied.
            Also, other loss functions like BCE loss can be used too.
        """
        return loss_func.BPRLoss()

    def _get_sampler(self, train_data):
        return sampler.UniformSampler(train_data.num_items)
