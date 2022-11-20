import torch
from recstudio.ann import sampler
from recstudio.data import dataset
from recstudio.model import basemodel, loss_func, scorer
from recstudio.model.module import data_augmentation
from .sasrec import SASRecQueryEncoder


r"""
CoSeRec
#############
    Contrastive Self-supervised Sequential Recommendation with Robust Augmentation
    Reference: 
        https://doi.org/10.48550/arXiv.2108.06479
"""
class CoSeRec(basemodel.BaseRetriever):
    r"""
    Model hyper parameters:
        - ``embed_dim(int)``: The dimension of embedding layers. Default: ``64``.
        - ``hidden_size(int)``: The output size of Transformer layer. Default: ``128``.
        - ``layer_num(int)``: The number of layers for the Transformer. Default: ``2``.
        - ``dropout_rate(float)``:  The dropout probablity for dropout layers after item embedding
         | and in Transformer layer. Default: ``0.5``.
        - ``head_num(int)``: The number of heads for MultiHeadAttention in Transformer. Default: ``2``.
        - ``activation(str)``: The activation function in transformer. Default: ``"gelu"``.
        - ``layer_norm_eps``: The layer norm epsilon in transformer. Default: ``1e-12``.
    """

    def _init_model(self, train_data:dataset.SeqToSeqDataset):
        super()._init_model(train_data)
        self.config['max_seq_len'] = train_data.config['max_seq_len']
        self.num_items = train_data.num_items
        self.augmentation_model = data_augmentation.CoSeRecAugmentation(self.config, train_data)
        
    def _get_dataset_class():
        return dataset.SeqToSeqDataset

    def _get_item_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_items + 1, self.embed_dim, padding_idx=0) # the last is masking 

    def _get_query_encoder(self, train_data):
        return SASRecQueryEncoder(
            fiid=self.fiid, embed_dim=self.embed_dim,
            max_seq_len=train_data.config['max_seq_len'], n_head=self.config['head_num'],
            hidden_size=self.config['hidden_size'], dropout=self.config['dropout_rate'],
            activation=self.config['activation'], layer_norm_eps=self.config['layer_norm_eps'],
            n_layer=self.config['layer_num'],
            training_pooling_type='origin',
            item_encoder=self.item_encoder
        )

    def _get_score_func(self):
        return scorer.InnerProductScorer()

    def _get_loss_func(self):
        return loss_func.BinaryCrossEntropyLoss()

    def _get_sampler(self, train_data):
        return sampler.UniformSampler(train_data.num_items)

    def training_step(self, batch):
        output = self.forward(batch, isinstance(self.loss_fn, loss_func.FullScoreLoss))
        cl_output = self.augmentation_model(batch, self.query_encoder)
        loss_value = self.loss_fn(batch[self.frating], **output['score']) + \
            self.config['cl_weight'] * cl_output['cl_loss'] 
        return loss_value

    def training_epoch(self, nepoch):
        if nepoch + 1 >= self.config['augmentation_warm_up_epochs'] + 1:
            self.augmentation_model.update_online_model(nepoch, self.item_encoder)
        return super().training_epoch(nepoch)