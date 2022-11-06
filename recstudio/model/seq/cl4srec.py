import torch
from recstudio.data import dataset
from recstudio.model.module import data_augmentation
from .sasrec import SASRec, SASRecQueryEncoder

r"""
CL4SRec
#############
    Contrastive Learning for Sequential Recommendation(SIGIR'21)
    Reference: 
        https://arxiv.org/abs/2010.14395
"""
class CL4SRec(SASRec):
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

    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.augmentation_model = data_augmentation.CL4SRecAugmentation(self.config, train_data)

    def _get_dataset_class():
        return dataset.SeqToSeqDataset

    def _get_item_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_items + 1, self.embed_dim, padding_idx=0) # the last item is mask 

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

    def training_step(self, batch):
        output = self.forward(batch, False)
        cl_output = self.augmentation_model(batch, self.query_encoder)
        loss_value = self.loss_fn(batch[self.frating], **output['score']) + self.config['cl_weight'] * cl_output['cl_loss']
        return loss_value
    
