import torch
from recstudio.data import dataset
from recstudio.model.module import data_augmentation
from recstudio.model.seq.sasrec import SASRec, SASRecQueryEncoder

r"""
ICLRec
#############
    Intent Contrastive Learning for Sequential Recommendation (WWW'22)
    Reference: 
        https://doi.org/10.1145/3485447.3512090
"""
class ICLRec(SASRec):

    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.augmentation_model = data_augmentation.ICLRecAugmentation(self.config, train_data)

    def _get_dataset_class():
        return dataset.SeqToSeqDataset

    def _get_train_loaders(self, train_data:dataset.SeqToSeqDataset, ddp=False):
        rec_train_loader = train_data.train_loader(batch_size = self.config['batch_size'], shuffle = True, \
            num_workers = self.config['num_workers'], drop_last = False, ddp=ddp)
        kmeans_train_loader = train_data.train_loader(batch_size = self.config['batch_size'], shuffle = False, \
            num_workers = self.config['num_workers'], drop_last = False, ddp=ddp)
        return [rec_train_loader, kmeans_train_loader]
        
    def current_epoch_trainloaders(self, nepoch):
        return self.trainloaders[0], False

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

    def training_step(self, batch):
        output = self.forward(batch, False, return_query=True)
        cl_output = self.augmentation_model(batch, output['query'], self.query_encoder)
        loss_value = self.loss_fn(batch[self.frating], **output['score']) \
           + self.config['cl_weight'] * cl_output['instance_cl_loss'] \
           + self.config['intent_cl_weight'] * cl_output['intent_cl_loss']
        return loss_value

    def training_epoch(self, nepoch):
        self.augmentation_model.train_kmeans(self.query_encoder, self.trainloaders[1], \
            self._parameter_device)
        return super().training_epoch(nepoch)

        


