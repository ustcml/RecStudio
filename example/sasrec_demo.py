import torch
from recstudio.ann import sampler
from recstudio.data import dataset
from recstudio.model import scorer, loss_func, basemodel, module

# Step1: load dataset

## use our default config
ml_100k_dataset = dataset.SeqDataset(name='ml-100k', default_config=True)
## or you can change some config with a dict or a config file(yaml)
dataset_config = {
    'use_fields': ['item_id'],
    'max_seq_len': 50
}

ml_100k_dataset = ml_100k_dataset.SeqDataset(name='ml-100k', default_config=False, config=dataset_config, config_path=None)


# Step2: build dataset
## here you can set split mode, split ratio and negative sampling.

# for SASRec, we use negative sampling in training instead of in dataset loading.
trn, val, tst = dataset.build(ratio_or_num=1, neg_count=None)


# Step3: build model

## First, construct two encoders

# Here we assume that we don't have SASRec Query Encoder Class
class SASRecQueryEncoder(torch.nn.Module):
    def __init__(self, embed_dim, max_seq_len, n_head, hidden_size, dropout, activation, n_layer, item_encoder) -> None:
        super().__init__()
        self.item_encoder = item_encoder
        self.position_emb = torch.nn.Embedding(max_seq_len, embed_dim)
        transformer_encoder = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_head,
            dim_feedforward=hidden_size,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=False
        )
        self.transformer_layer = torch.nn.TransformerEncoder(
            encoder_layer=transformer_encoder,
            num_layers=n_layer,
        )
        self.dropout = torch.nn.Dropout(p=dropout)
        self.gather_layer = module.SeqPoolingLayer(pooling_type='last')


    def forward(self, batch):
        # for SeqDataset and UserDataset,
        # user interaction list will be named as 'in_'+item_fields name
        user_hist = batch['in_'+'item_id']
        positions = torch.arange(user_hist.size(1), dtype=torch.long, device=user_hist.device)
        positions = positions.unsqueeze(0).expand_as(user_hist)
        position_embs = self.position_emb(positions)
        seq_embs = self.item_encoder(user_hist)

        mask4padding = user_hist==0
        L = user_hist.size(-1)
        attention_mask = ~torch.tril(torch.ones((L, L), dtype=torch.bool, device=user_hist.device))
        transformer_out = self.transformer_layer(
            src=self.dropout(seq_embs+position_embs),
            mask=attention_mask,
            src_key_padding_mask=mask4padding)

        # For SeqDataset, we have seqlen filed in batch
        return self.gather_layer(transformer_out, batch['seqlen'])


item_encoder = torch.nn.Embedding(trn.num_items, 64, 0)
query_encoder = SASRecQueryEncoder(
    embed_dim=64, max_seq_len=trn.max_seq_len,
    n_head=4, hidden_size=128, dropout=0.5, activation='gelu',
    n_layer=2, item_encoder=item_encoder
)

## Then, construct score function
sasrec_score_func = scorer.InnerProductScorer()


## Finally, contruct sampler and loss function
## in order to avoid sampling the padding item with index 0,
# so we set num_items as trn.num_items-1
sasrec_sampler = sampler.UniformSampler(num_items=trn.num_items-1, scorer_fn=score_func)
sasrec_loss_func = loss_func.BinaryCrossEntropyLoss()



SASRecModel = basemodel.BaseRetriever(
    item_encoder = item_encoder,
    query_encoder = query_encoder,
    scorer = sasrec_score_func,
    loss = sasrec_loss_func,
    sampler = sasrec_sampler
)


# Step4: Trainig:
## config: optimizer, learning_rate, neg_count, et al.
training_config = {
    'learner': 'Adam',
    'learning_rate': 0.001,
    'batch_size': 256,
    'epochs': 100,
    'gpu': [0,],
    'negative_count': 1,
    'eval_batch_size': 1024,
    'test_metrics': ['recall', 'precision', 'map', 'ndcg', 'mrr', 'hit'],
    'val_metrics': ['recall', 'ndcg'],
    'topk': 100,
    'cutoff': 10,
    'early_stop_mode': 'max',
    'early_stop_patience': 10,
}

SASRecModel.fit(config=training_config, train_data=trn, val_data=val)
SASRecModel.evaluate(test_data=tst)