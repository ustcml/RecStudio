## Quick Started


### Source Code
By downloading the source code, you can run th provided script `run.py` for initial usage of RecStudio.

```bash
python run.py
```

The initial config will train and evaluate BPR model on MovieLens-100k(ml-100k) dataset.

Generally speaking, the simple example will take less than one minute with GPUs. And the output will be
like below:

```bash
[2022-04-11 14:30:29] INFO (faiss.loader/MainThread) Loading faiss with AVX2 support.
[2022-04-11 14:30:29] INFO (faiss.loader/MainThread) Loading faiss.
[2022-04-11 14:30:29] INFO (faiss.loader/MainThread) Successfully loaded faiss.
[2022-04-11 14:30:30] INFO (pytorch_lightning.utilities.seed/MainThread) Global seed set to 42
[2022-04-11 14:30:30] INFO (pytorch_lightning/MainThread) learning_rate=0.001
weight_decay=0
learner=adam
scheduler=None
epochs=100
batch_size=2048
num_workers=0
gpu=None
ann=None
sampler=None
negative_count=1
dataset_sampling_count=None
embed_dim=64
item_bias=False
eval_batch_size=20
split_ratio=[0.8, 0.1, 0.1]
test_metrics=['recall', 'precision', 'map', 'ndcg', 'mrr', 'hit']
val_metrics=['recall', 'ndcg']
topk=100
cutoff=10
early_stop_mode=max
split_mode=user_entry
shuffle=True
use_fields=['user_id', 'item_id', 'rating']
[2022-04-11 14:30:30] INFO (pytorch_lightning/MainThread) save_dir:/home/RecStudio/
[2022-04-11 14:30:30] INFO (pytorch_lightning.utilities.distributed/MainThread) GPU available: True, used: False
[2022-04-11 14:30:30] INFO (pytorch_lightning.utilities.distributed/MainThread) TPU available: False, using: 0 TPU cores
[2022-04-11 14:30:30] INFO (pytorch_lightning.utilities.distributed/MainThread) IPU available: False, using: 0 IPUs
[2022-04-11 14:30:30] INFO (pytorch_lightning.utilities.distributed/MainThread) The following callbacks returned in `LightningModule.configure_callbacks` will override existing callbacks passed to Trainer: ModelCheckpoint
[2022-04-11 14:30:30] INFO (pytorch_lightning.core.lightning/MainThread) 
  | Name         | Type               | Params
----------------------------------------------------
0 | loss_fn      | BPRLoss            | 0     
1 | score_func   | InnerProductScorer | 0     
2 | item_encoder | Embedding          | 107 K 
3 | sampler      | UniformSampler     | 0     
4 | user_encoder | Embedding          | 60.4 K
----------------------------------------------------
168 K     Trainable params
0         Non-trainable params
168 K     Total params
0.673     Total estimated model params size (MB)
[2022-04-11 14:30:30] INFO (pytorch_lightning.callbacks.early_stopping/MainThread) Metric recall@10 improved. New best score: 0.007
[2022-04-11 14:30:30] INFO (pytorch_lightning/MainThread) Training: Epoch=  0 [recall@10=0.0074 ndcg@10=0.0129 train_loss=0.6932]
[2022-04-11 14:30:31] INFO (pytorch_lightning.callbacks.early_stopping/MainThread) Metric recall@10 improved by 0.006 >= min_delta = 0.0. New best score: 0.014
[2022-04-11 14:30:31] INFO (pytorch_lightning/MainThread) Training: Epoch=  1 [recall@10=0.0135 ndcg@10=0.0251 train_loss=0.6915]
[2022-04-11 14:30:32] INFO (pytorch_lightning.callbacks.early_stopping/MainThread) Metric recall@10 improved by 0.038 >= min_delta = 0.0. New best score: 0.051
...
[2022-04-11 14:31:26] INFO (pytorch_lightning/MainThread) Training: Epoch= 75 [recall@10=0.2074 ndcg@10=0.2942 train_loss=0.1909]
[2022-04-11 14:31:26] INFO (pytorch_lightning.callbacks.early_stopping/MainThread) Monitored metric recall@10 did not improve in the last 10 records. Best score: 0.211. Signaling Trainer to stop.
[2022-04-11 14:31:26] INFO (pytorch_lightning/MainThread) Training: Epoch= 76 [recall@10=0.2073 ndcg@10=0.2949 train_loss=0.1899]
[2022-04-11 14:31:26] INFO (pytorch_lightning.utilities.distributed/MainThread) The following callbacks returned in `LightningModule.configure_callbacks` will override existing callbacks passed to Trainer: EarlyStopping, ModelCheckpoint
[2022-04-11 14:31:27] INFO (pytorch_lightning/MainThread) Testing:  [recall@10=0.2439 precision@10=0.1893 map@10=0.5762 ndcg@10=0.3718 mrr@10=0.4487 hit@10=0.7815]
```

If you want to change models or datasets, command line is ready for you.
```bash
python run.py -m=NCF -d=ml-1m 
```

- Supported commandline arguments:

  |args|type|description|default|optional|
  |---|---|---|---|---|
  |-m,--model| str|model name|BPR|all the models in RecStudio|
  |-d,--dataset|str|dataset name|ml-100k|all the datasets supported by RecStudio|
  |--data_dir|str|dataset folder|datasets|folders that could be read by RecStudio|
  |mode|str|training mode|light|['light','detail','tune']|
  |--learning_rate|float|learning rate|0.001||
  |--learner|str|optimizer name|adam|['adam','sgd','adasgd','rmsprop','sparse_adam']|
  |--weight_decay|float|weight decay for optimizer|0||
  |--epochs|int|training epoch|20,50||
  |--batch_size|int|the size of mini batch in training|2048||
  |--eval_batch_size|int|the size of mini batch in evaluation|128||
  |--embed_dim|int|the output size of embedding layers|64||


- For `ItemTowerRecommender`, some extra args are supported:

  |args|type|description|default|optional|
  |---|---|---|---|---|
  |--sampler|str|sampler name|uniform|['uniform','maskedpos_uniform','popularity','midx_uni','midx_pop','cluster_uni','cluster_pop']|
  |--negative_count|int|number of negative samples|1|positive integer|

- For `TwoTowerRecommender`, some extra args are supported based on `ItemTowerRecommender`:

  |args|type|description|default|optional|
  |---|---|---|---|---|
  |--split_mode|str|split methods for the dataset|user_entry|['user','entry','user_entry']|

Here are some details of some unclear arguments.
  > 1. `mode` ：in `light` mode and `detail` mode, the output will displayed on the terminal, while 
  the latter provide more detailed info. `tune` mode will use Neural Network Intelligence(NNI) to show 
  a beautiful visual interface. You can run like `tune.sh` with a config file like `config.yaml`. For
  more details about NNI, please refer to [NNI Documentation](https://nni.readthedocs.io/zh/stable/).
  > 2. `sampler`: `uniform` stands for UniformSampler is used. `popularity` stands for sampling according
  to the item popularity (more popular items are sampled with higher probablities). `midx_uni`,`midx_pop`
  are `midx` dynamic sampler, please refer to [FastVAE](https://arxiv.org/abs/2109.05773) for more details.
  `cluster_uni`,`cluster_pop` are `cluster` dynamic sampler, please refer to
  [PRIS](https://dl.acm.org/doi/10.1145/3366423.3380187) for more details.
  > 3. `split_mode`: `user` means spliting all users into train/valid/test datasets, users in
  those datasets are disjoint. `entry` means spliting all the interactions in those three dataset.
  `user_entry` means spliting interaction of each user into three parts. 


### PyPi
If you install RecStudio from PyPi, you could easily write a script to run it like below:

```python
import recstudio
recstudio.run(model='BPR', dataset='ml-100k')
```

Here we provide a model list that we have implemented in RecStudio.

| Model | Type | Task | Paper Link | 
| --- | --- | --- | --- |
| Multi_DAE | ItemTowerRecommender | General Recommendation| [Paper](https://dl.acm.org/doi/pdf/10.1145/3178876.3186150) |
| Multi_VAE | ItemTowerRecommender | General Recommendation| [Paper](https://dl.acm.org/doi/pdf/10.1145/3178876.3186150) |
| Rec_VAE | ItemTowerRecommender | General Recommendation| [Paper](https://dl.acm.org/doi/abs/10.1145/3336191.3371831) |
| FM | TowerFreeRecommender | Feature-based Recommendation | [Paper](https://ieeexplore.ieee.org/document/5694074) |
| CKFG | TwoTowerRecommender | Knowledge-based Recommendation | [Paper](https://arxiv.org/abs/1803.06540) |
| CKE | TwoTowerRecommender | Knowledge-based Recommendation | [Paper](https://dl.acm.org/doi/10.1145/2939672.2939673) |
| KGAT | TwoTowerRecommender | Knowledge-based Recommendation | [Paper](https://dl.acm.org/doi/10.1145/3292500.3330989) |
| KGCN | TwoTowerRecommender | Knowledge-based Recommendation | [Paper](https://dl.acm.org/doi/10.1145/3308558.3313417) |
| KGNNLS | TwoTowerRecommender | Knowledge-based Recommendation | [Paper](https://doi.org/10.1145/3292500.3330836) |
| KTUP | TwoTowerRecommender | Knowledge-based Recommendation | [Paper](https://doi.org/10.1145/3308558.3313705) |
| MKR | TwoTowerRecommender | Knowledge-based Recommendation | [Paper](https://doi.org/10.1145/3308558.3313411) |
| RippleNet | TwoTowerRecommender | Knowledge-based Recommendation | [Paper](https://doi.org/10.1145/3269206.3271739) |
| BPR | TwoTowerRecommender | General Recommendation | [Paper](https://dl.acm.org/doi/10.5555/1795114.1795167) |
| CML | TwoTowerRecommender | General Recommendation | [Paper]() |
| EASE | TwoTowerRecommender | General Recommendation | [Paper](https://dl.acm.org/doi/10.1145/3308558.3313710) |
| IRGAN | TwoTowerRecommender | General Recommendation | [Paper](https://dl.acm.org/doi/10.1145/3077136.3080786) |
| ItemKNN | TwoTowerRecommender | General Recommendation | [Paper](https://dl.acm.org/doi/10.1145/963770.963776) |
| LogisticMF | TwoTowerRecommender | General Recommendation | [Paper](http://web.stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf) |
| NCF | TwoTowerRecommender | General Recommendation | [Paper](https://dl.acm.org/doi/abs/10.1145/3038912.3052569) |
| SLIM | TwoTowerRecommender | General Recommendation | [Paper](https://dl.acm.org/doi/10.1109/ICDM.2011.134) |
| WRMF | TwoTowerRecommender | General Recommendation | [Paper]() | 
| Caser | TwoTowerRecommender | Sequential Recommendation | [Paper](https://dl.acm.org/doi/abs/10.1145/3159652.3159656) |
| DIN | TwoTowerRecommender | Sequential Recommendation | [Paper](https://dl.acm.org/doi/10.1145/3219819.3219823) |
| FPMC | TwoTowerRecommender | Sequential Recommendation | [Paper](https://dl.acm.org/doi/10.1145/1772690.1772773) |
| GRU4Rec | ItemTowerRecommender | Sequential Recommendation | [Paper](https://arxiv.org/abs/1511.06939) |
| HGN | TwoTowerRecommender | Sequential Recommendation | [Paper](https://dl.acm.org/doi/abs/10.1145/3292500.3330984) |
| NARM | ItemTowerRecommender | Sequential Recommendation | [Paper](https://dl.acm.org/doi/10.1145/3132847.3132926) |
| NPE | TwoTowerRecommender | Sequential Recommendation | [Paper](https://www.ijcai.org/proceedings/2018/0219.pdf) |
| SASRec | ItemTowerRecommender | Sequential Recommendation | [Paper](https://ieeexplore.ieee.org/document/8594844/) |
| STAMP | ItemTowerRecommender | Sequential Recommendation | [Paper](https://dl.acm.org/doi/abs/10.1145/3219819.3219950) |
| TransRec | TwoTowerRecommender | Sequential Recommendation | [Paper](https://dl.acm.org/doi/10.1145/3109859.3109882) |



For the dataset, we put several common dataset configuration files in RecStudio, which you can use
directly. And if you want to use new dataset, you are required to create a dataset folder, where a
sub folder is created with the dataset name and configuration file and atomic files should be put in
it. The file structure should be like as below if the dataset is yelp:

```
dataset_folder
|
|--yelp
   |
   |---yelp.yaml    # config file
   |---yelp.inter   # interaction atomic file
   |---yelp.item    # item information file 
   |---yelp.user    # user information file
   |---yelp.kg      # knowledge graph file
   |---yelp.link    # links between entities in knowledge graph and item
```

> - `yelp.inter`: with columns ['user_id', 'item_id', 'rating', 'timestamp'], save the interactions
> - `yelp.item`: item features ['item_id', 'feature1', 'feature2',...]
> - `yelp.user`: user features ['user_id', 'feature1', 'feature2', ...]
> - `yelp.kg`: knowledge graph file, ['head_id', 'relation_id', 'tail_id']
> - `yelp.link`: links between entities in knowledge graph and item, ['item_id', 'entity_id']
> You can only provide the file that you need to use, usually `.inter` is required. 


For the dataset config, you could refer to the file as below:

```yaml
user_id_field: &u user_id:token # user id field name, do not change
item_id_field: &i item_id:token # item id field name, do not change
rating_field: &r rating:float # rating field name, do not change
time_field: &t timestamp:float # time stamp of interactions. If the time is not unix timestamp, please set as `timestamp:str`
inter_feat_field: [*u, *i, *r, *t]  # fields in interaction feature fields
user_feat_name: ~ #[ml-100k.user] # file to save user feature
user_feat_field: [[*u, age:token, gender:token, occupation:token]]  # fields name in user feature file
item_feat_name: ~ #[ml-100k.item] # file to save item feature
item_feat_field: [[*i, movie_title:token_seq, release_year:token, class:token_seq]] # fields name in item feature file
field_separator: "\t" # separator in all file
seq_separator: " "  # separator in string sequence, like the `class` feature in movielens
min_user_inter: 0   # filter out users whose interactions numbers < min_user_inter
min_item_inter: 0   # filter out items whose interactions numbers < min_item_inter
field_max_len: ~    # max length of the sequence field. 
rating_threshold: ~   # ratings below the threshold will be set as 0. If you set the args when using models except FM, please set `drop_low_rating` as `True`
drop_low_rating: ~    # drop interactions with rating below `rating_threshold`
max_seq_len: 20       # max length of the interactions for sequential recommendation

# network feature, including social network and knowledge graph, the first two fields are remapped the corresponding features
network_feat_name: ~ #[[social.txt], [ml-100k.kg, ml-100k.link]]
mapped_feat_field: [*u, *i]
network_feat_field: [[[source_id:token, target_id:token]], [[head_id:token, tail_id:token, relation_id:token], [*i, entity_id:token]]]

save_cache: True # whether to save processed dataset to cache.

```

And then you shoud assign data_dir to run RecStudio:

```python
recstudio.run(model='BPR', data_dir='dataset_folder', dataset='yelp')
```


If you want to modify the hyper parameters of the model and some training parameters, a config file
should be provided, here we give an example:

```yaml
# for training all models
learning_rate: 0.001
weight_decay: 0
learner: adam   # optimizer, optional: [adam, sgd, adasgd, rmsprop, sparse_adam]
scheduler: ~    # learning rate scheduler, optional: []
epochs: 100
batch_size: 2048
num_workers: 0 # please do not use this parameter, slowing down the training process
gpu: ~  # used gpu number, should be number if provided. If multiple gpu numbers are provided, data parallel will be used. e.g. [0, 1]

# used for training tower-based model
#ann: {index: 'IVFx,Flat', parameter: ~}  ## 1 HNSWx,Flat; 2 Flat; 3 IVFx,Flat ## {nprobe: 1}  {efSearch: 1}
ann: ~
sampler: ~  # which sampler to use, optional: [uniform, maskedpos_uniform, popularity, midx_uni, midx_pop, cluster_uni, cluster_pop]
negative_count: 1
dataset_sampler: maskedpos_uniform
dataset_sampling_count: ~ # sampling in dataset loader, optional: [uniform, maskedpos_uniform, popularity,]
embed_dim: 64
item_bias: False

# used for evaluating tower-based model
eval_batch_size: 128
split_ratio: [0.8,0.1,0.1]
test_metrics: [recall, precision, map, ndcg, mrr, hit]
val_metrics: [recall, ndcg]
topk: 100
cutoff: 10
early_stop_mode: max
```
