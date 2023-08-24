# RecStudio


<p float="left">
<img src="https://img.shields.io/badge/python-v3.7+-blue">
<img src="https://img.shields.io/badge/pytorch-v1.9+-blue">
<!-- <img src="https://img.shields.io/badge/pytorch--lightning-v1.4.9-red"> -->
<img src="https://img.shields.io/badge/License-MIT-blue">
</p>
<p align="left">
  <img src="assets/recstudio_logo.png" alt="RecStudio logo" width="300">
  <br>
</p>

RecStudio is a unified, highly-modularized and recommendation-efficient recommendation library based on PyTorch. All the algorithms are
categorized as follows according to recommendation tasks.
- General Recommendation
- Sequential Recommendation
- Knowledge-based Recommendation
- Feature-based Recommendation
- Social Recommendation

## Description

### Model Structure

At the core of the library, all recommendation models are grouped into three base classes:

- `TowerFreeRecommender`: The most flexible base class, which enables any complex feature-interaction modeling.
- `ItemTowerRecommender`: Item encoders are separated from recommender, enabling fast ANN and model-based negative sampling.
- `TwoTowerRecommender`: The subclass of `ItemTowerRecommender`, where recommenders only consist of user encoder and item encoder.

### Dataset Structure

For the dataset structure, the datasets are divided into five categories:

|Dataset    |Application   | Examples  |
|-----------|-----------|----------|
|TripletDataset|Dataset for providing user-item-rating triplet|BPR, NCF, CML et al.|
|UserDataset|Dataset for AutoEncoder-based ItemTowerRecommender|MultiVAE, RecVAE, et al.|
|SeqDataset|Dataset for Sequential recommenders with Causal Prediction|GRU4Rec, SASRec, et al.|
|Seq2SeqDataset|Dataset for Sequential recommenders with Masked Prediction|Bert4Rec, et al.|
|ALSDataset|Dataset for recommenders optimized by alternating least square |WRMF, et al.|

In order to accelerate dataset processing, processed dataset are automatically cached for repeatable training shortly.


### Model Evaluation

Almost all common metrics used in recommender systems are implemented in RecStudio based on PyTorch, such as `NDCG`, `Recall`, `Precision`, et al. All metric functions have the same interface, being fully implemented with tensor operators. Therefore, the evaluation procedure can be moved to GPU, leading to a remarkable speedup of evaluation.


### ANNs & Sampler

In order to accelerate training and evaluation, RecStudio integrates various Approximate Nearest Neighbor
search (ANNs) and negative samplers. By building indexes with ANNs, the topk operator based on Euclidean distance, inner product and cosine similarity can be significantly accelerated. Negative samplers consist of static sampler and model-based samplers developed by RecStudio
team. Static samplers consist of `Uniform Sampler` and `Popularity Sampler`. The model-based samplers are based on either quantization of item vectors or importance resampling. Moreover, we also implement static sampling in the dataset, which enables us to generate negatives when loading data.

### Loss & Score

In RecStudio, loss functions are categorized into three types:
    - `FullScoreLoss`: Calculating scores on the whole items, such as `SoftmaxLoss`.
    - `PairwiseLoss`: Calculating scores on positive and negative items, such as `BPRLoss`,
    `BinaryCrossEntropyLoss`, et al.
    - `PointwiseLoss`: Calculating scores for a single (user, item) interaction, such as `HingeLoss`.

Score functions are used to model users' preference on items. Various common score functions are
implemented in RecStudio, such as `InnerProduct`, `EuclideanDistance`, `CosineDistance`, `MLPScorer`,
et al.

| Loss | Math Type | Sampling Distribution | Calculation Complexity | Sampling Complexity | Convergence Speed | Related Metrics |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Softmax | <!-- $-\log \frac{\exp f_{\theta}(c, {\color{red}k})}{\sum_{i=1}^{N} \exp f_{\theta}(c, i)}$ --> <img style="transform: translateY(0.1em);" src="https://latex.codecogs.com/svg.image?\small&space;-\log&space;\frac{\exp&space;f_{\theta}(c,&space;{\color{red}k})}{\sum_{i=1}^{N}&space;\exp&space;f_{\theta}(c,&space;i)}">  | No sampling | <!-- $O(N)$ --> <img style="transform: translateY(0.1em);" src="https://latex.codecogs.com/svg.image?\small&space;O(N)"> | - | very fast | NDCG |
| Sampled Softmax | <!-- $-\log \frac{\exp \left(f_{\theta}(c, {\color{red}k})-\log Q({\color{red}k} \mid c)\right)}{\sum_{i \in S \cup\{k\}} \exp \left(f_{\theta}(c, i)-\log Q(i \mid c)\right)}$ --> <img style="transform: translateY(0.1em);" src="https://latex.codecogs.com/svg.image?\small&space;-\log&space;\frac{\exp&space;\left(f_{\theta}(c,&space;{\color{red}k})-\log&space;Q({\color{red}k}&space;\mid&space;c)\right)}{\sum_{i&space;\in&space;S&space;\cup\{k\}}&space;\exp&space;\left(f_{\theta}(c,&space;i)-\log&space;Q(i&space;\mid&space;c)\right)}"> | No sampling | <!-- $O(N)$ --> <img style="transform: translateY(0.1em);" src="https://latex.codecogs.com/svg.image?\small&space;O(N)"> | - | fast | NDCG |
| BPR | <!-- $-\log \left(\sigma\left(f_{\theta}(c, {\color{red}k})-f_{\theta}(c, {\color{red}j})\right)\right)$ --> <img style="transform: translateY(0.1em);" src="https://latex.codecogs.com/svg.image?\small&space;-\log&space;\left(\sigma\left(f_{\theta}(c,&space;{\color{red}k})-f_{\theta}(c,&space;{\color{red}j})\right)\right)"> | Uniform sampling | <!-- $O(1)$ --> <img style="transform: translateY(0.1em);" src="https://latex.codecogs.com/svg.image?\small&space;O(1)"> | <!-- $O(1)$ --> <img style="transform: translateY(0.1em);" src="https://render.githubusercontent.com/render/math?math=O(1)"> | slow | AUC |
| WARP | <!-- $L\left(\left\lfloor \frac{Y-1}{N}\right\rfloor\right)\lvert 1-f_{\theta}(c, {\color{red}k})+f_{\theta}(c, {\color{red}j})\rvert_{+}$ --> <img style="transform: translateY(0.1em);" src="https://latex.codecogs.com/svg.image?\small&space;L\left(\left\lfloor&space;\frac{Y-1}{N}\right\rfloor\right)\lvert&space;1-f_{\theta}(c,&space;{\color{red}k})&plus;f_{\theta}(c,&space;{\color{red}j})\rvert_{&plus;}"> | Reject Sampling | <!-- $O(1)$ --> <img style="transform: translateY(0.1em);" src="https://latex.codecogs.com/svg.image?\small&space;O(1)"> | slower and slower | slow | Precision |
| InfoNCE | <!-- $-\log \frac{\exp \left(f_{\theta}(c, {\color{red}k})\right)}{\sum_{i \in S \cup\{k\}} \exp \left(f_{\theta}(c, i)\right)}$ --> <img style="transform: translateY(0.1em);" src="https://latex.codecogs.com/svg.image?\small&space;-\log&space;\frac{\exp&space;\left(f_{\theta}(c,&space;{\color{red}k})\right)}{\sum_{i&space;\in&space;S&space;\cup\{k\}}&space;\exp&space;\left(f_{\theta}(c,&space;i)\right)}"> | Popularity sampling | <!-- $O(\lvert S\rvert)$ --> <img style="transform: translateY(0.1em);" src="https://latex.codecogs.com/svg.image?\small&space;O(\lvert&space;S\rvert)"> | <!-- $O(1)$ --> <img style="transform: translateY(0.1em);" src="https://latex.codecogs.com/svg.image?\small&space;O(1)"> | fast | DCG |
| WRMF | <!-- $\sum_{j} {\color{red}w_{c j}}\left(f_{\theta}(c, j)-y(c, j)\right)^{2}$ --> <img style="transform: translateY(0.1em);" src="https://latex.codecogs.com/svg.image?\small&space;\sum_{j}&space;{\color{red}w_{c&space;j}}\left(f_{\theta}(c,&space;j)-y(c,&space;j)\right)^{2}"> | No sampling | <!-- $O(N)$ --> <img style="transform: translateY(0.1em);" src="https://latex.codecogs.com/svg.image?\small&space;O(N)"> | - | very fast | - |
| PRIS | <!-- $-\sum_{j \in S} \frac{\exp \left(f_{\theta}(c, {\color{red}j})-\log Q({\color{red}j} \mid c)\right)}{\sum_{{j^{\prime}} \in S} \exp \left(f_{\theta}\left(c, {j^{\prime}}\right)-\log Q\left({j^{\prime}} \mid c\right)\right)} \log \left(\sigma\left(f_{\theta}(c, {\color{red}k})-f_{\theta}(c, {\color{red}j})\right)\right)$ --> <img style="transform: translateY(0.1em);" src="https://latex.codecogs.com/svg.image?-\sum_{j&space;\in&space;S}&space;\frac{\exp&space;\left(f_{\theta}(c,&space;{\color{red}j})-\log&space;Q({\color{red}j}&space;\mid&space;c)\right)}{\sum_{{j^{\prime}}&space;\in&space;S}&space;\exp&space;\left(f_{\theta}\left(c,&space;{j^{\prime}}\right)-\log&space;Q\left({j^{\prime}}&space;\mid&space;c\right)\right)}&space;\log&space;\left(\sigma\left(f_{\theta}(c,&space;{\color{red}k})-f_{\theta}(c,&space;{\color{red}j})\right)\right)"> | Cluster sampling | <!-- $O(\lvert S\rvert)$ --> <img style="transform: translateY(0.1em);" src="https://latex.codecogs.com/svg.image?\small&space;O(\lvert&space;S\rvert)"> | <!-- $O(K)$ --> <img style="transform: translateY(0.1em);" src="https://latex.codecogs.com/svg.image?\small&space;O(K)"> | very fast | DCG |


<p align="center">
  <img src="assets/recstudio_framework.png" alt="RecStudio v0.2 Framework" width="600">
  <br>
  <b>Figure</b>: RecStudio Framework
</p>


## Features

- **General Dataset Structure** A unified dataset config based on atomic data files and automatic data cache
are supported in RecStudio.
- **Modular Model Structure** By organizing the whole recommender into different modules, loss functions, scoring
functions, samplers and ANNs, you can customize your model like building blocks.
- **GPU Acceleration** The whole operation from model training to model evaluation could be easily moved to
on GPUs and distributed GPUs for running.
- **Simple Model Categorization** RecStudio categorizes all the models based on the number of encoders,
which is easy to understand and use. The taxonomy can cover all models.
- **Simple and Complex Negative Samplers** RecStudio integrates static and model-based samplers with only tensor operators.


## Quick Start
By downloading the source code, you can run the provided script `run.py` for initial usage of RecStudio.

```bash
python run.py
```

The initial config will train and evaluate BPR model on MovieLens-100k(ml-100k) dataset.

Generally speaking, the simple example will take less than one minute with GPUs. And the output will be
like below:

```bash
[2023-08-24 10:51:41] INFO Log saved in /home/RecStudio/log/BPR/ml-100k/2023-08-24-10-51-41-738329.log.
[2023-08-24 10:51:41] INFO Global seed set to 2022
[2023-08-24 10:51:41] INFO dataset is read from /home/RecStudio/recstudio/dataset_demo/ml-100k.
[2023-08-24 10:51:42] INFO 
Dataset Info: 

=============================================================================
interaction information: 
field      user_id    item_id    rating     timestamp  
type       token      token      float      float      
##         944        1575       -          -          
=============================================================================
user information: 
field      user_id    age        gender     occupation zip_code   
type       token      token      token      token      token      
##         944        62         3          22         795        
=============================================================================
item information: 
field      item_id    
type       token      
##         1575       
=============================================================================
Total Interactions: 82520
Sparsity: 0.944404
=============================================================================
timestamp=StandardScaler()
[2023-08-24 10:51:42] INFO 
Model Config: 

data:
	binarized_rating_thres=None
	fm_eval=False
	neg_count=0
	sampler=None
	shuffle=True
	split_mode=user_entry
	split_ratio=[0.8, 0.1, 0.1]
	fmeval=False
	binaried_rating_thres=0.0
eval:
	batch_size=20
	cutoff=[5, 10, 20]
	val_metrics=['ndcg', 'recall']
	val_n_epoch=1
	test_metrics=['ndcg', 'recall', 'precision', 'map', 'mrr', 'hit']
	topk=100
	save_path=./saved/
model:
	embed_dim=64
	item_bias=False
train:
	accelerator=gpu
	ann=None
	batch_size=512
	early_stop_mode=max
	early_stop_patience=10
	epochs=1000
	gpu=1
	grad_clip_norm=None
	init_method=xavier_normal
	item_batch_size=1024
	learner=adam
	learning_rate=0.001
	num_threads=10
	sampling_method=none
	sampler=uniform
	negative_count=1
	excluding_hist=False
	scheduler=None
	seed=2022
	weight_decay=0.0
	tensorboard_path=None
[2023-08-24 10:51:42] INFO Tensorboard log saved in ./tensorboard/BPR/ml-100k/2023-08-24-10-51-41-738329.
[2023-08-24 10:51:42] INFO The default fields to be used is set as [user_id, item_id, rating]. If more fields are needed, please use `self._set_data_field()` to reset.
[2023-08-24 10:51:42] INFO save_dir:./saved/
[2023-08-24 10:51:42] INFO BPR(
  (score_func): InnerProductScorer()
  (loss_fn): BPRLoss()
  (item_encoder): Embedding(1575, 64, padding_idx=0)
  (query_encoder): Embedding(944, 64, padding_idx=0)
  (sampler): UniformSampler()
)
[2023-08-24 10:51:42] INFO GPU id [8] are selected.
[2023-08-24 10:51:45] INFO Training: Epoch=  0 [ndcg@5=0.0111 recall@5=0.0044 train_loss_0=0.6931]
[2023-08-24 10:51:45] INFO Train time: 0.88524s. Valid time: 0.18036s. GPU RAM: 0.03/10.76 GB
[2023-08-24 10:51:45] INFO ndcg@5 improved. Best value: 0.0111
[2023-08-24 10:51:45] INFO Best model checkpoint saved in ./saved/BPR/ml-100k/2023-08-24-10-51-41-738329.ckpt.
...
[2023-08-24 10:52:08] INFO Training: Epoch= 34 [ndcg@5=0.1802 recall@5=0.1260 train_loss_0=0.1901]
[2023-08-24 10:52:08] INFO Train time: 0.41784s. Valid time: 0.32394s. GPU RAM: 0.03/10.76 GB
[2023-08-24 10:52:08] INFO Early stopped. Since the metric ndcg@5 haven't been improved for 10 epochs.
[2023-08-24 10:52:08] INFO The best score of ndcg@5 is 0.1807 on epoch 24
[2023-08-24 10:52:08] INFO Best model checkpoint saved in ./saved/BPR/ml-100k/2023-08-24-10-51-41-738329.ckpt.
[2023-08-24 10:52:08] INFO Testing:  [ndcg@5=0.2389 recall@5=0.1550 precision@5=0.1885 map@5=0.1629 mrr@5=0.3845 hit@5=0.5705 ndcg@10=0.2442 recall@10=0.2391 precision@10=0.1498 map@10=0.1447 mrr@10=0.4021 hit@10=0.6999 ndcg@20=0.2701 recall@20=0.3530 precision@20=0.1170 map@20=0.1429 mrr@20=0.4109 hit@20=0.8240]
```

If you want to change models or datasets, command line is ready for you.
```bash
python run.py -m=NCF -d=ml-1m
```

- Supported command line arguments:

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
  |--sampler|str|sampler name|uniform|['uniform','popularity','midx_uni','midx_pop','cluster_uni','cluster_pop']|
  |--negative_count|int|number of negative samples|1|positive integer|

- For `TwoTowerRecommender`, some extra args are supported based on `ItemTowerRecommender`:

  |args|type|description|default|optional|
  |---|---|---|---|---|
  |--split_mode|str|split methods for the dataset|user_entry|['user','entry','user_entry']|

Here are some details of some unclear arguments.
  > 1. `mode`: in `light` mode and `detail` mode, the output will displayed on the terminal, while
  the latter provide more detailed info. `tune` mode will use Neural Network Intelligence(NNI) to show
  a beautiful visual interface. You can run `tune.sh` with a config file like `config.yaml`. For
  more details about NNI, please refer to [NNI Documentation](https://nni.readthedocs.io/zh/stable/).
  > 2. `sampler`: `uniform` stands for UniformSampler is used. `popularity` stands for sampling according
  to the item popularity (more popular items are sampled with higher probabilities). `midx_uni`,`midx_pop`
  are `midx` dynamic sampler, please refer to [FastVAE](https://arxiv.org/abs/2109.05773) for more details.
  `cluster_uni`,`cluster_pop` are `cluster` dynamic sampler, please refer to
  [PRIS](https://dl.acm.org/doi/10.1145/3366423.3380187) for more details.
  > 3. `split_mode`: `user` means splitting all users into train/valid/test datasets, users in
  those datasets are disjoint. `entry` means spliting all the interactions in those three dataset.
  `user_entry` means spliting interaction of each user into three parts.


Also, you can install RecStudio from PyPi:

```bash
pip install recstudio
```

For basic usage like below:

```python
import recstudio
recstudio.run(model="BPR", data_dir="./datasets/", dataset='ml-100k')
```

For more detailed information, please refer to our documentation http://recstudio.org.cn/docs/.

## Automatic Hyper-parameter Tuning
RecStudio integrates with NNI module for tuning the hype-parameters automatically. For easy usage,
you can run the following command in bash:
```bash
nnictl create --config ./nni-experiments/config/bpr.yaml --port 2023
```
Configure `nni-experiments/config/bpr.yaml`  and `nni-experiments/search_space/bpr.yaml` for personal usage.

For more detailed information about NNI, please refer to
[NNI Documentation](https://nni.readthedocs.io/zh/stable/).


## Contributing
Please let us know if you encounter a bug or have any suggestions by
[submitting an issue](https://github.com/ustcml/RecStudio/issues).

We welcome all contributions from bug fixes to new features and extensions.

We expect all contributions firstly discussed in the issue tracker and then going through PRs.


## The Team
RecStudio is developed and maintained by USTC BigData Lab.

|User|Contributions|
|---|---|
|@[DefuLian](https://github.com/DefuLian)|Framework design and construction|
|@[AngusHuang17](https://github.com/AngusHuang17)|Sequential model, docs, bugs fixing|
|@[Xiuchen519](https://github.com/Xiuchen519)|Knowledge-based model, bugs fixing|
|@[JennahF](https://github.com/JennahF)|NCF,CML,logisticMF models|
|@[HERECJ](https://github.com/HERECJ)|AutoEncoder models|
|@[BinbinJin](https://github.com/BinbinJin)|IRGAN model|
|@[pepsi2222](https://github.com/pepsi2222)|Ranking models|
|@[echobelbo](https://github.com/echobelbo)|Docs|
|@[jinbaobaojhr](https://github.com/jinbaobaojhr)|Docs|

## License
RecStudio uses [MIT License](./LICENSE).
