# RecStudio

<p align="left">
  <img src="assets/recstudio_logo.png" alt="RecStudio logo" width="300">
  <br>
</p>


RecStudio 是一个基于 PyTorch 实现的，高效、统一、全面的推荐系统算法库。
我们根据任务的不同将推荐系统算法分为以下四类：

- General Recommendation
- Sequential Recommendation
- Knowledge-based Recommendation
- Social-Network-based Recommendation

在算法库的核心层，我们将所有的模型结构按照“塔”的数量分为 `TowerFree`, `ItemTower`, `TwoTower` 三大基类，分别表示：

- 没有显式的用户/物品塔
- 只有物品塔
- 具有用户和物品塔

在数据集结构方面，
我们根据任务的不同将数据集分为 `TripletDataset`, `UserDataset`, `SeqDataset`, `Seq2SeqDataset`,`ALSDataset` 
五大类，五类数据集应用如下：

|Dataset    |适用情形   | 使用模型举例  |
|-----------|-----------|----------|
|TripletDataset|矩阵分解系列模型|BPR, NCF 等|
|UserDataset|自动编码器系列模型|MultiVAE, RecVAE 等|
|SeqDataset|序列化推荐系列模型|GRU4Rec, SASRec等|
|Seq2SeqDataset|序列化推荐部分基于 MLM 的模型|Bert4Rec 等|
|ALSDataset|交替优化系列模型|CML 等|

且我们提供一系列基于原子文件的数据集文件，并附有处理好的数据集对象文件可直接读入。
用户可以简单快速的加载我们提供的数据集进行模型的训练和评估。

在模型评估上，我们基于 PyTorch 实现了统一的指标计算函数（如 `NDCG`, `Recall`, `Precsion` 等），所有函数计算均可在 GPU 上进行加速计算。

为了加速训练过程，我们提供了基于 faiss 的近似近邻搜索接口和丰富的采样器。近似近邻搜索在进行预测时可以为topk操作
提供显著的加速效果。采样器不仅包含常见的均匀采样和基于流行度的采样两种静态的采样方法，还包括我们自主研发的动态
的基于量化的采样方法。另外，为了方便使用，我们提供了数据集加载时采样和训练过程中采样两种采样方式，用户可以根据
需求简单修改配置文件即可使用两种方式。

在损失函数上，我们将常用的损失函数总结为三大类：`FullScoreLoss`, `PairwiseLoss`, `PointwiseLoss`。这三类
损失函数能够覆盖我们常用的 `SoftmaxLoss`, `BPRLoss` 等。打分函数上我们提供了常用的内积打分函数、欧氏距离打
分函数、余弦距离打分函数、MLP打分函数等。用户可以通过模块化的拼接去尝试不同的损失函数和打分函数以比较得出效果
更好的模型。

RecStudio 的总体架构如下：

<p align="center">
  <img src="assets/recstudio_framework.png" alt="RecStudio v0.2 架构" width="600">
  <br>
  <b>图片</b>: RecStudio 总体架构
</p>


## 特色

- **通用的数据集结构** 我们提供了基于原子文件配置以及在线 cache 的通用数据集配置。
- **模块化的模型配置** 我们将模型结构、损失函数、打分函数、采样器模块、近似近邻搜索模块进行模块化拆分，以达到积木化模型实现的效果。
- **高效的GPU加速技术** 框架中模型从训练到评估所有过程均基于 PyTorch 实现，可以非常简单的利用 GPU 进行加速。
- **易用的模型结构分类** 我们的基于塔的数量的模型分类非常简单易懂，且具有很高的通用性。
- **丰富的负样本采样器** 我们提供了静态和动态两大类型的负采样模块，且均可通过 GPU 计算得到很高性能的表现。


## 快速使用
通过从 GitHub 下载 RecStudio 的源码，可以使用我们提供的 `run.py` 的脚本文件快速运行：

```bash
python run.py
```

这个简单的默认配置将会在 MovieLens-100k 上进行 BPR 模型的训练和评估。

一般来说，在使用 GPU 的情况下，这个例子将运行不到 1 分钟的时间即可完成，将会得到如下类似的输出：

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

如果需要尝试其他的模型或采用不同的数据集训练，可以通过命令行简单设置：
```bash
python run.py -m=NCF -d=ml-1m
```

- 命令行还支持修改的参数有：

  |参数|数据类型|描述|默认值|可选值|
  |---|---|---|---|---|
  |-m,--model| str| 模型名称|BPR|RecStudio支持的模型|
  |-d,--dataset|str|数据集名称|ml-100k|RecStudio支持的模型或用户处理好的原子文件|
  |--data_dir|str|数据集文件夹|datasets|RecStudio可以访问到的文件夹|
  |mode|str|训练模式，分为简单 (light)、详细 (detail)和调参 (tune)|light|['light', 'detail', 'tune']|
  |--learning_rate|float|学习率|0.001||
  |--learner|str|优化器|adam|['adam', 'sgd', 'adasgd', 'rmsprop', 'sparse_adam']|
  |--weight_decay|float|权重衰减率|0||
  |--epochs|int|训练迭代轮数|20,50||
  |--batch_size|int|训练时 mini batch 的大小|2048||
  |--eval_batch_size|int|评估时 mini batch 的大小|128||
  |--embed_dim|int|embedding 层的输出维度|64||

- 对于单塔模型（`ItemTowerRecommender`），有额外的命令行参数：

  |参数|数据类型|描述|默认值|可选值|
  |---|---|---|---|---|
  |--sampler|str|负采样器名称|uniform|['uniform', 'popularity', 'midx_uni', 'midx_pop', 'cluster_uni', 'cluster_pop']|
  |--negative_count|int|采样的负样本个数|1|正整数|

- 对于双塔模型（`TwoTowerRecommender`），相比于单塔模型再添加一个额外命令行参数：

  |参数|数据类型|描述|默认值|可选值|
  |---|---|---|---|---|
  |--split_mode|str|数据集划分方式|user_entry|['user', 'entry', 'user_entry']|

  > 1. `mode` 参数说明：light 模式和 tune 模式均适用于命令行，区别在于显示的信息详略不同；tune 模式将使用
  Neural Network Intelligence（NNI）框架。详细使用规则请参考 [NNI文档](https://nni.readthedocs.io/zh/stable/).
  > 2. `sampler` 参数说明：`uniform` 表示使用均匀采样；`popularity` 表示基于商品流行度采样；`midx_uni`, `midx_pop` 为 `midx` 系列动态采样器，详细说明参见论文 [FastVAE](https://arxiv.org/abs/2109.05773)。`cluster_uni`, `cluster_pop`
  为 `cluster` 系列动态采样器，详细说明参见论文 [PRIS](https://dl.acm.org/doi/10.1145/3366423.3380187)。
  > 3. `split_mode` 参数说明：`user` 模式表示按照用户对数据集进行划分，训练集、验证集、测试集中用户均不同；
  `entry` 模式表示按照交互记录划分，随机将所有交互记录按照比例划分成训练、验证、测试集；`user_entry` 模式表
  示对每个用户，按一定比例划分其交互记录到训练、验证、测试集。


除了通过命令行设置参数外，我们更加推荐使用 YAML 配置文件来设置参数。

## 自动调参
RecStudio 集成了 NNI 的自动调参。你可以通过在 bash 中输入如下命令快速开始：
```bash
nnictl create --config ./nni-experiments/config/bpr.yaml --port 2023
```
个性化的使用请配置 `nni-experiments/config/bpr.yaml`  and `nni-experiments/search_space/bpr.yaml`。

## 贡献

如果您遇到错误或有任何建议，请通过 [Issue](https://github.com/ustcml/RecStudio/issues) 进行反馈。

我们欢迎关于修复错误、添加新特性的任何贡献。

如果想贡献代码，请先在 issue 中提出问题，然后再提 PR。


## 项目团队
RecStudio 由中国科学技术大学的同学和老师进行开发和维护。
|用户|贡献|
|---|---|
|@[DefuLian](https://github.com/DefuLian)|基础框架设计和搭建|
|@[AngusHuang17](https://github.com/AngusHuang17)|序列化推荐模型，文档编写，bug 修正|
|@[Xiuchen519](https://github.com/Xiuchen519)|基于知识图谱的模型，bug 修正|
|@[JennahF](https://github.com/JennahF)|NCF, CML, logisticMF 等矩阵分解模型|
|@[HERECJ](https://github.com/HERECJ)|自动编码器模型|
|@[BinbinJin](https://github.com/BinbinJin)|IRGAN 模型|
|@[pepsi2222](https://github.com/pepsi2222)|排序模型|
|@[echobelbo](https://github.com/echobelbo)|文档编写|
|@[jinbaobaojhr](https://github.com/jinbaobaojhr)|文档编写|


## 免责声明
RecStudio 基于 [MIT License](./LICENSE) 进行开发，本项目的所有数据和代码只能被用于学术目的。