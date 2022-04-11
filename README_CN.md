# RecStudio

<p align="left">
  <img src="assets/recstudio_logo.png" alt="RecStudio logo" width="300">
  <br>
</p>


RecStudio 是一个基于PyTorch实现的，高效、统一、全面的推荐系统算法库。
我们根据任务的不同将推荐系统算法分为以下四类：

- General Recommendation
- Sequential Recommendation
- Knowledge-based Recommendation
- Social-Network-based Recommendation

在算法库的核心层，我们将所有的模型结构按照“塔”的数量分为 `TowerFree`, `ItemTower`, `TwoTower`三大基类，
其中分别表示为：

- 没有显式的用户/物品塔
- 只有物品塔
- 具有用户和物品塔

在数据集结构方面，
我们根据任务的不同将数据集分为 `MFDataset`, `AEDataset`, `SeqDataset`, `Seq2SeqDataset`,`ALSDataset`
五大类，五类数据集应用如下：

|Dataset    |适用情形   | 使用模型举例  |
|-----------|-----------|----------|
|MFDataset|矩阵分解系列模型|BPR,NCF 等|
|AEDataset|自动编码器系列模型|MultiVAE,RecVAE等|
|SeqDataset|序列化推荐系列模型|GRU4Rec,SASRec等|
|Seq2SeqDataset|序列化推荐部分基于MLM的模型|Bert4Rec等|
|ALSDataset|交替优化系列模型|CML等|

且我们提供一系列基于原子文件的数据集文件，并附有处理好的数据集对象文件可直接读入。
用户可以简单快速的加载我们提供的数据集进行模型的训练和评估。

在模型评估上，我们基于PyTorch实现了统一的指标计算函数（如 `NDCG`, `Recall`, `Precsion` 等），所有函数计算
均可在GPU上进行加速计算。

为了加速训练过程，我们提供了基于faiss的近似近邻搜索接口和丰富的采样器。近似近邻搜索在进行预测时可以为topk操作
提供显著的加速效果。采样器不仅包含常见的均匀采样和基于流行度的采样两种静态的采样方法，还包括我们自主研发的动态
的基于量化的采样方法。另外，为了方便使用，我们提供了数据集加载时采样和训练过程中采样两种采样方式，用户可以根据
需求简单修改配置文件即可使用两种方式。

在损失函数上，我们将常用的损失函数总结为三大类：`FullScoreLoss`, `PairwiseLoss`, `PointwiseLoss`。这三类
损失函数能够覆盖我们常用的 `SoftmaxLoss`, `BPRLoss` 等。打分函数上我们提供了常用的内积打分函数、欧氏距离打
分函数、余弦距离打分函数、MLP打分函数等。用户可以通过模块化的拼接去尝试不同的损失函数和打分函数以比较得出效果
更好的模型。

RecStudio 的总体架构如下：

<p align="center">
  <img src="assets/recstudio_framework.png" alt="RecStudio v0.1 架构" width="600">
  <br>
  <b>图片</b>: RecStudio 总体架构
</p>


## 特色

- **通用的数据集结构** 我们提供了基于原子文件配置以及在线cache的通用数据集配置。
- **模块化的模型配置** 我们将模型结构、损失函数、打分函数、采样器模块、近似近邻搜索模块进行模块化拆分，以达到
积木化模型实现的效果。
- **高效的GPU加速技术** 框架中模型从训练到评估所有过程均基于PyTorch实现，可以非常简单的利用GPU进行加速。
- **易用的模型结构分类** 我们的基于塔的数量的模型分类非常简单易懂，且具有很高的通用性。
- **丰富的负样本采样器** 我们提供了静态和动态两大类型的负采样模块，且均可通过GPU计算得到很高性能的表现。


## 快速使用
通过从GitHub下载RecStudio的源码，可以使用我们提供的 `run.py` 的脚本文件快速运行：

```bash
python run.py
```

这个简单的默认配置将会在MovieLens-100k 上进行BPR模型的训练和评估。

一般来说，在使用GPU的情况下，这个例子将运行不到1分钟的时间即可完成，将会得到如下类似的输出：
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
/data1/home/huangxu/miniconda3/lib/python3.9/site-packages/pytorch_lightning/trainer/data_loading.py:105: UserWarning: The dataloader, test dataloader 0, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 80 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
  rank_zero_warn(
[2022-04-11 14:31:27] INFO (pytorch_lightning/MainThread) Testing:  [recall@10=0.2439 precision@10=0.1893 map@10=0.5762 ndcg@10=0.3718 mrr@10=0.4487 hit@10=0.7815]
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
  |mode|str|训练模式，分为简单(light)、详细(detail)和调参(tune)|light|['light','detail','tune']|
  |--learning_rate|float|学习率|0.001||
  |--learner|str|优化器|adam|['adam','sgd','adasgd','rmsprop','sparse_adam']|
  |--weight_decay|float|权重衰减率|0||
  |--epochs|int|训练迭代轮数|20,50||
  |--batch_size|int|训练时mini batch的大小|2048||
  |--eval_batch_size|int|评估时mini batch的大小|128||
  |--embed_dim|int|embedding层的输出维度|64||

- 对于单塔模型（`ItemTowerRecommender`），有额外的命令行参数：

  |参数|数据类型|描述|默认值|可选值|
  |---|---|---|---|---|
  |--sampler|str|负采样器名称|uniform|['uniform','popularity','midx_uni','midx_pop','cluster_uni','cluster_pop']|
  |--negative_count|int|采样的负样本个数|1|正整数|

- 对于双塔模型（`TwoTowerRecommender`），相比于单塔模型再添加一个额外命令行参数：

  |参数|数据类型|描述|默认值|可选值|
  |---|---|---|---|---|
  |--split_mode|str|数据集划分方式|user_entry|['user','entry','user_entry']|

  > 1. `mode` 参数说明：light模式和detail模式均适用于命令行，区别在于显示的信息详略不同；tune模式将使用
  Neural Network Intelligence（NNI）框架。详细使用规则请参考 [NNI文档](https://nni.readthedocs.io/zh/stable/).
  > 2. `sampler` 参数说明：`uniform`表示使用均匀采样；`popularity`表示基于商品流行度采样；`midx_uni`,`midx_pop`
  为`midx`系列动态采样器，详细说明参见论文 [FastVAE](https://arxiv.org/abs/2109.05773)。`cluster_uni`,`cluster_pop`
  为`cluster`系列动态采样器，详细说明参见论文 [PRIS](https://dl.acm.org/doi/10.1145/3366423.3380187)。
  > 3. `split_mode` 参数说明：`user` 模式表示按照用户对数据集进行划分，训练集、验证集、测试集中用户均不同；
  `entry` 模式表示按照交互记录划分，随机将所有交互记录按照比例划分成训练、验证、测试集；`user_entry` 模式表
  示对每个用户，按一定比例划分其交互记录到训练、验证、测试集。


除了通过命令行设置参数外，我们更加推荐使用YAML配置文件来设置参数。

## 自动调参


## 贡献

如果您遇到错误或有任何建议，请通过 [Issue](https://github.com/ustcml/RecStudio/issues) 进行反馈

我们欢迎关于修复错误、添加新特性的任何贡献。

如果想贡献代码，请先在issue中提出问题，然后再提PR。


## 项目团队
RecStudio由中国科学技术大学的同学和老师进行开发和维护。 



## 免责声明
RecStudio 基于 [MIT License](./LICENSE) 进行开发，本项目的所有数据和代码只能被用于学术目的。