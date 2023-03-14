from recstudio.model import scorer, loss_func   # 导入打分和损失函数模块
from recstudio.ann import sampler   # 导入采样器模块
from recstudio.model.basemodel import BaseRetriever  # 导入召回模型基类
from recstudio.data import dataset  # 导入数据集模块
import torch
import sys
sys.path.append(".")


ml_1m_data = dataset.TripletDataset(name='ml-100k')
trn, val, tst = ml_1m_data.build(split_ratio=[0.7, 0.2, 0.1])

bpr = BaseRetriever(
    item_encoder=torch.nn.Embedding(trn.num_items, 64, 0),
    query_encoder=torch.nn.Embedding(trn.num_users, 64, 0),
    scorer=scorer.InnerProductScorer(),
    loss=loss_func.BPRLoss(),
    sampler=sampler.UniformSampler(trn.num_items)
)

bpr.fit(trn, val, negative_count=1)
bpr.evaluate(tst)
