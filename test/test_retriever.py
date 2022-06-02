import sys
sys.path.append(".")

from recstudio.model import scorer, loss_func as loss
from recstudio.ann import sampler
from recstudio.model.basemodel import BaseRetriever
from recstudio.data import dataset
import torch


ml_1m_data = dataset.MFDataset(name='ml-1m')

trn, val, tst = ml_1m_data.build(split_ratio=[0.7, 0.2, 0.1])

bpr = BaseRetriever(
    item_encoder = torch.nn.Embedding(trn.num_items, 64, 0),
    query_encoder = torch.nn.Embedding(trn.num_users, 64, 0),
    scorer = scorer.InnerProductScorer(),
    loss = loss.BPRLoss(),
    sampler = sampler.UniformSampler(trn.num_items-1)
)

bpr.fit(trn, val, negative_count=1)

bpr.evaluate(tst)
