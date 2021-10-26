import numpy as np
from data.dataset import MFDataset, AEDataset, SeqDataset, parser_yaml
from ann.search import Index
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from model.MF import BPR
if __name__=='__main__':
    dataset = MFDataset(r"datasets/ml-100k.yaml")
    train, val, test = dataset.build([0.8, 0.1, 0.1])
    model = BPR.BPRRecommender(parser_yaml(r'model/basemodel.yaml'))
    model.fit(train, {'user_id', 'item_id', 'rating'}, val, True)
    #train.drop_feat(['user_id', 'item_id', 'age', 'gender', 'item_hist'])
    #freq = train.item_freq
    #print(freq.sum().item())
    #loader = train.item_feat.loader(10)
    #for batch in loader:
    #    print(batch)
    # b = 0
    #loader = val.loader(batch_size=2, shuffle=True, num_workers=1, drop_last=True)
    #for batch in loader:
    #   print(batch)
    #   break
    # print(b)
    #print(train[10])
    #print(train.item_freq.sum())
    #print(train.item_freq.sum() + val.item_freq.sum()+test.item_freq.sum())
    #train, val, test = dataset.build([dataset.fuid, dataset.fiid], [0.8, 0.1, 0.1])
    #print(len(train)+len(val)+len(test))