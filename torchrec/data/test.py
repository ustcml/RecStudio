from re import U
import numpy as np
from numpy.core.numeric import ones_like
from torchrec.data.dataset import MFDataset, AEDataset, SeqDataset
from torchrec.data.advance_dataset import ALSDataset
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchrec.model.mf.bpr import BPR


if __name__=='__main__':
    dataset = ALSDataset(r"datasets/ml-100k/ml-100k.yaml")
    train, val, test = dataset.build([0.8, 0.1, 0.1], shuffle=True, split_mode='user_entry')#, split_mode='entry')
    train_t = train.transpose()
    a = train.save()
    b = train_t.save()
    print(a.shape)
    print(b.shape)
    print((a - b).sum())
    # data = next(iter(train.loader(batch_size=len(train), shuffle=False)))
    # uid, iid, rating = data[train.fuid], data[train.fiid], data[train.frating]
    # val = []
    # for u, ids, rs in zip(uid, iid, rating):
    #     for id, r in zip(ids, rs):
    #         if id>0:
    #             val.append([u,id,r])
    # print(len(val))
    # val = sorted(val, key = lambda x: (x[0], x[1]))
    # for i in range(20):
    #     print(val[-i])
    
    # train.switch()
    # data = next(iter(train.loader(batch_size=len(train), shuffle=False)))
    # uid, iid, rating = data[train.fuid], data[train.fiid], data[train.frating]
    # val = []
    # for u, ids, rs in zip(uid, iid, rating):
    #     for id, r in zip(ids, rs):
    #         if id>0:
    #             val.append([id,u,r])
    # print(len(val))
    # val = sorted(val, key = lambda x: (x[0], x[1]))
    # for i in range(20):
    #     print(val[-i])
    

        
    #with open('../RecBole/abc.pkl', 'rb') as f:
    #    data_index = pickle.load(f)
    #train.data_index = torch.tensor(data_index)
    #train_mat = get_train_mat(train)
    #val_mat = get_test_mat(val)
    #model = BPR.BPRRecommender(parser_yaml(r'model/basemodel.yaml'))
    #model.fit(train, {'user_id', 'item_id', 'rating'}, val, True)
    #model.evaluate(test)
    #users = model.user_encoder.weight[1:].detach()
    #items = model.get_item_vector().detach()
    #output = Eval.evaluate_item(train_mat, val_mat, users.numpy(), items.numpy(), 10, 10)
    #print(output)
    #train.drop_feat(['user_id', 'item_id', 'age', 'gender', 'item_hist'])
    #freq = train.item_freq
    #print(freq.sum().item())
    #loader = train.item_feat.loader(10)
    #for batch in loader:
    #    print(batch)
    # b = 0
    #loader = val.eval_loader(batch_size=20)
    #for batch in loader:
    #  print(batch['item_id'].shape)
    #  break
    # print(b)
    #print(train[10])
    #print(train.item_freq.sum())
    #print(train.item_freq.sum() + val.item_freq.sum()+test.item_freq.sum())
    #train, val, test = dataset.build([dataset.fuid, dataset.fiid], [0.8, 0.1, 0.1])
    #print(len(train)+len(val)+len(test))