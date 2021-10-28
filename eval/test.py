import numpy as np
from numpy.core.numeric import ones_like
from data.dataset import MFDataset, AEDataset, SeqDataset, parser_yaml
from ann.search import Index
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from model.MF import BPR
#import scipy.sparse as ssp
#from eval.evaluation import Eval

# def get_train_mat(dataset):
#     for e in dataset.loader(len(dataset)):
#         uid, iid = e['user_id'].numpy(), e['item_id'].numpy()
#         break
#     return ssp.csr_matrix((np.ones_like(uid), (uid-1, iid-1)), (dataset.num_users-1, dataset.num_items-1))

# def get_test_mat(dataset):
#     for e in dataset.loader(len(dataset)):
#         user_list = []
#         for u, items in zip(e['user_id'].numpy(), e['item_id'].numpy()):
#             for i in items:
#                 if i>0:
#                     user_list.append((u, i))
#         break
#     uid, iid = zip(*user_list)
#     uid, iid = np.array(uid), np.array(iid)
#     return ssp.csr_matrix((np.ones_like(uid), (uid-1, iid-1)), (dataset.num_users-1, dataset.num_items-1))


if __name__=='__main__':
    dataset = MFDataset(r"datasets/ml-100k.yaml")
    train, val, test = dataset.build([0.8, 0.1, 0.1])
    #with open('../RecBole/abc.pkl', 'rb') as f:
    #    data_index = pickle.load(f)
    #train.data_index = torch.tensor(data_index)
    #train_mat = get_train_mat(train)
    #val_mat = get_test_mat(val)
    model = BPR.BPRRecommender(parser_yaml(r'model/basemodel.yaml'))
    model.fit(train, {'user_id', 'item_id', 'rating'}, val, True)
    model.evaluate(test)
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