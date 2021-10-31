import numpy as np
from data.dataset import MFDataset, AEDataset, SeqDataset
from ann.search import Index
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from model.MF.BPR import BPR
from utils.utils import parser_yaml
import nni
if __name__=='__main__':
    tune_para = nni.get_next_parameter()
    param = parser_yaml(r'model/basemodel.yaml')
    for k, v in tune_para.items():
        param[k.split('/')[1]] = v
    model = BPR(param)
    train, val, test = model.load_dataset(r"datasets/ml-100k.yaml")
    model.fit(train, val, 'tune')
    model.evaluate(test)
    

