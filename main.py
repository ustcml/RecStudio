import numpy as np
from data.dataset import MFDataset, AEDataset, SeqDataset
from ann.search import Index
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from model.MF import BPR
from utils.utils import parser_yaml
if __name__=='__main__':
    model = BPR.BPRRecommender(parser_yaml(r'model/basemodel.yaml'))
    train, val, test = model.load_dataset(r"datasets/ml-100k.yaml")
    model.fit(train, val, True)
    model.evaluate(test)