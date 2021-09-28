import torch
from torch import nn
class BaseRecommender(nn.Module):
    def __init__(self):
        pass

    def forward(self, batch_data):
        pass
    
    def fit(self, train_data):
        pass

    def eval(self, test_data):
        pass
