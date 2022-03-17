import torch
from torch import nn

class MLPModule(nn.Module):
    def __init__(self, mlp_layers, activation_func=nn.ReLU(), dropout=nn.Dropout(0)):
        super().__init__()
        self.mlp_layers = mlp_layers
        self.model = []
        for idx, layer in enumerate((zip(self.mlp_layers[ : -1], self.mlp_layers[1 : ]))):
            self.model.append(dropout)
            self.model.append(nn.Linear(*layer))
            self.model.append(activation_func)
        self.model = nn.Sequential(*self.model)
    
    def add_modules(self, *args):
        for block in args:
            self.model.add_module(str(len(self.model._modules)), block)

    def forward(self, input):
        return self.model(input)
