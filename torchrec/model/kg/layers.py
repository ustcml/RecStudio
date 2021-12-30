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
            self.model._modules[str(len(self.model._modules))] = block

    def forward(self, input):
        return self.model(input)

# layers = [32, 16, 16]
# mlp = MLPModule(layers)
# mlp.add_modules(nn.ReLU(), nn.Sigmoid())
# a = torch.rand(5, 2, 32)
# print(mlp._modules)
# print(mlp(a))

