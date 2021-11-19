import torch

class Order2Inter(torch.nn.Module):

    def __init__(self):
        super(Order2Inter, self).__init__()

    def forward(self, inputs):
        square_of_sum = torch.sum(inputs, dim=1) ** 2
        sum_of_square = torch.sum(inputs ** 2, dim=1)
        cross_term = 0.5 * (square_of_sum - sum_of_square)
        return cross_term