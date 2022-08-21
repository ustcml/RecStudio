import torch


class AIGRU(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 bias: bool = True,
                 batch_first: bool = True,
                 dropout: float = 0.0,
                 bidirectional: bool = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout_rate = dropout
        self.bidirectional = bidirectional
        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)

    def forward(self, input, weight, h_0=None):
        # input: [BxLxD] | [LxBxD]
        # weight: [BxL] | [LxB]
        assert input.shape[:-1] == weight.shape, ("`input` must have the same shape with `weight` at dimension 0"
                                                  "and 1, but get {} and {}".format(input.shape, weight.shape))
        weighted_input = input * weight.unsqueeze(2)
        if h_0 is not None:
            gru_out, _ = self.gru(weighted_input, h_0)
        else:
            gru_out, _ = self.gru(weighted_input)
        return gru_out

    def extra_repr(self) -> str:
        s = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)


class AGRUCell(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 bias: bool = True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.gru_cell = torch.nn.GRUCell(self.input_size, self.hidden_size, self.bias)

    def forward(self, input, hidden, weight):
        # input: BxD, hidden: BxD, weight: B
        weight = weight.view(-1, 1)
        hidden_o = self.gru_cell(input, hidden)
        hidden = (1 - weight) * hidden + weight * hidden_o
        return hidden


class AUGRUCell(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 bias: bool = True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.w_ir = torch.nn.Linear(self.input_size, self.hidden_size, bias=self.bias)
        self.w_hr = torch.nn.Linear(self.input_size, self.hidden_size, bias=self.bias)
        self.w_iz = torch.nn.Linear(self.input_size, self.hidden_size, bias=self.bias)
        self.w_hz = torch.nn.Linear(self.input_size, self.hidden_size, bias=self.bias)
        self.w_in = torch.nn.Linear(self.input_size, self.hidden_size, bias=self.bias)
        self.w_hn = torch.nn.Linear(self.input_size, self.hidden_size, bias=self.bias)
        self.sigma = torch.nn.Sigmoid()

    def forward(self, input, hidden, weight):
        weight = weight.view(-1, 1)
        r = self.sigma(self.w_ir(input) + self.w_hr(hidden))
        z = self.sigma(self.w_iz(input) + self.w_hz(hidden))
        z = weight * z
        n = torch.tanh(self.w_in(input) + r * self.w_hn(hidden))
        hidden = (1 - z) * n + z * hidden
        return hidden


class AGRU(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 bias: bool = True,
                 batch_first: bool = True,
                 dropout: float = 0.0,
                 bidirectional: bool = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout_rate = dropout
        self.bidirectional = bidirectional
        self.cell = self._set_gru_cell(self.input_size, self.hidden_size, self.bias)

    def _set_gru_cell(self, input_size, hidden_size, bias):
        return AGRUCell(input_size, hidden_size, bias)

    def forward(self, input, weight, h_0=None):
        if self.batch_first:
            input = input.contiguous().transpose(0, 1)
        L = input.size(0)

        if h_0 is None:
            num_directions = 2 if self.bidirectional else 1
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            hx = torch.zeros(self.num_layers * num_directions,
                             max_batch_size, self.hidden_size,
                             dtype=input.dtype, device=input.device)
        else:
            hx = h_0

        output = [None] * L
        for i in range(L):
            input_ = input[i]
            weight_ = weight[i]
            hx = self.cell(input_, hx, weight_)
            output[i] = hx

        if self.batch_first:
            output = torch.stack(output, dim=1)  # BxLxD
        else:
            output = torch.stack(output, dim=0)  # LxBxD

        return output, hx

    def extra_repr(self) -> str:
        s = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)


class AUGRU(AGRU):
    def _set_gru_cell(self, input_size, hidden_size, bias):
        return AUGRUCell(input_size, hidden_size, bias)


if __name__ == '__main__':
    input_size = 16,
    hidden_size = 32,
    num_layers = 2,
    bias = True,
    batch_first = True,
    dropout = 0.0,
    bidirectional = False

    gru = AIGRU(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)
    pass
