import torch
import torch.utils

from torch import nn
from torch.nn import functional as F


class RNNModel(nn.Module):
    """定义序列模型的框架"""
    def __init__(self, rnn, vocab_size):
        super().__init__()
        self.rnn = rnn
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        self.linear = nn.Linear(self.num_hiddens, vocab_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs, state):
        self.rnn.flatten_parameters()
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # ramdomly drop out some neurons 
        # Y = self.dropout(Y)
        output = self.linear(Y.reshape(-1, self.num_hiddens))
        return output, state
    
    def begin_state(self, device, batch_size):
        if isinstance(self.rnn, nn.RNN) or isinstance(self.rnn, nn.GRU):
            return torch.zeros((self.rnn.num_layers, batch_size, self.num_hiddens), device=device)
        else:
            return (torch.zeros(self.rnn.num_layers, batch_size, self.num_hiddens, device=device),
                    torch.zeros(self.rnn.num_layers, batch_size, self.num_hiddens, device=device))
