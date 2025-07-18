import torch
from torch import nn

class Decoder(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=None, z_dim=2):
        super().__init__()
        if hidden_size is None:
            hidden_size = [128, 64, 8]

        hidden_size.reverse()

        self.decoder = nn.ModuleList()
        prev_size = z_dim
        for size in hidden_size:
            self.decoder.append(nn.Linear(prev_size, size))
            prev_size = size

        self.decoder.append(nn.Linear(hidden_size[-1], input_size))

    def forward(self, x):
        for i in range(len(self.decoder) - 1):
            x = torch.relu(self.decoder[i](x))
        x = torch.sigmoid(self.decoder[-1](x))
        return x