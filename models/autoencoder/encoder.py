from torch import nn
import torch
class Encoder(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=None, z_dim=2):

        super().__init__()
        if hidden_size is None:
            hidden_size = [128, 64, 8]

        self.encoder = nn.ModuleList()
        prev_size = input_size
        for size in hidden_size:
            self.encoder.append(nn.Linear(prev_size, size))
            prev_size = size

        self.encoder.append(nn.Linear(hidden_size[-1], z_dim))

    def forward(self, x):

        for i in range(len(self.encoder) - 1):
            x = torch.relu(self.encoder[i](x))
        x = self.encoder[-1](x)
        return x



