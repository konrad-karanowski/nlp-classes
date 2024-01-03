import torch
from torch import nn


class Net(nn.Module):

    def __init__(self, in_dims: int, out_dims: int) -> None:
        super(Net, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dims, in_dims),
            nn.GELU(),
            nn.Linear(in_dims, out_dims)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
