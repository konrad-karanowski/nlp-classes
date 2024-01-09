from typing import List, Optional

import torch
from torch import nn



class DoubleLayer(nn.Module):

    def __init__(
        self, 
        in_dim: int, 
        out_dim: int,
        batch_norm: int,
        resnet: bool = True,
    ) -> None:
        super(DoubleLayer, self).__init__()

        layers1 = [
            nn.Linear(in_dim, out_dim)
        ]
        if batch_norm:
            layers1.append(nn.BatchNorm1d(out_dim))
        layers1.append(nn.ReLU())

        layers2 = [
            nn.Linear(out_dim, out_dim)
        ]
        if batch_norm:
            layers2.append(nn.BatchNorm1d(out_dim))
        layers2.append(nn.ReLU())

        self.l1 = nn.Sequential(*layers1)
        self.l2 = nn.Sequential(*layers2)

        self.resnet = resnet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        out1 = self.l1(x)
        out2 = self.l2(out1)

        if self.resnet:
            out2 += out1

        return out2



class Encoder(nn.Module):

    def __init__(
            self,
            in_dim: int,
            intermediate_dims: List[int],
            out_dim: int,
            resnet: bool,
            batch_norm: bool
        ) -> None:
        super(Encoder, self).__init__()

        assert len(intermediate_dims) > 0

        layers = []

        dims = [in_dim] + intermediate_dims

        for i in range(len(dims) - 1):
            in_d, out_d = dims[i], dims[i + 1]
            layers.append(
                DoubleLayer()
            )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass



class Decoder(nn.Module):

    def __init__(
            self,
            z_dim: int,
            out_size: int,
            batch_norm: bool = False
        ) -> None:
        super(Decoder, self).__init__()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass



class VAE(nn.Module):

    def __init__(self) -> None:
        super(VAE, self).__init__()


        self.encoder = Encoder()
        self.decoder = Decoder()


    def reparameterization(self, mu, var):
        epsilon = torch.randn_like(var)      # sampling epsilon        
        z = mu + var*epsilon                          # reparameterization trick
        return z    


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, log_var = self.encoder(x)

        # reparametrization trick
        eps = self.reparameterization(mu, torch.exp(0.5 * log_var))

        x_hat = self.decoder(eps)

        return x_hat, mu, log_var
    