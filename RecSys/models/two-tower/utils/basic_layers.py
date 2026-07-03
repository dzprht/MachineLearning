import torch
from torch import nn
from torch.nn import functional as F

from collections.abc import Iterable
from activations import ActivationFunction, activations_func

class MLP(nn.Module):
    def __init__(
        self,
        input_dims: int,
        is_output_layer: bool = False,
        dims: Iterable | None = None,
        dropout: float = 0.0,
        activation: ActivationFunction = "relu",
    ) -> None:
        if dims is None:
            dims = []
        layers = []

        for i_dim in dims:
            layers.append(nn.Linear(input_dims, i_dim))
            layers.append(nn.BatchNorm1d(input_dims))
            layers.append(activations_func(activation))
            layers.append(nn.Dropout(p=dropout))

            input_dims = i_dim
        if is_output_layer:
            layers.append(nn.Linear(input_dims, 1))
        
        self.mlp = nn.Sequential(*layers)
        

    def forward(self, x):
        return self.mlp(x)