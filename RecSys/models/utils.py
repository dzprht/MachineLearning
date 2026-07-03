import torch
from torch import nn

from collections.abc import Iterable
from typing import TypeAlias, Literal, Callable

ActivationLayer: TypeAlias = Literal["relu", "sigmoid", "tanh"]


def activation_layer(name: ActivationLayer) -> Callable:
    if name == "relu":
        return nn.ReLU()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh()
    else:
        raise ValueError("'name' must be 'relu', 'sigmoid' or 'tanh'")

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_layer: bool = False,
        dims: Iterable | None = None,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        if dims is None:
            dims = []
        layers = []

        for i_dim in dims:
            layers.append(nn.Linear(input_dim, i_dim))
            layers.append(nn.BatchNorm1d(i_dim))
            layers.append(activation_layer(activation))
            layers.append(nn.Dropout(p=dropout))
            input_dim = i_dim
        if output_layer:
            layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)