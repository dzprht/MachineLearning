import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from typing import Literal


class ELSA(nn.Module):
    def __init__(
        self,
        n_items: int,
        n_factors: int,
    ) -> None:
        super().__init__()

        if n_items <= 1:
            raise ValueError("...")

        if n_factors <= 0:
            raise ValueError("...")

        self.n_items = n_items
        self.n_factors = n_factors

        self.raw_item_embeddings = nn.Parameter(torch.empty(n_items, n_factors))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """initialize trainable item vectors"""
        nn.init.xavier_uniform_(self.raw_item_embeddings)

    def item_embeddings(self) -> torch.Tensor:
        """return nomalized item embeddings A
        Shape: (n_items, n_factors)"""

        return F.normalize(
            input=self.raw_item_embeddings,
            p=2,
            dim=1,
            eps=1e-12,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """convert user-item history to user embeddings"""

        self._validate_input(x)

        A = self.item_embeddings()
        return x @ A

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """calculate ELSA prediction via formula:
        X_hat = (X @ A) @ A.T - X"""

        self._validate_input(x)

        A = self.item_embeddings()
        user_embeddings = x @ A
        predictions = user_embeddings @ A.T - x

        return predictions

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 2:
            raise ValueError("'x' must be 2-dimensional")
        if x.shape[1] != self.n_items:
            raise ValueError(
                f"expeted {self.n_items} items; got {x.shape[1]!r} instead"
            )
        if not x.is_floating_point():
            raise TypeError("'x' must have a floating point dtype")
