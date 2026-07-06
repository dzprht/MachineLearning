import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from utils.basic_layers import EmbeddingLayer, SequenceFeature


def _get_casual_mask(n) -> torch.Tensor:
    return torch.triu(
        torch.ones(
            size=(n, n),
            dtype=torch.bool,
        ),
        diagonal=1,
    ).unsqueeze(0)


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float = 0.0) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim

        self.W_Q = nn.Linear()
        self.W_K = nn.Linear()
        self.W_V = nn.Linear()

        self.NormalizationLayer = nn.LayerNorm(normalized_shape=self.embedding_dim)

    def forward(self, E, padding_mask, dropout: float = 0.0):
        """E: shaped (batch_size, sequence_length, embedding_dim)"""
        batch_size, sequence_length, embedding_dim = E.shape

        if embedding_dim != self.embedding_dim:
            raise ValueError(
                f"'E' has size ({batch_size}, {sequence_length}, {embedding_dim}) but ({batch_size}, {sequence_length}, {self.embedding_dim}) is expected"
            )

        E_norm = self.NormalizationLayer(E)

        Q = E_norm @ self.W_Q  # shape: (B, L, d)
        K = E_norm @ self.W_K  # shape: (B, L, d)
        V = E_norm @ self.W_V  # shape: (B, L, d)

        Z = (Q @ K.swapaxes(-1, -2)) / np.sqrt(self.embedding_dim)  # shape: (B, L, L)

        causal_mask = _get_casual_mask(sequence_length).unsqueeze(0)  # shape: (1, L, L)
        query_padding_mask = padding_mask.unsqueeze(2) # shape: (B, L, 1)
        key_padding_mask = padding_mask.unsqueeze(1)  # shape: (B, 1, L)

        Z = Z.masked_fill(
            key_padding_mask,
            float("-inf"),
        )

        Z = Z.masked_fill(
            causal_mask,
            float("-inf"),
        )

        Z = Z.masked_fill(
            query_padding_mask,
            0.0,
        )
        
        A = F.softmax(Z + causal_mask + padding_mask, dim=2)  # shape: (B, L, L)

        S = A @ V  # shape: (B, L, d)

        S_drop = F.dropout(
            S,
            p=dropout,
            training=self.training,
        )

        output = E + S_drop

        output = output.masked_fill(
            query_padding_mask,
            0.0,
        )

        return output


class SASRecBlock(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()


class SASRec(nn.Module):
    def __init__(
        self,
        X,
        n: int,
        sa_blocks: int,
        *,
        position_embed_dim: int = 8,
    ) -> None:
        """X: interactions matrix, shaped (B, L), where B - batch size; L - sequence length"""

        super().__init__()

        self.item_embedding = EmbeddingLayer(X)
        self.position_embedding = nn.Embedding(n, 8, self._get_padding_idx(X, n))

        self.blocks = nn.ModuleList([SASRecBlock() for _ in range(sa_blocks)])

    def _get_padding_idx(self, X, n) -> torch.Tensor:
        pass
