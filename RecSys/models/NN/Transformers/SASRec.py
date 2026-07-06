import numpy as np

import torch
from torch import nn
from torch.nn import functional as F


def _get_causal_mask(n, device) -> torch.Tensor:
    return torch.triu(
        torch.ones(
            size=(n, n),
            dtype=torch.bool,
            device=device,
        ),
        diagonal=1,
    )


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float = 0.0) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim

        self.W_Q = nn.Linear(
            self.embedding_dim,
            self.embedding_dim,
            bias=False,
        )
        self.W_K = nn.Linear(
            self.embedding_dim,
            self.embedding_dim,
            bias=False,
        )
        self.W_V = nn.Linear(
            self.embedding_dim,
            self.embedding_dim,
            bias=False,
        )

        self.NormalizationLayer = nn.LayerNorm(normalized_shape=self.embedding_dim)

    def forward(self, E, padding_mask, dropout: float = 0.0):
        """E: shaped (batch_size, sequence_length, embedding_dim)"""
        batch_size, sequence_length, embedding_dim = E.shape

        if embedding_dim != self.embedding_dim:
            raise ValueError(
                f"'E' has size ({batch_size}, {sequence_length}, {embedding_dim}) but ({batch_size}, {sequence_length}, {self.embedding_dim}) is expected"
            )

        E_norm = self.NormalizationLayer(E)

        Q = self.W_Q(E_norm)  # shape: (B, L, d)
        K = self.W_K(E_norm)  # shape: (B, L, d)
        V = self.W_V(E_norm)  # shape: (B, L, d)

        Z = (Q @ K.swapaxes(-1, -2)) / np.sqrt(self.embedding_dim)  # shape: (B, L, L)

        causal_mask = _get_causal_mask(sequence_length, Z.device).unsqueeze(
            0
        )  # shape: (1, L, L)
        query_padding_mask = padding_mask.unsqueeze(2)  # shape: (B, L, 1)
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

        A = F.softmax(Z, dim=2)  # shape: (B, L, L)
        A_drop = F.dropout(
            A,
            p=dropout,
            training=self.training,
        )

        S = A_drop @ V  # shape: (B, L, d)


        output = E + S

        output = output.masked_fill(
            query_padding_mask,
            0.0,
        )

        return output


class SASRecBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        ffn_hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ffn_hidden_dim = ffn_hidden_dim
        self.dropout = dropout

        self.self_attention_layer = SelfAttention(embedding_dim, dropout)
        self.sa_normalizer = nn.LayerNorm(normalized_shape=embedding_dim)

        self.ffn = nn.Sequential(
            nn.Linear(
                self.embedding_dim,
                self.ffn_hidden_dim,
                bias=True,
            ),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(
                self.ffn_hidden_dim,
                self.embedding_dim,
                bias=True,
            ),
            nn.Dropout(p=dropout),
        )

    def forward(
        self,
        E,
        padding_mask,
    ) -> torch.Tensor:
        H_att = self.self_attention_layer(E, padding_mask, self.dropout)
        H_att_normalized = self.sa_normalizer(H_att)

        G = self.ffn(H_att_normalized)

        H_out = H_att + G

        return H_out


class SASRec(nn.Module):
    def __init__(
        self,
        *,
        n_items: int,
        max_sequence_length: int,
        embedding_dim: int,
        n_blocks: int,
        ffn_hidden_dim: int,
        padding_idx: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.n_items = n_items
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.n_blocks = n_blocks
        self.ffn_hidden_dim = ffn_hidden_dim
        self.padding_idx = padding_idx
        self.dropout = dropout

        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

        self.item_embedding = nn.Embedding(
            n_items + 1,
            embedding_dim,
            padding_idx=padding_idx,
        )

        self.positional_embedding = nn.Embedding(
            max_sequence_length,
            embedding_dim,
        )

        self.sasrec_blocs = nn.ModuleList(
            [
                SASRecBlock(embedding_dim, ffn_hidden_dim, dropout)
                for _ in range(n_blocks)
            ]
        )

    def forward(
        self,
        X: torch.Tensor | list,
    ) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            X = self._pad_or_truncate(X)
        elif isinstance(X, list):
            X = self._pad_or_truncate_list(X)
        else:
            raise ValueError(
                f"'X' must be either <torch.Tensor> or <list>; got {type(X)} instead"
            )

        padding_mask = self._get_padding_mask(X)

        item_embedding = self.item_embedding(X) * np.sqrt(self.embedding_dim)
        positional_ids = torch.arange(
            self.max_sequence_length,
            device=X.device,
            dtype=torch.long,
        )
        positional_embedding = self.positional_embedding(positional_ids)

        E = item_embedding + positional_embedding  # shape: (B, L, d)
        E = F.dropout(E, p=self.dropout, training=self.training)
        E[padding_mask] = 0.0

        for block in self.sasrec_blocs:
            E = block(E, padding_mask)
            E[padding_mask] = 0.0
        
        E = self.final_layer_norm(E)
        scores = E @ self.item_embedding.weight.T
        scores[:, :, self.padding_idx] = float("-inf")

        return scores

    def _get_padding_mask(self, X) -> torch.Tensor:
        return X == self.padding_idx

    def _pad_or_truncate(self, X) -> torch.Tensor:
        X = X[:, -self.max_sequence_length :]
        current_length = X.shape[1]
        if current_length < self.max_sequence_length:
            padding_length = self.max_sequence_length - current_length

            X = F.pad(
                X,
                pad=(padding_length, 0),
                mode="constant",
                value=self.padding_idx,
            )

        return X

    def _pad_or_truncate_list(self, X) -> torch.Tensor:
        processed_sequences = []

        for sequence in X:
            sequence = sequence[-self.max_sequence_length:]
            current_length = sequence.size(0)
            if current_length < self.max_sequence_length:
                padding_length = self.max_sequence_length - current_length

                sequence = F.pad(
                    sequence,
                    pad=(padding_length, 0),
                    mode="constant",
                    value=self.padding_idx,
                )

            processed_sequences.append(sequence)

        return torch.stack(processed_sequences)
