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


class ModifiedSelfAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        context_dim: int,
        context_hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if context_hidden_dim is None:
            context_hidden_dim = embedding_dim

        self.embedding_dim = embedding_dim
        self.context_dim = context_dim
        self.dropout = dropout

        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.W_Q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_K = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_V = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.context_mlp = nn.Sequential(
            nn.Linear(context_dim, context_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gate_Q = nn.Linear(context_hidden_dim, embedding_dim)
        self.gate_K = nn.Linear(context_hidden_dim, embedding_dim)
        self.gate_score = nn.Linear(context_hidden_dim, 1)

    def forward(self, E, C, padding_mask) -> torch.Tensor:
        batch_size, sequence_length, _ = E.shape

        if C.dim() == 2:
            C = C.unsqueeze(1).expand(-1, sequence_length, -1)
        if C.shape[:2] != (batch_size, sequence_length):
            raise ValueError("'C' must have the same batch and sequence length as 'E'")

        C = C.to(device=E.device, dtype=E.dtype)
        z_c = self.context_mlp(C)

        E_norm = self.layer_norm(E)
        Q = self.W_Q(E_norm) * (2 * torch.sigmoid(self.gate_Q(z_c)))
        K = self.W_K(E_norm) * (2 * torch.sigmoid(self.gate_K(z_c)))
        V = self.W_V(E_norm)

        scores = (Q @ K.swapaxes(-1, -2)) / np.sqrt(self.embedding_dim)
        scores = scores * (2 * torch.sigmoid(self.gate_score(z_c)))

        causal_mask = _get_causal_mask(sequence_length, scores.device).unsqueeze(0)
        scores = scores.masked_fill(padding_mask.unsqueeze(1), float("-inf"))
        scores = scores.masked_fill(causal_mask, float("-inf"))
        scores = scores.masked_fill(padding_mask.unsqueeze(2), 0.0)

        attention = F.softmax(scores, dim=-1)
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        output = E + attention @ V

        return output.masked_fill(padding_mask.unsqueeze(2), 0.0)


class ModifiedSASRecBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        ffn_hidden_dim: int,
        context_dim: int,
        context_hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.attention = ModifiedSelfAttention(
            embedding_dim=embedding_dim,
            context_dim=context_dim,
            context_hidden_dim=context_hidden_dim,
            dropout=dropout,
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, E, C, padding_mask) -> torch.Tensor:
        E = self.attention(E, C, padding_mask)
        return E + self.ffn(self.layer_norm(E))


class ModifiedSASRec(nn.Module):
    def __init__(
        self,
        *,
        n_items: int,
        max_sequence_length: int,
        embedding_dim: int,
        n_blocks: int,
        ffn_hidden_dim: int,
        context_dim: int,
        context_hidden_dim: int | None = None,
        padding_idx: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.context_dim = context_dim
        self.padding_idx = padding_idx
        self.dropout = dropout

        self.item_embedding = nn.Embedding(
            n_items + 1,
            embedding_dim,
            padding_idx=padding_idx,
        )
        self.positional_embedding = nn.Embedding(max_sequence_length, embedding_dim)
        self.blocks = nn.ModuleList(
            [
                ModifiedSASRecBlock(
                    embedding_dim=embedding_dim,
                    ffn_hidden_dim=ffn_hidden_dim,
                    context_dim=context_dim,
                    context_hidden_dim=context_hidden_dim,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, X: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        if X.dim() != 2:
            raise ValueError("'X' must be shaped (batch_size, sequence_length)")
        if C.dim() not in (2, 3):
            raise ValueError("'C' must be shaped (B, context_dim) or (B, L, context_dim)")
        if C.size(-1) != self.context_dim:
            raise ValueError(f"Expected context_dim={self.context_dim}, got {C.size(-1)}")

        X = X[:, -self.max_sequence_length :]
        if X.size(1) < self.max_sequence_length:
            X = F.pad(
                X,
                pad=(self.max_sequence_length - X.size(1), 0),
                value=self.padding_idx,
            )

        if C.dim() == 3:
            C = C[:, -self.max_sequence_length :, :]
            if C.size(1) < self.max_sequence_length:
                C = F.pad(C, pad=(0, 0, self.max_sequence_length - C.size(1), 0))

        padding_mask = X == self.padding_idx
        positional_ids = torch.arange(
            self.max_sequence_length,
            device=X.device,
            dtype=torch.long,
        )

        E = self.item_embedding(X) * np.sqrt(self.embedding_dim)
        E = E + self.positional_embedding(positional_ids)
        E = F.dropout(E, p=self.dropout, training=self.training)
        E = E.masked_fill(padding_mask.unsqueeze(2), 0.0)

        for block in self.blocks:
            E = block(E, C, padding_mask)
            E = E.masked_fill(padding_mask.unsqueeze(2), 0.0)

        E = self.final_layer_norm(E)
        scores = E @ self.item_embedding.weight.T
        scores[:, :, self.padding_idx] = float("-inf")

        return scores


ContextGatedSASRec = ModifiedSASRec


class LowRankContextSelfAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        context_dim: int,
        rank: int,
        context_hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if rank <= 0:
            raise ValueError("'rank' must be positive")
        if context_hidden_dim is None:
            context_hidden_dim = embedding_dim

        self.embedding_dim = embedding_dim
        self.context_dim = context_dim
        self.rank = rank
        self.dropout = dropout

        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.W_Q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_K = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_V = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.context_mlp = nn.Sequential(
            nn.Linear(context_dim, context_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Linear(context_hidden_dim, rank)

        self.A = nn.Parameter(torch.empty(embedding_dim, rank))
        self.B = nn.Parameter(torch.empty(embedding_dim, rank))
        nn.init.xavier_uniform_(self.A)
        nn.init.xavier_uniform_(self.B)

    def forward(self, E, C, padding_mask) -> torch.Tensor:
        batch_size, sequence_length, _ = E.shape

        if C.dim() == 2:
            C = C.unsqueeze(1).expand(-1, sequence_length, -1)
        if C.shape[:2] != (batch_size, sequence_length):
            raise ValueError("'C' must have the same batch and sequence length as 'E'")

        C = C.to(device=E.device, dtype=E.dtype)
        z_c = self.context_mlp(C)
        g = 2 * torch.sigmoid(self.gate(z_c))

        E_norm = self.layer_norm(E)
        Q = self.W_Q(E_norm)
        K = self.W_K(E_norm)
        V = self.W_V(E_norm)

        base_scores = Q @ K.swapaxes(-1, -2)
        low_rank_scores = ((Q @ self.A) * g) @ (K @ self.B).swapaxes(-1, -2)
        scores = (base_scores + low_rank_scores) / np.sqrt(self.embedding_dim)

        causal_mask = _get_causal_mask(sequence_length, scores.device).unsqueeze(0)
        scores = scores.masked_fill(padding_mask.unsqueeze(1), float("-inf"))
        scores = scores.masked_fill(causal_mask, float("-inf"))
        scores = scores.masked_fill(padding_mask.unsqueeze(2), 0.0)

        attention = F.softmax(scores, dim=-1)
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        output = E + attention @ V

        return output.masked_fill(padding_mask.unsqueeze(2), 0.0)


class LowRankContextSASRecBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        ffn_hidden_dim: int,
        context_dim: int,
        rank: int,
        context_hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.attention = LowRankContextSelfAttention(
            embedding_dim=embedding_dim,
            context_dim=context_dim,
            rank=rank,
            context_hidden_dim=context_hidden_dim,
            dropout=dropout,
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, E, C, padding_mask) -> torch.Tensor:
        E = self.attention(E, C, padding_mask)
        return E + self.ffn(self.layer_norm(E))


class LowRankContextSASRec(nn.Module):
    def __init__(
        self,
        *,
        n_items: int,
        max_sequence_length: int,
        embedding_dim: int,
        n_blocks: int,
        ffn_hidden_dim: int,
        context_dim: int,
        rank: int,
        context_hidden_dim: int | None = None,
        padding_idx: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.context_dim = context_dim
        self.padding_idx = padding_idx
        self.dropout = dropout

        self.item_embedding = nn.Embedding(
            n_items + 1,
            embedding_dim,
            padding_idx=padding_idx,
        )
        self.positional_embedding = nn.Embedding(max_sequence_length, embedding_dim)
        self.blocks = nn.ModuleList(
            [
                LowRankContextSASRecBlock(
                    embedding_dim=embedding_dim,
                    ffn_hidden_dim=ffn_hidden_dim,
                    context_dim=context_dim,
                    rank=rank,
                    context_hidden_dim=context_hidden_dim,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, X: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        if X.dim() != 2:
            raise ValueError("'X' must be shaped (batch_size, sequence_length)")
        if C.dim() not in (2, 3):
            raise ValueError("'C' must be shaped (B, context_dim) or (B, L, context_dim)")
        if C.size(-1) != self.context_dim:
            raise ValueError(f"Expected context_dim={self.context_dim}, got {C.size(-1)}")

        X = X[:, -self.max_sequence_length :]
        if X.size(1) < self.max_sequence_length:
            X = F.pad(
                X,
                pad=(self.max_sequence_length - X.size(1), 0),
                value=self.padding_idx,
            )

        if C.dim() == 3:
            C = C[:, -self.max_sequence_length :, :]
            if C.size(1) < self.max_sequence_length:
                C = F.pad(C, pad=(0, 0, self.max_sequence_length - C.size(1), 0))

        padding_mask = X == self.padding_idx
        positional_ids = torch.arange(
            self.max_sequence_length,
            device=X.device,
            dtype=torch.long,
        )

        E = self.item_embedding(X) * np.sqrt(self.embedding_dim)
        E = E + self.positional_embedding(positional_ids)
        E = F.dropout(E, p=self.dropout, training=self.training)
        E = E.masked_fill(padding_mask.unsqueeze(2), 0.0)

        for block in self.blocks:
            E = block(E, C, padding_mask)
            E = E.masked_fill(padding_mask.unsqueeze(2), 0.0)

        E = self.final_layer_norm(E)
        scores = E @ self.item_embedding.weight.T
        scores[:, :, self.padding_idx] = float("-inf")

        return scores


BilinearContextSASRec = LowRankContextSASRec
