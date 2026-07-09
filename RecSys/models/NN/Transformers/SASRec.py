import math

import torch
from torch import nn
from torch.nn import functional as F


def _split_heads(
    X: torch.Tensor,
    n_heads: int,
    head_dim: int,
) -> torch.Tensor:
    batch_size, sequence_length, _ = X.shape

    X = X.view(
        batch_size,
        sequence_length,
        n_heads,
        head_dim,
    )

    return X.transpose(1, 2)


def _get_causal_mask(
    n: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.triu(
        torch.ones(
            size=(n, n),
            dtype=torch.bool,
            device=device,
        ),
        diagonal=1,
    )


def _get_padding_mask(
    X: torch.Tensor,
    padding_idx: int,
    device: torch.device,
) -> torch.Tensor:
    return (X == padding_idx).to(device=device, dtype=torch.bool)


def _make_int_tensor(
    X: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    return X.to(device=device, dtype=torch.long)


def _pad_truncate_tensor(
    X: torch.Tensor,
    max_sequence_length: int,
    padding_idx: int,
    device: torch.device,
) -> torch.Tensor:
    X = X[:, -max_sequence_length:]

    current_length = X.shape[1]
    if current_length < max_sequence_length:
        padding_length = max_sequence_length - current_length

        X = F.pad(
            X,
            pad=(padding_length, 0),
            mode="constant",
            value=padding_idx,
        )

    return X.to(device=device)


def _pad_truncate_list(
    X: list,
    max_sequence_length: int,
    padding_idx: int,
    device: torch.device,
) -> torch.Tensor:
    processed_sequences = []

    for sequence in X:
        sequence = torch.as_tensor(sequence, device=device)
        sequence = sequence[-max_sequence_length:]
        current_length = sequence.size(0)
        if current_length < max_sequence_length:
            padding_length = max_sequence_length - current_length

            sequence = F.pad(
                sequence,
                pad=(padding_length, 0),
                mode="constant",
                value=padding_idx,
            )
        processed_sequences.append(sequence)

    return torch.stack(processed_sequences)


def _pad_truncate(
    X: torch.Tensor | list,
    max_sequence_length: int,
    padding_idx: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(X, torch.Tensor):
        return _pad_truncate_tensor(
            X,
            max_sequence_length=max_sequence_length,
            padding_idx=padding_idx,
            device=device,
        )
    if isinstance(X, list):
        return _pad_truncate_list(
            X,
            max_sequence_length=max_sequence_length,
            padding_idx=padding_idx,
            device=device,
        )
    else:
        raise ValueError("'X' must be <torch.Tensor> or <list>")


def _pad_truncate_context_tensor(
    context: torch.Tensor,
    max_sequence_length: int,
    context_dim: int,
    device: torch.device,
) -> torch.Tensor:
    if context.ndim != 3 or context.shape[-1] != context_dim:
        raise ValueError(
            "'context' must have shape "
            f"(batch_size, sequence_length, {context_dim})"
        )

    context = context[:, -max_sequence_length:, :]

    current_length = context.shape[1]
    if current_length < max_sequence_length:
        padding_length = max_sequence_length - current_length

        context = F.pad(
            context,
            pad=(0, 0, padding_length, 0),
            mode="constant",
            value=0.0,
        )

    return context.to(device=device)


def _pad_truncate_context_list(
    context: list,
    max_sequence_length: int,
    context_dim: int,
    device: torch.device,
) -> torch.Tensor:
    processed_context = []

    for sequence_context in context:
        sequence_context = torch.as_tensor(
            sequence_context,
            device=device,
        )

        if (
            sequence_context.ndim != 2
            or sequence_context.shape[-1] != context_dim
        ):
            raise ValueError(
                "Each context sequence must have shape "
                f"(sequence_length, {context_dim})"
            )

        sequence_context = sequence_context[-max_sequence_length:, :]

        current_length = sequence_context.shape[0]
        if current_length < max_sequence_length:
            padding_length = max_sequence_length - current_length

            sequence_context = F.pad(
                sequence_context,
                pad=(0, 0, padding_length, 0),
                mode="constant",
                value=0.0,
            )

        processed_context.append(sequence_context)

    return torch.stack(processed_context)


def _pad_truncate_context(
    context: torch.Tensor | list,
    max_sequence_length: int,
    context_dim: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(context, torch.Tensor):
        return _pad_truncate_context_tensor(
            context,
            max_sequence_length=max_sequence_length,
            context_dim=context_dim,
            device=device,
        )
    if isinstance(context, list):
        return _pad_truncate_context_list(
            context,
            max_sequence_length=max_sequence_length,
            context_dim=context_dim,
            device=device,
        )
    else:
        raise ValueError("'context' must be <torch.Tensor> or <list>")


def _validate_context_alignment(
    X: torch.Tensor | list,
    context: torch.Tensor | list,
) -> None:
    if len(X) != len(context):
        raise ValueError("'X' and 'context' must have the same batch size")

    for sequence, sequence_context in zip(X, context):
        if len(sequence) != len(sequence_context):
            raise ValueError(
                "Each sequence in 'X' and 'context' must have "
                "the same sequence length"
            )


class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        items_count: int,
        embedding_dim: int,
        max_sequence_length: int,
        dropout: float = 0.0,
        device: torch.device | None = None,
        n_heads: int = 1,
    ) -> None:
        super().__init__()

        if n_heads <= 0:
            raise ValueError("'n_heads' must be greater than 0")
        if embedding_dim % n_heads != 0:
            raise ValueError("'embedding_dim' must be divisible by 'n_heads'")

        self.items_count = items_count
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.dropout = dropout
        self.device = device
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads

        self.W_q = nn.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim,
            bias=False,
            device=device,
        )
        self.W_k = nn.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim,
            bias=False,
            device=device,
        )
        self.W_v = nn.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim,
            bias=False,
            device=device,
        )
        self.W_o = nn.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim,
            bias=False,
            device=device,
        )

    def forward(
        self,
        E: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """E: normalized Embedding Matrix; (B, L, d)"""
        E = E.to(device=self.W_q.weight.device)
        padding_mask = padding_mask.to(device=E.device)

        batch_size, sequence_length, _ = E.shape

        Q = self.W_q(E)
        K = self.W_k(E)
        V = self.W_v(E)

        Q = _split_heads(Q, self.n_heads, self.head_dim)
        K = _split_heads(K, self.n_heads, self.head_dim)
        V = _split_heads(V, self.n_heads, self.head_dim)

        Z = (Q @ K.swapaxes(-1, -2)) / math.sqrt(self.head_dim)  # (B, n_heads, L, L)

        causal_mask = _get_causal_mask(sequence_length, E.device)  # (L, L)

        query_padding_mask = padding_mask[:, None, :, None]  # (B, 1, L, 1)
        key_padding_mask = padding_mask[:, None, None, :]  # (B, 1, 1, L)

        Z = Z.masked_fill(
            causal_mask,
            float("-inf"),
        )
        Z = Z.masked_fill(
            key_padding_mask,
            float("-inf"),
        )

        Z = Z.masked_fill(
            query_padding_mask,
            0.0,
        )

        A = F.softmax(
            Z,
            dim=-1,
        )
        A_dropout = F.dropout(
            A,
            p=self.dropout,
            training=self.training,
        )

        S = A_dropout @ V  # (B, n_heads, L, head_dim)

        S = S.transpose(1, 2)
        S = S.contiguous().view(
            batch_size,
            sequence_length,
            self.embedding_dim,
        )  # (B, L, d)

        S = self.W_o(S)

        S = S.masked_fill(
            padding_mask.unsqueeze(-1),
            0.0,
        )

        return S

class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        items_count: int,
        embedding_dim: int,
        max_sequence_length: int,
        ffn_hidden_dim: int,
        dropout: float = 0.0,
        device: torch.device | None = None,
        n_heads: int = 1,
    ) -> None:
        super().__init__()

        self.items_count = items_count
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.ffn_hidden_dim = ffn_hidden_dim
        self.dropout = dropout
        self.n_heads = n_heads

        if device is None:
            device = torch.get_default_device()
        self.device = device

        self.SelfAttentionLayer = SelfAttentionLayer(
            items_count=items_count,
            embedding_dim=embedding_dim,
            max_sequence_length=max_sequence_length,
            dropout=dropout,
            device=device,
            n_heads=n_heads,
        )

        self.ffn = nn.Sequential(
            nn.Linear(
                in_features=embedding_dim,
                out_features=ffn_hidden_dim,
                device=device,
            ),
            nn.ReLU(),
            nn.Dropout(dropout=dropout),
            nn.Linear(
                in_features=ffn_hidden_dim,
                out_features=embedding_dim,
                device=device,
            ),
            nn.Dropout(dropout=dropout),
        )

        self.sa_normalizer = nn.LayerNorm(
            normalized_shape=embedding_dim,
            device=device,
        )
        self.ffn_normalizer = nn.LayerNorm(
            normalized_shape=embedding_dim,
            device=device,
        )

    def forward(
        self,
        E: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        E_norm = self.sa_normalizer(E)

        S = self.SelfAttentionLayer(E_norm, padding_mask)
        H = E + S
        H_norm = self.ffn_normalizer(H)

        G = self.ffn(H_norm)

        H_out = H + G

        return H_out


class SASRec(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        items_count: int,
        embedding_dim: int,
        max_sequence_length: int,
        ffn_hidden_dim: int,
        dropout: float = 0.0,
        padding_idx: int = 0,
        device: torch.device | None = None,
        n_heads: int = 1,
    ) -> None:
        super().__init__()

        self.n_blocks = n_blocks
        self.items_count = items_count
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.ffn_hidden_dim = ffn_hidden_dim
        self.dropout = dropout
        self.padding_idx = padding_idx
        self.n_heads = n_heads

        if device is None:
            device = torch.get_default_device()
        self.device = device

        self.ItemEmbedding = nn.Embedding(
            num_embeddings=items_count,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            device=device,
        )

        self.PositionalEmbedding = nn.Embedding(
            num_embeddings=max_sequence_length + 1,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            device=device,
        )

        self.SABlocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    items_count=items_count,
                    embedding_dim=embedding_dim,
                    max_sequence_length=max_sequence_length,
                    ffn_hidden_dim=ffn_hidden_dim,
                    dropout=dropout,
                    device=device,
                    n_heads=n_heads,
                )
                for _ in range(n_blocks)
            ]
        )

        self.LayerNormalization = nn.LayerNorm(
            self.embedding_dim,
            device=device,
        )

    def log2feats(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        device = self.ItemEmbedding.weight.device

        X = _pad_truncate(
            X,
            self.max_sequence_length,
            self.padding_idx,
            device,
        )
        X = _make_int_tensor(X, device)

        padding_mask = _get_padding_mask(X, self.padding_idx, device)

        item_embedding = self.ItemEmbedding(X) * math.sqrt(self.embedding_dim)

        positional_ids = torch.arange(
            1,
            self.max_sequence_length + 1,
            device=device,
            dtype=torch.long,
        )
        positional_embedding = self.PositionalEmbedding(positional_ids)

        E = item_embedding + positional_embedding
        E = F.dropout(E, p=self.dropout, training=self.training)
        E[padding_mask] = 0.0

        for block in self.SABlocks:
            E = block(E, padding_mask)
            E[padding_mask] = 0.0

        E = self.LayerNormalization(E)
        E[padding_mask] = 0.0

        return E

    def forward(
        self,
        log_seqs: torch.Tensor,
        positive_seqs: torch.Tensor,
        negative_seqs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """returns tuple[positive logits, negative logits], dtype=torch.Tensor"""
        log_feats = self.log2feats(log_seqs)

        device = self.ItemEmbedding.weight.device

        positive_seqs = _pad_truncate(
            positive_seqs,
            self.max_sequence_length,
            self.padding_idx,
            device,
        )
        positive_seqs = _make_int_tensor(positive_seqs, device)
        negative_seqs = _pad_truncate(
            negative_seqs,
            self.max_sequence_length,
            self.padding_idx,
            device,
        )
        negative_seqs = _make_int_tensor(negative_seqs, device)

        positives = self.ItemEmbedding(positive_seqs)
        negatives = self.ItemEmbedding(negative_seqs)

        pos_logits = (log_feats * positives).sum(dim=-1)
        neg_logits = (log_feats * negatives).sum(dim=-1)

        return pos_logits, neg_logits

# ---
# Начнем с гейтов (код расклонировал ради кайфа)
# ---


class GatedSelfAttentionLayer(nn.Module):
    def __init__(
        self,
        items_count: int,
        embedding_dim: int,
        context_dim: int,
        context_hidden_dim: int,
        max_sequence_length: int,
        dropout: float = 0.0,
        device: torch.device | None = None,
        n_heads: int = 1,
    ) -> None:
        super().__init__()

        if n_heads <= 0:
            raise ValueError("'n_heads' must be greater than 0")
        if embedding_dim % n_heads != 0:
            raise ValueError("'embedding_dim' must be divisible by 'n_heads'")

        self.items_count = items_count
        self.embedding_dim = embedding_dim
        self.context_dim = context_dim
        self.context_hidden_dim = context_hidden_dim
        self.max_sequence_length = max_sequence_length
        self.dropout = dropout
        self.device = device
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads

        self.W_q = nn.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim,
            bias=False,
            device=device,
        )
        self.W_k = nn.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim,
            bias=False,
            device=device,
        )
        self.W_v = nn.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim,
            bias=False,
            device=device,
        )
        self.W_o = nn.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim,
            bias=False,
            device=device,
        )
        self.context_mlp = nn.Sequential(
            nn.Linear(
                in_features=context_dim,
                out_features=context_hidden_dim,
                device=device,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gate_Q = nn.Linear(
            context_hidden_dim,
            embedding_dim,
            device=device,
        )
        self.gate_K = nn.Linear(
            context_hidden_dim,
            embedding_dim,
            device=device,
        )
        self.gate_score = nn.Linear(
            context_hidden_dim,
            embedding_dim,
            device=device,
        )

    def forward(
        self,
        E: torch.Tensor,
        context: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """E: normalized Embedding Matrix; (B, L, d)"""
        E = E.to(device=self.W_q.weight.device)
        context = context.to(
            device=self.W_q.weight.device,
            dtype=E.dtype,
        )
        padding_mask = padding_mask.to(device=E.device)

        batch_size, sequence_length, _ = E.shape

        if context.shape != (batch_size, sequence_length, self.context_dim):
            raise ValueError(
                "'context' must have shape "
                f"({batch_size}, {sequence_length}, {self.context_dim})"
            )

        z_context = self.context_mlp(context)

        Q = self.W_q(E) * (2 * torch.sigmoid(self.gate_Q(z_context)))
        K = self.W_k(E) * (2 * torch.sigmoid(self.gate_K(z_context)))
        V = self.W_v(E)

        Q = _split_heads(Q, self.n_heads, self.head_dim)
        K = _split_heads(K, self.n_heads, self.head_dim)
        V = _split_heads(V, self.n_heads, self.head_dim)

        Z = (Q @ K.swapaxes(-1, -2)) / math.sqrt(self.head_dim)  # (B, n_heads, L, L)

        causal_mask = _get_causal_mask(sequence_length, E.device)  # (L, L)

        query_padding_mask = padding_mask[:, None, :, None]  # (B, 1, L, 1)
        key_padding_mask = padding_mask[:, None, None, :]  # (B, 1, 1, L)

        Z = Z.masked_fill(
            causal_mask,
            float("-inf"),
        )
        Z = Z.masked_fill(
            key_padding_mask,
            float("-inf"),
        )

        Z = Z.masked_fill(
            query_padding_mask,
            0.0,
        )

        A = F.softmax(
            Z,
            dim=-1,
        )
        A_dropout = F.dropout(
            A,
            p=self.dropout,
            training=self.training,
        )

        S = A_dropout @ V  # (B, n_heads, L, head_dim)

        S = S.transpose(1, 2)
        S = S.contiguous().view(
            batch_size,
            sequence_length,
            self.embedding_dim,
        )  # (B, L, d)

        S = self.W_o(S) * (2 * torch.sigmoid(self.gate_score(z_context)))

        S = S.masked_fill(
            padding_mask.unsqueeze(-1),
            0.0,
        )

        return S

class GatedSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        items_count: int,
        embedding_dim: int,
        context_dim: int,
        context_hidden_dim: int,
        max_sequence_length: int,
        ffn_hidden_dim: int,
        dropout: float = 0.0,
        device: torch.device | None = None,
        n_heads: int = 1,
    ) -> None:
        super().__init__()

        self.items_count = items_count
        self.embedding_dim = embedding_dim
        self.context_hidden_dim = context_hidden_dim
        self.max_sequence_length = max_sequence_length
        self.ffn_hidden_dim = ffn_hidden_dim
        self.dropout = dropout
        self.n_heads = n_heads

        if device is None:
            device = torch.get_default_device()
        self.device = device

        self.SelfAttentionLayer = GatedSelfAttentionLayer(
            items_count=items_count,
            embedding_dim=embedding_dim,
            context_dim=context_dim,
            context_hidden_dim=context_hidden_dim,
            max_sequence_length=max_sequence_length,
            dropout=dropout,
            device=device,
            n_heads=n_heads,
        )

        self.ffn = nn.Sequential(
            nn.Linear(
                in_features=embedding_dim,
                out_features=ffn_hidden_dim,
                device=device,
            ),
            nn.ReLU(),
            nn.Dropout(dropout=dropout),
            nn.Linear(
                in_features=ffn_hidden_dim,
                out_features=embedding_dim,
                device=device,
            ),
            nn.Dropout(dropout=dropout),
        )

        self.sa_normalizer = nn.LayerNorm(
            normalized_shape=embedding_dim,
            device=device,
        )
        self.ffn_normalizer = nn.LayerNorm(
            normalized_shape=embedding_dim,
            device=device,
        )

    def forward(
        self,
        E: torch.Tensor,
        context: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        E_norm = self.sa_normalizer(E)

        S = self.SelfAttentionLayer(E_norm, context, padding_mask)
        H = E + S
        H_norm = self.ffn_normalizer(H)

        G = self.ffn(H_norm)

        H_out = H + G

        return H_out


class GSASRec(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        items_count: int,
        embedding_dim: int,
        context_dim: int,
        context_hidden_dim: int,
        max_sequence_length: int,
        ffn_hidden_dim: int,
        dropout: float = 0.0,
        padding_idx: int = 0,
        device: torch.device | None = None,
        n_heads: int = 1,
    ) -> None:
        super().__init__()

        self.n_blocks = n_blocks
        self.items_count = items_count
        self.embedding_dim = embedding_dim
        self.context_dim = context_dim
        self.context_hidden_dim = context_hidden_dim
        self.max_sequence_length = max_sequence_length
        self.ffn_hidden_dim = ffn_hidden_dim
        self.dropout = dropout
        self.padding_idx = padding_idx
        self.n_heads = n_heads

        if device is None:
            device = torch.get_default_device()
        self.device = device

        self.ItemEmbedding = nn.Embedding(
            num_embeddings=items_count,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            device=device,
        )

        self.PositionalEmbedding = nn.Embedding(
            num_embeddings=max_sequence_length + 1,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            device=device,
        )

        self.SABlocks = nn.ModuleList(
            [
                GatedSelfAttentionBlock(
                    items_count=items_count,
                    embedding_dim=embedding_dim,
                    context_dim=context_dim,
                    context_hidden_dim=context_hidden_dim,
                    max_sequence_length=max_sequence_length,
                    ffn_hidden_dim=ffn_hidden_dim,
                    dropout=dropout,
                    device=device,
                    n_heads=n_heads,
                )
                for _ in range(n_blocks)
            ]
        )

        self.LayerNormalization = nn.LayerNorm(
            self.embedding_dim,
            device=device,
        )

    def log2feats(
        self,
        X: torch.Tensor,
        context: torch.Tensor | list,
    ) -> torch.Tensor:
        device = self.ItemEmbedding.weight.device

        _validate_context_alignment(X, context)

        X = _pad_truncate(
            X,
            self.max_sequence_length,
            self.padding_idx,
            device,
        )
        X = _make_int_tensor(X, device)
        context = _pad_truncate_context(
            context,
            self.max_sequence_length,
            self.context_dim,
            device,
        )
        context = context.to(
            device=device,
            dtype=self.ItemEmbedding.weight.dtype,
        )

        padding_mask = _get_padding_mask(X, self.padding_idx, device)

        item_embedding = self.ItemEmbedding(X) * math.sqrt(self.embedding_dim)

        positional_ids = torch.arange(
            1,
            self.max_sequence_length + 1,
            device=device,
            dtype=torch.long,
        )
        positional_embedding = self.PositionalEmbedding(positional_ids)

        E = item_embedding + positional_embedding
        E = F.dropout(E, p=self.dropout, training=self.training)
        E[padding_mask] = 0.0

        for block in self.SABlocks:
            E = block(E, context, padding_mask)
            E[padding_mask] = 0.0

        E = self.LayerNormalization(E)
        E[padding_mask] = 0.0

        return E

    def forward(
        self,
        log_seqs: torch.Tensor,
        context: torch.Tensor | list,
        positive_seqs: torch.Tensor,
        negative_seqs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """returns tuple[positive logits, negative logits], dtype=torch.Tensor"""
        log_feats = self.log2feats(log_seqs, context)

        device = self.ItemEmbedding.weight.device

        positive_seqs = _pad_truncate(
            positive_seqs,
            self.max_sequence_length,
            self.padding_idx,
            device,
        )
        positive_seqs = _make_int_tensor(positive_seqs, device)
        negative_seqs = _pad_truncate(
            negative_seqs,
            self.max_sequence_length,
            self.padding_idx,
            device,
        )
        negative_seqs = _make_int_tensor(negative_seqs, device)

        positives = self.ItemEmbedding(positive_seqs)
        negatives = self.ItemEmbedding(negative_seqs)

        pos_logits = (log_feats * positives).sum(dim=-1)
        neg_logits = (log_feats * negatives).sum(dim=-1)

        return pos_logits, neg_logits
