import numpy as np

from torch import nn

from typing import Callable, TypeAlias, Literal
from collections.abc import Iterable


PoolingType: TypeAlias = Literal["sum", "mean", "concat"]


class NormalInitizalizer:
    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0,
    ) -> None:
        self.mean = mean
        self.std = std

    def __call__(
        self,
        vocab_size: int,
        embedding_dim: int,
    ) -> nn.Embedding:
        embed = nn.Embedding(vocab_size, embedding_dim)
        nn.init.normal_(
            embed,
            mean=self.mean,
            std=np.std,
        )


class SparseFeature:
    def __init__(
        self,
        name: str,
        vocab_size: int,
        embedding_dim: int | None = None,
        padding_idx: Iterable | None = None,
        initializer: Callable = NormalInitizalizer,
    ) -> None:
        self.name = name
        self.vocab_size = vocab_size

        if embedding_dim is None:
            self.embedding_dim = get_auto_embedding_dim(vocab_size)
        else:
            self.embedding_dim = embedding_dim

        self.padding_idx = padding_idx
        self.initializer = initializer

        def __repr__(self):
            return f'<SparseFeature {self.name} with Embedding shape ({self.vocab_size}, {self.embedding_dim})>'
        
        def get_embedding_layer(self):
            if not hasattr(self, "embed"):
                self.embed = self.initializer(self.vocab_size, self.embedding_dim)
            return self.embed
        

class SequenceFeature:
    def __init__(
        self,
        name: str,
        vocab_size: int,
        embedding_dim: int | None = None,
        pooling: PoolingType = "mean", # maybe otdelniy TypeAlias is needed
        padding_idx: Iterable | None = None,
        shared_with: Iterable | str | None = None,
        initializer: Callable = NormalInitizalizer,
    ) -> None:
        self.name = name
        self.vocab_size = vocab_size

        if embedding_dim is None:
            self.embedding_dim = get_auto_embedding_dim(vocab_size)
        else:
            self.embedding_dim = embedding_dim

        self.padding_idx = padding_idx
        self.pooling = pooling
        self.shared_with = shared_with
        self.initializer = initializer

        def __repr__(self):
            return f'<SequenceFeature {self.name} with Embedding shape ({self.vocab_size}, {self.embedding_dim})>'

        def get_embedding_layer(self):
            if not hasattr(self, "embed"):
                self.embed = self.initializer(self.vocab_size, self.embedding_dim)
            return self.embed


class DenseFeature:
    def __init__(
        self,
        name: str,
    ) -> None:
        self.name = name
        self.embedding_dim = 1

    def __repr__(self):
        return f'<DenseFeature {self.name}>'
    



def get_auto_embedding_dim(num_classes):
    """
    Calculate the dim of embedding vector according to the number of classes in the category
    emb_dim = [6 * (num_classes)^(1/4)]
    reference: Deep & Cross Network for Ad Click Predictions.(ADKDD'17)
    """
    return np.floor(6 * np.pow(num_classes, 0.26))
