import torch
from torch import nn
from torch.nn import functional as F

from collections.abc import Iterable

from utils.basic_layers import EmbeddingLayer, MLP


class DSSM(nn.Module):
    def __init__(
        self,
        user_features: Iterable,
        item_features: Iterable,
        user_mlp_params: Iterable,
        item_mlp_params: Iterable,
        temperature: float = 1.0,
    ) -> None:
        self.user_features = user_features
        self.item_features = item_features
        self.temperature = temperature

        self.user_embed_dim = sum([f.embed for f in user_features])
        self.item_embed_dim = sum([f.embed for f in item_features])

        self.embedding = EmbeddingLayer(user_features + item_features)
        self.user_mlp = MLP(
            self.user_embed_dim, is_output_layer=False, *user_mlp_params
        )
        self.item_mlp = MLP(
            self.item_embed_dim, is_output_layer=False, *item_mlp_params
        )

    def forward(
        self,
        x: Iterable,
    ) -> torch.Tensor:
        user_embeddings = self.user_tower(x)
        item_embeddings = self.item_tower(x)

        predictions = (
            torch.sum(user_embeddings * item_embeddings, dim=1) * self.temperature
        )

        return predictions

    def user_tower(
        self,
        x: Iterable,
    ) -> torch.Tensor:
        input_user = self.embedding(
            x,
            self.user_features,
            squeeze_dim=True,
        )

        user_embedding = self.user_mlp(input_user)

        user_embedding = F.normalize(
            user_embedding,
            p=2,
            dim=1,
        )

        return user_embedding

    def item_tower(
        self,
        x: Iterable,
    ) -> torch.Tensor:
        input_item = self.embedding(
            x,
            self.item_features,
            squeeze_dim=True,
        )

        item_embedding = self.item_mlp(input_item)

        item_embedding = F.normalize(
            item_embedding,
            p=2,
            dim=1,
        )

        return item_embedding
