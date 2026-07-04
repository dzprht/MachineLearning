import torch
from torch import nn

from collections.abc import Iterable
from activations import ActivationFunction, activations_func

from features import SparseFeature, SequenceFeature, DenseFeature


class InputMask(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, features):
        mask = []

        if not isinstance(features, list):
            features = [features]
        for feature in features:
            if isinstance(feature, SparseFeature) or isinstance(
                feature, SequenceFeature
            ):
                if feature.padding_idx is not None:
                    feature_mask = x[feature.name].long() != feature.padding_idx
                else:
                    feature_mask = x[feature.name].long() != -1
                mask.append(feature_mask.unsqueeze(1).float())
            else:
                raise ValueError(
                    "Only supports <SparseFeature> or <SequenceFeature> to get mask"
                )

        return torch.cat(mask, dim=1)


class SumPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        if mask is None:
            return torch.sum(x, 1)
        else:
            return torch.bmm(mask, x).squeeze(1)


class AveragePooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        if mask is None:
            return torch.mean(x, dim=1)
        else:
            sum_pooling_matrix = torch.bmm(mask, x).squeeze(1)
            non_padding_length = mask.sum(dim=-1)

            return sum_pooling_matrix / (non_padding_length.float() + 1e-16)


class ConcatPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        return x


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


class EmbeddingLayer(nn.Module):
    """forward() должен возвращать эмбеддинги стандартизированного вида"""

    def __init__(
        self,
        features,
    ) -> None:
        self.features = features
        self.embed_dict = nn.ModuleDict()
        self.n_dense = 0

        for feature in features:
            if feature in self.embed_dict:
                continue
            if isinstance(feature, SparseFeature) and feature.shared_with is None:
                self.embed_dict[feature.name] = feature.get_embedding_layer()
            elif isinstance(feature, SequenceFeature) and feature.shared_with is None:
                self.embed_dict[feature.name] = feature.get_embedding_layer()
            elif isinstance(feature, DenseFeature):
                self.n_dense += 1  # пока не используется но может как нибудь потом будет полезен для каких то чеков; не знаю

    def forward(
        self,
        x,
        features: Iterable,
        squeeze_dim: bool = True,
    ) -> torch.Tensor:
        sparse_embeddings, dense_values = [], []
        sparse_exists, dense_exists = False, False

        for feature in features:
            if isinstance(feature, SparseFeature):
                if feature.shared_with is None:
                    sparse_embeddings.append(
                        self.embed_dict[feature.name](x[feature.name].long()).unsqueeze(
                            1
                        )
                    )
                else:
                    sparse_embeddings.append(
                        self.embed_dict[feature.shared_with](
                            x[feature.name].long()
                        ).unsqeeze(1)
                    )
            elif isinstance(feature, SequenceFeature):
                if feature.pooling == "sum":
                    pooling_layer = SumPooling()
                elif feature.pooling == "mean":
                    pooling_layer = AveragePooling()
                elif feature.pooling == "concat":
                    pooling_layer = ConcatPooling()
                else:
                    raise ValueError(
                        "Sequence pooling method supports only pooling in %s, got %s."
                        % (["sum", "mean"], feature.pooling)
                    )
                feature_mask = InputMask()(x, feature)

                if feature.shared_with is None:
                    sparse_embeddings.append(
                        pooling_layer(
                            self.embed_dict[feature.name](
                                x[feature.name].long(), feature_mask
                            ).unsqeeze(1)
                        )
                    )
                else:
                    sparse_embeddings.append(
                        pooling_layer(
                            self.embed_dict[feature.shared_with](
                                x[feature.name].long(), feature_mask
                            ).unsqeeze(1)
                        )
                    )
            else:
                dense_values.append(x[feature.name].float().unsqueeze(1))

        if len(sparse_embeddings) > 0:
            sparse_exists = True
            sparse_embeddings = torch.cat(sparse_embeddings, dim=1)
        if len(dense_values) > 0:
            dense_exists = True
            dense_values = torch.cat(dense_values, dim=1)

        if squeeze_dim:
            if dense_exists and not sparse_exists:
                return dense_values
            elif not dense_exists and sparse_exists:
                return sparse_embeddings.flatten(start_dim=1)
            elif dense_exists and sparse_exists:
                return torch.cat((sparse_embeddings.flatten(start_dim=1), dense_values), dim=1)
            else:
                raise ValueError("The input features can note be empty")
        else:
            if dense_exists:
                UserWarning("Dense features require squeeze_dim=True")
            if sparse_exists:
                return sparse_embeddings

            raise ValueError("Features must not be empty")

        pass
