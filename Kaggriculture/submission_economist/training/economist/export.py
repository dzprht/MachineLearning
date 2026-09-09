"""Export SB3 actor parameters and run dependency-light NumPy inference.

Same three-affine-layer v1 architecture and export/load contract as
`training/seed_buyer/export.py`, duplicated (not imported) because the two
packages' `ACTION_NAMES`/`FEATURE_SCHEMA_VERSION` differ and each is meant to
stay a self-contained submission bundle (see agents/README.md).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .constants import ACTION_NAMES, FEATURE_SCHEMA_VERSION


class NumpyActor:
    def __init__(self, weights: list[np.ndarray], biases: list[np.ndarray]):
        if len(weights) != 3 or len(biases) != 3:
            raise ValueError("The v1 actor requires exactly three affine layers")
        self.weights = [np.asarray(weight, dtype=np.float32) for weight in weights]
        self.biases = [np.asarray(bias, dtype=np.float32) for bias in biases]

    @classmethod
    def load(cls, path) -> "NumpyActor":
        with np.load(path, allow_pickle=False) as data:
            schema_version = int(data["schema_version"])
            actions = tuple(str(item) for item in data["action_names"])
            if schema_version != FEATURE_SCHEMA_VERSION:
                raise ValueError(f"Unsupported feature schema version: {schema_version}")
            if actions != ACTION_NAMES:
                raise ValueError(f"Unexpected action order: {actions}")
            return cls(
                [data[f"weight_{index}"] for index in range(3)],
                [data[f"bias_{index}"] for index in range(3)],
            )

    def logits(self, observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float32)
        for index in range(2):
            value = np.tanh(value @ self.weights[index].T + self.biases[index])
        return value @ self.weights[2].T + self.biases[2]

    def predict(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        mask = np.asarray(action_mask, dtype=bool)
        if mask.shape != (len(ACTION_NAMES),) or not mask.any():
            raise ValueError(f"action_mask must enable at least one of the {len(ACTION_NAMES)} actions")
        logits = self.logits(observation)
        return int(np.argmax(np.where(mask, logits, -np.inf)))


def export_actor(model, weights_path, schema: dict, schema_path) -> None:
    """Export the two policy layers and final action head from MaskablePPO."""
    import torch

    hidden = [
        layer
        for layer in model.policy.mlp_extractor.policy_net
        if isinstance(layer, torch.nn.Linear)
    ]
    layers = [*hidden, model.policy.action_net]
    if len(layers) != 3:
        raise ValueError(f"Expected three actor affine layers, found {len(layers)}")
    payload = {
        "schema_version": np.asarray(FEATURE_SCHEMA_VERSION, dtype=np.int64),
        "action_names": np.asarray(ACTION_NAMES, dtype="U16"),
    }
    for index, layer in enumerate(layers):
        payload[f"weight_{index}"] = layer.weight.detach().cpu().numpy().astype(np.float32)
        payload[f"bias_{index}"] = layer.bias.detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(weights_path, **payload)
    Path(schema_path).write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
