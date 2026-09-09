"""The fixed MaskablePPO v1 architecture and hyperparameters for the Economist.

Hyperparameters are the same starting point as `training/seed_buyer/ppo.py`
(IDEA-020 contract point 6: "Остальные стартовые параметры взять из
training/seed_buyer/ppo.py, включая gamma=1"); only `net_arch` grows to
separate actor/critic networks as the contract specifies, and the action
head now has `len(ACTION_NAMES)` (7) outputs instead of 6.
"""

from __future__ import annotations

from .constants import ACTION_NAMES

PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 256,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 1.0,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}


def build_model(env, seed: int = 8008, device: str = "cpu"):
    """Construct the approved model; this function never starts learning."""
    import torch
    from sb3_contrib import MaskablePPO

    policy_kwargs = {
        "activation_fn": torch.nn.Tanh,
        "net_arch": {"pi": [128, 128], "vf": [128, 128]},
    }
    assert env.action_space.n == len(ACTION_NAMES)
    return MaskablePPO(
        "MlpPolicy",
        env,
        seed=int(seed),
        device=device,
        policy_kwargs=policy_kwargs,
        verbose=1,
        **PPO_CONFIG,
    )
