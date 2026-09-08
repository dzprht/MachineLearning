import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from kaggle_environments import make
from stable_baselines3.common.env_checker import check_env

from training.seed_buyer.constants import ACTION_NAMES, CROPS, TARGET_TILE
from training.seed_buyer.encoder import FEATURE_COUNT, ObservationEncoder
from training.seed_buyer.executor import SeedBuyerExecutor
from training.seed_buyer.export import NumpyActor, export_actor
from training.seed_buyer.gym_env import KaggricultureSeedBuyerEnv
from training.seed_buyer.ppo import build_model
from training.seed_buyer.runtime import SeedBuyerRuntime
from training.seed_buyer.train import build_vector_env


class SeedBuyerTest(unittest.TestCase):
    def test_encoder_schema_is_finite_and_role_relative(self):
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 11})
        obs = env.state[0].observation
        encoder = ObservationEncoder(env.configuration)
        executor = SeedBuyerExecutor(env.configuration)
        encoded = encoder.encode(obs, executor.state)
        self.assertEqual(encoded.shape, (FEATURE_COUNT,))
        self.assertEqual(len(encoder.feature_names), FEATURE_COUNT)
        self.assertTrue(np.isfinite(encoded).all())
        self.assertTrue((encoded >= -1).all() and (encoded <= 1).all())

        swapped = copy.deepcopy(obs)
        swapped["farms"] = [copy.deepcopy(obs["farms"][1]), copy.deepcopy(obs["farms"][0])]
        swapped["player"] = 1
        np.testing.assert_allclose(encoded, encoder.encode(swapped, executor.state))

        hidden = copy.deepcopy(obs)
        hidden["farms"][1]["private"] = {"seeds": {"MELON": 99999}}
        np.testing.assert_allclose(encoded, encoder.encode(hidden, executor.state))

    def test_wait_is_exactly_one_day(self):
        env = KaggricultureSeedBuyerEnv("train_v1", master_seed=8008)
        try:
            env.reset(seed=20)
            _, reward, terminated, truncated, info = env.step(ACTION_NAMES.index("WAIT"))
            self.assertEqual(info["primitive_steps"], 24)
            self.assertEqual(reward, 0.0)
            self.assertFalse(terminated)
            self.assertFalse(truncated)
        finally:
            env.close()

    def test_every_crop_completes_with_one_purchase_and_all_yields(self):
        expected_harvests = {crop: (4 if crop in ("TOMATO", "STRAWBERRY") else 1) for crop in CROPS}
        for index, crop in enumerate(CROPS):
            with self.subTest(crop=crop):
                env = KaggricultureSeedBuyerEnv("train_v1", master_seed=8008)
                try:
                    env.reset(seed=100 + index)
                    _, _, terminated, _, _ = env.step(index)
                    self.assertFalse(terminated)
                    position = env._selection.position
                    actions = [state[position].get("action", {}) for state in env._kaggle_env.steps]
                    buys = [
                        order
                        for action in actions
                        for order in action.get("market", [])
                        if order[:2] == ["BUY_SEED", crop]
                    ]
                    harvests = [
                        action for action in actions if action.get("farmer") == ["HARVEST"]
                    ]
                    self.assertEqual(len(buys), 1)
                    self.assertEqual(len(harvests), expected_harvests[crop])
                    private = env._obs["private"]
                    x, y = TARGET_TILE
                    self.assertIsNone(env._obs["farms"][position]["tiles"][y][x])
                    self.assertEqual(private["seeds"].get(crop, 0), 0)
                    self.assertEqual(private["shed"].get(crop, 0), 0)
                    self.assertEqual(private["inventories"][0].get(crop, 0), 0)
                    self.assertTrue(env.executor.needs_decision())
                finally:
                    env.close()

    def test_end_of_season_masks_incomplete_cycles(self):
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 12})
        obs = env.state[0].observation
        obs["step"] = 400
        obs["day"] = 16
        obs["hour"] = 16
        executor = SeedBuyerExecutor(env.configuration)
        mask = executor.action_mask(obs)
        self.assertFalse(mask[ACTION_NAMES.index("STRAWBERRY")])
        self.assertTrue(mask[ACTION_NAMES.index("CARROT")])
        self.assertTrue(mask[ACTION_NAMES.index("WAIT")])

    def test_interval_rewards_telescope_to_money_change(self):
        env = KaggricultureSeedBuyerEnv("train_v1", master_seed=8008)
        try:
            env.reset(seed=30)
            initial_money = env._money(env._obs)
            total_reward = 0.0
            terminated = False
            while not terminated:
                mask = env.action_masks()
                carrot = ACTION_NAMES.index("CARROT")
                action = carrot if mask[carrot] else ACTION_NAMES.index("WAIT")
                _, reward, terminated, _, _ = env.step(action)
                total_reward += reward
            final_money = env._money(env._obs)
            self.assertAlmostEqual(total_reward, 0.01 * (final_money - initial_money), places=6)
        finally:
            env.close()

    def test_gym_contract_and_numpy_export_match_sb3(self):
        env = KaggricultureSeedBuyerEnv("train_v1", master_seed=8008)
        try:
            check_env(env, warn=True)
            obs, _ = env.reset(seed=40)
            model = build_model(env, seed=8008)
            actor_layers = [
                layer.out_features
                for layer in model.policy.mlp_extractor.policy_net
                if isinstance(layer, torch.nn.Linear)
            ]
            critic_layers = [
                layer.out_features
                for layer in model.policy.mlp_extractor.value_net
                if isinstance(layer, torch.nn.Linear)
            ]
            self.assertEqual(actor_layers, [128, 128])
            self.assertEqual(critic_layers, [128, 128])
            self.assertEqual(model.gamma, 1.0)
            with tempfile.TemporaryDirectory() as temporary:
                weights = Path(temporary) / "actor.npz"
                schema = Path(temporary) / "feature_schema.json"
                export_actor(model, weights, env.encoder.schema(), schema)
                actor = NumpyActor.load(weights)
                with torch.no_grad():
                    features = model.policy.extract_features(
                        torch.as_tensor(obs).reshape(1, -1)
                    )
                    latent_pi, _ = model.policy.mlp_extractor(features)
                    expected = model.policy.action_net(latent_pi).cpu().numpy()[0]
                actual = actor.logits(obs)
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)
                mask = env.action_masks()
                self.assertEqual(actor.predict(obs, mask), int(np.argmax(np.where(mask, expected, -np.inf))))
                self.assertTrue(schema.is_file())
        finally:
            env.close()

    def test_subprocess_vector_env_exposes_masks(self):
        from sb3_contrib.common.maskable.utils import get_action_masks

        env = build_vector_env("train_v1", master_seed=8008, n_envs=2)
        try:
            observations = env.reset()
            self.assertEqual(observations.shape, (2, FEATURE_COUNT))
            masks = get_action_masks(env)
            self.assertEqual(masks.shape, (2, len(ACTION_NAMES)))
            self.assertTrue(masks.all())
            _, rewards, dones, infos = env.step(
                np.asarray([ACTION_NAMES.index("WAIT"), ACTION_NAMES.index("WAIT")])
            )
            np.testing.assert_array_equal(rewards, np.zeros(2))
            np.testing.assert_array_equal(dones, np.zeros(2, dtype=bool))
            self.assertEqual([info["primitive_steps"] for info in infos], [24, 24])
        finally:
            env.close()

    def test_numpy_runtime_completes_a_full_kaggle_season(self):
        source_env = KaggricultureSeedBuyerEnv("train_v1", master_seed=8008)
        try:
            source_env.reset(seed=50)
            model = build_model(source_env, seed=8008)
            with tempfile.TemporaryDirectory() as temporary:
                weights = Path(temporary) / "actor.npz"
                schema = Path(temporary) / "feature_schema.json"
                export_actor(model, weights, source_env.encoder.schema(), schema)
                runtime = SeedBuyerRuntime(weights)
                game = make(
                    "kaggriculture",
                    configuration={"episodeSteps": 720, "seed": 51},
                    debug=True,
                )
                baseline = (
                    Path(__file__).resolve().parents[1]
                    / "agents"
                    / "baselines"
                    / "carrot_loop_v1"
                    / "main.py"
                )
                game.run([runtime.act, str(baseline)])
                self.assertEqual([state.status for state in game.steps[-1]], ["DONE", "DONE"])
        finally:
            source_env.close()


if __name__ == "__main__":
    unittest.main()
