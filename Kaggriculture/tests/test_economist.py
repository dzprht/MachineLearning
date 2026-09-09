import copy
import inspect
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import torch
from kaggle_environments import make
from stable_baselines3.common.env_checker import check_env

from training.economist import phi as phi_module
from training.economist.constants import ACTION_NAMES, CROPS, MAX_HANDS
from training.economist.encoder import FEATURE_COUNT, ObservationEncoder
from training.economist.executor import EconomistExecutor
from training.economist.export import NumpyActor, export_actor
from training.economist.gym_env import KaggricultureEconomistEnv
from training.economist.phi import compute_phi
from training.economist.ppo import build_model
from training.economist.pricing import batch_revenue, market_price
from training.economist.runtime import EconomistRuntime
from training.economist.train import build_vector_env, run_training


class PricingTest(unittest.TestCase):
    def test_matches_real_environment_curve(self):
        from kaggle_environments.envs.kaggriculture.kaggriculture import PRODUCTS as REAL_PRODUCTS
        from kaggle_environments.envs.kaggriculture.kaggriculture import market_price as real_price

        for item in REAL_PRODUCTS:
            for inventory in (0, 1, 100, 5000, 9999, 10000, 10001, 15000, 50000, 200000):
                self.assertEqual(market_price(item, inventory), real_price(item, inventory))

    def test_batch_revenue_is_a_sum_of_unit_prices(self):
        price0 = market_price("CARROT", 10000)
        price1 = market_price("CARROT", 10001)
        expected = price0 + price1
        self.assertEqual(batch_revenue("CARROT", 10000, 2), expected)

    def test_fractional_batch_interpolates(self):
        whole = batch_revenue("WHEAT", 10000, 1)
        half = batch_revenue("WHEAT", 10000, 0.5)
        self.assertAlmostEqual(half, 0.5 * whole, places=6)


class EncoderTest(unittest.TestCase):
    def test_schema_is_finite_and_role_relative(self):
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 11})
        obs = env.state[0].observation
        encoder = ObservationEncoder(env.configuration)
        executor = EconomistExecutor(env.configuration)
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


class PhiStructuralTest(unittest.TestCase):
    def test_never_calls_env_step(self):
        source = inspect.getsource(phi_module)
        for forbidden in ("kaggle_environments", ".step(", "KaggricultureEconomistEnv", "make("):
            self.assertNotIn(forbidden, source)

    def test_empty_portfolio_is_zero(self):
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 5})
        obs = env.state[0].observation
        self.assertEqual(compute_phi(obs, EconomistExecutor(env.configuration).state, env.configuration), 0.0)

    def test_deterministic_and_finite(self):
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 6})
        obs = env.state[0].observation
        obs["farms"][0]["tiles"][0][0] = {
            "kind": "PLANT", "crop": "CARROT", "planted_day": 0, "watered_today": True,
            "consecutive_unwatered": 0, "yield_units": 1, "max_lifespan_step": 96, "fertilized_until_day": -1,
        }
        state = EconomistExecutor(env.configuration).state
        first = compute_phi(obs, state, env.configuration)
        second = compute_phi(obs, state, env.configuration)
        self.assertEqual(first, second)
        self.assertTrue(np.isfinite(first))

    def test_shed_batch_matches_pricing_directly(self):
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 7})
        obs = env.state[0].observation
        obs["private"]["shed"]["CARROT"] = 5
        state = EconomistExecutor(env.configuration).state
        phi = compute_phi(obs, state, env.configuration)
        inventory = obs["market"]["inventory"]["CARROT"]
        expected = batch_revenue("CARROT", inventory, 5)
        self.assertAlmostEqual(phi, expected, places=4)

    def test_no_days_left_is_zero(self):
        env = make("kaggriculture", configuration={"episodeSteps": 48, "seed": 8})
        obs = env.state[0].observation
        obs["step"] = 47
        obs["day"] = 1
        obs["hour"] = 23
        obs["private"]["shed"]["CARROT"] = 5
        state = EconomistExecutor(env.configuration).state
        self.assertEqual(compute_phi(obs, state, env.configuration), 0.0)

    def test_reasonably_fast(self):
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 9})
        obs = env.state[0].observation
        state = EconomistExecutor(env.configuration).state
        for x, y in [(0, 0), (1, 0), (2, 0), (0, 1)]:
            obs["farms"][0]["tiles"][y][x] = {
                "kind": "PLANT", "crop": "TOMATO", "planted_day": -5, "watered_today": True,
                "consecutive_unwatered": 0, "yield_units": 1, "max_lifespan_step": -1, "fertilized_until_day": -1,
            }
        started = time.perf_counter()
        for _ in range(20):
            compute_phi(obs, state, env.configuration)
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 2.0)


class ExecutorMaskTest(unittest.TestCase):
    def test_crop_masks_require_money_free_cell_and_time(self):
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 12})
        obs = env.state[0].observation
        obs["farms"][0]["money"] = 5  # below every seed cost
        executor = EconomistExecutor(env.configuration)
        mask = executor.action_mask(obs)
        self.assertFalse(any(mask[: len(CROPS)]))
        self.assertTrue(mask[ACTION_NAMES.index("WAIT")])

    def test_hire_mask_respects_cap_and_money(self):
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 13})
        obs = env.state[0].observation
        executor = EconomistExecutor(env.configuration)
        self.assertTrue(executor.action_mask(obs)[ACTION_NAMES.index("HIRE_ONE")])
        obs["farms"][0]["hands"] = [[5, 5]] * MAX_HANDS
        self.assertFalse(executor.action_mask(obs)[ACTION_NAMES.index("HIRE_ONE")])
        obs["farms"][0]["hands"] = []
        obs["farms"][0]["money"] = 0
        self.assertFalse(executor.action_mask(obs)[ACTION_NAMES.index("HIRE_ONE")])

    def test_pending_reservation_blocks_further_crop_decisions(self):
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 14})
        obs = env.state[0].observation
        executor = EconomistExecutor(env.configuration)
        executor.register_decision(ACTION_NAMES.index("CARROT"), obs)
        self.assertIsNotNone(executor.state.pending_crop)
        mask = executor.action_mask(obs)
        self.assertFalse(any(mask[: len(CROPS)]))
        self.assertTrue(mask[ACTION_NAMES.index("WAIT")])
        self.assertTrue(mask[ACTION_NAMES.index("HIRE_ONE")])

    def test_reset_clears_pending_reservation(self):
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 16})
        obs = env.state[0].observation
        executor = EconomistExecutor(env.configuration)
        executor.register_decision(ACTION_NAMES.index("WHEAT"), obs)
        self.assertIsNotNone(executor.state.pending_crop)
        executor.reset()
        self.assertIsNone(executor.state.pending_crop)
        self.assertIsNone(executor.state.pending_cell)
        self.assertFalse(executor.state.hire_requested)

    def test_end_of_season_masks_crops_that_cannot_finish(self):
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 15})
        obs = env.state[0].observation
        obs["step"] = 696
        obs["day"] = 29
        obs["hour"] = 0
        executor = EconomistExecutor(env.configuration)
        mask = executor.action_mask(obs)
        self.assertFalse(mask[ACTION_NAMES.index("MELON")])


class EconomistEnvTest(unittest.TestCase):
    def test_wait_is_one_primitive_turn(self):
        env = KaggricultureEconomistEnv("train_v1", master_seed=8008, phi_enabled=False)
        try:
            env.reset(seed=20)
            _, reward, terminated, truncated, info = env.step(ACTION_NAMES.index("WAIT"))
            self.assertEqual(info["total_primitive_steps"], 1)
            self.assertFalse(terminated)
            self.assertFalse(truncated)
        finally:
            env.close()

    def test_every_crop_eventually_completes_a_cycle(self):
        for crop in CROPS:
            with self.subTest(crop=crop):
                env = KaggricultureEconomistEnv("train_v1", master_seed=8008, phi_enabled=False)
                try:
                    env.reset(seed=100)
                    action = ACTION_NAMES.index(crop)
                    terminated = False
                    harvested = False
                    turns = 0
                    while not terminated and turns < 600:
                        mask = env.action_masks()
                        chosen = action if mask[action] else ACTION_NAMES.index("WAIT")
                        _, _, terminated, _, _ = env.step(chosen)
                        turns += 1
                        if env.executor.state.pending_crop is None and turns > 5:
                            action = ACTION_NAMES.index("WAIT")
                        if float(env._obs["private"]["shed"].get(crop, 0)) > 0 or float(
                            env._obs["private"]["seeds"].get(crop, 0)
                        ) == 0 and turns > 5:
                            harvested = True
                    self.assertTrue(harvested)
                finally:
                    env.close()

    def test_several_cells_grow_simultaneously_over_a_season(self):
        env = KaggricultureEconomistEnv(
            "train_v1", master_seed=8008, configuration={"episodeSteps": 400}, phi_enabled=False,
        )
        try:
            env.reset(seed=101)
            wheat = ACTION_NAMES.index("WHEAT")
            max_simultaneous = 0
            terminated = False
            turns = 0
            while not terminated and turns < 400:
                mask = env.action_masks()
                action = wheat if mask[wheat] else ACTION_NAMES.index("WAIT")
                _, _, terminated, _, _ = env.step(action)
                turns += 1
                farm = env._obs["farms"][env._selection.position]
                planted = sum(
                    isinstance(tile, dict) and tile.get("kind") == "PLANT"
                    for row in farm["tiles"]
                    for tile in row
                )
                max_simultaneous = max(max_simultaneous, planted)
            self.assertGreaterEqual(max_simultaneous, 2)
        finally:
            env.close()

    def test_money_only_reward_telescopes_to_money_change(self):
        env = KaggricultureEconomistEnv(
            "train_v1", master_seed=8008,
            configuration={"episodeSteps": 96}, phi_enabled=False,
        )
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
            self.assertAlmostEqual(total_reward, 0.01 * (final_money - initial_money), places=4)
        finally:
            env.close()

    def test_analytic_v1_reward_telescopes_with_phi_correction(self):
        env = KaggricultureEconomistEnv(
            "train_v1", master_seed=8008,
            configuration={"episodeSteps": 96}, phi_enabled=True,
        )
        try:
            env.reset(seed=31)
            initial_money = env._money(env._obs)
            initial_phi = env._phi
            total_reward = 0.0
            terminated = False
            while not terminated:
                mask = env.action_masks()
                carrot = ACTION_NAMES.index("CARROT")
                action = carrot if mask[carrot] else ACTION_NAMES.index("WAIT")
                _, reward, terminated, _, _ = env.step(action)
                total_reward += reward
            final_money = env._money(env._obs)
            self.assertEqual(env._phi, 0.0)  # real terminal forces Phi=0
            expected = 0.01 * (final_money - initial_money - initial_phi)
            self.assertAlmostEqual(total_reward, expected, places=4)
        finally:
            env.close()

    def test_invalid_action_falls_back_to_wait(self):
        env = KaggricultureEconomistEnv("train_v1", master_seed=8008, phi_enabled=False)
        try:
            env.reset(seed=32)
            position = env._selection.position
            env._obs["farms"][position]["money"] = 0  # forces every crop action off
            mask = env.action_masks()
            blocked = ACTION_NAMES.index("WHEAT")
            self.assertFalse(mask[blocked])
            _, _, _, _, info = env.step(blocked)
            self.assertEqual(info["invalid_action"], blocked)
        finally:
            env.close()

    def test_gym_contract_and_numpy_export_match_sb3(self):
        env = KaggricultureEconomistEnv("train_v1", master_seed=8008, phi_enabled=False)
        try:
            check_env(env, warn=True)
            obs, _ = env.reset(seed=40)
            model = build_model(env, seed=8008)
            actor_layers = [
                layer.out_features
                for layer in model.policy.mlp_extractor.policy_net
                if isinstance(layer, torch.nn.Linear)
            ]
            self.assertEqual(actor_layers, [128, 128])
            self.assertEqual(model.gamma, 1.0)
            with tempfile.TemporaryDirectory() as temporary:
                weights = Path(temporary) / "actor.npz"
                schema = Path(temporary) / "feature_schema.json"
                export_actor(model, weights, env.encoder.schema(), schema)
                actor = NumpyActor.load(weights)
                with torch.no_grad():
                    features = model.policy.extract_features(torch.as_tensor(obs).reshape(1, -1))
                    latent_pi, _ = model.policy.mlp_extractor(features)
                    expected = model.policy.action_net(latent_pi).cpu().numpy()[0]
                actual = actor.logits(obs)
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)
                mask = env.action_masks()
                self.assertEqual(actor.predict(obs, mask), int(np.argmax(np.where(mask, expected, -np.inf))))
        finally:
            env.close()

    def test_subprocess_vector_env_exposes_masks(self):
        from sb3_contrib.common.maskable.utils import get_action_masks

        env = build_vector_env("train_v1", master_seed=8008, n_envs=2, phi_enabled=False)
        try:
            observations = env.reset()
            self.assertEqual(observations.shape, (2, FEATURE_COUNT))
            masks = get_action_masks(env)
            self.assertEqual(masks.shape, (2, len(ACTION_NAMES)))
            _, rewards, dones, infos = env.step(
                np.asarray([ACTION_NAMES.index("WAIT"), ACTION_NAMES.index("WAIT")])
            )
            np.testing.assert_array_equal(rewards, np.zeros(2))
            np.testing.assert_array_equal(dones, np.zeros(2, dtype=bool))
            self.assertEqual([info["total_primitive_steps"] for info in infos], [1, 1])
        finally:
            env.close()

    def test_numpy_runtime_completes_a_full_kaggle_season(self):
        source_env = KaggricultureEconomistEnv("train_v1", master_seed=8008, phi_enabled=False)
        try:
            source_env.reset(seed=50)
            model = build_model(source_env, seed=8008)
            with tempfile.TemporaryDirectory() as temporary:
                weights = Path(temporary) / "actor.npz"
                schema = Path(temporary) / "feature_schema.json"
                export_actor(model, weights, source_env.encoder.schema(), schema)
                runtime = EconomistRuntime(weights)
                game = make(
                    "kaggriculture", configuration={"episodeSteps": 720, "seed": 51}, debug=True,
                )
                baseline = (
                    Path(__file__).resolve().parents[1]
                    / "agents" / "baselines" / "carrot_loop_v1" / "main.py"
                )
                game.run([runtime.act, str(baseline)])
                self.assertEqual([state.status for state in game.steps[-1]], ["DONE", "DONE"])
        finally:
            source_env.close()


class TrainSmokeTest(unittest.TestCase):
    def test_short_smoke_train_and_resume(self):
        import json
        import shutil

        with tempfile.TemporaryDirectory() as temporary:
            # Path("repo_root") / "<absolute path>" resolves to the absolute
            # path itself (pathlib semantics), so run_training's REPO_ROOT
            # join works unchanged with an out-of-repo scratch directory.
            output_dir = Path(temporary) / "smoke_run"
            config_path = Path(temporary) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "train_pool": "train_v1",
                        "master_seed": 1,
                        "primitive_budget": 300,
                        "n_envs": 2,
                        "output_dir": str(output_dir),
                        "phi_enabled": False,
                    }
                )
            )
            result_dir = run_training(config_path)
            for name in ("actor.npz", "feature_schema.json", "sb3_model.zip", "progress.json", "run_manifest.json"):
                self.assertTrue((result_dir / name).exists(), name)
            progress = json.loads((result_dir / "progress.json").read_text())
            self.assertGreaterEqual(progress["primitive_steps"], 300)

            # Resuming with a bigger budget continues instead of restarting.
            config_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "train_pool": "train_v1",
                        "master_seed": 1,
                        "primitive_budget": 600,
                        "n_envs": 2,
                        "output_dir": str(output_dir),
                        "phi_enabled": False,
                    }
                )
            )
            run_training(config_path)
            progress_after = json.loads((result_dir / "progress.json").read_text())
            self.assertGreater(progress_after["primitive_steps"], progress["primitive_steps"])
            shutil.rmtree(result_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
