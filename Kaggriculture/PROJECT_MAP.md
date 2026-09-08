# Project Map

The repository contains task documentation, a frozen deterministic baseline and implemented RL training infrastructure. Training has not been run and no RL weights exist yet.

- `docs/README.md` — detailed game rules, observation/action formats, market curves and default configuration. Season: 720 turns, 24 turns/day; terminal bank balance determines the winner.
- `docs/CONTEXT.md` — read-only task context / getting-started guide, example wheat agent and local/submission workflow. Its claim that BUY_PRODUCT prices are fixed conflicts with README; verify against the environment implementation before coding.
- `docs/IDEAS.md` — experiment proposals and outcomes; consult before experiments.
- `main.py` — current Kaggle-compatible entry point; currently exports the frozen `carrot_loop_v1` baseline.
- `agents/README.md`, `agents/MODEL_CATALOG.md` — versioning/metadata convention and a behavioral index for selecting frozen agents into experiment pools.
- `agents/baselines/carrot_loop_v1/main.py` — self-contained deterministic one-tile carrot policy (`agent(obs)`). Buys one seed ahead, plants, waters daily, harvests after watering on day 3 and sells shed stock; digs a blocking weed.
- `agents/baselines/carrot_loop_v1/metadata.json` — immutable provenance, policy scope, supported observations/actions and evaluation record for the baseline.
- `agents/baselines/rule_based_v1/` — 19 frozen rule-based entrypoints from IDEA-009–018. A shared deterministic executor handles routing, daily work, inventory, hiring and land; per-agent configs isolate crop, market, opponent-aware, livestock and expansion rules. Every entrypoint has its own `*.metadata.json` and hash including the shared executor.
- `training/catalog.py`, `training/pools.py`, `training/poolctl.py` — discover and hash frozen agents; validate, sample and atomically edit named weighted opponent pools. Pool roles are experiment-specific; sampling is deterministic per master/episode seed.
- `training/opponent_pools.json` — current pool definitions. Only `train_v1` exists and currently contains `carrot_loop_v1`.
- `training/seed_buyer/` — 249-feature observation encoder, deterministic full-cycle crop executor, semi-Markov Gymnasium wrapper, fixed MaskablePPO factory, primitive-turn budget callback, NumPy actor export/runtime and explicit future training command.
- `training/README.md` — pool management commands and the deferred experiment-config format.
- `requirements-rl.txt` — pinned SB3 training-only dependencies.
- `tests/` — baseline, rule-based raw-entrypoint/integration behavior, catalog/pool, encoder, executor, reward, Gymnasium, subprocess masking and NumPy-export checks.
- `main.ipynb` — imports `kaggle_environments.make` and creates Kaggriculture with `episodeSteps=720`, `debug=True`. Includes a prior dataset-download path; contains no agent implementation.

Game dynamics, rewards and built-in agents belong to the external `kaggle_environments` package. This map reflects repository files, not a verification of that package's implementation.
