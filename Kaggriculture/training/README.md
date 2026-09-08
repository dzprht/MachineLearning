# RL training infrastructure

Текущий pipeline реализован, но обучение намеренно не запускалось и сохранённых весов нет. Корневой `main.py` по-прежнему использует `carrot_loop_v1`.

## Opponent pools

Замороженные агенты автоматически находятся по `agents/**/metadata.json` и `agents/**/*.metadata.json`. Текущий состав пулов находится в `opponent_pools.json`.

```bash
../ml_venv/bin/python -m training.poolctl list-agents
../ml_venv/bin/python -m training.poolctl list-pools
../ml_venv/bin/python -m training.poolctl validate
../ml_venv/bin/python -m training.poolctl sample train_v1 --episodes 10 --seed 8008
```

Добавление и перевзвешивание версии:

```bash
../ml_venv/bin/python -m training.poolctl add train_v1 AGENT_ID --weight 2
../ml_venv/bin/python -m training.poolctl set-weight train_v1 AGENT_ID 0.5
../ml_venv/bin/python -m training.poolctl remove train_v1 AGENT_ID
```

Новый непустой пул создаётся сразу с первым агентом:

```bash
../ml_venv/bin/python -m training.poolctl create-pool validation_v1 \
  --role validation --agent AGENT_ID --weight 1
```

Выбор соперника и позиции детерминирован парой `(master_seed, episode_seed)`. Пересечение ролей выводится как предупреждение, но не блокирует запуск.

## Seed buyer

`seed_buyer` содержит 249-признаковый encoder, одно-клеточный executor всех пяти культур, Gymnasium environment, фиксированную конфигурацию MaskablePPO, ограничитель бюджета внутренних ходов и экспорт actor в NumPy.

Запуск требует отдельный JSON с конфигурацией эксперимента. `training/experiments/seed_buyer_ppo_v1.json` подготовлен (2026-09-08) и ссылается на заполненный `train_v1`; обучение по нему ещё не запускалось:

```json
{
  "schema_version": 1,
  "train_pool": "train_v1",
  "master_seed": 8008,
  "primitive_budget": 5000000,
  "n_envs": 8,
  "output_dir": "runs/seed_buyer_ppo_v1"
}
```

`primitive_budget` — плейсхолдер из документации, не результат подбора; пересмотреть перед реальным запуском при необходимости.

Обучение запускается только явно:

```bash
../ml_venv/bin/python -m training.seed_buyer.train --config training/experiments/seed_buyer_ppo_v1.json
```

2026-09-08: прогон по этому конфигу завершён (`runs/seed_buyer_ppo_v1/`, `actual_primitive_steps=5000567`), см. результат IDEA-004/008 в `docs/IDEAS.md`. Веса и SB3-чекпоинт лежат в `runs/seed_buyer_ppo_v1/{actor.npz,sb3_model.zip}`; автономный submission собран в `submission/`.

Начиная с этого прогона `train.py` сохраняет прогресс каждые ~1% бюджета (`sb3_model.zip`, `actor.npz`, `progress.json` в `output_dir`) и **продолжает** с последнего чекпоинта, если запустить ту же команду с тем же `--config` повторно — не запускает обучение заново. Повторный запуск после уже достигнутого бюджета — no-op. Чтобы обучить версию с нуля, используйте новый `output_dir`.

Команда создаёт новый output-каталог, записывает разрешённые пулы и версии зависимостей в `run_manifest.json`, а после обучения сохраняет SB3 checkpoint, NumPy actor и схему признаков.
