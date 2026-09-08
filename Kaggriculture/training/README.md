# RL training infrastructure

Текущий pipeline реализован, но обучение намеренно не запускалось и сохранённых весов нет. Корневой `main.py` по-прежнему использует `carrot_loop_v1`.

## Opponent pools

Замороженные агенты автоматически находятся по `agents/**/metadata.json`. Текущий состав пулов находится в `opponent_pools.json`.

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

Будущий запуск требует отдельный JSON, которого в репозитории пока нет:

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

После наполнения независимых пулов обучение запускается только явно:

```bash
../ml_venv/bin/python -m training.seed_buyer.train --config EXPERIMENT.json
```

Команда создаёт новый output-каталог, записывает разрешённые пулы и версии зависимостей в `run_manifest.json`, а после обучения сохраняет SB3 checkpoint, NumPy actor и схему признаков.
