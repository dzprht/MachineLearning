try:
    from engine import run_agent
except ImportError:
    from agents.baselines.rule_based_v1.engine import run_agent
CONFIG = {"mode": "animals", "animal": "COW", "animals": 30, "animal_interval": 1, "workers": 8, "expand": True, "cash_reserve": 900, "capacity_per_quadrant": 10}
def agent(obs):
    return run_agent(obs, CONFIG)
