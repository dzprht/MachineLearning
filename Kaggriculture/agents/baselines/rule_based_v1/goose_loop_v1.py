try:
    from engine import run_agent
except ImportError:
    from agents.baselines.rule_based_v1.engine import run_agent
CONFIG = {"mode": "animals", "animal": "GOOSE", "animals": 1, "workers": 0, "cash_reserve": 100}
def agent(obs):
    return run_agent(obs, CONFIG)
