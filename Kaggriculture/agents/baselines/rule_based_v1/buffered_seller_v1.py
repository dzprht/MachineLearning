try:
    from engine import run_agent
except ImportError:
    from agents.baselines.rule_based_v1.engine import run_agent
CONFIG = {"mode": "crops", "crop": "CARROT", "cells": 12, "workers": 4, "buffered": True, "sell_fraction": 0.92, "sell_batch": 8, "max_wait_days": 3}
def agent(obs):
    return run_agent(obs, CONFIG)
