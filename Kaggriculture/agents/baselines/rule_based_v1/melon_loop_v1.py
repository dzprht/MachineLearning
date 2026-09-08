try:
    from engine import run_agent
except ImportError:
    from agents.baselines.rule_based_v1.engine import run_agent
CONFIG = {"mode": "crops", "crop": "MELON", "cells": 1, "workers": 0}
def agent(obs):
    return run_agent(obs, CONFIG)
