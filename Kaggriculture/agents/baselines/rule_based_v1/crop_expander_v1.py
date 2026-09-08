try:
    from engine import run_agent
except ImportError:
    from agents.baselines.rule_based_v1.engine import run_agent
CONFIG = {"mode": "crops", "selector": "demand", "cells": 60, "workers": 8, "expand": True, "cash_reserve": 800, "capacity_per_quadrant": 20}
def agent(obs):
    return run_agent(obs, CONFIG)
