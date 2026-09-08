try:
    from engine import run_agent
except ImportError:
    from agents.baselines.rule_based_v1.engine import run_agent
CONFIG = {"mode": "mixed", "selector": "staggered", "mix": ("WHEAT", "WHEAT", "CARROT", "TOMATO"), "cells": 12, "animals": 8, "animal_interval": 3, "animal_offset": 12, "workers": 8, "expand": True, "expansion_demand": 35, "cash_reserve": 900, "capacity_per_quadrant": 20}
def agent(obs):
    return run_agent(obs, CONFIG)
