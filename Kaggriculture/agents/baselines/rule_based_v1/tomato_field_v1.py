try:
    from engine import run_agent
except ImportError:
    from agents.baselines.rule_based_v1.engine import run_agent
CONFIG = {"mode": "crops", "crop": "TOMATO", "cells": 12, "workers": 4}
def agent(obs):
    return run_agent(obs, CONFIG)
