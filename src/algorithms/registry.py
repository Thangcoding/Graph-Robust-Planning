from __future__ import annotations

from typing import Callable

import gymnasium as gym
from .base import BasePlannerConfig

PLANNERS: dict[str, type] = {}

def register_planner(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        if name in PLANNERS:
            raise KeyError(f"Planner '{name}' already registered")
        cls.name = name
        PLANNERS[name] = cls
        return cls
    return deco

def make_planner(name: str, env: gym.Env | None = None, config: BasePlannerConfig | dict | None = None,
                 **overrides):
    if name not in PLANNERS:
        raise KeyError(f"Unknown planner '{name}'. Available: {sorted(PLANNERS)}")
    cls = PLANNERS[name]
    if config is None:
        config = cls.config_cls()
    elif isinstance(config, dict):
        config = cls.config_cls.from_dict(config)
    if overrides:
        config = config.replace(**overrides)
    return cls(env, config)