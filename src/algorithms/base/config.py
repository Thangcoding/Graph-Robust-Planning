from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any

import gymnasium as gym

FINAL_ACTIONS = ("most_visited", "max_q")

@dataclass
class BasePlannerConfig:

    num_simulations: int = 1000     
    max_depth: int = 50            
    rollout_depth: int = 20    
    gamma: float = 0.99          
    exploration_const: float = 1.414  
    final_action: str = "most_visited" 
    seed: int | None = None 

    def __post_init__(self):
        if self.num_simulations < 1:
            raise ValueError(f"num_simulations must be >= 1, got {self.num_simulations}")
        if self.max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {self.max_depth}")
        if self.rollout_depth < 0:
            raise ValueError(f"rollout_depth must be >= 0, got {self.rollout_depth}")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")
        if self.exploration_const < 0:
            raise ValueError(f"exploration_const must be >= 0, got {self.exploration_const}")
        if self.final_action not in FINAL_ACTIONS:
            raise ValueError(f"final_action must be one of {FINAL_ACTIONS}, got {self.final_action!r}")

    def to_dict(self) -> dict[str, Any]:
        """Hyper-parameters only (no env), e.g. for logging or saving with results."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BasePlannerConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
        return cls(**d)

    def replace(self, **overrides) -> "BasePlannerConfig":
        return replace(self, **overrides)
