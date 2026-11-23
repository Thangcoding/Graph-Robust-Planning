from __future__ import annotations

import math
from dataclasses import dataclass

from .gs_robust_uct_f import GSFRobustUCT, GSFRobustUCTPlannerConfig
from ..registry import register_planner


def adaptive_horizon(num_simulations: int, gamma: float) -> int:
    """H = ceil(log N / (2 log(1/gamma))), as GSP-UCT-F+ (gspuct/algorithms/gsp_uct_f.py, plus=True)."""
    return max(1, math.ceil(math.log(max(2, num_simulations)) / (2.0 * math.log(1.0 / gamma))))


@dataclass
class GSFRobustUCTPlusPlannerConfig(GSFRobustUCTPlannerConfig):
    name = 'gsf-robust-uct-plus-config'

    def __post_init__(self):
        # the search depth is set by N and gamma; any max_depth passed in (e.g. tune.FIXED's 50) is replaced
        if self.gamma >= 1.0:
            raise ValueError("adaptive horizon needs gamma < 1")
        self.max_depth = adaptive_horizon(self.num_simulations, self.gamma)
        super().__post_init__()


@register_planner("gsf-robust-uct-plus")
class GSFRobustUCTPlus(GSFRobustUCT):
    """GSF-Robust-UCT with the adaptive search horizon H = ceil(log N / (2 log(1/gamma))) instead of a fixed
    max_depth (gamma 0.99: H = 379 at N = 2000, 413 at N = 4000). Everything else is GSF-Robust-UCT."""
    name: str = 'gsf-robust-uct-plus'
    config_cls = GSFRobustUCTPlusPlannerConfig
