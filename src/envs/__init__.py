from .frozen_lake import FrozenLake
from .gambler import Gambler
from .cliffwalking import CliffWalking
from .american_option import AmericanOption
from .inventory import Inventory

# name -> env class; add new environments here to make them available to scripts/run_plan.py
ENVS: dict[str, type] = {
    "frozen_lake": FrozenLake,
    "gambler": Gambler,
    "cliff_walking": CliffWalking,
    "american_option": AmericanOption,
    "inventory": Inventory,
}


def make_env(name: str, **kwargs):
    if name not in ENVS:
        raise KeyError(f"Unknown env '{name}'. Available: {sorted(ENVS)}")
    return ENVS[name](**kwargs)


__all__ = ["FrozenLake", "Gambler", "CliffWalking", "AmericanOption", "Inventory", "ENVS", "make_env"]












