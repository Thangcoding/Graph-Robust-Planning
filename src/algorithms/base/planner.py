from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, override

import gymnasium as gym

from .config import BasePlannerConfig

@dataclass(eq=False, slots=True)
class Action:
    T: int = 0                          # T(s, a)
    Q: float = 0.0                      # Q(s, a)
    R: float | list[float] = 0.0        # R(s, a)
    next: dict[Node, int] = field(default_factory=dict)  # T(s,a,s')

@dataclass(eq=False, slots=True)
class Node:

    id: int          
    state: Any       
    depth: int       
    terminal: bool         
    T: int = 0        # T(s)
    V: float = 0.0    # V(s)
    actions: dict[int, Action] | None = field(default=None, repr=False)  # None until expanded

    @property
    def is_expanded(self) -> bool:
        return self.actions is not None 

# one step of a simulation: (node, action, reward, child)
PathStep = tuple[Node, int, float, Node]

class BasePlanner(ABC):

    config_cls: type[BasePlannerConfig] = BasePlannerConfig
    name: str = "base_search"

    def __init__(self, env: gym.Env | None = None, config: BasePlannerConfig | None = None):
        self.config = config if config is not None else self.config_cls()
        self.env = env
        self.rng = random.Random(self.config.seed)
        self.root: Node | None = None
        self.reset_tables()

    def reset_tables(self) -> None:
        self.nodes: list[Node] = []           # id -> node
        self.transition_table: dict[Any, Node] = {}    # lookup_key -> node (every node except the root)

    def new_node(self, s, depth: int, terminal: bool) -> Node:
        node = Node(len(self.nodes), s, depth, terminal)
        self.nodes.append(node)
        return node


    @abstractmethod
    def lookup_key(self, parent: Node, a: int, s_next):
        ''' Key to verity tree or graph ''' 

    def get_or_create(self, parent: Node, a: int, s_next: Any, terminal: bool) -> tuple[Node, bool]:
        """Return (child, is_new_node) and register the edge (parent, a) -> child with count 0.

        The transition count T(s, a, s') is increased in backup, together with T(s, a)."""
        key = self.lookup_key(parent, a, s_next)
        child = self.transition_table.get(key)   
        is_new = child is None   
        if is_new:
            child = self.transition_table[key] = self.new_node(s_next, parent.depth + 1, terminal)
        parent.actions[a].next.setdefault(child, 0)
        return child, is_new

    @override 
    def expand(self, node: Node, env: gym.Env) -> None:
        node.actions = {a: Action() for a in range(env.action_space.n)}

    @abstractmethod
    def select_action(self, node: Node) -> int:
        """Tree policy at an expanded node."""

    @abstractmethod
    def backup(self, path: list[PathStep], leaf: Node, leaf_value: float) -> None:
        """Update the statistics after one simulation."""

    def rollout_policy(self, env: gym.Env) -> int:
        return self.rng.randrange(env.action_space.n)

    def rollout(self, env: gym.Env) -> float:

        return_val, discount = 0.0, 1.0
        for _ in range(self.config.rollout_depth):
            _, reward, terminated, truncated, _ = env.step(self.rollout_policy(env))
            return_val += discount * reward 
            discount *= self.config.gamma
            if terminated or truncated:
                break
    
        return return_val 

    def leaf_value(self, leaf: Node, env: gym.Env) -> float:
        """Bootstrap value of the leaf: 0 for a terminal node, otherwise a rollout estimate."""
        return 0.0 if leaf.terminal else self.rollout(env)
    
    @abstractmethod 
    def simulation(self, env: gym.Env) -> None:
        """ simulate four step Monte Carlo Tree Search """

    def argmax(self, items, key):
        """argmax with uniform random tie-breaking."""
        best, best_val = [], -math.inf
        for it in items:
            val = key(it)
            if val > best_val:
                best, best_val = [it], val
            elif val == best_val:
                best.append(it)
        return self.rng.choice(best)

    def get_plan(self, root: Node) -> int:
        stats = root.actions
        if self.config.final_action == "most_visited":
            key = lambda a: stats[a].T
        else:
            key = lambda a: stats[a].Q if stats[a].T > 0 else -math.inf
        return self.argmax(stats, key)

    def plan(self, env: gym.Env | None = None) -> int:
        env =  env if env is not None else self.env 
        self.reset_tables()
        self.root = self.new_node(env.get_state(), 0, False)
        for _ in range(self.config.num_simulations):
            self.simulation(env.clone())
        return self.get_plan(self.root)

    def reset(self) -> None:
        self.root = None
        self.rng = random.Random(self.config.seed)
        self.reset_tables()

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    def root_stats(self) -> dict[int, dict[str, float]]:
        if self.root is None or not self.root.is_expanded:
            return {}
        return {a: {"visits": st.T, "q": st.Q} for a, st in self.root.actions.items()}

    def get_name(self) -> str:
        return self.name