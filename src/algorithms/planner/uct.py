from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np 

from ..base import BasePlannerConfig, BasePlanner, Node, PathStep
from ..registry import register_planner

@dataclass
class UCTPlannerConfig(BasePlannerConfig):
    name = "uct-config"

@register_planner("uct")
class UCTMCTS(BasePlanner):
    """UCT (Kocsis & Szepesvari, 2006): vanilla MCTS on a tree.

    Nodes     : every (parent node, action, next state) is its own node.
    Selection : a* = argmax_a  Q(s, a) + c * sqrt(ln T(s) / T(s, a))   (untried actions first)
    Backup    : G = r + gamma * G along the path (leaf -> root),
                Q(s, a) <- Q(s, a) + (G - Q(s, a)) / T(s, a)
    
    """

    name: str = "uct"
    config_cls = UCTPlannerConfig

    def __init__(self, env: gym.Env, config: UCTPlannerConfig):
        super().__init__(env, config)

    def lookup_key(self, parent: Node, a: int, s_next):
        return (parent.id, a, s_next)

    def exploration_bonus(self, node: Node, a: int) -> float:
        '''
            UCT policy 
        '''
        T_sa = node.actions[a].T
        if T_sa == 0:
            # try every action once first
            return math.inf 
        return self.config.exploration_const * math.sqrt(math.log(node.T) / T_sa)

    def select_action(self, node: Node) -> int:
        action_space = node.actions.keys()
        UCT_function = lambda a : node.actions[a].Q + self.exploration_bonus(node,a)

        return self.argmax(action_space, key = UCT_function)

    def backup(self, path: list[PathStep], leaf: Node, leaf_value: float) -> None:
        # leaf node: its value is the mean of the leaf values from it (always 0 if terminal)
        leaf.T += 1 
        leaf.V += (leaf_value - leaf.V)/ leaf.T 

        v_back = leaf_value
        for node, a, reward, child in reversed(path):
            G = reward + self.config.gamma * v_back

            # Q-value backup
            action = node.actions[a]         
            action.T += 1                    
            action.Q += (G - action.Q) / action.T      
            action.R += (reward - action.R) / action.T 

            # V-value backup                   
            node.T += 1     
            tried = [act for act in node.actions.values() if act.T > 0]
            list_Q = np.array([act.Q for act in tried])
            weight = np.array([act.T for act in tried]) / sum(act.T for act in tried)

            node.V = float(np.sum(weight * list_Q))
            v_back = node.V

    def simulation(self, env: gym.Env) -> None:
        node = self.root
        path: list[PathStep] = []

        # selection + expansion
        while not node.terminal and node.depth < self.config.max_depth:
            if not node.is_expanded:
                self.expand(node, env)
            a = self.select_action(node)
            obs, reward, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            child, is_new = self.get_or_create(node, a, env.get_state(), done)
            path.append((node, a, reward, child))
            node = child
            if is_new:
                break

        # random rollout at leaf node
        leaf_value = self.leaf_value(node, env)

        # backup
        self.backup(path, node, leaf_value)

