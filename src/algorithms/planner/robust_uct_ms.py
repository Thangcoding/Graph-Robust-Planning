from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import math

from .robust_uct import compute_robust_reward, compute_robust_value
from ..base import Action, BasePlanner, BasePlannerConfig, Node, PathStep
from ..registry import register_planner


@dataclass
class RobustUCTMSPlannerConfig(BasePlannerConfig):
    name = 'robust-uct-ms-config'
    uncertainty_budget : float = 0.6
    uncertainty_type : str = 'tv'
    p : float = 2 # power value


class _ModelSharingRobustUCT(BasePlanner):
    """Robust UCT with model sharing: every edge with the same (s, a) shares one model
    (transition counts T(s,a,s') and reward samples), while Q, V and visit counts stay local to each node.

    The nodes are not merged by the model: lookup_key (tree or graph) is set by the subclass.
    """
    config_cls = RobustUCTMSPlannerConfig

    def reset_tables(self) -> None:
        super().reset_tables()
        # (s, a) -> {"N": T(s,a), "next": {s': T(s,a,s')}, "R": reward samples}
        self.model: dict[Any, dict] = {}

    def lookup_key_model(self, node: Node, a : int):
        return (node.state, a)

    def expand(self, node: Node, env: gym.Env) -> None:
        node.actions = {a: Action() for a in range(env.action_space.n)}

    def exploration_bonus(self, node: Node, a: int) -> float:
        '''
        Stochastic UCT policy
        '''
        T_sa = node.actions[a].T
        if T_sa == 0:
            # try every action once first
            return math.inf
        return self.config.exploration_const * math.sqrt(math.sqrt(node.T) / T_sa)

    def select_action(self, node: Node) -> int:

        Stochastic_UCT_function = lambda a: node.actions[a].Q + self.exploration_bonus(node, a)
        return self.argmax(node.actions,
                           key= Stochastic_UCT_function)

    def backup(self, path, leaf, leaf_value):
        ''' Robust Bellman backup with the shared model: Q(s,a) = robust(R(s,a)) + gamma * robust(V(s')) '''
        leaf.T += 1
        leaf.V += (leaf_value - leaf.V) / leaf.T

        for node , a , reward , child in reversed(path):

            #-----------Q-value backup-----------------
            action = node.actions[a]
            action.T += 1                # local T(s, a)
            action.next[child] += 1      # local T(s, a, s')

            # shared model of (s, a)
            model = self.model.setdefault(self.lookup_key_model(node, a), {"N": 0, "next": {}, "R": []})
            model["N"] += 1
            model["next"][child.state] = model["next"].get(child.state, 0) + 1
            model["R"].append(reward)

            # robust reward from the shared reward samples
            robust_reward = compute_robust_reward(model["R"],
                                                  self.config.uncertainty_budget,
                                                  self.config.uncertainty_type)

            # robust value: shared P(s'|s,a) restricted to the children of this node, renormalised
            # (skip children registered by get_or_create but not backed up yet)
            local = [next_node for next_node, count in action.next.items() if count > 0]
            counts = np.array([model["next"][next_node.state] for next_node in local], dtype=float)
            probs = list(counts / counts.sum())
            v_values = [next_node.V for next_node in local]

            robust_value = compute_robust_value(v_values,
                                                probs,
                                                self.config.uncertainty_type,
                                                self.config.uncertainty_budget)

            # backup with robust bellman
            action.Q = float(robust_reward + self.config.gamma * robust_value)

            #------------ V-value backup ------------
            node.T += 1
            tried = [act for act in node.actions.values() if act.T > 0]
            list_Q = np.array([act.Q for act in tried])
            weight = np.array([act.T for act in tried]) / sum(act.T for act in tried)

            if np.isinf(self.config.p):   # p -> inf: the power mean becomes the max
                node.V = float(np.max(list_Q))
            else:
                node.V = float(np.power(np.sum(weight * np.power(list_Q, self.config.p)), 1 / self.config.p))

    def simulation(self, env : gym.Env) -> None:
        node = self.root
        path : list[PathStep] = []

        # selection + expand
        while not node.terminal and node.depth < self.config.max_depth:
            if not node.is_expanded:
                self.expand(node,env)

            a = self.select_action(node)
            obs, reward, terminated , truncated , info = env.step(a)
            done = terminated or truncated
            child , is_new = self.get_or_create(node, a , env.get_state(), done)
            path.append((node, a , reward, child))
            node = child
            if is_new:
                break

        # random rollout at leaf node
        leaf_value = self.leaf_value(node, env)

        # backup
        self.backup(path, node, leaf_value)

@register_planner("robust-uct-ms")
class RobustUCTMS(_ModelSharingRobustUCT):
    ''' tree (as robust-uct) + model sharing '''
    name: str = 'robust-uct-ms'

    def lookup_key(self, parent: Node, a : int, s_next):
        return (parent.id, a, s_next)


@register_planner("gs-robust-uct-ms")
class GSRobustUCTMS(_ModelSharingRobustUCT):
    ''' nodes merged by (depth, s) (as gs-robust-uct) + model sharing across depths '''
    name: str = 'gs-robust-uct-ms'

    def lookup_key(self, parent: Node, a : int, s_next):
        return (parent.depth + 1, s_next)
