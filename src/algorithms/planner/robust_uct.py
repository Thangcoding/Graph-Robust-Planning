from __future__ import annotations 

import math 
from dataclasses import dataclass 

import gymnasium as gym 
import numpy as np 

from ..base import Action, BasePlanner, BasePlannerConfig, Node, PathStep
from ..registry import register_planner


def tv_worst_case(values: list[float], probs: list[float], budget: float) -> float:
    ''' TV worst case of a few values: move up to `budget` mass from the highest values to the lowest one
    (same result as compute_robust_value with 'tv', in pure python: numpy is slower on 2-4 values) '''
    order = sorted(range(len(values)), key=values.__getitem__)
    lowest = values[order[0]]
    result, budget_remaining = 0.0, budget
    for i in reversed(order):
        movable_mass = min(probs[i], budget_remaining)
        result += values[i] * (probs[i] - movable_mass) + lowest * movable_mass
        budget_remaining -= movable_mass
    return result


@dataclass
class RobustUCTPlannerConfig(BasePlannerConfig): 
    name = 'robust-uct-config'
    uncertainty_budget : float = 0.6
    uncertainty_type : str = 'tv'
    p : float = 2 # power value 

@register_planner("robust-uct")
class RobustUCT(BasePlanner): 
    name: str = 'robust-uct' 
    config_cls = RobustUCTPlannerConfig

    def lookup_key(self, parent: Node, a : int, s_next): 
        return (parent.id, a , s_next)

    def expand(self, node: Node, env: gym.Env) -> None:
        # R counts the reward samples of (s, a) by value, {reward: count}, for the robust reward
        node.actions = {a: Action(R={}) for a in range(env.action_space.n)}

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
        ''' Robust Bellman backup: Q(s,a) = robust(R) + gamma * robust(V(s')) '''
        leaf.T += 1 
        leaf.V += (leaf_value - leaf.V) / leaf.T 

        for node , a , reward , child in reversed(path): 

            #-----------Q-value backup-----------------
            action = node.actions[a]
            action.T += 1                # T(s, a)
            action.next[child] += 1      # T(s, a, s')
            action.R[reward] = action.R.get(reward, 0) + 1   # update reward sample count

            # robust reward over the empirical reward distribution
            rewards = list(action.R)
            if self.config.uncertainty_type == 'tv':
                robust_reward = tv_worst_case(rewards, [action.R[r] / action.T for r in rewards],
                                              self.config.uncertainty_budget)
            else:
                robust_reward = compute_robust_reward([r for r in rewards for _ in range(action.R[r])],
                                                      self.config.uncertainty_budget,
                                                      self.config.uncertainty_type)

            # robust value 
            # P(s'|s,a) = T(s,a,s') / T(s,a); skip children registered by get_or_create but not backed up yet
            # (on a graph with cycles the same (s, a) can appear again later in the path)
            seen = [(next_node, count) for next_node, count in action.next.items() if count > 0]
            probs = [count / action.T for _, count in seen]
            v_values = [next_node.V for next_node, _ in seen] 

            assert len(probs) == len(v_values)

            if self.config.uncertainty_type == 'tv':
                robust_value = tv_worst_case(v_values, probs, self.config.uncertainty_budget)
            else:
                robust_value = compute_robust_value(v_values,
                                                    probs,
                                                    self.config.uncertainty_type,
                                                    self.config.uncertainty_budget)

            # backup with robust bellman
            action.Q = float(robust_reward + self.config.gamma * robust_value)

            #------------ V-value backup ------------
            node.T += 1
            tried = [act for act in node.actions.values() if act.T > 0]
            if math.isinf(self.config.p):   # p -> inf: the power mean becomes the max
                node.V = max(act.Q for act in tried)
            else:
                total = sum(act.T for act in tried)
                node.V = sum(act.T / total * act.Q ** self.config.p for act in tried) ** (1 / self.config.p)


    def simulation(self, env : gym.Env) -> None: 
        node = self.root 
        path : list[PathStep] = []

        # selection + extension 
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

def compute_robust_reward(reward_samples: list[float],
                          uncertainty_budget: float = 2.0, 
                          uncertainty_type: str = 'tv'):
    if len(reward_samples) == 0: 
        return 0.0 

    rewards = np.array(reward_samples)
    n = len(rewards)
    epsilon = uncertainty_budget 

    if uncertainty_type == 'tv':
        sorted_rewards = np.sort(rewards)
        probs = np.ones(n) / n

        remaining_budget = epsilon 
        worst_case_probs = np.copy(probs)

        for i in reversed(range(1, n)):
            movable = min(worst_case_probs[i], remaining_budget)
            worst_case_probs[0] += movable
            worst_case_probs[i] -= movable
            remaining_budget -= movable
            if remaining_budget <= 1e-9:
                break

        return float(np.dot(sorted_rewards, worst_case_probs))

    elif uncertainty_type == 'chi2': 
        mean = np.mean(rewards)
        std = np.std(rewards) if n > 1 else 0.0
        return mean - math.sqrt(epsilon) * std

    elif uncertainty_type == 'wasserstein': 
        sorted_rewards = np.sort(rewards)
        cdf = np.arange(1, n + 1) / n
        transport_cost = np.sum(np.abs(np.diff(sorted_rewards)) * cdf[:-1])
        return np.mean(rewards) - epsilon * transport_cost

    else: 
        raise ValueError(
            f"Unknown uncertainty type: {uncertainty_type}"
        )

def compute_robust_value(v_values : list[float], 
                         probs: list[float],
                         uncertainty_type: str, 
                         uncertainty_budget: float):

    if len(v_values) == 0 or len(probs) == 0: 
        return 0.0 

    v = np.array(v_values)
    q = np.array(probs)
    budget = uncertainty_budget

    if uncertainty_type == 'tv': 
        # move up to `budget` probability mass from the highest values to the lowest one
        sorted_idx = np.argsort(v)
        v_sorted = v[sorted_idx]
        q_sorted = q[sorted_idx]
        
        budget_remaining = uncertainty_budget
        result = 0.0
        
        for i in reversed(range(len(v_sorted))):
            movable_mass = min(q_sorted[i], budget_remaining)
        
            result += v_sorted[i] * (q_sorted[i] - movable_mass)
            result += v_sorted[0] * movable_mass
        
            budget_remaining -= movable_mass
            if budget_remaining <= 0:
                # budget used up: the lower values (not processed yet) keep their mass
                result += v_sorted[:i] @ q_sorted[:i]
                break
        return float(result)

    elif uncertainty_type == "chi2": 
        sorted_idx = np.argsort(v)
        v_sorted = v[sorted_idx]
        q_sorted = q[sorted_idx]

        mean = np.dot(q_sorted, v_sorted)
        diff = v_sorted - mean
        std = np.sqrt(np.dot(q_sorted, diff**2))

        ball_radius = budget * std

        return mean - ball_radius

    elif uncertainty_type == 'wasserstein': 
        sorted_idx = np.argsort(v)
        v_sorted = v[sorted_idx] 
        q_sorted = q[sorted_idx] 


        Q = np.cumsum(q_sorted)

        cost = 0.0
        for i in range(len(v_sorted) - 1):
            cost += abs(v_sorted[i + 1] - v_sorted[i]) * Q[i]

        return np.dot(q, v) - budget * cost

    else: 
        raise ValueError(f'Unknown uncertainty type: {uncertainty_type}')
