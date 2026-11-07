from __future__ import annotations 

from dataclasses import dataclass 

from ..base import Node
from ..registry import register_planner 

import gymnasium as gym 
import numpy as np 
import math 

from .robust_uct import compute_robust_reward_counts , compute_robust_value
from ..base import Action, BasePlanner, BasePlannerConfig, Node, PathStep
from ..registry import register_planner 

@dataclass 
class GSRobustUCTPlannerConfig(BasePlannerConfig): 
    name = 'gs-robust-uct-config'
    uncertainty_budget : float = 0.6 
    uncertainty_type : str = 'tv'
    p : float = 2 # power value 

@register_planner("gs-robust-uct")
class GSRobustUCT(BasePlanner): 
    name: str = 'gs-robust-uct' 
    config_cls = GSRobustUCTPlannerConfig

    def lookup_key(self, parent: Node, a : int, s_next): 
        return (parent.depth + 1, s_next)

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

            # robust reward 
            robust_reward = compute_robust_reward_counts(action.R,
                                                         self.config.uncertainty_budget,
                                                         self.config.uncertainty_type)
            
            # robust value 
            # P(s'|s,a) = T(s,a,s') / T(s,a); skip children registered by get_or_create but not backed up yet
            # (on a graph with cycles the same (s, a) can appear again later in the path)
            seen = [(next_node, count) for next_node, count in action.next.items() if count > 0]
            probs = [count / action.T for _, count in seen]
            v_values = [next_node.V for next_node, _ in seen] 

            assert len(probs) == len(v_values)

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
