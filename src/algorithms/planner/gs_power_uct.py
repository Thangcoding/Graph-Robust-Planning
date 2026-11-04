from __future__ import annotations 

import math 
from dataclasses import dataclass 
import numpy as np 

import gymnasium as gym 
from ..base import BasePlanner, BasePlannerConfig, Node, PathStep
from ..registry import register_planner 

@dataclass
class GSPowerUCTPlannerConfig(BasePlannerConfig): 
    name = 'gs-power-uct-config'
    p : float = 2 # power value 

    
@register_planner("gs-power-uct")
class GSPowerUCT(BasePlanner): 
    name: str = 'gs-power-uct'
    config_cls = GSPowerUCTPlannerConfig

    def __init__(self, env : gym.Env, config: GSPowerUCTPlannerConfig | None = None): 
        super().__init__(env, config)
        pass

    def lookup_key(self, parent: Node , a : int, s_next): 
        return (parent.depth + 1, s_next)

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

    def backup(self, path: list[PathStep], leaf: Node, leaf_value: float) -> None:
        '''power mean backup '''
        leaf.T += 1
        leaf.V += (leaf_value - leaf.V) / leaf.T

        v_back = leaf_value 
        for node , a , reward , child in reversed(path): 
            G = reward + self.config.gamma * v_back

            # Q-value backup
            action = node.actions[a]         
            action.T += 1                    
            action.Q += (G - action.Q) / action.T      
            action.R += (reward - action.R) / action.T 

            #  V-value backup with power mean operator: 
            #  V(s) = (sum_a n(s,a)/n(s) * Q(s,a)^p)^(1/p)
            node.T += 1 
            tried = [act for act in node.actions.values() if act.T > 0]
            list_Q = np.array([act.Q for act in tried])
            weight = np.array([act.T for act in tried]) / sum(act.T for act in tried)
            
            if np.isinf(self.config.p):   # p -> inf: the power mean becomes the max
                node.V = float(np.max(list_Q))
            else:
                node.V = float(np.power(np.sum(weight * np.power(list_Q, self.config.p)), 1 / self.config.p))

            # the parent backs up the power mean value of this node
            v_back = node.V 

    def simulation(self, env : gym.Env) -> None:
        node = self.root 
        path : list[PathStep] = []

        # selection + expansion 
        while not node.terminal and node.depth < self.config.max_depth: 
            if not node.is_expanded: 
                self.expand(node, env)
            a = self.select_action(node)
            obs, reward, terminated, truncated , info = env.step(a)
            done = terminated or truncated 
            child , is_new = self.get_or_create(node, a , env.get_state(), done)
            path.append((node, a, reward, child))
            node = child 
            if is_new: 
                break 

        # random rollout at leaf node 
        leaf_value = self.leaf_value(node, env)

        # backup 
        self.backup(path, node , leaf_value)
    







