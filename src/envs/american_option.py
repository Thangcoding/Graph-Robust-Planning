import copy
from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# action decode
HOLD, EXERCISE = 0, 1

class AmericanOption(gym.Env):

    def __init__(self, p_up=0.5, n_steps=20, s0=100.0, strike=100.0, up=1.1, seed=None):
        super().__init__() 
        self.p_up = p_up
        self.n_steps = n_steps
        self.s0 = s0
        self.strike = strike
        self.up = up
        self.time_limit = n_steps + 1
        self.t = 0

        self.num_nodes = (n_steps + 1) * (n_steps + 2) // 2
        self.done = self.num_nodes
        self.num_states = self.num_nodes + 1
        self.nodes = [(t, u) for t in range(n_steps + 1) for u in range(t + 1)]

        self.start = 0
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(self.num_states)

        # true model: P[s][a] = list of (prob, next_state)
        self.P = [[self.transitions(s, a) for a in range(2)] for s in range(self.num_states)]

        # separate RNG for clones, so planning does not change the real episode's randomness
        self.clone_rng = np.random.default_rng(None if seed is None else seed + 1)
        self.reset(seed=seed)

    def node_id(self, t, u):
        return t * (t + 1) // 2 + u

    def price(self, s):
        t, u = self.nodes[s]
        return self.s0 * self.up ** (2 * u - t)

    def payoff(self, s):
        return max(self.strike - self.price(s), 0.0) / self.strike 

    def is_terminal(self, s):
        return s == self.done

    def reward(self, s, a, s_next):
        """Reward of (s, a, s'): the payoff when exercising, 0 otherwise."""
        return self.payoff(s) if a == EXERCISE and not self.is_terminal(s) else 0.0

    def transitions(self, s, a):
        """List of (prob, next_state) for taking action a in state s."""
        if self.is_terminal(s) or a == EXERCISE:
            return [(1.0, self.done)]
        t, u = self.nodes[s]
        if t == self.n_steps:   # held to maturity: the option expires
            return [(1.0, self.done)]
        return [(self.p_up, self.node_id(t + 1, u + 1)),
                (1 - self.p_up, self.node_id(t + 1, u))]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.s = self.start
        self.t = 0
        return self.s, {}

    def step(self, action):
        probs, next_states = zip(*self.P[self.s][action])
        i = self.np_random.choice(len(probs), p=probs)
        reward = self.reward(self.s, action, int(next_states[i]))
        self.s = int(next_states[i])
        self.t += 1

        terminated = self.is_terminal(self.s)
        truncated = self.t >= self.time_limit
        return self.s, reward, terminated, truncated, {}

    def render(self):
        if self.is_terminal(self.s):
            return "done\n"
        t, _ = self.nodes[self.s]
        return f"t={t}/{self.n_steps}  price={self.price(self.s):.2f}  payoff={self.payoff(self.s):.4f}\n"

    def clone(self):
        env = copy.copy(self)  # shares nodes and P (read-only); s, t are plain ints
        env.np_random = np.random.default_rng(self.clone_rng.integers(2**63))
        env.clone_rng = np.random.default_rng(self.clone_rng.integers(2**63))
        return env

    def get_state(self):
        return self.s


