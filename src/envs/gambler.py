import copy

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class Gambler(gym.Env):

    def __init__(self, p_h=0.4, goal=10, time_limit=100, seed=None):
        super().__init__()
        self.p_h = p_h
        self.goal = goal
        self.num_states = goal + 1
        self.time_limit = time_limit
        self.t = 0

        self.start = goal // 2
        self.observation_space = spaces.Discrete(self.num_states)
        self.action_space = spaces.Discrete(self.max_bet(self.start))  # resized every step

        # true model: P[s][a] = list of (prob, next_state), a = 0 .. goal // 2 - 1
        self.P = [[self.transitions(s, a) for a in range(goal // 2)] for s in range(self.num_states)]

        # separate RNG for clones, so planning does not change the real episode's randomness
        self.clone_rng = np.random.default_rng(None if seed is None else seed + 1)
        self.reset(seed=seed)

    def is_terminal(self, s):
        return s == 0 or s == self.goal

    def max_bet(self, s):
        return min(s, self.goal - s)

    def transitions(self, s, a):
        """List of (prob, next_state) for staking a + 1 in state s."""
        if self.is_terminal(s):
            return [(1.0, s)]
        stake = min(a + 1, self.max_bet(s))
        return [(self.p_h, s + stake),
                (1 - self.p_h, s - stake)]

    def update_action_space(self):
        if not self.is_terminal(self.s):
            self.action_space = spaces.Discrete(self.max_bet(self.s))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.s = self.start
        self.t = 0
        self.update_action_space()
        return self.s, {}

    def step(self, action):
        probs, next_states = zip(*self.P[self.s][action])
        i = self.np_random.choice(len(probs), p=probs)
        self.s = int(next_states[i])
        self.t += 1
        self.update_action_space()

        reward = 1.0 if self.s == self.goal else 0.0
        terminated = self.is_terminal(self.s)
        truncated = self.t >= self.time_limit
        return self.s, reward, terminated, truncated, {}

    def render(self):
        return f"[{'#' * self.s}{'.' * (self.goal - self.s)}] {self.s}/{self.goal}\n"

    def clone(self):
        env = copy.copy(self)  # shares P (read-only); s, t are plain ints
        env.np_random = np.random.default_rng(self.clone_rng.integers(2**63))
        env.clone_rng = np.random.default_rng(self.clone_rng.integers(2**63))
        return env

    def get_state(self):
        return self.s



