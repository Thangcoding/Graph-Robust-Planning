import copy
from typing import Any, Optional 

import numpy as np
import gymnasium as gym 
from gymnasium import spaces

# action decode  
LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3  

# frozenlake 8x8 map 
MAP = ["SFFFFFFF",    
       "FFFFFFFF",    
       "FFFHFFFF",     
       "FFFFFHFF",     
       "FFFHFFFF",     
       "FHHFFFHF",     
       "FHFFHFHF",     
       "FFFHFFFG"]     

class FrozenLake(gym.Env):

    def __init__(self, is_slippery=True, slip_rate = 2/3, time_limit=200, seed=None):
        super().__init__()
        self.map = np.array([list(row) for row in MAP])
        self.nrow, self.ncol = self.map.shape
        self.num_states = self.nrow * self.ncol
        self.is_slippery = is_slippery
        self.slip_rate = slip_rate
        self.time_limit = time_limit
        self.t = 0.0 

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.num_states)

        self.start = int(np.flatnonzero(self.map == "S")[0])

        # true model: P[s][a] = list of (prob, next_state)
        self.P = [[self.transitions(s, a) for a in range(4)] for s in range(self.num_states)]

        # separate RNG for clones, so planning does not change the real episode's randomness
        self.clone_rng = np.random.default_rng(None if seed is None else seed + 1)
        self.reset(seed=seed)

    def tile(self, s):
        return self.map.flat[s]

    def move(self, s, a):
        row, col = divmod(s, self.ncol)
        if a == LEFT:
            col = max(col - 1, 0)
        elif a == DOWN:
            row = min(row + 1, self.nrow - 1)
        elif a == RIGHT:
            col = min(col + 1, self.ncol - 1)
        elif a == UP:
            row = max(row - 1, 0)
        return row * self.ncol + col

    def transitions(self, s, a):
        """List of (prob, next_state) for taking action a in state s."""
        if self.tile(s) in "HG":
            return [(1.0, s)]
        if not self.is_slippery:
            return [(1.0, self.move(s, a))]
        success_rate = 1 - self.slip_rate
        return [(self.slip_rate/2, self.move(s, (a - 1) % 4)),
                (success_rate, self.move(s, a)),
                (self.slip_rate/2, self.move(s, (a + 1) % 4))]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  
        self.s = self.start
        self.t = 0
        return self.s, {}

    def step(self, action):
        probs, next_states = zip(*self.P[self.s][action])
        i = self.np_random.choice(len(probs), p=probs)
        self.s = int(next_states[i])
        self.t += 1 

        reward = 1.0 if self.tile(self.s) == "G" else 0.0
        terminated = self.tile(self.s) in "HG"
        truncated = self.t >= self.time_limit
        return self.s, reward, terminated, truncated, {}

    def render(self):
        grid = self.map.copy()
        row, col = divmod(self.s, self.ncol)
        grid[row, col] = "@"
        return "\n".join("".join(r) for r in grid) + "\n"

    def clone(self):
        env = copy.copy(self)  # shares map and P (read-only); s, t are plain ints
        env.np_random = np.random.default_rng(self.clone_rng.integers(2**63))
        env.clone_rng = np.random.default_rng(self.clone_rng.integers(2**63))
        return env

    def get_state(self):
        return self.s

