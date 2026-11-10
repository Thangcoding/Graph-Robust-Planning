import copy 
from math import comb, exp

import numpy as np
import gymnasium as gym
from gymnasium import spaces

ORDER_SIZES = (0, 5, 10, 15)

class Inventory(gym.Env):
    """Single-product inventory control with lost sales and Poisson demand.

    State     inventory level I in {0, ..., i_max}                  (31 states for i_max = 30)
    Action    index into ORDER_SIZES = (0, 5, 10, 15)                (units ordered, arrive at once)
    Demand    demand="levels" (default): D = d_step * B, B ~ Binomial(n_levels, lam / (d_step * n_levels)), so
              D in {0, 5, 10, 15, 20} with mean lam (lam = 10: probs 1/16, 4/16, 6/16, 4/16, 1/16);
              demand="poisson": D ~ Poisson(lam), demand above d_max lumped into d_max
    Dynamics  stock = min(I + order, i_max), sold = min(stock, D),
              I' = stock - sold; demand above the stock is lost
    Reward    (price * sold - cost * received - holding * I' - lost_penalty * (D - sold)) / scale

    Why levels: the TV worst case puts eps mass on the worst outcome however rare it is. With Poisson demand the
    worst one (D = 0, or the cut-off d_max) has a tiny probability, the planner almost never samples it, so its
    robust backup (over the sampled outcomes) does not match the exact robust one and drifts with the number of
    samples. With 5 levels, each one >= 1/16 likely at lam = 10, a node sees all of them after a few dozen visits.

    The reward of (s, a) is stochastic: it depends on the demand, not only on the next state (every D >= stock
    ends in I' = 0 with a different lost-sales penalty), so a robust reward backup has something to act on.
    There is no terminal state: an episode ends at time_limit (truncation). theta = lam (planning lam_0 = 10).

    P[s][a] lists (prob, next_state) with the demands that lead to the same next state merged; reward(s, a, s')
    is the expected reward given s' (exact for a nominal backup). outcomes(s, a) lists (prob, next_state, reward)
    per demand value, for an exact robust reward.
    """

    def __init__(self, lam=10.0, i_max=30, i_0=10, price=2.0, cost=1.0, holding=0.1, lost_penalty=1.0,
                 scale=None, demand="levels", d_step=5, n_levels=4, d_max=None, time_limit=50, seed=None):
        super().__init__()
        self.lam = lam
        self.i_max = i_max
        self.i_0 = i_0
        self.price, self.cost, self.holding, self.lost_penalty = price, cost, holding, lost_penalty
        self.scale = price * i_max if scale is None else scale   # rewards in about [-1, 1]
        self.demand = demand
        self.time_limit = time_limit
        self.t = 0

        self.num_states = i_max + 1
        self.start = i_0
        self.action_space = spaces.Discrete(len(ORDER_SIZES))
        self.observation_space = spaces.Discrete(self.num_states)

        # demand_values[k] has probability demand_pmf[k]
        if demand == "levels":
            q = lam / (d_step * n_levels)
            if not 0.0 <= q <= 1.0:
                raise ValueError(f"lam must be in [0, {d_step * n_levels}] for demand='levels'")
            self.d_step, self.n_levels = d_step, n_levels
            self.demand_values = np.arange(n_levels + 1) * d_step
            self.demand_pmf = np.array([comb(n_levels, k) * q**k * (1 - q)**(n_levels - k) for k in range(n_levels + 1)])
        elif demand == "poisson":
            self.d_max = 2 * i_max if d_max is None else d_max      # demand above d_max lumped into d_max
            pmf = [exp(-lam)]
            for d in range(1, self.d_max):
                pmf.append(pmf[-1] * lam / d)
            pmf.append(max(0.0, 1.0 - sum(pmf)))
            self.demand_values = np.arange(self.d_max + 1)
            self.demand_pmf = np.array(pmf)
        else:
            raise ValueError(f"unknown demand model '{demand}'")

        # true model, per demand: (prob, next_state, reward); merged by next state: P and the conditional reward
        self.outcome_table = [[self._outcomes(s, a) for a in range(len(ORDER_SIZES))] for s in range(self.num_states)]
        self.P, self.R = [], []
        for s in range(self.num_states):
            P_s, R_s = [], []
            for a in range(len(ORDER_SIZES)):
                mass, total = {}, {}
                for p, s_next, r in self.outcome_table[s][a]:
                    mass[s_next] = mass.get(s_next, 0.0) + p
                    total[s_next] = total.get(s_next, 0.0) + p * r
                P_s.append([(m, s_next) for s_next, m in mass.items()])
                R_s.append({s_next: total[s_next] / mass[s_next] for s_next in mass})
            self.P.append(P_s)
            self.R.append(R_s)

        # separate RNG for clones, so planning does not change the real episode's randomness
        self.clone_rng = np.random.default_rng(None if seed is None else seed + 1)
        self.reset(seed=seed)

    def _sale(self, s, a, d):
        """(next_state, reward) when demand d arrives after ordering ORDER_SIZES[a] in state s."""
        stock = min(s + ORDER_SIZES[a], self.i_max)
        received = stock - s
        sold = min(stock, d)
        left = stock - sold
        r = self.price * sold - self.cost * received - self.holding * left - self.lost_penalty * (d - sold)
        return left, r / self.scale

    def _outcomes(self, s, a):
        return [(float(p), *self._sale(s, a, int(d))) for d, p in zip(self.demand_values, self.demand_pmf) if p > 0]

    def outcomes(self, s, a):
        """List of (prob, next_state, reward), one per demand value."""
        return self.outcome_table[s][a]

    def is_terminal(self, s):
        return False

    def reward(self, s, a, s_next):
        """Expected reward of (s, a) given the next state s' (the demand is not identified by s' when s' = 0)."""
        return self.R[s][a][s_next]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.s = self.start
        self.t = 0
        return self.s, {}

    def step(self, action):
        if self.demand == "levels":
            d = self.d_step * int(self.np_random.binomial(self.n_levels, self.lam / (self.d_step * self.n_levels)))
        else:
            d = min(int(self.np_random.poisson(self.lam)), self.d_max)
        self.s, reward = self._sale(self.s, action, d)
        self.t += 1
        return self.s, reward, False, self.t >= self.time_limit, {"demand": d}

    def render(self):
        return f"t={self.t}/{self.time_limit}  inventory={self.s}\n"

    def clone(self):
        env = copy.copy(self)  # shares the model tables (read-only); s, t are plain ints
        env.np_random = np.random.default_rng(self.clone_rng.integers(2**63))
        env.clone_rng = np.random.default_rng(self.clone_rng.integers(2**63))
        return env

    def get_state(self):
        return self.s
