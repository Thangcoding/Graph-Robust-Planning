"""Exact (robust) value iteration for the tabular environments (Gambler, FrozenLake).

    Q(s, a) = R(s, a) + gamma * W(V(s'))        V(s) = max_a Q(s, a)        V(terminal) = 0

R is the expected reward and W the expectation over s' ~ P(.|s, a) when eps = 0 (non-robust). When eps > 0 both are
replaced by their worst case over the total-variation ball of radius eps around P(.|s, a) (robust), in the same split
form as the backup of the robust planners (robust reward + gamma * robust value).

    q_star = solve(env, gamma=0.99, eps=0.1)
    q_star[state]        # array: Q*(state, a) for every legal action a
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.algorithms.planner.robust_uct import compute_robust_value  # noqa: E402


def is_terminal(env, s: int) -> bool:
    return env.is_terminal(s) if hasattr(env, "is_terminal") else env.tile(s) in "HG"

def is_goal(env, s: int) -> bool:
    return s == env.goal if hasattr(env, "goal") else env.tile(s) == "G"

def legal_actions(env, s: int) -> range:
    return range(env.max_bet(s)) if hasattr(env, "max_bet") else range(env.action_space.n)

def mean_under(values: np.ndarray, probs: np.ndarray, eps: float) -> float:
    """E_p[values] when eps = 0, else its worst case (lowest value) over the TV ball of radius eps around p."""
    return float(probs @ values) if eps == 0 else float(compute_robust_value(values, probs, "tv", eps))

def solve(env, gamma: float, eps: float, tol: float = 1e-10, max_iter: int = 100_000) -> dict[int, np.ndarray]:
    """Value iteration on the exact model env.P. Returns {state: Q*(state, a) over the legal actions} for every
    non-terminal state."""
    # (s, a) -> (probs, next states, next state is not terminal, expected reward); nothing here changes over iterations
    model = {}
    for s in range(env.num_states):
        if is_terminal(env, s):
            continue
        for a in legal_actions(env, s):
            probs = np.array([p for p, _ in env.P[s][a]])
            nexts = np.array([n for _, n in env.P[s][a]])
            rewards = np.array([env.reward(s, a, n) if hasattr(env, "reward") else float(is_goal(env, n))
                                for n in nexts])
            alive = np.array([not is_terminal(env, n) for n in nexts])
            model[s, a] = (probs, nexts, alive, mean_under(rewards, probs, eps))

    V = np.zeros(env.num_states)
    for _ in range(max_iter):
        Q = {sa: reward + gamma * mean_under(V[nexts] * alive, probs, eps)
             for sa, (probs, nexts, alive, reward) in model.items()}
        V_new = np.zeros(env.num_states)
        for (s, _), q in Q.items():
            V_new[s] = max(V_new[s], q)   # values are >= 0, terminal states stay 0
        converged = np.max(np.abs(V_new - V)) < tol
        V = V_new
        if converged:
            break
    else:
        raise RuntimeError(f"value iteration did not converge in {max_iter} iterations (eps={eps})")

    return {s: np.array([Q[s, a] for a in legal_actions(env, s)])
            for s in range(env.num_states) if not is_terminal(env, s)}



