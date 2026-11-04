import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from algorithms.planner.uct import UCTMCTS, UCTPlannerConfig  # noqa: E402
from envs import FrozenLake  # noqa: E402


def test_uct_frozen_lake():
    # 1. config
    config = UCTPlannerConfig(num_simulations=500, max_depth=50, rollout_depth=20,
                              gamma=0.99, exploration_const=1.414, seed=0)

    # 2. env
    env = FrozenLake(is_slippery=True, seed=0)

    # 3. algorithm
    planner = UCTMCTS(env, config)

    # 4. run one episode: plan at every real step
    env.reset(seed=0)
    total_reward = 0.0
    steps = 0
    done = False
    while not done:
        action = planner.plan(env)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

    print(f"return = {total_reward}, steps = {steps}, nodes in last search = {planner.num_nodes}")
    print("root stats:", planner.root_stats())


if __name__ == "__main__":
    test_uct_frozen_lake()
