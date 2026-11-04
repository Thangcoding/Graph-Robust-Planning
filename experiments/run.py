from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tune  # noqa: E402
from tune import FIXED, PARAM_NAMES, ROOT, init_worker, make_env, make_planner, regret, set_state  # noqa: E402
from value_iteration import solve  # noqa: E402

OUT = os.path.join(ROOT, "experiments", "results")

# ---------------------------------------------------------------- settings

PLAN_THETAS = tune.THETAS        # nominal models the planner plans on (the ones it was tuned on)
EXEC_THETAS = tune.EXEC_THETAS   # envs the actions are executed in (the ones it was tuned for)
SEEDS = list(range(100))
EPISODES = 2                     # per seed, config, theta_plan and theta_exec
RUN_TAG = 104729                 # random streams of these runs, apart from the tuning ones
OPT_TOL = 1e-6                   # regret below this counts as an optimal action

KEYS = ["group", "config_id", "theta_plan", "theta_exec"]   # one row of summary.csv


# ---------------------------------------------------------------- inputs

def load_best_configs(env: str, algo: str, sims: int, eps: float | None = None) -> list[dict]:
    """[{group, config_id, params}]: the single rank-1 config of tune.py (group = its uncertainty_budget, 0 for a
    non-robust planner)."""
    path = os.path.join(tune.result_dir(tune.OUT, algo, env, sims, eps), "top3.json")
    if not os.path.exists(path):
        eps_arg = "" if eps is None else f" --eps {eps:g}"
        raise SystemExit(f"{path} not found: run first  python experiments/tune.py --env {env} --algo {algo} "
                         f"--sims {sims}{eps_arg}")
    with open(path) as f:
        best = next(entry for entry in json.load(f)["top"] if entry["rank"] == 1)
    return [{"group": tune.group_of(best["params"]), "config_id": algo, "params": best["params"]}]


def solve_models(env: str, configs: list[dict]) -> dict:
    """The Q* run_episode scores against: (theta_plan, eps of every config) for `regret`, (theta_exec, 0) for
    `regret_true`."""
    epsilons = {tune.group_of(c["params"]) for c in configs}
    needed = {(theta, eps) for theta in PLAN_THETAS[env] for eps in epsilons} | \
             {(theta, 0.0) for theta in EXEC_THETAS[env]}
    return {(theta, eps): solve(make_env(env, theta), FIXED["gamma"], eps) for theta, eps in sorted(needed)}

# ---------------------------------------------------------------- runs

def run_episode(job: dict) -> dict:
    """One episode: the planner plans on the nominal model theta_plan and its actions are executed in the real env
    theta_exec (a new search at every real step, only the root action is executed). Every executed action is scored
    twice with exact Q* from value iteration:
        regret       against Q*(theta_plan, eps): the problem the planner is given (robust if eps > 0)
        regret_true  against Q*(theta_exec, 0):   how good the action really is in the env it is executed in
    """
    seeds = np.random.SeedSequence([job["seed"], job["episode"], RUN_TAG]).spawn(3)
    env_seed, plan_seed, planner_seed = (int(s.generate_state(1)[0]) for s in seeds)
    config = job["config"]
    q_target = tune.TRUTH[job["theta_plan"], tune.group_of(config)]
    q_true = tune.TRUTH[job["theta_exec"], 0.0]

    real_env = make_env(job["env"], job["theta_exec"], seed=env_seed)   # where the actions are executed
    plan_env = make_env(job["env"], job["theta_plan"], seed=plan_seed)  # the model the planner searches on
    planner = make_planner(job["algo"], plan_env, **config, seed=planner_seed)
    real_env.reset(seed=env_seed)

    reward, disc_reward, discount, steps = 0.0, 0.0, 1.0, 0
    regrets, regrets_true = [], []
    start = time.time()
    done = False
    while not done:
        state = real_env.s
        set_state(plan_env, state, real_env.t)   # the planning model starts from the real state
        action = planner.plan(plan_env)
        regrets.append(regret(q_target, state, action))
        regrets_true.append(regret(q_true, state, action))

        _, r, terminated, truncated, _ = real_env.step(action)
        reward += r
        disc_reward += discount * r
        discount *= config["gamma"]
        steps += 1
        done = terminated or truncated
    elapsed = time.time() - start

    return {
        # which run
        "env": job["env"], "algo": job["algo"], "config_id": job["config_id"],
        # seeds: (seed, episode) -> the three random streams below
        "seed": job["seed"], "episode": job["episode"],
        "env_seed": env_seed, "plan_seed": plan_seed, "planner_seed": planner_seed,
        # models: the planner plans on theta_plan, its actions are executed in theta_exec
        "theta_plan": job["theta_plan"], "theta_exec": job["theta_exec"],
        # planner parameters
        **config,
        # results of the execution
        "success": float(reward > 0), "reward": reward, "disc_reward": disc_reward, "steps": steps,
        "regret_true": np.mean(regrets_true), "regret": np.mean(regrets),
        "opt_rate": np.mean(np.array(regrets_true) < OPT_TOL),
        "time_per_search": elapsed / steps,
    }


def run_all(env: str, algo: str, configs: list[dict], workers: int, truth: dict) -> pd.DataFrame:
    jobs = [{"env": env, "algo": algo, "config_id": c["config_id"], "config": c["params"],
             "theta_plan": theta_plan, "theta_exec": theta_exec, "seed": seed, "episode": episode}
            for c in configs for theta_plan in PLAN_THETAS[env] for theta_exec in EXEC_THETAS[env]
            for seed in SEEDS for episode in range(EPISODES)]
    print(f"{len(configs)} configs x {len(PLAN_THETAS[env])} theta_plan x {len(EXEC_THETAS[env])} theta_exec "
          f"x {len(SEEDS)} seeds x {EPISODES} episodes = {len(jobs)} episodes", flush=True)

    rows, start = [], time.time()
    with Pool(workers, initializer=init_worker, initargs=(truth,)) as pool:
        for done, row in enumerate(pool.imap_unordered(run_episode, jobs, chunksize=2), 1):
            rows.append(row)
            if done % max(1, len(jobs) // 10) == 0:
                print(f"  {done}/{len(jobs)}  ({time.time() - start:.0f}s)", flush=True)

    episodes = pd.DataFrame(rows)
    labels = pd.DataFrame([{k: c[k] for k in ("config_id", "group")} for c in configs])
    episodes = labels.merge(episodes, on="config_id")
    return episodes.sort_values(["group", "theta_plan", "theta_exec", "seed", "episode"]).reset_index(drop=True)


def summarize(episodes: pd.DataFrame) -> pd.DataFrame:
    """One row per (config, theta_plan, theta_exec). Means over episodes, standard errors over per-seed means."""
    per_seed = episodes.groupby([*KEYS, "seed"])[["success", "reward", "disc_reward", "regret_true"]].mean()
    errors = per_seed.groupby(KEYS).sem().fillna(0.0).add_suffix("_se")
    means = episodes.groupby(KEYS).agg(
        success_rate=("success", "mean"), mean_reward=("reward", "mean"), mean_disc_reward=("disc_reward", "mean"),
        mean_regret_true=("regret_true", "mean"), mean_regret=("regret", "mean"), opt_rate=("opt_rate", "mean"),
        mean_steps=("steps", "mean"), time_per_search=("time_per_search", "mean"),
        n_seeds=("seed", "nunique"), n_episodes=("success", "size"))
    params = episodes.groupby(KEYS)[[n for n in PARAM_NAMES if n in episodes.columns]].first()
    return params.join(means).join(errors).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", required=True, choices=sorted(tune.ENVS))
    parser.add_argument("--algo", required=True, choices=sorted(tune.PLANNERS))
    parser.add_argument("--sims", type=int, default=100, help="simulations per search (must have been tuned)")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--eps", type=float, default=None, help="run the config tuned with this fixed eps")
    args = parser.parse_args()

    configs = load_best_configs(args.env, args.algo, args.sims, args.eps)
    truth = solve_models(args.env, configs)
    episodes = run_all(args.env, args.algo, configs, args.workers, truth)
    summary = summarize(episodes)

    out_dir = tune.result_dir(OUT, args.algo, args.env, args.sims, args.eps)
    os.makedirs(out_dir, exist_ok=True)
    episodes.to_csv(os.path.join(out_dir, "episodes.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    for theta_plan, rows in summary.groupby("theta_plan"):
        table = rows.pivot(index="config_id", columns="theta_exec", values="success_rate").round(3)
        print(f"\nsuccess rate, plan on theta = {theta_plan}  (columns: theta_exec)\n{table.to_string()}")
    print(f"\nsaved {out_dir}")


if __name__ == "__main__":
    main()
