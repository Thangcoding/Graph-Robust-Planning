
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.algorithms import PLANNERS, make_planner  # noqa: E402
from src.envs import CliffWalking, FrozenLake, Gambler  # noqa: E402
from value_iteration import solve  # noqa: E402

OUT = os.path.join(ROOT, "experiments", "tune_config")

# planner name -> folder name of its results (the planner's source file name)
ALGO_DIRS = {"uct": "uct", "stochastic-power-uct": "st_power_uct",
             "gs-power-uct": "gs_power_uct", "gsf-power-uct": "gs_power_uct_f",
             "robust-uct": "robust_uct", "gs-robust-uct": "gs_robust_uct", "gsf-robust-uct": "gs_robust_uct_f"}


def result_dir(root: str, algo: str, env: str, sims: int, eps: float | None = None) -> str:
    """<root>/<algo folder>/<env>/sims<sims>[/eps<eps>]: the layout of both tune_config/ and results/."""
    out = os.path.join(root, ALGO_DIRS.get(algo, algo.replace("-", "_")), env, f"sims{sims}")
    return out if eps is None else os.path.join(out, f"eps{eps:g}")

# ---------------------------------------------------------------- settings

# env name -> (class, name of its model parameter theta)
ENVS = {"gambler": (Gambler, "p_h"), "frozen_lake": (FrozenLake, "slip_rate"), "cliffwalking": (CliffWalking, "slip_rate")}
# nominal models the planners plan on, and the real envs their actions are scored in (run.py uses the same ones)
THETAS = {"gambler": [0.4, 0.6, 0.8], "frozen_lake": [0.1, 0.2, 0.3, 0.4], "cliffwalking": [0.1, 0.2, 0.3, 0.4]}
EXEC_THETAS = {
    "gambler": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "frozen_lake": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "cliffwalking": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
}

# values tried for the hyper-parameters a planner has; the others are not searched
SEARCH_SPACE = {
    "exploration_const": [0.1, 0.25, 0.5, 1.0, 2.0],
    "p": [1, 4, 8, 16, 32, np.inf],
    "uncertainty_budget": [0.05, 0.1, 0.2, 0.3, 0.5],
}
# same for every config
FIXED = {"gamma": 0.99, "rollout_depth": 100, "max_depth": 50, "final_action": "most_visited",
         "uncertainty_type": "tv"}
PARAM_NAMES = ["num_simulations", *SEARCH_SPACE, *FIXED]   # planner parameters, one column each in the tables

SCREEN_SEEDS = [0, 1, 2, 3, 4]
REFINE_SEEDS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
REFINE_TOP = 8      # configs that go to round 2
TOP_K = 3           # configs that are kept
TUNE_TAG = 7919     # tuning searches get their own random streams
FLAT_TOL = 1e-6     # a state whose Q* values differ by less than this is not scored


# ---------------------------------------------------------------- environments

def make_env(env: str, theta: float, seed: int | None = None):
    env_class, parameter = ENVS[env]
    return env_class(**{parameter: theta}, seed=seed)


def set_state(env, state: int, t: int = 0) -> None:
    """Put a model in the given state (the gambler's legal bets depend on it)."""
    env.s, env.t = state, t
    if hasattr(env, "update_action_space"):
        env.update_action_space()


# ---------------------------------------------------------------- configs

def make_configs(algo: str, sims: int, eps: float | None = None) -> list[dict]:
    """Every combination of the searched hyper-parameters that this planner's config has (uncertainty_budget fixed
    to eps when given)."""
    fields = PLANNERS[algo].config_cls.__dataclass_fields__
    space = SEARCH_SPACE if eps is None else {**SEARCH_SPACE, "uncertainty_budget": [eps]}
    searched = {name: values for name, values in space.items() if name in fields}
    fixed = {name: value for name, value in FIXED.items() if name in fields}

    configs = [{**fixed, "num_simulations": sims}]
    for name, values in searched.items():
        configs = [{**config, name: value} for config in configs for value in values]
    return configs


def group_of(config: dict) -> float:
    """The uncertainty budget (0 for a planner without one)."""
    return config.get("uncertainty_budget", 0.0)


def params_of(row: pd.Series) -> dict:
    """The planner parameters of a result row, as plain Python values (ready for make_planner)."""
    return {name: row[name].item() if hasattr(row[name], "item") else row[name]
            for name in PARAM_NAMES if name in row.index}


# ---------------------------------------------------------------- exact values

TRUTH: dict = {}   # theta_exec -> {state: Q*(state, .) over the legal actions}; set in every worker process


def solve_models(env: str) -> dict:
    """Q* of every real env (ordinary value iteration: the true value of an action there)."""
    return {theta: solve(make_env(env, theta), FIXED["gamma"], 0.0) for theta in EXEC_THETAS[env]}


def init_worker(truth: dict) -> None:
    TRUTH.update(truth)


def regret(q_star: dict, state: int, action: int) -> float:
    return float(q_star[state].max() - q_star[state][action])


def scored_states(truth: dict) -> list[int]:
    """The states where the actions really differ in at least one real env; elsewhere every action is optimal and
    the regret is always 0."""
    states = sorted(next(iter(truth.values())))
    return [s for s in states if any(q[s].max() - q[s].min() > FLAT_TOL for q in truth.values())]


# ---------------------------------------------------------------- searches

def search_once(job: dict) -> list[dict]:
    """One search from a fixed state of the nominal model; the root action is scored in every real env."""
    seeds = np.random.SeedSequence([job["seed"], job["state"], round(job["theta_plan"] * 1e6), TUNE_TAG]).spawn(2)
    env_seed, planner_seed = (int(s.generate_state(1)[0]) for s in seeds)
    config = job["config"]

    plan_env = make_env(job["env"], job["theta_plan"], seed=env_seed)
    plan_env.reset(seed=env_seed)
    set_state(plan_env, job["state"])
    planner = make_planner(job["algo"], plan_env, **config, seed=planner_seed)
    action = planner.plan(plan_env)

    search = {"config_id": job["config_id"], "round": job["round"],
              "seed": job["seed"], "env_seed": env_seed, "planner_seed": planner_seed,
              "theta_plan": job["theta_plan"], "state": job["state"], **config, "action": action}
    return [{**search, "theta_exec": theta, "regret_true": regret(q_star, job["state"], action)}
            for theta, q_star in TRUTH.items()]


def evaluate(env: str, algo: str, configs: dict[int, dict], seeds: list[int], round_name: str,
             workers: int, truth: dict) -> pd.DataFrame:
    """One search per config, nominal theta, scored state and seed, in parallel."""
    jobs = [{"env": env, "algo": algo, "round": round_name, "config_id": config_id, "config": config,
             "theta_plan": theta, "state": state, "seed": seed}
            for config_id, config in configs.items() for theta in THETAS[env]
            for state in scored_states(truth) for seed in seeds]
    print(f"[{round_name}] {len(configs)} configs, {len(jobs)} searches", flush=True)

    rows = []
    with Pool(workers, initializer=init_worker, initargs=(truth,)) as pool:
        for done, search_rows in enumerate(pool.imap_unordered(search_once, jobs, chunksize=8), 1):
            rows.extend(search_rows)
            if done % max(1, len(jobs) // 10) == 0:
                print(f"  {done}/{len(jobs)}", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- ranking

def summarize(searches: pd.DataFrame) -> pd.DataFrame:
    """One row per config. The mean regret is computed on each (theta_plan, theta_exec) pair, then averaged over the
    pairs, so every pair weighs the same. The standard error is over the per-seed values of that average."""
    pairs = ["theta_plan", "theta_exec"]
    per_pair = searches.groupby(["config_id", *pairs]).regret_true.mean()
    per_seed = searches.groupby(["config_id", "seed", *pairs]).regret_true.mean().groupby(["config_id", "seed"]).mean()
    scores = pd.DataFrame({"mean_regret": per_pair.groupby("config_id").mean(),
                           "regret_se": per_seed.groupby("config_id").sem().fillna(0.0),
                           "n_searches": searches.groupby("config_id").size() // searches.theta_exec.nunique()})
    params = searches.groupby("config_id")[[n for n in PARAM_NAMES if n in searches.columns]].first()
    return params.join(scores).reset_index()


def rank(summary: pd.DataFrame) -> pd.DataFrame:
    """Lowest mean regret first; config_id makes ties deterministic."""
    ranked = summary.sort_values(["mean_regret", "config_id"]).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked


# ---------------------------------------------------------------- tuning

def tune(env: str, algo: str, sims: int, workers: int, eps: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (all searches, ranked results)."""
    configs = dict(enumerate(make_configs(algo, sims, eps)))
    truth = solve_models(env)

    # round 1: every config
    screen = evaluate(env, algo, configs, SCREEN_SEEDS, "screen", workers, truth)
    screened = rank(summarize(screen))

    # round 2: the best configs, on new seeds
    finalists = screened[screened["rank"] <= REFINE_TOP].config_id.tolist()
    refine = evaluate(env, algo, {i: configs[i] for i in finalists}, REFINE_SEEDS, "refine", workers, truth)

    # finalists are ranked on both rounds pooled; the configs dropped after the screen keep their ranks after them
    pooled = rank(summarize(pd.concat([screen[screen.config_id.isin(finalists)], refine])))
    dropped = screened[~screened.config_id.isin(finalists)]

    results = pd.concat([pooled.assign(refined=True), dropped.assign(refined=False)], ignore_index=True)
    return pd.concat([screen, refine], ignore_index=True), results


# ---------------------------------------------------------------- output

def top_configs(results: pd.DataFrame) -> list[dict]:
    """The TOP_K best configs: rank, parameters and score."""
    return [{"rank": int(row["rank"]), "params": params_of(row), "mean_regret": float(row["mean_regret"]),
             "regret_se": float(row["regret_se"]), "n_searches": int(row["n_searches"])}
            for _, row in results[results["rank"] <= TOP_K].sort_values("rank").iterrows()]


def save(env: str, algo: str, sims: int, searches: pd.DataFrame, results: pd.DataFrame,
         eps: float | None = None) -> str:
    out_dir = result_dir(OUT, algo, env, sims, eps)
    os.makedirs(out_dir, exist_ok=True)
    searches.to_csv(os.path.join(out_dir, "searches.csv"), index=False)
    results.sort_values("rank").to_csv(os.path.join(out_dir, "results.csv"), index=False)

    summary = {"env": env, "algo": algo, "num_simulations": sims, "thetas_plan": THETAS[env],
               "thetas_exec": EXEC_THETAS[env], "top": top_configs(results)}
    with open(os.path.join(out_dir, "top3.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", required=True, choices=sorted(ENVS))
    parser.add_argument("--algo", required=True, choices=sorted(PLANNERS))
    parser.add_argument("--sims", type=int, default=100, help="simulations per search")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--eps", type=float, default=None,
                        help="fix uncertainty_budget to this value (output in sims<N>/eps<eps>/); default: tune it")
    args = parser.parse_args()

    searches, results = tune(args.env, args.algo, args.sims, args.workers, args.eps)
    out_dir = save(args.env, args.algo, args.sims, searches, results, args.eps)

    best = results[results["rank"] <= TOP_K].sort_values("rank")
    columns = ["rank", "exploration_const", "p", "uncertainty_budget", "mean_regret", "regret_se", "n_searches"]
    table = best[[c for c in columns if c in best.columns]].round(5).to_string(index=False)
    print(f"\ntop {TOP_K}:\n{table}\n\nsaved {out_dir}")


if __name__ == "__main__":
    main()
