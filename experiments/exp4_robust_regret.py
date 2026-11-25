"""Measure root robust regret under execution-model mismatch.

Each planner builds its tree at theta_plan. Its chosen root action is scored
by exact robust value iteration at theta_exec using the configuration's TV
budget. Tune epsilon, exploration constant and power separately for each
model pair, then evaluate the best configuration on independent seeds.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tune  # noqa: E402
from src.envs import Inventory  # noqa: E402
from value_iteration import solve  # noqa: E402

# at module level: the worker processes import this file again, they would not see a change made in main()
tune.ENVS["inventory"] = (Inventory, "lam")

ALGOS = ["robust-uct", "gs-robust-uct", "gsf-robust-uct", "gsf-robust-uct-plus"]
# Planning models match Exp3 (wall-clock time per search).
THETA_PLAN = {"frozen_lake": 0.1, "cliffwalking": 0.1, "gambler": 0.7, "inventory": 10.0}
SIMS = [2 ** exponent for exponent in range(4, 12)]
THETA_EXEC = [0.4, 0.6, 0.8]
EVAL_SEEDS = 100
EVAL_TAG = 15485863   # eval searches get their own random streams, apart from tune (TUNE_TAG) and run (RUN_TAG)
OPT_TOL = 1e-6        # robust regret below this counts as a robust-optimal action
OUT = os.path.join(tune.ROOT, "experiments", "results", "experiement_4")

TRUTH: dict = {}      # "robust" / "nominal" -> {state: Q*(state, .)}; set in every worker process


def init_worker(truth: dict) -> None:
    TRUTH.update(truth)


def root_states(robust_models: dict, n: int | None) -> list[int]:
    states = tune.scored_states(robust_models)
    if n is None or n >= len(states):
        return states
    return sorted(int(s) for s in np.random.default_rng(0).choice(states, size=n, replace=False))


# ---------------------------------------------------------------- tune (tune.py rounds, robust regret as the score)

def tune_robust(env, algo, sims, eps, states, workers, truth):
    configs = dict(enumerate(tune.make_configs(algo, sims, eps)))

    def score(configs, seeds, stage):
        searches = evaluate(env, algo, configs, states, seeds, workers, truth, stage, tune.TUNE_TAG)
        return searches.assign(regret_true=searches.robust_regret)

    screen = score(configs, tune.SCREEN_SEEDS, "screen")
    screened = tune.rank(tune.summarize(screen))
    finalists = screened[screened["rank"] <= tune.REFINE_TOP].config_id.tolist()
    refine = score({i: configs[i] for i in finalists}, tune.REFINE_SEEDS, "refine")
    pooled = tune.rank(tune.summarize(pd.concat([screen[screen.config_id.isin(finalists)], refine])))
    dropped = screened[~screened.config_id.isin(finalists)]
    results = pd.concat([pooled.assign(refined=True), dropped.assign(refined=False)], ignore_index=True)
    return pd.concat([screen, refine], ignore_index=True), results


def save_tune(out_dir: str, env: str, algo: str, sims: int, theta_exec: float, states: list[int], searches, results) -> None:
    os.makedirs(out_dir, exist_ok=True)
    searches.to_csv(os.path.join(out_dir, "searches.csv"), index=False)
    results.sort_values("rank").to_csv(os.path.join(out_dir, "results.csv"), index=False)
    summary = {"env": env, "algo": algo, "num_simulations": sims, "theta_plan": tune.THETAS[env][0],
               "theta_exec": theta_exec, "score": "robust regret at the root", "states": states, "top": tune.top_configs(results)}
    temporary = os.path.join(out_dir, "top3.json.tmp")
    with open(temporary, "w") as f:
        json.dump(summary, f, indent=2)
    os.replace(temporary, os.path.join(out_dir, "top3.json"))


# ---------------------------------------------------------------- eval

def search_once(job: dict) -> dict:
    seeds = np.random.SeedSequence([job["seed"], job["state"], round(job["theta_plan"] * 1e6), job["tag"]]).spawn(2)
    env_seed, planner_seed = (int(s.generate_state(1)[0]) for s in seeds)
    plan_env = tune.make_env(job["env"], job["theta_plan"], seed=env_seed)
    plan_env.reset(seed=env_seed)
    tune.set_state(plan_env, job["state"])
    planner = tune.make_planner(job["algo"], plan_env, **job["config"], seed=planner_seed)
    action = planner.plan(plan_env)
    config = job["config"]
    robust = TRUTH["robust"][config["uncertainty_budget"]]
    r_rob = tune.regret(robust, job["state"], action)
    return {**config, "config_id": job["config_id"], "round": job["round"],
            "theta_plan": job["theta_plan"], "theta_exec": TRUTH["theta_exec"],
            "seed": job["seed"], "state": job["state"], "action": action, "env_seed": env_seed,
            "planner_seed": planner_seed, "robust_regret": r_rob, "robust_opt": float(r_rob < OPT_TOL),
            "nominal_regret": tune.regret(TRUTH["nominal"], job["state"], action)}


def evaluate(env, algo, configs, states, seeds, workers, truth, stage, tag):
    jobs = [{"env": env, "algo": algo, "config": config, "config_id": config_id,
             "theta_plan": tune.THETAS[env][0], "state": state, "seed": seed, "round": stage, "tag": tag}
            for config_id, config in configs.items() for state in states for seed in seeds]
    print(f"[{stage}] {len(configs)} configs, {len(jobs)} searches", flush=True)
    rows = []
    with Pool(workers, initializer=init_worker, initargs=(truth,)) as pool:
        for done, row in enumerate(pool.imap_unordered(search_once, jobs, chunksize=4), 1):
            rows.append(row)
            if done % max(1, len(jobs) // 10) == 0:
                print(f"  {done}/{len(jobs)}", flush=True)
    return pd.DataFrame(rows).sort_values(["state", "seed"]).reset_index(drop=True)


def summarize(searches: pd.DataFrame, config: dict, env: str, algo: str) -> pd.DataFrame:
    """One row: per-state means averaged over the states; SE over the seeds of the per-seed state average."""
    metrics = ["robust_regret", "nominal_regret", "robust_opt"]
    per_state = searches.groupby("state")[metrics].mean().mean()
    per_seed = searches.groupby("seed")[metrics].mean()
    row = {"env": env, "algo": algo, "theta_plan": tune.THETAS[env][0], "theta_exec": searches.theta_exec.iat[0], **config,
           **{f"mean_{m}": per_state[m] for m in metrics},
           **{f"{m}_se": per_seed[m].sem() for m in metrics},
           "n_states": searches.state.nunique(), "n_seeds": searches.seed.nunique()}
    return pd.DataFrame([row])


# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--envs", nargs="+", default=["frozen_lake", "cliffwalking", "gambler"], choices=sorted(tune.ENVS))
    parser.add_argument("--algos", nargs="+", default=ALGOS, choices=ALGOS)
    parser.add_argument("--sims", nargs="+", type=int, default=SIMS)
    parser.add_argument("--eps", type=float, help="fix the TV budget instead of tuning it")
    parser.add_argument("--theta-exec", nargs="+", type=float, default=THETA_EXEC)
    parser.add_argument("--theta-plan", type=float, default=None, help="default: THETA_PLAN of the env")
    parser.add_argument("--states", type=int, default=None, help="fixed random sample of this many root states")
    parser.add_argument("--seeds", type=int, default=EVAL_SEEDS, help="eval seeds per state")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()
    if args.seeds < 2:
        parser.error("--seeds must be at least 2 to estimate standard error")

    for env in args.envs:
        theta = THETA_PLAN[env] if args.theta_plan is None else args.theta_plan
        tune.THETAS[env] = [theta]
        budgets = tune.SEARCH_SPACE["uncertainty_budget"] if args.eps is None else [args.eps]
        for theta_exec in args.theta_exec:
            model = tune.make_env(env, theta_exec)
            robust = {eps: solve(model, tune.FIXED["gamma"], eps) for eps in budgets}
            truth = {"robust": robust, "nominal": solve(model, tune.FIXED["gamma"], 0.0),
                     "theta_exec": theta_exec}
            states = root_states(robust, args.states)
            if not states:
                raise ValueError(f"No scored root states for {env} at theta_exec={theta_exec:g}")
            pair = os.path.join(args.out, env, f"plan{theta:g}", f"exec{theta_exec:g}")
            print(f"######## {env}: plan={theta:g}, exec={theta_exec:g}, {len(states)} root states", flush=True)
            for sims in args.sims:
                for algo in args.algos:
                    print(f"================ {env} | {algo} | sims {sims} ================", flush=True)
                    tune_dir = tune.result_dir(os.path.join(pair, "tune"), algo, env, sims, args.eps)
                    tune_path = os.path.join(tune_dir, "top3.json")
                    if not os.path.exists(tune_path):
                        searches, results = tune_robust(env, algo, sims, args.eps, states, args.workers, truth)
                        save_tune(tune_dir, env, algo, sims, theta_exec, states, searches, results)
                    run_dir = tune.result_dir(os.path.join(pair, "run"), algo, env, sims, args.eps)
                    summary_path = os.path.join(run_dir, "summary.csv")
                    if os.path.exists(summary_path):
                        print("already run", run_dir, flush=True)
                        continue
                    with open(tune_path) as file:
                        config = next(entry for entry in json.load(file)["top"] if entry["rank"] == 1)["params"]
                    if algo == "gsf-robust-uct-plus":
                        config["max_depth"] = tune.PLANNERS[algo].config_cls(**config).max_depth
                    searches = evaluate(env, algo, {0: config}, states, range(args.seeds), args.workers,
                                        truth, "eval", EVAL_TAG)
                    summary = summarize(searches, config, env, algo)
                    os.makedirs(run_dir, exist_ok=True)
                    searches.to_csv(os.path.join(run_dir, "searches.csv"), index=False)
                    temporary = summary_path + ".tmp"
                    summary.to_csv(temporary, index=False)
                    os.replace(temporary, summary_path)
                    print(summary[["uncertainty_budget", "mean_robust_regret", "robust_regret_se"]]
                          .round(5).to_string(index=False), flush=True)
    print("exp4 done", flush=True)


if __name__ == "__main__":
    main()
