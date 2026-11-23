"""Experiment 4: robust regret at the root, against the exact robust values of value iteration.

    python experiments/exp4_robust_regret.py                                   # 3 envs, 3 robust planners, all N
    python experiments/exp4_robust_regret.py --envs gambler --sims 100 300 --seeds 5 --out /tmp/test

Question: how close is the root action of each robust planner to the exact robust optimum of the problem it is given
(nominal model theta_plan, TV ball of radius eps)? No model mismatch and no episodes: one search from each fixed
root state of the planning model, the chosen action a scored with Q*_rob = robust value iteration on theta_plan, eps
(value_iteration.solve, the same TV worst case and the same split backup as the planners, but over the full support):

    robust regret(s, a) = max_b Q*_rob(s, b) - Q*_rob(s, a)      (>= 0, 0 = an exact robust-optimal action)

Pipeline, as tune.py / run.py, for every (env, planner, N):
  1. tune   exploration_const and p (eps fixed) on the robust regret itself, with tune.py's screen + refine rounds
            (SCREEN_SEEDS, REFINE_SEEDS) -> tune/<algo folder>/<env>/sims<N>/eps<eps>/{searches,results}.csv, top3.json
  2. eval   the rank-1 config on new seeds (EVAL_SEEDS, own random streams) at the same fixed root states
            -> run/<algo folder>/<env>/sims<N>/eps<eps>/{searches,summary}.csv
Root states: every non-terminal state of the planning model where the robust actions really differ (robust Q* range
> FLAT_TOL; elsewhere every action is robust-optimal), optionally a fixed random sample of --states of them (seed 0,
same states for every planner and N).
Average: robust regret per state averaged over the seeds, then over the states (every state weighs the same);
SE over the seeds of the per-seed state average. Also kept: the nominal regret (Q* of theta_plan, eps = 0) of the same
action, and the rate of exact robust-optimal actions.
A stage whose output exists is skipped, so an interrupted run continues by running the same command again.
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

ALGOS = ["robust-uct", "gs-robust-uct", "gsf-robust-uct"]
# planning model and N grid of Exp3 (wall-clock time per search vs N), so the regret of Exp4 and the time of Exp3 are
# measured on the same searches' settings (regret vs time); CliffWalking as FrozenLake
THETA_PLAN = {"frozen_lake": 0.1, "cliffwalking": 0.1, "gambler": 0.7, "inventory": 10.0}
SIMS = [100, 300, 500, 700, 1000, 1500, 2000]
EPS = 0.2
EVAL_SEEDS = 100
EVAL_TAG = 15485863   # eval searches get their own random streams, apart from tune (TUNE_TAG) and run (RUN_TAG)
OPT_TOL = 1e-6        # robust regret below this counts as a robust-optimal action
OUT = os.path.join(tune.ROOT, "experiments", "results", "experiement_4")

TRUTH: dict = {}      # "robust" / "nominal" -> {state: Q*(state, .)}; set in every worker process


def init_worker(truth: dict) -> None:
    TRUTH.update(truth)


def root_states(q_rob: dict, n: int | None) -> list[int]:
    states = [s for s in sorted(q_rob) if q_rob[s].max() - q_rob[s].min() > tune.FLAT_TOL]
    if n is None or n >= len(states):
        return states
    return sorted(int(s) for s in np.random.default_rng(0).choice(states, size=n, replace=False))


# ---------------------------------------------------------------- tune (tune.py rounds, robust regret as the score)

def tune_robust(env: str, algo: str, sims: int, eps: float, states: list[int], workers: int, q_rob: dict):
    """tune.tune with the robust Q* of theta_plan as the only 'real env' and the root states fixed to `states`."""
    theta = tune.THETAS[env][0]
    truth = {theta: {s: q_rob[s] for s in states}}   # scored_states(truth) = states (all have a robust gap)
    configs = dict(enumerate(tune.make_configs(algo, sims, eps)))
    screen = tune.evaluate(env, algo, configs, tune.SCREEN_SEEDS, "screen", workers, truth)
    screened = tune.rank(tune.summarize(screen))
    finalists = screened[screened["rank"] <= tune.REFINE_TOP].config_id.tolist()
    refine = tune.evaluate(env, algo, {i: configs[i] for i in finalists}, tune.REFINE_SEEDS, "refine", workers, truth)
    pooled = tune.rank(tune.summarize(pd.concat([screen[screen.config_id.isin(finalists)], refine])))
    dropped = screened[~screened.config_id.isin(finalists)]
    results = pd.concat([pooled.assign(refined=True), dropped.assign(refined=False)], ignore_index=True)
    return pd.concat([screen, refine], ignore_index=True), results


def save_tune(out_dir: str, env: str, algo: str, sims: int, states: list[int], searches, results) -> None:
    os.makedirs(out_dir, exist_ok=True)
    searches.to_csv(os.path.join(out_dir, "searches.csv"), index=False)
    results.sort_values("rank").to_csv(os.path.join(out_dir, "results.csv"), index=False)
    summary = {"env": env, "algo": algo, "num_simulations": sims, "theta_plan": tune.THETAS[env][0],
               "score": "robust regret at the root", "states": states, "top": tune.top_configs(results)}
    with open(os.path.join(out_dir, "top3.json"), "w") as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------- eval

def eval_once(job: dict) -> dict:
    seeds = np.random.SeedSequence([job["seed"], job["state"], round(job["theta_plan"] * 1e6), EVAL_TAG]).spawn(2)
    env_seed, planner_seed = (int(s.generate_state(1)[0]) for s in seeds)
    plan_env = tune.make_env(job["env"], job["theta_plan"], seed=env_seed)
    plan_env.reset(seed=env_seed)
    tune.set_state(plan_env, job["state"])
    planner = tune.make_planner(job["algo"], plan_env, **job["config"], seed=planner_seed)
    action = planner.plan(plan_env)
    r_rob = tune.regret(TRUTH["robust"], job["state"], action)
    return {"seed": job["seed"], "state": job["state"], "action": action, "env_seed": env_seed,
            "planner_seed": planner_seed, "robust_regret": r_rob, "robust_opt": float(r_rob < OPT_TOL),
            "nominal_regret": tune.regret(TRUTH["nominal"], job["state"], action)}


def evaluate(env: str, algo: str, config: dict, states: list[int], n_seeds: int, workers: int, truth: dict):
    jobs = [{"env": env, "algo": algo, "config": config, "theta_plan": tune.THETAS[env][0], "state": s, "seed": seed}
            for s in states for seed in range(n_seeds)]
    print(f"[eval] {len(jobs)} searches", flush=True)
    rows = []
    with Pool(workers, initializer=init_worker, initargs=(truth,)) as pool:
        for done, row in enumerate(pool.imap_unordered(eval_once, jobs, chunksize=4), 1):
            rows.append(row)
            if done % max(1, len(jobs) // 10) == 0:
                print(f"  {done}/{len(jobs)}", flush=True)
    return pd.DataFrame(rows).sort_values(["state", "seed"]).reset_index(drop=True)


def summarize(searches: pd.DataFrame, config: dict, env: str, algo: str) -> pd.DataFrame:
    """One row: per-state means averaged over the states; SE over the seeds of the per-seed state average."""
    metrics = ["robust_regret", "nominal_regret", "robust_opt"]
    per_state = searches.groupby("state")[metrics].mean().mean()
    per_seed = searches.groupby("seed")[metrics].mean()
    row = {"env": env, "algo": algo, "theta_plan": tune.THETAS[env][0], **config,
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
    parser.add_argument("--eps", type=float, default=EPS)
    parser.add_argument("--theta-plan", type=float, default=None, help="default: THETA_PLAN of the env")
    parser.add_argument("--states", type=int, default=None, help="fixed random sample of this many root states")
    parser.add_argument("--seeds", type=int, default=EVAL_SEEDS, help="eval seeds per state")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()

    for env in args.envs:
        theta = THETA_PLAN[env] if args.theta_plan is None else args.theta_plan
        tune.THETAS[env] = [theta]
        model = tune.make_env(env, theta)
        q_rob = solve(model, tune.FIXED["gamma"], args.eps)
        truth = {"robust": q_rob, "nominal": solve(model, tune.FIXED["gamma"], 0.0)}
        states = root_states(q_rob, args.states)
        print(f"######## {env}: theta_plan {theta:g}, eps {args.eps:g}, {len(states)} root states {states}", flush=True)
        for sims in args.sims:
            for algo in args.algos:
                print(f"================ {env} | {algo} | sims {sims} | eps {args.eps:g} ================", flush=True)
                tune_dir = tune.result_dir(os.path.join(args.out, "tune"), algo, env, sims, args.eps)
                if os.path.exists(os.path.join(tune_dir, "top3.json")):
                    print("already tuned", tune_dir, flush=True)
                else:
                    searches, results = tune_robust(env, algo, sims, args.eps, states, args.workers, q_rob)
                    save_tune(tune_dir, env, algo, sims, states, searches, results)
                    print(results.sort_values("rank").head(3)[["exploration_const", "p", "mean_regret", "regret_se"]]
                          .round(5).to_string(index=False), flush=True)
                run_dir = tune.result_dir(os.path.join(args.out, "run"), algo, env, sims, args.eps)
                if os.path.exists(os.path.join(run_dir, "summary.csv")):
                    print("already run", run_dir, flush=True)
                    continue
                with open(os.path.join(tune_dir, "top3.json")) as f:
                    config = next(e for e in json.load(f)["top"] if e["rank"] == 1)["params"]
                searches = evaluate(env, algo, config, states, args.seeds, args.workers, truth)
                summary = summarize(searches, config, env, algo)
                os.makedirs(run_dir, exist_ok=True)
                searches.to_csv(os.path.join(run_dir, "searches.csv"), index=False)
                summary.to_csv(os.path.join(run_dir, "summary.csv"), index=False)
                print(summary[["exploration_const", "p", "mean_robust_regret", "robust_regret_se", "mean_robust_opt",
                               "mean_nominal_regret"]].round(5).to_string(index=False), flush=True)
    print("exp4 done", flush=True)


if __name__ == "__main__":
    main()
