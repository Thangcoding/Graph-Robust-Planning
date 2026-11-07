"""Ablation: full merge and depth merge in the robust setting, on one nominal model and a fixed budget.

    python experiments/ablation_merge.py                        # frozen_lake and american_option, N = 2000
    python experiments/ablation_merge.py --envs american_option --sims 100 --out /tmp/test

Nominal model 0.5, eps 0.5 for the robust planners, executed in theta 0.1 ... 0.8 (slip_rate on FrozenLake, p_up on
the American option). The tune.py and run.py logic is reused with these settings, tune.THETAS and the shared results
are not changed: output goes to experiments/results/experiment_1/{tune,run}/<algo folder>/<env>/sims<N>[/eps<e>]/.
A stage whose output exists is skipped, so an interrupted ablation continues by running the same command again (the
run stage saves at its end: an interrupted one starts over, with the same seeds).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run  # noqa: E402
import tune  # noqa: E402
from src.envs import AmericanOption  # noqa: E402

# at module level: the worker processes import this file again, they would not see a change made in main()
tune.ENVS["american_option"] = (AmericanOption, "p_up")

THETA_PLAN = 0.5
EPS = 0.5
THETA_EXEC = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# algo -> eps
METHODS = {"robust-uct": EPS, "gs-power-uct": None, "gsf-power-uct": None, "gs-robust-uct": EPS, "gsf-robust-uct": EPS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--envs", nargs="+", default=["frozen_lake", "american_option"])
    parser.add_argument("--algos", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--sims", type=int, default=2000)
    parser.add_argument("--eps", type=float, default=EPS, help="uncertainty budget of the robust planners")
    parser.add_argument("--tune-only", action="store_true", help="stop after tune: the regret, no episodes")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--out", default=os.path.join(tune.ROOT, "experiments", "results", "experiment_1"))
    args = parser.parse_args()

    tune.OUT, run.OUT = os.path.join(args.out, "tune"), os.path.join(args.out, "run")

    for env in args.envs:
        tune.THETAS[env] = [THETA_PLAN]
        tune.EXEC_THETAS[env] = THETA_EXEC
        for algo in args.algos:
            eps = None if METHODS[algo] is None else args.eps
            print(f"================ {env} | {algo} | sims {args.sims} | eps {eps} ================", flush=True)
            tune_dir = tune.result_dir(tune.OUT, algo, env, args.sims, eps)
            if os.path.exists(os.path.join(tune_dir, "top3.json")):
                print("already tuned", tune_dir, flush=True)
            else:
                searches, results = tune.tune(env, algo, args.sims, args.workers, eps)
                print("tuned", tune.save(env, algo, args.sims, searches, results, eps), flush=True)
                print(results.sort_values("rank").head(3)[["exploration_const", "p", "mean_regret", "regret_se"]]
                      .round(5).to_string(index=False), flush=True)

            if args.tune_only:
                continue
            out_dir = tune.result_dir(run.OUT, algo, env, args.sims, eps)
            if os.path.exists(os.path.join(out_dir, "summary.csv")):
                print("already run", out_dir, flush=True)
                continue
            configs = run.load_best_configs(env, algo, args.sims, eps)
            episodes = run.run_all(env, algo, configs, args.workers, run.solve_models(env, configs))
            summary = run.summarize(episodes)
            os.makedirs(out_dir, exist_ok=True)
            episodes.to_csv(os.path.join(out_dir, "episodes.csv"), index=False)
            summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
            print(summary[["theta_exec", "success_rate", "mean_reward", "mean_disc_reward", "mean_regret_true",
                           "mean_steps"]].round(4).to_string(index=False), flush=True)
    print("ablation done", flush=True)


if __name__ == "__main__":
    main()
