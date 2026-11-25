"""Run the standard tune/run pipeline independently for each model pair."""

import argparse
import itertools
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

import tune

PLAN = {"gambler": (0.2, 0.4, 0.6, 0.8), "frozen_lake": (0.2, 0.3, 0.4, 0.5)}
EXEC = tuple(i / 10 for i in range(1, 9))
ROOT = Path(tune.ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)))
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/results/main_comparision/per_theta_pair")
    args = parser.parse_args()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    averages = []
    for sims in (2 ** exponent for exponent in range(4, 12)):
        for env, plans in PLAN.items():
            for algo in tune.PLANNERS:
                rewards = []
                for theta_plan, theta_exec in itertools.product(plans, EXEC):
                    pair = output / env / f"plan{theta_plan:g}" / f"exec{theta_exec:g}"
                    summary = Path(tune.result_dir(pair / "run", algo, env, sims)) / "summary.csv"
                    if not summary.is_file():
                        print(f"{env} {algo} N={sims} plan={theta_plan:g} exec={theta_exec:g}", flush=True)
                        settings = dict(
                            ENVS=env, ALGOS=algo, SIMS=str(sims), WORKERS=str(args.workers),
                            THETA_PLAN=str(theta_plan), THETA_EXEC=str(theta_exec), EPS="none", P_VALUES="none",
                            TUNE_ROOT=str(pair / "tune"), RUN_ROOT=str(pair / "run"), SKIP_TUNE="",
                            LOG_DIR=str(ROOT / "experiments/logs/main_comparison" / env / pair.parent.name / pair.name),
                            PATH=f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
                        )
                        subprocess.run(["bash", "scripts/run_pipeline.sh"], cwd=ROOT,
                                       env={**os.environ, **settings}, check=True)
                    rows = pd.read_csv(summary)
                    if (len(rows) != 1 or rows.theta_plan.iat[0] != theta_plan
                            or rows.theta_exec.iat[0] != theta_exec or rows.num_simulations.iat[0] != sims
                            or rows.config_id.iat[0] != algo or pd.isna(rows.mean_reward.iat[0])):
                        raise ValueError(f"Unexpected pair summary in {summary}")
                    rewards.append(rows.mean_reward.iat[0])
                averages.append({"env": env, "algo": algo, "num_simulations": sims,
                                 "n_pairs": len(rewards), "average_mean_reward": sum(rewards) / len(rewards)})
                temporary = output / "mean_reward.csv.tmp"
                pd.DataFrame(averages).to_csv(temporary, index=False)
                temporary.replace(output / "mean_reward.csv")
    print(f"Main Comparison complete. Results in {output / 'mean_reward.csv'}", flush=True)


if __name__ == "__main__":
    main()
