"""Plot one Experiment 4 figure per environment and execution parameter."""
import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ENVS = {"frozen_lake": "FrozenLake", "cliffwalking": "CliffWalking", "gambler": "Gambler"}
PLANS = {"frozen_lake": 0.1, "cliffwalking": 0.1, "gambler": 0.7}
# same colors as the Exp1 / Exp2 plots
ALGOS = [("robust_uct", "Robust-UCT", "#2a78d6"),
         ("gs_robust_uct", "GS-Robust-UCT", "#eda100"),
         ("gs_robust_uct_f", "GSF-Robust-UCT", "#e87ba4"),
         ("gsf_robust_uct_plus", "GSF-Robust-UCT+", "#00836b")]
ROWS = [("robust_regret", "robust regret at the root"), ("robust_opt", "robust-optimal action rate")]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def plot(root, env, title, theta_exec):
    theta_plan = PLANS[env]
    pair = os.path.join(root, env, f"plan{theta_plan:g}", f"exec{theta_exec:g}")
    series = []
    for folder, label, color in ALGOS:
        pattern = os.path.join(pair, "run", folder, env, "sims*", "summary.csv")
        paths = glob.glob(pattern)
        if not paths:
            raise FileNotFoundError(f"No summaries for {label} in {pair}")
        results = pd.concat(map(pd.read_csv, paths)).sort_values("num_simulations")
        for metric, _ in ROWS:
            if results[f"{metric}_se"].isna().any() or results.n_seeds.lt(2).any():
                raise ValueError(f"Standard error requires at least two evaluation seeds in {pattern}")
        series.append((label, color, results))

    ticks = series[0][2].num_simulations.tolist()
    if any(results.num_simulations.tolist() != ticks for _, _, results in series):
        raise ValueError(f"Simulation budgets differ across planners in {pair}")

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, layout="constrained")
    for ax, (metric, ylabel) in zip(axes, ROWS):
        for label, color, results in series:
            x, y, se = results.num_simulations, results[f"mean_{metric}"], results[f"{metric}_se"]
            ax.fill_between(x, y - se, y + se, color=color, alpha=0.22, linewidth=0)
            ax.errorbar(x, y, yerr=se, color=color, linewidth=1.8, elinewidth=0.9,
                        capsize=2, marker="o", markersize=4, label=label)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].legend(loc="best", ncols=2, frameon=False)
    axes[-1].set_xscale("log", base=2)
    axes[-1].set_xticks(ticks, labels=[str(n) for n in ticks])
    axes[-1].minorticks_off()
    axes[-1].set_xlabel("Simulations per search")
    fig.suptitle(f"{title} — θ_plan = {theta_plan:g}, θ_exec = {theta_exec:g} (±1 SE)", color=INK)
    output = os.path.join(root, f"exp4_robust_regret_{env}_plan{theta_plan:g}_exec{theta_exec:g}")
    for ext in ("png", "pdf"):
        fig.savefig(f"{output}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", f"{output}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=HERE)
    args = parser.parse_args()
    for env, title in ENVS.items():
        for theta_exec in (0.4, 0.6, 0.8):
            plot(args.results, env, title, theta_exec)
