"""Exp4 (robust regret at the root) plot: mean robust regret vs N (top row) and rate of exact robust-optimal actions
vs N (bottom row), one column per env, 3 robust planners. Band = +-1 SE over the seeds.
    python experiments/results/experiement_4/plot_exp4.py"""
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "run")
ENVS = {"frozen_lake": "FrozenLake", "cliffwalking": "CliffWalking", "gambler": "Gambler"}
EPS = 0.2
# same colors as the Exp1 / Exp2 plots
ALGOS = [("robust_uct", "Robust-UCT", "#2a78d6"),
         ("gs_robust_uct", "GS-Robust-UCT", "#eda100"),
         ("gs_robust_uct_f", "GSF-Robust-UCT", "#e87ba4")]
ROWS = [("robust_regret", "robust regret at the root"), ("robust_opt", "robust-optimal action rate")]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, len(ENVS), figsize=(16, 7.6), squeeze=False)
for col, (env, title) in enumerate(ENVS.items()):
    info = ""
    for folder, label, color in ALGOS:
        paths = glob.glob(os.path.join(RUN, folder, env, "sims*", f"eps{EPS:g}", "summary.csv"))
        if not paths:
            continue
        s = pd.concat(map(pd.read_csv, paths)).sort_values("num_simulations")
        info = f"θ_plan = {s.theta_plan.iloc[0]:g}, ε = {EPS:g}, {s.n_states.iloc[0]} states × {s.n_seeds.iloc[0]} seeds"
        for row, (metric, ylabel) in enumerate(ROWS):
            ax, x, y, se = axes[row, col], s.num_simulations, s[f"mean_{metric}"], s[f"{metric}_se"]
            ax.fill_between(x, y - se, y + se, color=color, alpha=0.25, linewidth=0)
            ax.plot(x, y, "-", color=color, lw=2, marker="o", ms=5, mec="white", mew=1.2, label=label)
            ax.set_ylabel(ylabel)
    for row in range(len(ROWS)):
        ax = axes[row, col]
        ax.set_xscale("log"); ax.minorticks_off(); ax.xaxis.set_major_formatter("{x:g}")
        ticks = sorted({int(n) for line in ax.get_lines() for n in line.get_xdata()})
        if ticks:
            ax.set_xticks(ticks)
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        if row == len(ROWS) - 1:
            ax.set_xlabel("simulations per search N")
    axes[0, col].set_title(f"{title}  ({info})", color=INK, fontsize=10, loc="left")
handles, labels = max((ax.get_legend_handles_labels() for ax in axes[0]), key=lambda hl: len(hl[1]))
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
fig.suptitle("Experiment 4: robust regret of the root action vs exact robust value iteration (band = ±1 SE)",
             y=1.06, color=INK)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, f"exp4_robust_regret.{ext}"), dpi=200, bbox_inches="tight")
print("saved", os.path.join(HERE, "exp4_robust_regret.png"))
