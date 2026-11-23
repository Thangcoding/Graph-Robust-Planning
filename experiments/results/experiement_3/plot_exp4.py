"""Exp4 plot: wall-clock time of one search vs N (top) and time relative to UCT (bottom), FrozenLake + Gambler.
    python experiments/results/experiement_4/plot_exp4.py"""
import os
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ENVS = {"frozen_lake": "FrozenLake (θ_plan = 0.1, 10 root states × 3 seeds)",
        "gambler": "Gambler (θ_plan = 0.7, 9 root states × 3 seeds)"}
# algo -> (label, color, linestyle); the 5 Exp1 planners keep their Exp1 colors, tree baselines are added
ALGOS = [("uct", "UCT", "#6b6a64", ":"),
         ("stochastic-power-uct", "Stochastic-Power-UCT", "#8b5cf6", ":"),
         ("robust-uct", "Robust-UCT", "#2a78d6", "-"),
         ("gs-power-uct", "GS-Power-UCT", "#eb6834", "--"),
         ("gsf-power-uct", "GSF-Power-UCT", "#1baf7a", "--"),
         ("gs-robust-uct", "GS-Robust-UCT", "#eda100", "-"),
         ("gsf-robust-uct", "GSF-Robust-UCT", "#e87ba4", "-")]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
for col, (env, title) in enumerate(ENVS.items()):
    s = pd.read_csv(os.path.join(HERE, env, "summary.csv"))
    base = s[s.algo == "uct"].set_index("sims")["mean"]
    for algo, label, color, ls in ALGOS:
        d = s[s.algo == algo].sort_values("sims")
        x, y, se = d.sims, d["mean"] * 1000, d["sem"] * 1000
        ax = axes[0, col]
        ax.fill_between(x, y - se, y + se, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, y, ls, color=color, lw=2, marker="o", ms=5, mec="white", mew=1.2, label=label)
        axes[1, col].plot(x, d["mean"].values / base.loc[d.sims].values, ls, color=color, lw=2, marker="o", ms=4,
                          mec="white", mew=1)
    axes[0, col].set_title(title, color=INK, fontsize=11, loc="left")
    axes[0, col].set_ylabel("ms / search")
    axes[1, col].axhline(1, color=MUTED, lw=1)
    axes[1, col].set_ylabel("time / UCT time")
    axes[1, col].set_xlabel("number of simulations N")
    for ax in axes[:, col]:
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        ax.set_xticks([100, 500, 1000, 1500, 2000])
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.04))
fig.suptitle("Experiment 4: wall-clock time of one search vs N (single process, fixed c = 1, p = 1, ε = 0.2, "
             "band = ±1 SE)", y=1.07, color=INK)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, f"exp4_walltime.{ext}"), dpi=200, bbox_inches="tight")
print("saved", os.path.join(HERE, "exp4_walltime.png"))
