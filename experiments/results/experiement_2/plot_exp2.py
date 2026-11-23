"""Exp2 (simulation budget) plot: mean reward averaged over theta_exec 0.1-0.8 vs N, one panel per env, 6 planners.
Band = +-1 SE of that average: mean reward of each seed over all its episodes, SE over seeds.
    python experiments/results/experiement_2/plot_exp2.py
N grid 2^4 ... 2^12 (user, 2026-11-20). Five points reuse runs at a nearby N and are plotted at 2^k:
2^7 <- 100, 2^8 <- 300, 2^10 <- 1000, 2^11 <- 2000, 2^12 <- 4000 (FrozenLake/CliffWalking copied from the Exp1 N = 4000
run, experiement_1/n4000/plan0.5; Gambler from experiments/logs/exp2_pow2.sh). Missing points are skipped."""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "run")
ENVS = {"frozen_lake": "FrozenLake", "cliffwalking": "CliffWalking", "gambler": "Gambler"}
# x position 2^k -> sims<N> folder it is read from
SIMS = {2**4: 16, 2**5: 32, 2**6: 64, 2**7: 100, 2**8: 300, 2**9: 512, 2**10: 1000, 2**11: 2000, 2**12: 4000}
EPS = 0.2
# planner -> (folder, robust, label, color, linestyle); same colors as the Exp1 plot, dashes = power (non-robust)
ALGOS = [("st_power_uct", False, "Power-UCT", "#8a8983", "--"),
         ("robust_uct", True, "Robust-UCT", "#2a78d6", "-"),
         ("gs_power_uct", False, "GS-Power-UCT", "#eb6834", "--"),
         ("gs_power_uct_f", False, "GSF-Power-UCT", "#1baf7a", "--"),
         ("gs_robust_uct", True, "GS-Robust-UCT", "#eda100", "-"),
         ("gs_robust_uct_f", True, "GSF-Robust-UCT", "#e87ba4", "-"),
         ("gsf_robust_uct_plus", True, "GSF-Robust-UCT+ (adaptive H)", "#7b3fb5", "-")]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
for ax, (env, title) in zip(axes, ENVS.items()):
    plan = None
    for folder, robust, label, color, ls in ALGOS:
        xs, ys, ses = [], [], []
        for n, src in SIMS.items():
            path = os.path.join(RUN, folder, env, f"sims{src}", *([f"eps{EPS:g}"] if robust else []), "summary.csv")
            if not os.path.exists(path):
                continue
            s = pd.read_csv(path).dropna(subset=["theta_exec"])   # skip stray non-data lines
            plan = s.theta_plan.iloc[0]
            per_seed = pd.read_csv(path.replace("summary.csv", "episodes.csv")).groupby("seed").reward.mean()
            xs.append(n); ys.append(s.mean_reward.mean()); ses.append(per_seed.std(ddof=1) / np.sqrt(len(per_seed)))
        if not xs:
            continue
        xs, ys, ses = map(np.array, (xs, ys, ses))
        ax.fill_between(xs, ys - ses, ys + ses, color=color, alpha=0.25, linewidth=0)
        ax.plot(xs, ys, ls, color=color, lw=2, marker="o", ms=5, mec="white", mew=1.2, label=label)
    ax.set_xscale("log", base=2); ax.set_xticks(list(SIMS))
    ax.set_xticklabels([f"$2^{{{n.bit_length() - 1}}}$" for n in SIMS]); ax.minorticks_off()
    ax.set_title(f"{title}  (θ_plan = {plan:g}, ε = {EPS:g})", color=INK, fontsize=11, loc="left")
    ax.set_xlabel("simulations per search N"); ax.set_ylabel("mean reward, avg over θ_exec 0.1–0.8")
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 1.02))
fig.suptitle("Experiment 2: simulation budget, average mean reward vs N (band = ±1 SE)", y=1.08, color=INK)
fig.text(0.5, -0.04, "2^7, 2^8, 2^10, 2^11, 2^12 use the runs at N = 100, 300, 1000, 2000, 4000", ha="center",
         color=MUTED, fontsize=8.5)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, f"exp2_mean_reward.{ext}"), dpi=200, bbox_inches="tight")
print("saved", os.path.join(HERE, "exp2_mean_reward.png"))
