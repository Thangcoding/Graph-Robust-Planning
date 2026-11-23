"""Exp3 (uncertainty budget sweep) plot, N = 2000. Rows: FrozenLake (plan slip 0.02), Gambler (plan p_h 0.6).
Columns: GS-Robust-UCT and GSF-Robust-UCT, mean reward vs theta_exec, one line per eps (eps 0 = the graph power
planner with the same merge, the user's nominal baseline); last column: summary vs eps.
Error bars = +-1 SE; for the average over theta_exec it is the SE over seeds of each seed's mean reward.
    python experiments/results/experiement_3/plot_exp3.py"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "run")
# env -> (title, theta name, eps grid, degenerate eps (exact V_rob = 0, all actions tied), summary metric)
ENVS = {"frozen_lake": ("FrozenLake", "slip rate", [0.1, 0.2, 0.3, 0.4, 0.5], [], "avg"),
        "gambler": ("Gambler", "$p_h$", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], [0.6, 0.7], 0.2)}
# merge -> (robust folder, power folder, label, summary color); colors as in the Exp1/Exp2 plots
MERGES = [("gs_robust_uct", "gs_power_uct", "GS", "#eda100"), ("gs_robust_uct_f", "gs_power_uct_f", "GSF", "#e87ba4")]
RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281"]
INK, MUTED, GRID, BASE = "#1f1f1e", "#6b6a64", "#e4e3de", "#8a8983"

def load(folder, env, eps=None):
    path = os.path.join(RUN, folder, env, "sims2000", *([f"eps{eps:g}"] if eps else []), "summary.csv")
    s = pd.read_csv(path).dropna(subset=["theta_exec"]).sort_values("theta_exec")
    # SE of the average over theta_exec: mean of each seed over all its episodes, SE over seeds (not independent cells)
    per_seed = pd.read_csv(path.replace("summary.csv", "episodes.csv")).groupby("seed").reward.mean()
    s.attrs["avg_se"] = per_seed.std(ddof=1) / np.sqrt(len(per_seed))
    return s

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 3, figsize=(15, 8.4), gridspec_kw={"width_ratios": [1, 1, 0.85]})
for row, (env, (title, theta, grid, degenerate, metric)) in zip(axes, ENVS.items()):
    colors = [RAMP[round(i * (len(RAMP) - 1) / max(len(grid) - 1, 1))] for i in range(len(grid))]
    summary = {}
    for ax, (rob, power, label, scolor) in zip(row[:2], MERGES):
        base = load(power, env)
        plan = base.theta_plan.iloc[0]
        ax.plot(base.theta_exec, base.mean_reward, "--", color=BASE, lw=2, marker="o", ms=4.5, mec="white",
                label=f"ε = 0 ({label}-Power)")
        pick = lambda s: s.mean_reward.mean() if metric == "avg" else s.set_index("theta_exec").mean_reward[metric]
        pick_se = lambda s: (s.attrs["avg_se"] if metric == "avg"
                             else s.set_index("theta_exec").reward_se[metric])
        xs, ys, ses = [0.0], [pick(base)], [pick_se(base)]
        for eps, color in zip(grid, colors):
            s = load(rob, env, eps)
            deg = eps in degenerate
            ax.plot(s.theta_exec, s.mean_reward, ":" if deg else "-", color=color, lw=1.8, marker="o", ms=4.5,
                    mec="white", label=f"ε = {eps:g}" + (" (degenerate)" if deg else ""))
            xs.append(eps); ys.append(pick(s)); ses.append(pick_se(s))
        summary[label] = (np.array(xs), np.array(ys), np.array(ses), scolor)
        ax.axvline(plan, color=MUTED, lw=1, ls=":")
        ax.annotate("θ_plan", (plan, 0.0), xycoords=("data", "axes fraction"), xytext=(4, 4),
                    textcoords="offset points", color=MUTED, fontsize=8.5)
        ax.set_title(f"{title}: {label}-Robust-UCT  (θ_plan = {plan:g})", color=INK, fontsize=11, loc="left")
        ax.set_xlabel(f"θ_exec ({theta})"); ax.set_ylabel("mean reward")
        ax.set_ylim(-0.03, 1.03); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8.5, loc="best")
    ax = row[2]
    for label, (xs, ys, ses, color) in summary.items():
        ax.fill_between(xs, ys - ses, ys + ses, color=color, alpha=0.25, linewidth=0)
        ax.errorbar(xs, ys, yerr=ses, fmt="none", ecolor=color, elinewidth=1.2, capsize=3)
        ax.plot(xs, ys, "-", color=color, lw=2, marker="o", ms=5.5, mec="white", mew=1.2, label=f"{label}-Robust-UCT")
    if degenerate:
        ax.axvspan(min(degenerate) - 0.05, max(degenerate) + 0.05, color=GRID, alpha=0.6, lw=0)
        ax.annotate("degenerate ε", (np.mean(degenerate), 0.96), xycoords=("data", "axes fraction"), ha="center",
                    color=MUTED, fontsize=8.5)
    ax.set_xticks([0.0] + grid)
    ax.set_title(f"{title}: " + ("avg over θ_exec" if metric == "avg" else f"worst case, θ_exec = {metric:g}"),
                 color=INK, fontsize=11, loc="left")
    ax.set_xlabel("ε  (0 = power baseline)"); ax.set_ylabel("mean reward")
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=9)
fig.suptitle("Experiment 3: uncertainty budget ε sweep (TV, N = 2000; band = ±1 SE)", color=INK, y=1.0)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, f"exp3_eps_sweep.{ext}"), dpi=200, bbox_inches="tight")
print("saved", os.path.join(HERE, "exp3_eps_sweep.png"))
