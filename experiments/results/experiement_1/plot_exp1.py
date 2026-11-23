"""Exp1 (merge ablation) plot: mean reward vs theta_exec, one panel per env, 5 planners, +-1 SE band.
    python experiments/results/experiement_1/plot_exp1.py            -> exp1_mean_reward.{png,pdf} (run/)
    python experiments/results/experiement_1/plot_exp1.py plan0.5    -> exp1_mean_reward_plan0.5.{png,pdf}:
        FrozenLake and CliffWalking from plan0.5/run (theta_plan 0.5, restored from git 75e7111), Gambler from run/
    python experiments/results/experiement_1/plot_exp1.py fl0.2      -> exp1_mean_reward_fl0.2.{png,pdf}:
        FrozenLake at theta_plan 0.2 from the main comparison runs (same pipeline, theta_exec 0.1-0.5 only; the
        Exp1 plan 0.2 run with exec 0.1-0.8 did not finish), CliffWalking from plan0.5/run, Gambler from run/
    python experiments/results/experiement_1/plot_exp1.py plan0.2    -> exp1_mean_reward_plan0.2.{png,pdf}:
        FrozenLake and CliffWalking at theta_plan 0.2 from the main comparison runs (theta_exec 0.1-0.5), Gambler from
        run/ (theta_plan 0.7). The Exp1 plan 0.2 runs themselves were stopped (user, 2026-11-19).
    python experiments/results/experiement_1/plot_exp1.py n4000      -> exp1_mean_reward_n4000.{png,pdf}:
        FrozenLake and CliffWalking at N = 4000, theta_plan 0.5, from n4000/plan0.5/run (each planner tuned at
        N = 4000). No Gambler panel: Gambler was not run at N = 4000."""
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "run")
VARIANT = sys.argv[1] if len(sys.argv) > 1 else ""
P05 = os.path.join(HERE, "plan0.5", "run")
MAIN = lambda env, plan: os.path.join(HERE, "..", "main_comparision", env, f"plan{plan}", "run")
RUN_OF = {"plan0.5": {"frozen_lake": P05, "cliffwalking": P05},
          "fl0.2": {"frozen_lake": MAIN("frozen_lake", 0.2), "cliffwalking": P05},
          "plan0.2": {"frozen_lake": MAIN("frozen_lake", 0.2), "cliffwalking": MAIN("cliffwalking", 0.2)},
          "n4000": {"frozen_lake": os.path.join(HERE, "n4000", "plan0.5", "run"),
                    "cliffwalking": os.path.join(HERE, "n4000", "plan0.5", "run")}}.get(VARIANT, {})
SIMS = 4000 if VARIANT == "n4000" else 2000
# env -> (title, theta name, eps of the robust planners)
ENVS = {"frozen_lake": ("FrozenLake", "slip rate", 0.2), "cliffwalking": ("CliffWalking", "slip rate", 0.2),
        "gambler": ("Gambler", "$p_h$", 0.2)}
if VARIANT == "n4000":
    del ENVS["gambler"]
# planner -> (folder, robust, label, color, linestyle); hues in fixed categorical order, dashes = power (non-robust)
ALGOS = [("st_power_uct", False, "Power-UCT", "#4a4945", "--"),
         ("robust_uct", True, "Robust-UCT", "#2a78d6", "-"),
         ("gs_power_uct", False, "GS-Power-UCT", "#eb6834", "--"),
         ("gs_power_uct_f", False, "GSF-Power-UCT", "#1baf7a", "--"),
         ("gs_robust_uct", True, "GS-Robust-UCT", "#eda100", "-"),
         ("gs_robust_uct_f", True, "GSF-Robust-UCT", "#e87ba4", "-")]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, len(ENVS), figsize=(16 * len(ENVS) / 3 + (1.5 if len(ENVS) < 3 else 0), 4.6))
for ax, (env, (title, theta, eps)) in zip(axes.flat, ENVS.items()):
    plan = None
    for folder, robust, label, color, ls in ALGOS:
        # main comparison FrozenLake was flattened to run/<algo>/[eps/]summary.csv; the rest is run/<algo>/<env>/sims2000/
        base = os.path.join(RUN_OF.get(env, RUN), folder)
        sub = [env, f"sims{SIMS}"] if os.path.isdir(os.path.join(base, env)) else []
        path = os.path.join(base, *sub, *([f"eps{eps:g}"] if robust else []), "summary.csv")
        if not os.path.exists(path):
            continue
        s = pd.read_csv(path).sort_values("theta_exec")
        plan = s.theta_plan.iloc[0]
        x, y, se = s.theta_exec, s.mean_reward, s.reward_se
        ax.fill_between(x, y - se, y + se, color=color, alpha=0.15, linewidth=0)
        # Power-UCT on top with square markers: on Gambler it lies on GSF-Power-UCT and Robust-UCT
        top = folder == "st_power_uct"
        ax.plot(x, y, ls, color=color, lw=2, marker="s" if top else "o", ms=5, mec="white", mew=1.2,
                zorder=5 if top else 2,
                label=f"{label}" + (f" (ε={eps:g})" if robust else ""))
    ax.axvline(plan, color=MUTED, lw=1, ls=":")
    ax.annotate("θ_plan", (plan, 0.0),
                xycoords=("data", "axes fraction"), xytext=(4, 4), textcoords="offset points", color=MUTED, fontsize=8.5)
    ax.set_title(f"{title}  (θ_plan = {plan:g}, ε = {eps:g})", color=INK, fontsize=11, loc="left")
    ax.set_xticks(x); ax.xaxis.set_major_formatter("{x:g}")
    ax.set_xlabel(f"θ_exec ({theta})"); ax.set_ylabel("mean reward")
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, [l.replace(" (ε=0.2)", "") for l in labels], loc="upper center", ncol=6 if len(ENVS) == 3 else 3, frameon=False,
           bbox_to_anchor=(0.5, 1.0))
fig.suptitle(f"Experiment 1: merge ablation, mean reward vs θ_exec (N = {SIMS}, band = ±1 SE)", y=1.1, color=INK)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, f"exp1_mean_reward{'_' + VARIANT if VARIANT else ''}.{ext}"), dpi=200, bbox_inches="tight")
print("saved", os.path.join(HERE, f"exp1_mean_reward{'_' + VARIANT if VARIANT else ''}.png"))
