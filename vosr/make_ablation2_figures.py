"""Plots Ablation 2 (cost-limit sensitivity). Reads cost_limit_10 and
cost_limit_70 from ablation_logs_2/, and reuses the existing cost_limit=25
"deployed" arm from ablation_logs/deployed/ (Ablation 1) rather than
retraining it. Output: final_figures/ablations_2/*.png
"""
import csv
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

LOG_ROOT_2 = "ablation_logs_2"
LOG_ROOT_1 = "ablation_logs"
OUT_DIR = os.path.join("final_figures", "ablations_2")

ENVS = ["SafetyPointGoal1-v0", "SafetyPointButton1-v0"]
ENV_SHORT = {"SafetyPointGoal1-v0": "PointGoal1", "SafetyPointButton1-v0": "PointButton1"}

ARMS = ["cost_limit_10", "deployed_cl25", "cost_limit_70"]
ARM_LOG_DIR = {
    "cost_limit_10": os.path.join(LOG_ROOT_2, "cost_limit_10"),
    "deployed_cl25": os.path.join(LOG_ROOT_1, "deployed"),
    "cost_limit_70": os.path.join(LOG_ROOT_2, "cost_limit_70"),
}
ARM_COST_LIMIT = {"cost_limit_10": 10.0, "deployed_cl25": 25.0, "cost_limit_70": 70.0}
ARM_LABEL = {"cost_limit_10": "$c_{lim}=10$ (tight)", "deployed_cl25": "$c_{lim}=25$ (deployed)",
             "cost_limit_70": "$c_{lim}=70$ (relaxed)"}
ARM_COLOR = {"cost_limit_10": "#d62728", "deployed_cl25": "#1f77b4", "cost_limit_70": "#2ca02c"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.linewidth": 0.75, "axes.edgecolor": "#333333",
    "axes.grid": True, "grid.linewidth": 0.35, "grid.color": "#dddddd", "grid.alpha": 0.7,
    "legend.frameon": False, "legend.fontsize": 10, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_arm_env(arm, env):
    runs = []
    for f in glob.glob(os.path.join(ARM_LOG_DIR[arm], f"{env}__sac_lag_vosr__seed*.csv")):
        steps, rets, costs = [], [], []
        with open(f) as fh:
            for r in csv.DictReader(fh):
                try:
                    steps.append(int(r["step"])); rets.append(float(r["eval_return"])); costs.append(float(r["eval_cost"]))
                except (ValueError, KeyError):
                    continue
        if steps:
            runs.append({"steps": steps, "returns": rets, "costs": costs})
    return runs


def mean_curve(seed_runs, key, n_points=20):
    if not seed_runs:
        return [], [], []
    max_step = min(max(r["steps"]) for r in seed_runs)
    if max_step <= 0:
        return [], [], []
    grid = np.linspace(0, max_step, n_points)
    series = []
    for r in seed_runs:
        xs = np.array(r["steps"], dtype=float)
        ys = np.array(r[key], dtype=float)
        series.append(np.interp(grid, xs, ys))
    series = np.stack(series, axis=0)
    return grid.tolist(), series.mean(axis=0).tolist(), series.std(axis=0).tolist()


def plot_cell(ax, arm_runs, key, cost_line_value=None):
    any_series = False
    for arm in ARMS:
        runs = arm_runs.get(arm, [])
        gx, gy, gs = mean_curve(runs, key)
        if not gx:
            continue
        any_series = True
        gx, gy, gs = np.array(gx), np.array(gy), np.array(gs)
        ax.plot(gx, gy, color=ARM_COLOR[arm], linewidth=2.2, linestyle="-", label=ARM_LABEL[arm], zorder=3)
        ax.fill_between(gx, gy - gs, gy + gs, color=ARM_COLOR[arm], alpha=0.16, linewidth=0)
        if cost_line_value is not None:
            ax.axhline(ARM_COST_LIMIT[arm], color=ARM_COLOR[arm], linewidth=1.3, linestyle=":", zorder=2, alpha=0.8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if not any_series:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes,
                 fontsize=8, color="#999999")


def make_env_figure(env):
    arm_runs = {arm: load_arm_env(arm, env) for arm in ARMS}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    plot_cell(axes[0], arm_runs, "returns")
    plot_cell(axes[1], arm_runs, "costs", cost_line_value=True)
    axes[0].set_title(f"{ENV_SHORT[env]} — reward")
    axes[1].set_title(f"{ENV_SHORT[env]} — cost (dotted = that arm's own limit)")
    axes[0].set_ylabel("Evaluation return")
    axes[1].set_ylabel("Evaluation cost")
    axes[0].set_xlabel("Environment steps")
    axes[1].set_xlabel("Environment steps")

    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08), fontsize=10.5)
    fig.suptitle(f"Ablation 2: cost-limit sensitivity — {ENV_SHORT[env]}, SAC-Lagrangian + VOSR, "
                 f"mean $\\pm$ std, 2 seeds", fontsize=13, y=1.03)
    fig.tight_layout(rect=[0, 0.1, 1, 0.98])
    out_path = os.path.join(OUT_DIR, f"ablation2_costlimit_{env}.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for env in ENVS:
        make_env_figure(env)


if __name__ == "__main__":
    main()
