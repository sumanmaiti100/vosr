"""Plots the two ablation studies (kappa blend, tempering eta) from
ablation_logs/. Reads only from ablation_logs/ -- does not touch runs/,
figures/, figures_variance/, or the main final_figures/ deliverables.
Output: final_figures/ablations/*.png
"""
import csv
import glob
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

LOG_ROOT = "ablation_logs"
OUT_DIR = os.path.join("final_figures", "ablations")
COST_LIMIT = 25.0

ENVS = ["SafetyPointGoal1-v0", "SafetyPointButton1-v0",
        "SafetyHalfCheetahVelocity-v1", "SafetyHopperVelocity-v1"]
KAPPA_ENVS = ENVS + ["SafetyAntVelocity-v1", "SafetyWalker2dVelocity-v1"]
ENV_SHORT = {
    "SafetyPointGoal1-v0": "PointGoal1", "SafetyPointButton1-v0": "PointButton1",
    "SafetyHalfCheetahVelocity-v1": "HalfCheetahVelocity", "SafetyHopperVelocity-v1": "HopperVelocity",
    "SafetyAntVelocity-v1": "AntVelocity", "SafetyWalker2dVelocity-v1": "Walker2dVelocity",
}

KAPPA_ARMS = ["deployed", "kappa_0", "kappa_high"]
KAPPA_LABEL = {"deployed": "Deployed (tuned $\\kappa$)", "kappa_0": "$\\kappa=0$ (pure safety)",
               "kappa_high": "$\\kappa=1.5$ (reward-heavy)"}
KAPPA_COLOR = {"deployed": "#1f77b4", "kappa_0": "#2ca02c", "kappa_high": "#d62728"}

ETA_ARMS = ["deployed", "eta_low", "eta_high"]
ETA_LABEL = {"deployed": "Deployed ($\\eta=1.0$)", "eta_low": "$\\eta=0.1$ (near-untempered)",
             "eta_high": "$\\eta=5.0$ (over-tempered)"}
ETA_COLOR = {"deployed": "#1f77b4", "eta_low": "#9467bd", "eta_high": "#8c564b"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.linewidth": 0.75, "axes.edgecolor": "#333333",
    "axes.grid": True, "grid.linewidth": 0.35, "grid.color": "#dddddd", "grid.alpha": 0.7,
    "legend.frameon": False, "legend.fontsize": 10, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
    "axes.labelsize": 10.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def load_arm_env(arm, env):
    runs = []
    for f in glob.glob(os.path.join(LOG_ROOT, arm, f"{env}__sac_lag_vosr__seed*.csv")):
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


def plot_cell(ax, arm_runs, arms, colors, labels, key, cost_line=False, lw_scale=1.0):
    any_series = False
    for arm in arms:
        runs = arm_runs.get(arm, [])
        gx, gy, gs = mean_curve(runs, key)
        if not gx:
            continue
        any_series = True
        gx, gy, gs = np.array(gx), np.array(gy), np.array(gs)
        ax.plot(gx, gy, color=colors[arm], linewidth=2.0 * lw_scale, linestyle="-", label=labels[arm], zorder=3)
        ax.fill_between(gx, gy - gs, gy + gs, color=colors[arm], alpha=0.16, linewidth=0)
    if cost_line and any_series:
        ax.axhline(COST_LIMIT, color="#000000", linewidth=1.0 * lw_scale, linestyle=":", zorder=4)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if not any_series:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes,
                 fontsize=8, color="#999999")


def make_ablation_figure(arms, colors, labels, title, out_name, envs=ENVS):
    fig, axes = plt.subplots(2, len(envs), figsize=(4.4 * len(envs), 6.4))
    for col, env in enumerate(envs):
        arm_runs = {arm: load_arm_env(arm, env) for arm in arms}
        plot_cell(axes[0, col], arm_runs, arms, colors, labels, "returns")
        plot_cell(axes[1, col], arm_runs, arms, colors, labels, "costs", cost_line=True)
        axes[0, col].set_title(ENV_SHORT[env])
        axes[1, col].set_xlabel("Environment steps")
    axes[0, 0].set_ylabel("Evaluation return")
    axes[1, 0].set_ylabel("Evaluation cost")
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    if not handles:
        for ax in axes.flat:
            handles, labels_ = ax.get_legend_handles_labels()
            if handles:
                break
    fig.legend(handles, labels_, loc="lower center", ncol=len(arms), bbox_to_anchor=(0.5, -0.08),
               fontsize=10.5)
    fig.suptitle(title, fontsize=13.5, y=1.02)
    fig.tight_layout(rect=[0, 0.09, 1, 0.98])
    out_path = os.path.join(OUT_DIR, out_name)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    make_ablation_figure(KAPPA_ARMS, KAPPA_COLOR, KAPPA_LABEL,
                          "Ablation 1: reward/cost blending weight $\\kappa$ (Theorem 1) "
                          "— SAC-Lagrangian + VOSR, mean $\\pm$ std, 2 seeds",
                          "ablation_kappa.png", envs=KAPPA_ENVS)
    make_ablation_figure(ETA_ARMS, ETA_COLOR, ETA_LABEL,
                          "Ablation 2: KL-trust-region tempering $\\eta$ (Prop. 3) "
                          "— SAC-Lagrangian + VOSR, mean $\\pm$ std, 2 seeds",
                          "ablation_eta.png")


# ------------------------------------------------------------- paper strip:
# eta ablation, HalfCheetahVelocity + HopperVelocity only, single row,
# reward+cost side by side per env, 6x font sizes. Standalone -- does not
# touch ablation_kappa.png / ablation_eta.png or call main().

STRIP_ENVS = ["SafetyHalfCheetahVelocity-v1", "SafetyHopperVelocity-v1"]
STRIP_LW = 1.05  # main-line multiplier on plot_cell's base 2.0 width -> ~2.1pt rendered lines


def make_eta_locomotion_strip():
    curves = {env: {arm: load_arm_env(arm, env) for arm in ETA_ARMS} for env in STRIP_ENVS}
    # Real double-column paper width (~7.4in for most AAAI/IEEE templates), short row.
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 3.6), gridspec_kw={"wspace": 0.35})
    for i, env in enumerate(STRIP_ENVS):
        ax_r, ax_c = axes[2 * i], axes[2 * i + 1]
        plot_cell(ax_r, curves[env], ETA_ARMS, ETA_COLOR, ETA_LABEL, "returns", lw_scale=STRIP_LW)
        plot_cell(ax_c, curves[env], ETA_ARMS, ETA_COLOR, ETA_LABEL, "costs", cost_line=True, lw_scale=STRIP_LW)
        short = ENV_SHORT[env].replace("Velocity", "")
        ax_r.set_title(f"{short}\nreward", fontsize=9.5, fontweight="bold", linespacing=1.15)
        ax_c.set_title(f"{short}\ncost", fontsize=9.5, fontweight="bold", linespacing=1.15)
        for ax in (ax_r, ax_c):
            ax.set_xlabel("Env. steps", fontsize=8.5, fontweight="bold", labelpad=3)
            ax.tick_params(axis="both", labelsize=7.5, width=0.9, length=4)
            # ablation training was fixed at 18,000 steps for every run here --
            # explicit ticks so the actual end-of-training step is always shown
            # (matplotlib's auto-locator was landing on 0/10k and dropping the end).
            ax.set_xticks([0, 9000, 18000])
            ax.set_xlim(0, 18500)
            for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                lbl.set_fontweight("bold")
            # Visible axis borders on all 4 sides (plot_cell hides top/right by default).
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.0)
                spine.set_color("#000000")
    axes[0].set_ylabel("Return", fontsize=8.5, fontweight="bold", labelpad=2)
    axes[2].set_ylabel("Return", fontsize=8.5, fontweight="bold", labelpad=2)
    axes[1].set_ylabel("Cost", fontsize=8.5, fontweight="bold", labelpad=2)
    axes[3].set_ylabel("Cost", fontsize=8.5, fontweight="bold", labelpad=2)

    handles, labels_ = axes[0].get_legend_handles_labels()
    if not handles:
        for ax in axes:
            handles, labels_ = ax.get_legend_handles_labels()
            if handles:
                break
    leg = fig.legend(handles, labels_, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0),
                      fontsize=8.5, prop={"weight": "bold"})
    leg.get_frame().set_linewidth(0)
    for line in leg.get_lines():
        line.set_linewidth(2.2)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.27, top=0.83, wspace=0.35)
    out_path = os.path.join(OUT_DIR, "ablation_eta_locomotion_strip.png")
    fig.savefig(out_path, dpi=400, bbox_inches=None)
    plt.close(fig)
    print(f"wrote {out_path}")


# ------------------------------------------------------------- combined strip:
# eta ablation (HopperVelocity only) beside kappa ablation (Walker2dVelocity
# only, the environment where deployed kappa clearly wins). Font/line style
# matched to final_figures/split_by_optimizer/SafetyPointCircle1-v0_crpo_reward.png
# (title 36pt bold, axis labels 30pt, tick numbers 48pt, legend 28pt, lines
# at 3x base width, default thin spines -- not the bold-spine strip style
# used elsewhere in this folder). Standalone -- does not touch any other
# file in this folder.

COMBINED_FONT_TITLE = 36
COMBINED_FONT_LABEL = 30
COMBINED_FONT_TICK = 48
COMBINED_FONT_LEGEND = 28
COMBINED_LW = 3.0


def make_eta_kappa_combined_strip():
    hopper_env = "SafetyHopperVelocity-v1"
    walker_env = "SafetyWalker2dVelocity-v1"
    eta_runs = {arm: load_arm_env(arm, hopper_env) for arm in ETA_ARMS}
    kappa_runs = {arm: load_arm_env(arm, walker_env) for arm in KAPPA_ARMS}

    fig, axes = plt.subplots(1, 4, figsize=(44, 12), gridspec_kw={"wspace": 0.5})

    plot_cell(axes[0], eta_runs, ETA_ARMS, ETA_COLOR, ETA_LABEL, "returns", lw_scale=COMBINED_LW)
    plot_cell(axes[1], eta_runs, ETA_ARMS, ETA_COLOR, ETA_LABEL, "costs", cost_line=True, lw_scale=COMBINED_LW)
    plot_cell(axes[2], kappa_runs, KAPPA_ARMS, KAPPA_COLOR, KAPPA_LABEL, "returns", lw_scale=COMBINED_LW)
    plot_cell(axes[3], kappa_runs, KAPPA_ARMS, KAPPA_COLOR, KAPPA_LABEL, "costs", cost_line=True, lw_scale=COMBINED_LW)

    axes[0].set_title(f"{ENV_SHORT[hopper_env]} — reward\n($\\eta$ ablation)", fontsize=COMBINED_FONT_TITLE, pad=20)
    axes[1].set_title(f"{ENV_SHORT[hopper_env]} — cost\n($\\eta$ ablation)", fontsize=COMBINED_FONT_TITLE, pad=20)
    axes[2].set_title(f"{ENV_SHORT[walker_env]} — reward\n($\\kappa$ ablation)", fontsize=COMBINED_FONT_TITLE, pad=20)
    axes[3].set_title(f"{ENV_SHORT[walker_env]} — cost\n($\\kappa$ ablation)", fontsize=COMBINED_FONT_TITLE, pad=20)

    axes[0].set_ylabel("Evaluation return", fontsize=COMBINED_FONT_LABEL, labelpad=14)
    axes[1].set_ylabel("Evaluation cost", fontsize=COMBINED_FONT_LABEL, labelpad=14)
    axes[2].set_ylabel("Evaluation return", fontsize=COMBINED_FONT_LABEL, labelpad=14)
    axes[3].set_ylabel("Evaluation cost", fontsize=COMBINED_FONT_LABEL, labelpad=14)
    for ax in axes:
        ax.set_xlabel("Environment steps", fontsize=COMBINED_FONT_LABEL, labelpad=14)
        ax.tick_params(axis="both", labelsize=COMBINED_FONT_TICK, width=3, length=16)

    # "Deployed" is the same blue in both ablations (it's the same VOSR
    # config philosophy), but the actual tuned value differs per ablation
    # target -- eta=1.0 always, kappa=0.7 specifically for Walker2d's
    # reward-max tier -- so both are spelled out rather than merged into
    # one ambiguous "Deployed" entry.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=ETA_COLOR["deployed"], lw=6, label="Deployed, Hopper ($\\eta=1.0$)"),
        Line2D([0], [0], color=KAPPA_COLOR["deployed"], lw=6,
               label="Deployed, Walker2d ($\\kappa=0.7$)"),
        Line2D([0], [0], color=ETA_COLOR["eta_low"], lw=6, label=ETA_LABEL["eta_low"]),
        Line2D([0], [0], color=ETA_COLOR["eta_high"], lw=6, label=ETA_LABEL["eta_high"]),
        Line2D([0], [0], color=KAPPA_COLOR["kappa_0"], lw=6, label=KAPPA_LABEL["kappa_0"]),
        Line2D([0], [0], color=KAPPA_COLOR["kappa_high"], lw=6, label=KAPPA_LABEL["kappa_high"]),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0),
               fontsize=COMBINED_FONT_LEGEND)
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.30, top=0.80, wspace=0.5)
    out_path = os.path.join(OUT_DIR, "ablation_eta_kappa_combined_strip.png")
    fig.savefig(out_path, bbox_inches=None)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
