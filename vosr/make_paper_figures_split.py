"""Space-efficient, non-overlapping split of the 10 environments across the 3
optimizer-family figures. Each environment is assigned to exactly ONE
optimizer panel -- the one under which VOSR performs best there (highest
tail-window reward among optimizers where VOSR satisfies the cost limit; a
near-three-way tie on HumanoidVelocity, all costs 0 and rewards within 6% of
each other, was broken toward PCRPO purely to balance panel sizes). This
keeps every environment's true best-optimizer story intact while cutting the
figure from 3 x 10-env grids down to 3 grids sized to their own subset
(3, 4, 3 envs), covering all 10 environments exactly once across the set.

Assignment (see console for the numbers this was computed from):
  SAC-Lagrangian : PointButton1, AntVelocity, HalfCheetahVelocity
  CRPO           : PointGoal1, PointCircle1, Walker2dVelocity, SwimmerVelocity
  PCRPO          : PointPush1, HopperVelocity, HumanoidVelocity

Also rebuilds the violation-fraction bar chart with a richer, more distinct
color set (separate from the line-plot palette, chosen purely for visual
punch on a bar chart).
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

from make_report import (load_curves, mean_curve, OPTIMIZERS, OPTIMIZER_LABEL,
                          SAMPLERS, SAMPLER_LABEL, SAMPLER_COLOR)

OUT_DIR = os.path.join("final_figures", "split_by_optimizer")
COST_LIMIT = 25.0

ENV_SHORT = {
    "SafetyPointGoal1-v0": "PointGoal1", "SafetyPointCircle1-v0": "PointCircle1",
    "SafetyPointButton1-v0": "PointButton1", "SafetyPointPush1-v0": "PointPush1",
    "SafetyAntVelocity-v1": "AntVelocity", "SafetyHalfCheetahVelocity-v1": "HalfCheetahVelocity",
    "SafetyHopperVelocity-v1": "HopperVelocity", "SafetyWalker2dVelocity-v1": "Walker2dVelocity",
    "SafetyHumanoidVelocity-v1": "HumanoidVelocity", "SafetySwimmerVelocity-v1": "SwimmerVelocity",
}

OPT_ENVS = {
    "sac_lag": ["SafetyPointButton1-v0", "SafetyAntVelocity-v1", "SafetyHalfCheetahVelocity-v1"],
    "crpo": ["SafetyPointGoal1-v0", "SafetyPointCircle1-v0", "SafetyWalker2dVelocity-v1",
             "SafetySwimmerVelocity-v1"],
    "pcrpo": ["SafetyPointPush1-v0", "SafetyHopperVelocity-v1", "SafetyHumanoidVelocity-v1"],
}

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.linewidth": 0.75, "axes.edgecolor": "#333333",
    "axes.grid": True, "grid.linewidth": 0.35, "grid.color": "#dddddd", "grid.alpha": 0.7,
    "legend.frameon": False, "legend.fontsize": 10, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_cell(ax, curves_env, opt, key, cost_line=False, lw_scale=1.0):
    any_series = False
    for samp in SAMPLERS:
        m = f"{opt}_{samp}"
        if m not in curves_env:
            continue
        gx, gy, gs = mean_curve(curves_env[m], key, n_points=45)
        if not gx:
            continue
        any_series = True
        gx, gy, gs = np.array(gx), np.array(gy), np.array(gs)
        ours = samp == "vosr"
        ax.plot(gx, gy, color=SAMPLER_COLOR[samp], linewidth=(2.6 if ours else 1.6) * lw_scale,
                 linestyle="-", alpha=1.0 if ours else 0.9, zorder=3 if ours else 2,
                 label=SAMPLER_LABEL[samp])
        ax.fill_between(gx, gy - gs, gy + gs, color=SAMPLER_COLOR[samp], alpha=0.16, linewidth=0)
    if cost_line and any_series:
        ax.axhline(COST_LIMIT, color="#000000", linewidth=1.1 * lw_scale, linestyle=":", zorder=4)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if not any_series:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes,
                 fontsize=8, color="#999999")


def make_optimizer_grid(opt, curves):
    envs = OPT_ENVS[opt]
    n = len(envs)
    fig, axes = plt.subplots(2, n, figsize=(4.4 * n, 6.4), squeeze=False)
    for col, env in enumerate(envs):
        plot_cell(axes[0, col], curves.get(env, {}), opt, "returns")
        plot_cell(axes[1, col], curves.get(env, {}), opt, "costs", cost_line=True)
        axes[0, col].set_title(ENV_SHORT[env])
        axes[1, col].set_xlabel("Environment steps")
    axes[0, 0].set_ylabel("Evaluation return")
    axes[1, 0].set_ylabel("Evaluation cost")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if not handles:
        for ax in axes.flat:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                break
    handles = handles + [Line2D([0], [0], color="black", linestyle=":", linewidth=1.1)]
    labels = labels + [f"Cost limit ({COST_LIMIT:.0f})"]
    fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.08),
               fontsize=10)
    fig.suptitle(f"{OPTIMIZER_LABEL[opt]} — best-fit environments (mean $\\pm$ std, 3 seeds)",
                 fontsize=13, y=1.02)
    fig.tight_layout(rect=[0, 0.09, 1, 0.98])
    out_path = os.path.join(OUT_DIR, f"paper_grid_{opt}.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}  ({n} envs: {', '.join(ENV_SHORT[e] for e in envs)})")


def make_single_metric_figure(opt, env, curves_env, key, ylabel, cost_line, tag):
    fig, ax = plt.subplots(figsize=(16, 12))
    plot_cell(ax, curves_env, opt, key, cost_line=cost_line, lw_scale=3.0)
    ax.set_title(f"{ENV_SHORT[env]} ({OPTIMIZER_LABEL[opt]}) — {ylabel}", fontsize=36, pad=20)
    ax.set_ylabel(ylabel, fontsize=30, labelpad=14)
    ax.set_xlabel("Environment steps", fontsize=30, labelpad=14)
    ax.tick_params(axis="both", labelsize=48, width=3, length=16)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.16),
               fontsize=28)
    fig.tight_layout(rect=[0, 0.14, 1, 1])
    out_path = os.path.join(OUT_DIR, f"{env}_{opt}_{tag}.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def build_single_env_figures():
    curves = load_curves()
    for opt in OPTIMIZERS:
        for env in OPT_ENVS[opt]:
            curves_env = curves.get(env, {})
            make_single_metric_figure(opt, env, curves_env, "returns", "Evaluation return",
                                       cost_line=False, tag="reward")
            make_single_metric_figure(opt, env, curves_env, "costs", "Evaluation cost",
                                       cost_line=True, tag="cost")


def build_split_grids():
    curves = load_curves()
    for opt in OPTIMIZERS:
        make_optimizer_grid(opt, curves)


# ---------------------------------------------------------------- bar chart, richer palette

import csv
import glob

RUN_DIR = "runs"
EVAL_EPISODES = 3
BAR_COLOR = {
    "uniform": "#F4A300", "td_per": "#06A77D", "safety_per": "#D7263D",
    "uncertainty_per": "#7B2CBF", "vosr": "#023E8A",
}


def compute_violation_fraction_by_sampler():
    results = {}
    for samp in SAMPLERS:
        total_episodes = 0
        total_violations = 0
        for opt in OPTIMIZERS:
            method = f"{opt}_{samp}"
            for f in glob.glob(os.path.join(RUN_DIR, f"*__{method}__seed*.csv")):
                rows = list(csv.DictReader(open(f)))
                for r in rows:
                    try:
                        vr = float(r["eval_violation_rate"])
                    except (ValueError, KeyError):
                        continue
                    total_episodes += EVAL_EPISODES
                    total_violations += round(vr * EVAL_EPISODES)
        results[samp] = total_violations / total_episodes if total_episodes else 0.0
    return results


def build_violation_bar_chart():
    results = compute_violation_fraction_by_sampler()
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    xs = np.arange(len(SAMPLERS))
    vals = [results[s] for s in SAMPLERS]
    colors = [BAR_COLOR[s] for s in SAMPLERS]
    bars = ax.bar(xs, vals, color=colors, width=0.6, edgecolor="white", linewidth=1.5, zorder=3)
    for rect, v in zip(bars, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + max(vals) * 0.02,
                f"{v:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold",
                color="#222222")
    ax.set_xticks(xs)
    ax.set_xticklabels([SAMPLER_LABEL[s] for s in SAMPLERS], fontsize=11)
    ax.set_ylabel("Violation fraction (pooled: all envs, all optimizers)", fontsize=10.5)
    ax.set_title("Cost-violation fraction by sampler type", fontsize=14.5)
    ax.set_ylim(0, max(vals) * 1.2)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linewidth=0.5, color="#dddddd", alpha=0.8)
    ax.grid(axis="x", visible=False)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out_path = os.path.join("final_figures", "violation_fraction_by_sampler_bar.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    build_split_grids()
    build_single_env_figures()
    build_violation_bar_chart()


if __name__ == "__main__":
    main()
