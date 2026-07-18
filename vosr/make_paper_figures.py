"""Paper-ready, space-efficient figures for the manuscript.

Space problem this solves: putting 10 separate per-environment figures (each
with reward+cost across 3 optimizer panels) in a paper is impossible under a
page budget. Instead, following the P2BPO reference figure's layout (Fig. 3:
one shared, dense grid, reward stacked over cost per environment, one shared
legend), this produces ONE compact grid PER OPTIMIZER FAMILY -- so the whole
10-environment sweep for SAC-Lagrangian, CRPO, and PCRPO each fits in a single
page-width figure. 3 figures total cover everything, preserving the
per-family separation while being paper-space-realistic.

Layout per figure (one optimizer): 4 rows x 5 cols.
  row 0: reward, envs 1-5      row 1: cost, envs 1-5
  row 2: reward, envs 6-10     row 3: cost, envs 6-10
Each subplot shows the 5 samplers as solid lines (P2BPO-style tab10 colors)
with mean +/- std shaded bands, and a black dotted cost-limit reference line
on cost rows.

Also (re)builds the violation-fraction bar chart with the same palette.
"""
import csv
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from make_report import (load_curves, mean_curve, OPTIMIZERS, OPTIMIZER_LABEL,
                          SAMPLERS, SAMPLER_LABEL, SAMPLER_COLOR)

OUT_DIR = "final_figures"
RUN_DIR = "runs"
COST_LIMIT = 25.0

ENV_ORDER = ["SafetyPointGoal1-v0", "SafetyPointCircle1-v0", "SafetyPointButton1-v0",
             "SafetyPointPush1-v0", "SafetyAntVelocity-v1", "SafetyHalfCheetahVelocity-v1",
             "SafetyHopperVelocity-v1", "SafetyWalker2dVelocity-v1", "SafetyHumanoidVelocity-v1",
             "SafetySwimmerVelocity-v1"]
ENV_SHORT = {
    "SafetyPointGoal1-v0": "PointGoal1", "SafetyPointCircle1-v0": "PointCircle1",
    "SafetyPointButton1-v0": "PointButton1", "SafetyPointPush1-v0": "PointPush1",
    "SafetyAntVelocity-v1": "AntVelocity", "SafetyHalfCheetahVelocity-v1": "HalfCheetahVelocity",
    "SafetyHopperVelocity-v1": "HopperVelocity", "SafetyWalker2dVelocity-v1": "Walker2dVelocity",
    "SafetyHumanoidVelocity-v1": "HumanoidVelocity", "SafetySwimmerVelocity-v1": "SwimmerVelocity",
}
ENV_ROW1, ENV_ROW2 = ENV_ORDER[:5], ENV_ORDER[5:]

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.linewidth": 0.7, "axes.edgecolor": "#333333",
    "axes.grid": True, "grid.linewidth": 0.35, "grid.color": "#dddddd", "grid.alpha": 0.7,
    "legend.frameon": False, "legend.fontsize": 9, "xtick.labelsize": 6.3, "ytick.labelsize": 6.3,
    "axes.titlesize": 8, "axes.titleweight": "bold", "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_cell(ax, curves_env, opt, key, cost_line=False):
    any_series = False
    for samp in SAMPLERS:
        m = f"{opt}_{samp}"
        if m not in curves_env:
            continue
        gx, gy, gs = mean_curve(curves_env[m], key, n_points=40)
        if not gx:
            continue
        any_series = True
        gx, gy, gs = np.array(gx), np.array(gy), np.array(gs)
        ours = samp == "vosr"
        ax.plot(gx, gy, color=SAMPLER_COLOR[samp], linewidth=1.9 if ours else 1.1,
                 linestyle="-", alpha=1.0 if ours else 0.9, zorder=3 if ours else 2,
                 label=SAMPLER_LABEL[samp])
        ax.fill_between(gx, gy - gs, gy + gs, color=SAMPLER_COLOR[samp], alpha=0.15, linewidth=0)
    if cost_line and any_series:
        ax.axhline(COST_LIMIT, color="#000000", linewidth=1.0, linestyle=":", zorder=4)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if not any_series:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes,
                 fontsize=7, color="#999999")


def make_optimizer_grid(opt, curves):
    fig, axes = plt.subplots(4, 5, figsize=(16, 9.5))
    for col, env in enumerate(ENV_ROW1):
        plot_cell(axes[0, col], curves.get(env, {}), opt, "returns")
        plot_cell(axes[1, col], curves.get(env, {}), opt, "costs", cost_line=True)
        axes[0, col].set_title(ENV_SHORT[env])
    for col, env in enumerate(ENV_ROW2):
        plot_cell(axes[2, col], curves.get(env, {}), opt, "returns")
        plot_cell(axes[3, col], curves.get(env, {}), opt, "costs", cost_line=True)
        axes[2, col].set_title(ENV_SHORT[env])
    axes[0, 0].set_ylabel("Reward")
    axes[1, 0].set_ylabel("Cost")
    axes[2, 0].set_ylabel("Reward")
    axes[3, 0].set_ylabel("Cost")
    for row in range(4):
        for col in range(5):
            if row in (1, 3):
                axes[row, col].set_xlabel("Env. steps", fontsize=6.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if not handles:
        for ax in axes.flat:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                break
    from matplotlib.lines import Line2D
    handles = handles + [Line2D([0], [0], color="black", linestyle=":", linewidth=1.0)]
    labels = labels + [f"Cost limit ({COST_LIMIT:.0f})"]
    fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.015),
               fontsize=9.5)
    fig.suptitle(f"{OPTIMIZER_LABEL[opt]} — reward & cost across all 10 environments "
                 f"(mean $\\pm$ std over 3 seeds)", fontsize=12.5, y=1.005)
    fig.tight_layout(rect=[0, 0.035, 1, 0.98])
    out_path = os.path.join(OUT_DIR, f"paper_grid_{opt}.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def build_paper_grids():
    curves = load_curves()
    for opt in OPTIMIZERS:
        make_optimizer_grid(opt, curves)


# ---------------------------------------------------------------- violation-fraction bar chart

EVAL_EPISODES = 3


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
    fig, ax = plt.subplots(figsize=(7.2, 5.3))
    xs = np.arange(len(SAMPLERS))
    vals = [results[s] for s in SAMPLERS]
    colors = [SAMPLER_COLOR[s] for s in SAMPLERS]
    bars = ax.bar(xs, vals, color=colors, width=0.6, edgecolor="#222222", linewidth=1.0, zorder=2)
    for rect, v in zip(bars, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + max(vals) * 0.015,
                f"{v:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([SAMPLER_LABEL[s] for s in SAMPLERS], fontsize=10.5)
    ax.set_ylabel("Violation fraction (pooled: all envs, all optimizers)")
    ax.set_title("Cost-violation fraction by sampler type", fontsize=13.5)
    ax.set_ylim(0, max(vals) * 1.18)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "violation_fraction_by_sampler_bar.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    build_paper_grids()
    build_violation_bar_chart()
    # remove the earlier, unreadable 15-lines-per-axes attempt
    for stale in ("reward_cost_group1.png", "reward_cost_group2.png"):
        p = os.path.join(OUT_DIR, stale)
        if os.path.exists(p):
            os.remove(p)
            print(f"removed stale {p}")


if __name__ == "__main__":
    main()
