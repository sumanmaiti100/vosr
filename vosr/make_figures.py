"""Generates PNG figures (return + cost per environment, grouped by base
optimizer so each panel shows one optimizer's 5 samplers clearly) in
AAAI-paper style using matplotlib, saved to a figures/ folder.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from vosr.make_report import (load_curves, mean_curve, OPTIMIZERS, OPTIMIZER_LABEL,
                               SAMPLERS, METHOD_COLOR, METHOD_LABEL, IS_OURS)

OUT_DIR = os.environ.get("VOSR_FIGURES_DIR", "figures")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "axes.grid": True,
    "grid.linewidth": 0.4,
    "grid.color": "#dddddd",
    "grid.alpha": 0.7,
    "legend.frameon": False,
    "legend.fontsize": 7,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


COST_LIMIT = 25.0


def plot_optimizer_metric(ax, curves_env, opt, key, ylabel, cost_line=False):
    max_x_seen = 0.0
    any_series = False
    for samp in SAMPLERS:
        m = f"{opt}_{samp}"
        if m not in curves_env:
            continue
        gx, gy, gs = mean_curve(curves_env[m], key, n_points=50)
        if not gx:
            continue
        any_series = True
        gx, gy, gs = np.array(gx), np.array(gy), np.array(gs)
        max_x_seen = max(max_x_seen, gx.max())
        ours = IS_OURS[m]
        ax.plot(gx, gy, color=METHOD_COLOR[m], label=SAMPLER_LABEL_SHORT.get(samp, samp),
                linewidth=2.4 if ours else 1.5,
                linestyle="-" if ours else "--",
                alpha=1.0 if ours else 0.9, zorder=3 if ours else 2)
        ax.fill_between(gx, gy - gs, gy + gs, color=METHOD_COLOR[m], alpha=0.18, linewidth=0)
    if cost_line and any_series:
        ax.axhline(COST_LIMIT, color="#000000", linewidth=1.3, linestyle=":", zorder=4,
                    label=f"Cost limit ({COST_LIMIT:.0f})")
    ax.set_ylabel(ylabel, fontsize=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if not any_series:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes,
                 fontsize=8, color="#999999")


SAMPLER_LABEL_SHORT = {
    "uniform": "Uniform", "td_per": "TD-PER", "safety_per": "Safety-PER",
    "uncertainty_per": "Uncertainty-PER", "vosr": "VOSR (ours)",
}


def make_env_figure(env, curves_env, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.2))
    for j, opt in enumerate(OPTIMIZERS):
        plot_optimizer_metric(axes[0, j], curves_env, opt, "returns", "Evaluation return")
        plot_optimizer_metric(axes[1, j], curves_env, opt, "costs", "Evaluation cost", cost_line=True)
        axes[0, j].set_title(OPTIMIZER_LABEL[opt])
        axes[1, j].set_xlabel("Environment steps")

    seen, handles, labels = set(), [], []
    for ax in list(axes[0, :]) + list(axes[1, :]):
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in seen:
                seen.add(li)
                handles.append(hi)
                labels.append(li)
    fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(env, fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    curves = load_curves()
    envs = sorted(curves.keys())
    for env in envs:
        out_path = os.path.join(OUT_DIR, f"{env}.png")
        make_env_figure(env, curves[env], out_path)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
