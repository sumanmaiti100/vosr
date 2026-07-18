"""Plots sigma_c^2(q) vs training step for each environment, split into 3
panels (SAC-Lagrangian, CRPO, PCRPO) so the optimizer-specific behavior is
visible rather than averaged away, comparing VOSR against the 4 heuristic
samplers and the untempered Theorem-1 optimum. Log-scale y-axis (values span
many orders of magnitude). Separate figures_variance/ folder -- does not
touch any existing reward/cost plots or data.
"""
import csv
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import os as _os
OUT_DIR = _os.environ.get("VOSR_VARFIG_OUT", "figures_variance")
LOG_DIR = _os.environ.get("VOSR_VARFIG_LOG", "variance_diagnostic_logs")
OPTIMIZERS = ["sac_lag", "crpo", "pcrpo"]
OPTIMIZER_LABEL = {"sac_lag": "SAC-Lagrangian", "crpo": "CRPO", "pcrpo": "PCRPO"}

PALETTE = {
    "blue": "#1f77b4", "orange": "#ff7f0e", "green": "#2ca02c", "red": "#d62728",
    "purple": "#9467bd", "gray": "#7f7f7f",
}
SERIES = ["uniform", "td_per", "safety_per", "uncertainty_per", "vosr", "vosr_theoretical_optimum"]
COLOR = {
    "uniform": PALETTE["orange"], "td_per": PALETTE["green"], "safety_per": PALETTE["red"],
    "uncertainty_per": PALETTE["purple"], "vosr": PALETTE["blue"], "vosr_theoretical_optimum": PALETTE["gray"],
}
LABEL = {
    "uniform": "Uniform", "td_per": "TD-PER", "safety_per": "Safety-PER",
    "uncertainty_per": "Uncertainty-PER", "vosr": "VOSR (ours, deployed)",
    "vosr_theoretical_optimum": "VOSR (Theorem 1 exact optimum)",
}
STYLE = {
    "uniform": "-", "td_per": "-", "safety_per": "-", "uncertainty_per": "-",
    "vosr": "-", "vosr_theoretical_optimum": "-",
}
WIDTH = {
    "uniform": 1.6, "td_per": 1.6, "safety_per": 1.6, "uncertainty_per": 1.6,
    "vosr": 2.8, "vosr_theoretical_optimum": 1.8,
}

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.linewidth": 0.8, "axes.edgecolor": "#333333",
    "axes.grid": True, "grid.linewidth": 0.4, "grid.color": "#dddddd", "grid.alpha": 0.7,
    "legend.frameon": False, "legend.fontsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.titlesize": 10, "axes.titleweight": "bold", "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_env_optimizer_data(env, opt):
    steps_all = {}
    for f in glob.glob(os.path.join(LOG_DIR, f"{env}__{opt}__seed*.csv")):
        rows = list(csv.DictReader(open(f)))
        for r in rows:
            step = int(r["step"])
            steps_all.setdefault(step, {s: [] for s in SERIES})
            for s in SERIES:
                v = float(r[s])
                if np.isfinite(v):
                    steps_all[step][s].append(v)
    return steps_all


def plot_panel(ax, data):
    steps = sorted(data.keys())
    if not steps:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes,
                 fontsize=8, color="#999999")
        return
    for s in SERIES:
        means = [np.mean(data[st][s]) if data[st][s] else np.nan for st in steps]
        means = np.clip(means, 1e-6, None)
        ax.plot(steps, means, color=COLOR[s], linestyle=STYLE[s], linewidth=WIDTH[s],
                 marker="o", markersize=3.5 if s in ("vosr", "vosr_theoretical_optimum") else 2.5,
                 label=LABEL[s], zorder=3 if "vosr" in s else 2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def make_env_figure(env, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    any_data = False
    for j, opt in enumerate(OPTIMIZERS):
        data = load_env_optimizer_data(env, opt)
        if data:
            any_data = True
        plot_panel(axes[j], data)
        axes[j].set_title(OPTIMIZER_LABEL[opt])
        axes[j].set_xlabel("Environment steps")
        if j == 0:
            axes[j].set_ylabel(r"$\sigma_c^2(q)$  (log scale)")
    if not any_data:
        plt.close(fig)
        return False
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        for ax in axes[1:]:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                break
    fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f"{env} — cost-gradient estimator variance by sampler (3 seeds each)", fontsize=12, y=1.03)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_path)
    plt.close(fig)
    return True


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    envs = sorted(set(os.path.basename(f).split("__")[0] for f in glob.glob(os.path.join(LOG_DIR, "*.csv"))))
    for env in envs:
        out_path = os.path.join(OUT_DIR, f"{env}_variance.png")
        ok = make_env_figure(env, out_path)
        print(f"{'wrote' if ok else 'skip (no data)'} {out_path}")


if __name__ == "__main__":
    main()
