"""Builds a self-contained HTML report from results/*.json (produced by
aggregate.py) and the raw per-run CSVs (for learning curves). Charts are
plain inline SVG polylines -- no external JS/CSS dependencies."""
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

RUN_DIR = os.environ.get("VOSR_RUN_DIR", "runs")
RESULTS_DIR = os.environ.get("VOSR_RESULTS_DIR", "results")
OUT = os.path.join(RESULTS_DIR, "report.html")

PALETTE = {
    "blue": "#1f77b4", "orange": "#ff7f0e", "green": "#2ca02c", "red": "#d62728",
    "purple": "#9467bd", "brown": "#8c564b", "pink": "#e377c2", "gray": "#7f7f7f",
}
OPTIMIZERS = ["sac_lag", "crpo", "pcrpo"]
OPTIMIZER_LABEL = {"sac_lag": "SAC-Lagrangian", "crpo": "CRPO", "pcrpo": "PCRPO"}
SAMPLERS = ["uniform", "td_per", "safety_per", "uncertainty_per", "vosr"]
SAMPLER_LABEL = {
    "uniform": "Uniform", "td_per": "TD-PER", "safety_per": "Safety-PER",
    "uncertainty_per": "Uncertainty-PER", "vosr": "VOSR (ours)",
}
# Color encodes the sampler (the axis of comparison within each optimizer panel);
# fixed order, never cycled, matplotlib tab10-derived (clean, print-safe, paper-style).
SAMPLER_COLOR = {
    "uniform": PALETTE["orange"], "td_per": PALETTE["green"], "safety_per": PALETTE["red"],
    "uncertainty_per": PALETTE["purple"], "vosr": PALETTE["blue"],
}

METHOD_ORDER = [f"{opt}_{samp}" for opt in OPTIMIZERS for samp in SAMPLERS]
METHOD_COLOR = {f"{opt}_{samp}": SAMPLER_COLOR[samp] for opt in OPTIMIZERS for samp in SAMPLERS}
METHOD_LABEL = {f"{opt}_{samp}": f"{OPTIMIZER_LABEL[opt]} + {SAMPLER_LABEL[samp]}" if samp != "uniform"
                 else f"{OPTIMIZER_LABEL[opt]} (uniform)" for opt in OPTIMIZERS for samp in SAMPLERS}
IS_OURS = {m: m.endswith("_vosr") for m in METHOD_ORDER}


def load_curves():
    curves = defaultdict(lambda: defaultdict(list))  # curves[env][method] = list of per-seed (steps, returns, costs)
    for path in glob.glob(os.path.join(RUN_DIR, "*.csv")):
        base = os.path.basename(path)[:-4]
        parts = base.split("__")
        if len(parts) != 3:
            continue
        env, method, seed_s = parts
        steps, rets, costs, viol = [], [], [], []
        with open(path) as f:
            for r in csv.DictReader(f):
                try:
                    steps.append(int(r["step"])); rets.append(float(r["eval_return"]))
                    costs.append(float(r["eval_cost"])); viol.append(float(r["eval_violation_rate"]))
                except (ValueError, KeyError):
                    continue
        if steps:
            curves[env][method].append({"steps": steps, "returns": rets, "costs": costs, "viol": viol})
    return curves


def mean_curve(seed_runs, key, n_points=40):
    """Interpolate each seed run onto a common step grid (0..min max step) and
    return (steps, mean, std) across seeds."""
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
    mean = series.mean(axis=0)
    std = series.std(axis=0)
    return grid.tolist(), mean.tolist(), std.tolist()


def svg_line_chart(series_list, width=760, height=340, pad_l=54, pad_b=34, pad_t=14, pad_r=14, title="", y_label="", x_label="environment steps"):
    """series_list: list of dicts {label, color, x, y, std (optional), ours (bool)}.
    'ours' series render as thicker solid lines; baselines as dashed lines --
    the standard proposed-vs-baseline convention, so identity survives even
    without reading the legend colors."""
    all_x = [v for s in series_list for v in s["x"]]
    all_y_lo = [y - (s.get("std", [0] * len(s["y"]))[i]) for s in series_list for i, y in enumerate(s["y"])]
    all_y_hi = [y + (s.get("std", [0] * len(s["y"]))[i]) for s in series_list for i, y in enumerate(s["y"])]
    if not all_x:
        return f'<div class="chart-empty">{title}: no data yet</div>'
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y_lo), max(all_y_hi)
    if x_max - x_min < 1e-9:
        x_max = x_min + 1
    if y_max - y_min < 1e-9:
        y_max += 1e-6
        y_min -= 1e-6
    y_pad = (y_max - y_min) * 0.08
    y_min -= y_pad
    y_max += y_pad

    def sx(v):
        return pad_l + (v - x_min) / (x_max - x_min) * (width - pad_l - pad_r)

    def sy(v):
        return height - pad_b + (v - y_min) / (y_max - y_min) * -(height - pad_b - pad_t)

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img" aria-label="{title}">']
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="var(--surface-1)"/>')
    n_grid = 5
    for i in range(n_grid):
        gy = pad_t + i * (height - pad_b - pad_t) / (n_grid - 1)
        yv = y_max - i * (y_max - y_min) / (n_grid - 1)
        parts.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{width-pad_r}" y2="{gy:.1f}" stroke="var(--grid)" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l-6}" y="{gy+3:.1f}" text-anchor="end" class="chart-tick">{yv:.3g}</text>')
    parts.append(f'<line x1="{pad_l}" y1="{height-pad_b}" x2="{width-pad_r}" y2="{height-pad_b}" stroke="var(--axis)" stroke-width="1"/>')
    parts.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height-pad_b}" stroke="var(--axis)" stroke-width="1"/>')
    # x ticks (min / max step)
    parts.append(f'<text x="{sx(x_min):.1f}" y="{height-8}" text-anchor="start" class="chart-tick">{int(x_min):,}</text>')
    parts.append(f'<text x="{sx(x_max):.1f}" y="{height-8}" text-anchor="end" class="chart-tick">{int(x_max):,}</text>')

    for s in series_list:
        std = s.get("std")
        if std:
            top = " ".join(f"{sx(x):.1f},{sy(y+e):.1f}" for x, y, e in zip(s["x"], s["y"], std))
            bot = " ".join(f"{sx(x):.1f},{sy(y-e):.1f}" for x, y, e in zip(reversed(s["x"]), reversed(s["y"]), reversed(std)))
            parts.append(f'<polygon points="{top} {bot}" fill="{s["color"]}" opacity="0.10" stroke="none"/>')
    for s in series_list:
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(s["x"], s["y"]))
        ours = s.get("ours", False)
        sw = 2.6 if ours else 1.6
        dash = "" if ours else ' stroke-dasharray="5,3"'
        opacity = 1.0 if ours else 0.85
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{s["color"]}" stroke-width="{sw}" opacity="{opacity}" stroke-linecap="round" stroke-linejoin="round"{dash}/>')
    parts.append(f'<text x="{pad_l}" y="{pad_t-2}" class="chart-title">{title}</text>')
    parts.append(f'<text x="{(pad_l+width-pad_r)/2:.0f}" y="{height-2}" text-anchor="middle" class="chart-axis">{x_label}</text>')
    parts.append(f'<text x="12" y="{(pad_t+height-pad_b)/2:.0f}" text-anchor="middle" class="chart-axis" transform="rotate(-90 12 {(pad_t+height-pad_b)/2:.0f})">{y_label}</text>')
    parts.append("</svg>")
    return "".join(parts)


def legend_html(methods):
    items = []
    for m in methods:
        ours = IS_OURS.get(m, False)
        dash_attr = "" if ours else 'stroke-dasharray="5,3"'
        items.append(
            f'<span class="legend-item"><svg width="20" height="10"><line x1="0" y1="5" x2="20" y2="5" '
            f'stroke="{METHOD_COLOR[m]}" stroke-width="{2.6 if ours else 1.6}" '
            f'{dash_attr}/></svg>{METHOD_LABEL[m]}</span>'
        )
    return '<div class="legend">' + "".join(items) + '</div>'


def build_html():
    summary_path = os.path.join(RESULTS_DIR, "summary_by_method.json")
    pairs_path = os.path.join(RESULTS_DIR, "pairs.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    pairs = json.load(open(pairs_path)) if os.path.exists(pairs_path) else {}
    curves = load_curves()
    envs = sorted(curves.keys())

    rows_html = []
    for m in METHOD_ORDER:
        if m not in summary:
            continue
        s = summary[m]
        is_ours = "vosr" in m
        cls = ' class="ours"' if is_ours else ""
        rows_html.append(
            f"<tr{cls}><td>{METHOD_LABEL[m]}</td><td>{s['n_runs']}</td>"
            f"<td>{s['return_norm_iqm']:.3f} <span class='ci'>[{s['return_norm_ci'][0]:.3f}, {s['return_norm_ci'][1]:.3f}]</span></td>"
            f"<td>{s['cost_norm_iqm']:.3f} <span class='ci'>[{s['cost_norm_ci'][0]:.3f}, {s['cost_norm_ci'][1]:.3f}]</span></td>"
            f"<td>{s['violation_iqm']:.3f}</td>"
            f"<td>{'%.3g' % s['sigma_c2_median'] if s['sigma_c2_median'] is not None else '—'}</td></tr>"
        )
    table_html = "".join(rows_html)

    pair_rows = []
    for k, v in pairs.items():
        good = (v["return_delta"] >= -0.02) and (v["cost_delta"] <= 0.02) and (v["violation_delta"] <= 0.02)
        status = '<span class="status-good">better/equal</span>' if good else '<span class="status-warn">mixed</span>'
        pair_rows.append(
            f"<tr><td>{k.replace('_vs_', ' → ')}</td>"
            f"<td class='{'delta-good' if v['return_delta']>=0 else 'delta-bad'}'>{v['return_delta']:+.3f}</td>"
            f"<td class='{'delta-good' if v['cost_delta']<=0 else 'delta-bad'}'>{v['cost_delta']:+.3f}</td>"
            f"<td class='{'delta-good' if v['violation_delta']<=0 else 'delta-bad'}'>{v['violation_delta']:+.3f}</td>"
            f"<td>{status}</td></tr>"
        )
    pairs_html = "".join(pair_rows)

    env_sections = []
    for env in envs:
        present = [m for m in METHOD_ORDER if m in curves[env]]
        ret_series, cost_series = [], []
        for m in present:
            gx, gy, gs = mean_curve(curves[env][m], "returns")
            if gx:
                ret_series.append({"label": METHOD_LABEL[m], "color": METHOD_COLOR[m], "x": gx, "y": gy, "std": gs, "ours": IS_OURS[m]})
            gx2, gy2, gs2 = mean_curve(curves[env][m], "costs")
            if gx2:
                cost_series.append({"label": METHOD_LABEL[m], "color": METHOD_COLOR[m], "x": gx2, "y": gy2, "std": gs2, "ours": IS_OURS[m]})
        ret_chart = svg_line_chart(ret_series, title="Evaluation return", y_label="eval return")
        cost_chart = svg_line_chart(cost_series, title="Evaluation cost", y_label="eval cost")
        env_sections.append(
            f'<section class="env-section"><h3>{env}</h3>'
            f'<div class="chart-row"><div class="chart-col">{ret_chart}</div><div class="chart-col">{cost_chart}</div></div>'
            f'{legend_html(present)}</section>'
        )

    html = f"""<style>
.viz-root {{
  --surface-1: #fcfcfb; --page: #f9f9f7; --text-primary: #0b0b0b; --text-secondary: #52514e;
  --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7; --good: #006300; --bad: #d03b3b;
}}
@media (prefers-color-scheme: dark) {{
  .viz-root {{ --surface-1: #1a1a19; --page: #0d0d0d; --text-primary: #ffffff; --text-secondary: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --axis: #383835; --good: #0ca30c; --bad: #e66767; }}
}}
:root[data-theme="dark"] .viz-root {{ --surface-1: #1a1a19; --page: #0d0d0d; --text-primary: #ffffff; --text-secondary: #c3c2b7; --grid: #2c2c2a; --axis: #383835; --good: #0ca30c; --bad: #e66767; }}
:root[data-theme="light"] .viz-root {{ --surface-1: #fcfcfb; --page: #f9f9f7; --text-primary: #0b0b0b; --text-secondary: #52514e; --grid: #e1e0d9; --axis: #c3c2b7; --good: #006300; --bad: #d03b3b; }}
.viz-root {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif; background: var(--page); color: var(--text-primary); padding: 24px 32px 64px; max-width: 1100px; margin: 0 auto; }}
h1 {{ font-size: 1.5rem; margin-bottom: 4px; }}
h2 {{ font-size: 1.15rem; margin-top: 40px; border-bottom: 1px solid var(--grid); padding-bottom: 6px; }}
h3 {{ font-size: 1rem; color: var(--text-secondary); margin-bottom: 8px; }}
.subtitle {{ color: var(--text-secondary); margin-bottom: 24px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; margin-top: 12px; }}
th, td {{ text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--grid); }}
th {{ color: var(--text-secondary); font-weight: 600; }}
tr.ours td:first-child {{ font-weight: 600; }}
.ci {{ color: var(--muted); font-size: 0.8em; }}
.delta-good {{ color: var(--good); font-weight: 600; }}
.delta-bad {{ color: var(--bad); font-weight: 600; }}
.status-good {{ color: var(--good); font-weight: 600; }}
.status-warn {{ color: #c98500; font-weight: 600; }}
.chart-row {{ display: flex; gap: 16px; margin: 10px 0 10px; flex-wrap: wrap; }}
.chart-col {{ flex: 1 1 360px; min-width: 320px; border: 1px solid var(--grid); border-radius: 6px; overflow: hidden; }}
.chart-title {{ font-size: 12px; fill: var(--text-primary); font-weight: 700; }}
.chart-axis {{ font-size: 10px; fill: var(--muted); }}
.chart-tick {{ font-size: 9px; fill: var(--muted); font-variant-numeric: tabular-nums; }}
.legend {{ margin-top: 8px; display: flex; flex-wrap: wrap; gap: 4px 16px; }}
.legend-item {{ display: inline-flex; align-items: center; gap: 6px; font-size: 0.78rem; color: var(--text-secondary); }}
.env-section {{ margin-top: 32px; padding-top: 16px; border-top: 1px solid var(--grid); }}
.chart-empty {{ color: var(--muted); font-size: 0.85rem; padding: 20px; }}
.note {{ font-size: 0.82rem; color: var(--muted); margin-top: 8px; }}
</style>
<div class="viz-root">
<h1>VOSR: Variance-Optimal Sampled Replay — Results</h1>
<div class="subtitle">Safety Gym benchmarks · {len(envs)} environments · 8 methods · IQM with stratified bootstrap 95% CI</div>

<h2>Headline: does VOSR improve safety without hurting reward?</h2>
<table>
<tr><th>Base optimizer comparison</th><th>Δ Return (norm.)</th><th>Δ Cost (norm.)</th><th>Δ Violation rate</th><th>Verdict</th></tr>
{pairs_html}
</table>
<div class="note">Δ Return &gt; 0 is better, Δ Cost / Δ Violation &lt; 0 is better (VOSR minus baseline). "better/equal" requires return not meaningfully worse and both safety metrics not meaningfully worse.</div>

<h2>Table 1 — All methods (per-environment min-max normalized, IQM across envs × seeds)</h2>
<table>
<tr><th>Method</th><th>n runs</th><th>Return (norm.)</th><th>Cost (norm.)</th><th>Violation rate</th><th>σ²_c (median, diagnostic)</th></tr>
{table_html}
</table>

<h2>Learning curves by environment</h2>
{"".join(env_sections)}
</div>
"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build_html()
