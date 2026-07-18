"""Aggregates run CSVs into paper-style summaries: per-environment min-max
normalization, interquartile mean (IQM) with stratified bootstrap CIs over
(environment, seed), and paired baseline-vs-VOSR deltas.
"""
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

RUN_DIR = os.environ.get("VOSR_RUN_DIR", "runs")
FINAL_FRAC = 0.3  # average over the last 30% of eval points as "final performance"

BASE_OPT_PAIRS = {
    "sac_lag": {"baseline": ["sac_lag_uniform", "sac_lag_td_per", "sac_lag_safety_per", "sac_lag_uncertainty_per"], "vosr": "sac_lag_vosr"},
    "crpo": {"baseline": ["crpo_uniform", "crpo_td_per", "crpo_safety_per", "crpo_uncertainty_per"], "vosr": "crpo_vosr"},
    "pcrpo": {"baseline": ["pcrpo_uniform", "pcrpo_td_per", "pcrpo_safety_per", "pcrpo_uncertainty_per"], "vosr": "pcrpo_vosr"},
}
ALL_METHODS = ["sac_lag_uniform", "sac_lag_td_per", "sac_lag_safety_per", "sac_lag_uncertainty_per", "sac_lag_vosr",
               "crpo_uniform", "crpo_td_per", "crpo_safety_per", "crpo_uncertainty_per", "crpo_vosr",
               "pcrpo_uniform", "pcrpo_td_per", "pcrpo_safety_per", "pcrpo_uncertainty_per", "pcrpo_vosr"]


def load_runs():
    records = []
    for path in glob.glob(os.path.join(RUN_DIR, "*.csv")):
        base = os.path.basename(path)[:-4]
        parts = base.split("__")
        if len(parts) != 3:
            continue
        env, method, seed_s = parts
        seed = int(seed_s.replace("seed", ""))
        rows = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    rows.append({
                        "step": int(r["step"]),
                        "eval_return": float(r["eval_return"]),
                        "eval_cost": float(r["eval_cost"]),
                        "eval_violation_rate": float(r["eval_violation_rate"]),
                        "sigma_c2": float(r["sigma_c2"]) if r.get("sigma_c2") not in (None, "", "None") else None,
                    })
                except (ValueError, KeyError):
                    continue
        if not rows:
            continue
        records.append({"env": env, "method": method, "seed": seed, "rows": rows})
    return records


def final_performance(rows, step_cap=None):
    """Performance averaged over the trailing portion of TRAINING, matched by
    environment-step count (not wall-clock row index / time) so that methods
    with heavier per-step compute (e.g. VOSR's JVP scoring) are compared at
    the same amount of data/updates rather than being penalized for wall-clock
    overhead, which the paper reports separately as a runtime metric."""
    usable = [r for r in rows if step_cap is None or r["step"] <= step_cap]
    if not usable:
        usable = rows[:1]
    n = len(usable)
    k = max(1, int(np.ceil(n * FINAL_FRAC)))
    tail = usable[-k:]
    ret = np.mean([r["eval_return"] for r in tail])
    cost = np.mean([r["eval_cost"] for r in tail])
    viol = np.mean([r["eval_violation_rate"] for r in tail])
    sig = [r["sigma_c2"] for r in tail if r["sigma_c2"] is not None and np.isfinite(r["sigma_c2"])]
    sigma_c2 = float(np.median(sig)) if sig else None
    return ret, cost, viol, sigma_c2, usable[-1]["step"]


def iqm(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n < 4:
        return float(np.mean(x)) if n else float("nan")
    lo, hi = int(np.floor(n * 0.25)), int(np.ceil(n * 0.75))
    mid = x[lo:hi]
    return float(np.mean(mid)) if len(mid) else float(np.mean(x))


def bootstrap_iqm_ci(x, n_boot=2000, alpha=0.05, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.array([iqm(rng.choice(x, size=len(x), replace=True)) for _ in range(n_boot)])
    lo = np.percentile(boots, 100 * alpha / 2)
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return iqm(x), lo, hi


def main():
    records = load_runs()
    print(f"Loaded {len(records)} runs")

    # Step-matched comparison: every method in an env is scored using only the
    # data/updates that the SLOWEST method (usually VOSR, due to JVP scoring
    # overhead) managed to complete in the shared wall-clock budget. This
    # removes the wall-clock-vs-step-count confound from the headline numbers.
    max_step_by_run = {}
    for rec in records:
        max_step_by_run[(rec["env"], rec["method"], rec["seed"])] = max(r["step"] for r in rec["rows"])
    common_step_by_env = {}
    for env in set(rec["env"] for rec in records):
        maxes = [v for (e, m, s), v in max_step_by_run.items() if e == env]
        common_step_by_env[env] = min(maxes)

    per_run = []
    for rec in records:
        cap = common_step_by_env[rec["env"]]
        ret, cost, viol, sigma_c2, used_step = final_performance(rec["rows"], step_cap=cap)
        per_run.append({"env": rec["env"], "method": rec["method"], "seed": rec["seed"],
                         "return": ret, "cost": cost, "violation": viol, "sigma_c2": sigma_c2,
                         "n_evals": len(rec["rows"]), "step_cap": cap, "used_step": used_step,
                         "own_max_step": max_step_by_run[(rec["env"], rec["method"], rec["seed"])]})

    print("\n=== Step-matched comparison budget per environment (min over methods/seeds of max step reached) ===")
    for env, cap in sorted(common_step_by_env.items()):
        print(f"  {env}: {cap} steps")

    envs = sorted(set(r["env"] for r in per_run))
    # per-env min-max normalization of return and cost across all methods/seeds present
    norm_stats = {}
    for env in envs:
        rets = [r["return"] for r in per_run if r["env"] == env]
        costs = [r["cost"] for r in per_run if r["env"] == env]
        norm_stats[env] = {
            "ret_min": min(rets), "ret_max": max(rets),
            "cost_min": min(costs), "cost_max": max(costs),
        }

    def norm(v, lo, hi):
        if hi - lo < 1e-9:
            return 0.5
        return (v - lo) / (hi - lo)

    for r in per_run:
        st = norm_stats[r["env"]]
        r["return_norm"] = norm(r["return"], st["ret_min"], st["ret_max"])
        r["cost_norm"] = norm(r["cost"], st["cost_min"], st["cost_max"])

    # aggregate per method across (env, seed)
    by_method = defaultdict(list)
    for r in per_run:
        by_method[r["method"]].append(r)

    summary = {}
    for method, rows in by_method.items():
        ret_iqm, ret_lo, ret_hi = bootstrap_iqm_ci([r["return_norm"] for r in rows])
        cost_iqm, cost_lo, cost_hi = bootstrap_iqm_ci([r["cost_norm"] for r in rows])
        viol_iqm, viol_lo, viol_hi = bootstrap_iqm_ci([r["violation"] for r in rows])
        raw_ret_iqm = iqm([r["return"] for r in rows])
        raw_cost_iqm = iqm([r["cost"] for r in rows])
        sig_vals = [r["sigma_c2"] for r in rows if r["sigma_c2"] is not None]
        summary[method] = {
            "n_runs": len(rows),
            "return_norm_iqm": ret_iqm, "return_norm_ci": [ret_lo, ret_hi],
            "cost_norm_iqm": cost_iqm, "cost_norm_ci": [cost_lo, cost_hi],
            "violation_iqm": viol_iqm, "violation_ci": [viol_lo, viol_hi],
            "raw_return_iqm": raw_ret_iqm, "raw_cost_iqm": raw_cost_iqm,
            "sigma_c2_median": float(np.median(sig_vals)) if sig_vals else None,
        }

    results_dir = os.environ.get("VOSR_RESULTS_DIR", "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "per_run.json"), "w") as f:
        json.dump(per_run, f, indent=2)
    with open(os.path.join(results_dir, "summary_by_method.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # paired baseline-vs-VOSR deltas
    pairs_out = {}
    for opt, cfg in BASE_OPT_PAIRS.items():
        vosr_m = cfg["vosr"]
        if vosr_m not in summary:
            continue
        for base_m in cfg["baseline"]:
            if base_m not in summary:
                continue
            pairs_out[f"{base_m}_vs_{vosr_m}"] = {
                "baseline_return_norm": summary[base_m]["return_norm_iqm"],
                "vosr_return_norm": summary[vosr_m]["return_norm_iqm"],
                "baseline_cost_norm": summary[base_m]["cost_norm_iqm"],
                "vosr_cost_norm": summary[vosr_m]["cost_norm_iqm"],
                "baseline_violation": summary[base_m]["violation_iqm"],
                "vosr_violation": summary[vosr_m]["violation_iqm"],
                "return_delta": summary[vosr_m]["return_norm_iqm"] - summary[base_m]["return_norm_iqm"],
                "cost_delta": summary[vosr_m]["cost_norm_iqm"] - summary[base_m]["cost_norm_iqm"],
                "violation_delta": summary[vosr_m]["violation_iqm"] - summary[base_m]["violation_iqm"],
            }
    with open(os.path.join(results_dir, "pairs.json"), "w") as f:
        json.dump(pairs_out, f, indent=2)

    print("\n=== Summary (normalized IQM, bootstrap 95% CI) ===")
    for m in ALL_METHODS:
        if m not in summary:
            continue
        s = summary[m]
        print(f"{m:26s} n={s['n_runs']:3d}  return={s['return_norm_iqm']:.3f} "
              f"[{s['return_norm_ci'][0]:.3f},{s['return_norm_ci'][1]:.3f}]  "
              f"cost={s['cost_norm_iqm']:.3f} [{s['cost_norm_ci'][0]:.3f},{s['cost_norm_ci'][1]:.3f}]  "
              f"violation={s['violation_iqm']:.3f}")

    print("\n=== Baseline vs VOSR deltas (positive return_delta / negative cost&violation_delta = VOSR better) ===")
    for k, v in pairs_out.items():
        print(f"{k:45s} d_return={v['return_delta']:+.3f}  d_cost={v['cost_delta']:+.3f}  d_violation={v['violation_delta']:+.3f}")

    # runtime overhead (paper's own separate "Runtime" metric): wall time per 1e4 steps
    runtime_by_method = defaultdict(list)
    for path in glob.glob(os.path.join(RUN_DIR, "*.json")):
        try:
            meta = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            continue
        if meta.get("steps_completed", 0) > 0:
            sec_per_1e4 = meta["wall_time"] / meta["steps_completed"] * 1e4
            runtime_by_method[meta["method"]].append(sec_per_1e4)
    runtime_summary = {m: float(np.mean(v)) for m, v in runtime_by_method.items()}
    with open(os.path.join(results_dir, "runtime.json"), "w") as f:
        json.dump(runtime_summary, f, indent=2)
    print("\n=== Runtime overhead: mean wall-clock seconds per 1e4 env steps ===")
    for m in ALL_METHODS:
        if m in runtime_summary:
            print(f"  {m:26s} {runtime_summary[m]:.1f} s / 1e4 steps")


if __name__ == "__main__":
    main()
