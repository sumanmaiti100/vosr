"""Single, complete entry point to reproduce the full main-campaign dataset
(15 methods x 10 environments x 3 seeds = 450 runs) from a clean clone.

This did not exist before: full_campaign.py (kept as-is, for provenance)
only fills in gaps assuming 6 environments' non-VOSR baselines already
exist from an earlier, undocumented campaign -- it cannot rebuild runs/
from nothing. This script builds the entire grid explicitly, using
train.py's current logic (tiering, shield, episodic cost tracking, all
already permanent in train.py -- nothing here reaches into history).

Usage:
    python -m vosr.reproduce_all                      # writes to runs/
    VOSR_LOG_DIR=runs_check python -m vosr.reproduce_all   # write elsewhere

Step budgets and eval intervals are the exact values the paper's reported
runs/ data was produced with (copied from full_campaign.py's TARGET_STEPS /
EVAL_INTERVAL, which remain the source of truth for those numbers).
Per-run wall-clock caps use the same measured/estimated throughput table as
full_campaign.py, so a full run finishes in bounded time regardless of
which environment is slowest.
"""
import itertools
import json
import os
import subprocess
import sys
import time

OPTIMIZERS = ["sac_lag", "crpo", "pcrpo"]
SAMPLERS = ["uniform", "td_per", "safety_per", "uncertainty_per", "vosr"]
SEEDS = [0, 1, 2]

TARGET_STEPS = {
    "SafetyPointGoal1-v0": 77_500,
    "SafetyPointCircle1-v0": 95_000,
    "SafetyHalfCheetahVelocity-v1": 138_000,
    "SafetyHopperVelocity-v1": 144_000,
    "SafetyWalker2dVelocity-v1": 138_000,
    "SafetyAntVelocity-v1": 132_000,
    "SafetyHumanoidVelocity-v1": 120_000,
    "SafetySwimmerVelocity-v1": 120_000,
    "SafetyPointButton1-v0": 70_000,
    "SafetyPointPush1-v0": 85_000,
}
EVAL_INTERVAL = {
    "SafetyPointGoal1-v0": 2500, "SafetyPointCircle1-v0": 2500,
    "SafetyHalfCheetahVelocity-v1": 6000, "SafetyHopperVelocity-v1": 6000,
    "SafetyWalker2dVelocity-v1": 6000, "SafetyAntVelocity-v1": 6000,
    "SafetyHumanoidVelocity-v1": 6000, "SafetySwimmerVelocity-v1": 6000,
    "SafetyPointButton1-v0": 3500, "SafetyPointPush1-v0": 4000,
}
# measured (baseline, vosr) s/1e4steps rates for 6 envs; estimated for the
# other 4 -- same table as full_campaign.py, used only to size timeouts.
RATES = {
    "SafetyPointGoal1-v0": (261.1, 320.3), "SafetyPointCircle1-v0": (214.9, 365.1),
    "SafetyHalfCheetahVelocity-v1": (147.5, 289.9), "SafetyHopperVelocity-v1": (140.4, 280.5),
    "SafetyWalker2dVelocity-v1": (146.6, 286.3), "SafetyAntVelocity-v1": (156.7, 298.2),
    "SafetyHumanoidVelocity-v1": (250.0, 480.0), "SafetySwimmerVelocity-v1": (130.0, 260.0),
    "SafetyPointButton1-v0": (450.0, 520.0), "SafetyPointPush1-v0": (220.0, 340.0),
}

N_PARALLEL = int(os.environ.get("VOSR_N_PARALLEL", 24))
LOG_DIR = os.environ.get("VOSR_LOG_DIR", "runs")
MANIFEST = os.path.join(LOG_DIR, "manifest_reproduce_all.jsonl")
SAFETY_TIME_MULT = 2.2


def build_grid():
    grid = []
    for env in TARGET_STEPS:
        for opt, samp in itertools.product(OPTIMIZERS, SAMPLERS):
            for seed in SEEDS:
                grid.append({"env": env, "method": f"{opt}_{samp}", "seed": seed})
    return grid


def estimated_seconds(env, method):
    baseline_rate, vosr_rate = RATES[env]
    rate = vosr_rate if method.endswith("_vosr") else baseline_rate
    return TARGET_STEPS[env] / 1e4 * rate


def already_done(cfg):
    run_name = f"{cfg['env']}__{cfg['method']}__seed{cfg['seed']}"
    return os.path.exists(os.path.join(LOG_DIR, run_name + ".csv"))


def launch(cfg, env_vars):
    run_name = f"{cfg['env']}__{cfg['method']}__seed{cfg['seed']}"
    cmd = [sys.executable, "-m", "vosr.train",
           "--env", cfg["env"], "--method", cfg["method"], "--seed", str(cfg["seed"]),
           "--total_steps", str(TARGET_STEPS[cfg["env"]]),
           "--eval_interval", str(EVAL_INTERVAL[cfg["env"]]),
           "--eval_episodes", "3",
           "--start_steps", "1000",
           "--log_dir", LOG_DIR,
           "--device", "cpu",
           "--train_every", "4",
           "--time_budget_sec", str(int(estimated_seconds(cfg["env"], cfg["method"]) * SAFETY_TIME_MULT))]
    out_path = os.path.join(LOG_DIR, run_name + ".stdout.log")
    out_f = open(out_path, "w")
    p = subprocess.Popen(cmd, cwd=".", env=env_vars, stdout=out_f, stderr=subprocess.STDOUT)
    deadline = time.time() + estimated_seconds(cfg["env"], cfg["method"]) * SAFETY_TIME_MULT + 120
    return {"proc": p, "out_f": out_f, "name": run_name, "cfg": cfg, "start": time.time(), "deadline": deadline}


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    grid = build_grid()
    skip_existing = os.environ.get("VOSR_SKIP_EXISTING", "1") == "1"
    if skip_existing:
        before = len(grid)
        grid = [c for c in grid if not already_done(c)]
        print(f"Skipping {before - len(grid)} runs already present in {LOG_DIR}/ "
              f"(set VOSR_SKIP_EXISTING=0 to force full rebuild)")

    total_est = sum(estimated_seconds(c["env"], c["method"]) for c in grid)
    print(f"Total runs: {len(grid)}  |  estimated compute: {total_est/3600:.2f} run-hours  |  "
          f"parallelism: {N_PARALLEL}  |  est. wall time: {total_est/N_PARALLEL/60:.0f} min "
          f"({total_est/N_PARALLEL/3600:.2f} hours)  |  writing to {LOG_DIR}/")

    env_vars = os.environ.copy()
    env_vars["OMP_NUM_THREADS"] = "1"
    env_vars["MKL_NUM_THREADS"] = "1"

    grid.sort(key=lambda c: -estimated_seconds(c["env"], c["method"]))

    campaign_t0 = time.time()
    manifest_f = open(MANIFEST, "a")
    running = []
    pending = list(grid)

    while pending or running:
        while pending and len(running) < N_PARALLEL:
            running.append(launch(pending.pop(0), env_vars))
        time.sleep(2)
        still_running = []
        for r in running:
            rc = r["proc"].poll()
            timed_out = time.time() > r["deadline"] and rc is None
            if rc is not None or timed_out:
                if timed_out:
                    r["proc"].kill()
                    r["proc"].wait()
                    status = "killed_timeout"
                else:
                    status = "ok" if rc == 0 else f"exit_{rc}"
                r["out_f"].close()
                rec = {"run": r["name"], "status": status, "wall_time": time.time() - r["start"], **r["cfg"]}
                manifest_f.write(json.dumps(rec) + "\n")
                manifest_f.flush()
                elapsed = time.time() - campaign_t0
                print(f"[{elapsed/60:.1f} min] {r['name']} -> {status}  ({len(pending)} pending, {len(running)-1} running)")
            else:
                still_running.append(r)
        running = still_running

    manifest_f.close()
    print(f"reproduce_all complete. Total wall time {(time.time()-campaign_t0)/60:.1f} min")


if __name__ == "__main__":
    main()
