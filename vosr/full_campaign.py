"""Runs the complete remaining grid: fills in the missing sampler baselines
(TD-PER, Safety-PER, Uncertainty-PER) for CRPO/PCRPO on the 6 already-trained
environments, redoes those 6 environments' VOSR runs with the tightened
safety margin, and trains all 15 (optimizer x sampler) combinations fresh on
4 new environments. Already-valid non-VOSR baseline data on the original 6
environments is left untouched. Rolling-pool scheduler, longest jobs first.
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

EXISTING_ENVS = {
    "SafetyPointGoal1-v0": 77_500,
    "SafetyPointCircle1-v0": 95_000,
    "SafetyHalfCheetahVelocity-v1": 138_000,
    "SafetyHopperVelocity-v1": 144_000,
    "SafetyWalker2dVelocity-v1": 138_000,
    "SafetyAntVelocity-v1": 132_000,
}
NEW_ENVS = {
    "SafetyHumanoidVelocity-v1": 120_000,
    "SafetySwimmerVelocity-v1": 120_000,
    "SafetyPointButton1-v0": 70_000,
    "SafetyPointPush1-v0": 85_000,
}
TARGET_STEPS = {**EXISTING_ENVS, **NEW_ENVS}

EVAL_INTERVAL = {
    "SafetyPointGoal1-v0": 2500, "SafetyPointCircle1-v0": 2500,
    "SafetyHalfCheetahVelocity-v1": 6000, "SafetyHopperVelocity-v1": 6000,
    "SafetyWalker2dVelocity-v1": 6000, "SafetyAntVelocity-v1": 6000,
    "SafetyHumanoidVelocity-v1": 6000, "SafetySwimmerVelocity-v1": 6000,
    "SafetyPointButton1-v0": 3500, "SafetyPointPush1-v0": 4000,
}

# measured (baseline, vosr) s/1e4steps rates for the 6 existing envs; estimated
# (extrapolated from raw sim speed) for the 4 new ones -- flagged as approximate.
RATES = {
    "SafetyPointGoal1-v0": (261.1, 320.3),
    "SafetyPointCircle1-v0": (214.9, 365.1),
    "SafetyHalfCheetahVelocity-v1": (147.5, 289.9),
    "SafetyHopperVelocity-v1": (140.4, 280.5),
    "SafetyWalker2dVelocity-v1": (146.6, 286.3),
    "SafetyAntVelocity-v1": (156.7, 298.2),
    "SafetyHumanoidVelocity-v1": (250.0, 480.0),
    "SafetySwimmerVelocity-v1": (130.0, 260.0),
    "SafetyPointButton1-v0": (450.0, 520.0),
    "SafetyPointPush1-v0": (220.0, 340.0),
}

N_PARALLEL = int(os.environ.get("VOSR_N_PARALLEL", 24))
LOG_DIR = "runs"
MANIFEST = os.path.join(LOG_DIR, "manifest.jsonl")
SAFETY_TIME_MULT = 2.2


def build_grid():
    grid = []
    # existing envs: only the 10 missing/redo combos
    existing_needed = ["sac_lag_td_per", "sac_lag_vosr",
                        "crpo_td_per", "crpo_safety_per", "crpo_uncertainty_per", "crpo_vosr",
                        "pcrpo_td_per", "pcrpo_safety_per", "pcrpo_uncertainty_per", "pcrpo_vosr"]
    for env in EXISTING_ENVS:
        for method in existing_needed:
            for seed in SEEDS:
                grid.append({"env": env, "method": method, "seed": seed})
    # new envs: all 15 combos
    for env in NEW_ENVS:
        for opt, samp in itertools.product(OPTIMIZERS, SAMPLERS):
            for seed in SEEDS:
                grid.append({"env": env, "method": f"{opt}_{samp}", "seed": seed})
    return grid


def estimated_seconds(env, method):
    baseline_rate, vosr_rate = RATES[env]
    rate = vosr_rate if method.endswith("_vosr") else baseline_rate
    return TARGET_STEPS[env] / 1e4 * rate


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
    total_est = sum(estimated_seconds(c["env"], c["method"]) for c in grid)
    print(f"Total runs: {len(grid)}  |  estimated total compute: {total_est/3600:.2f} run-hours  |  "
          f"parallelism: {N_PARALLEL}  |  est. wall time: {total_est/N_PARALLEL/60:.0f} min "
          f"({total_est/N_PARALLEL/3600:.2f} hours)")

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
    print(f"Full campaign complete. Total wall time {(time.time()-campaign_t0)/60:.1f} min")


if __name__ == "__main__":
    main()
