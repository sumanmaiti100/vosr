"""Two ablations isolating VOSR's own design choices (Theorem 1's kappa
blend, Proposition 3's tempering), NOT a comparison against baselines. Writes
exclusively to ablation_logs/ -- runs/, figures/, figures_variance/, and
final_figures/ are never touched or read by this script. Quick-scale (18k
steps, matching the earlier variance-diagnostic scale), 4 representative
environments spanning all 3 tiers, 2 seeds, single base optimizer
(SAC-Lagrangian) to isolate the ablated component from optimizer variation.

Arms:
  deployed    -- no override: tier-tuned kappa, eta=1.0 (the actual shipped config)
  kappa_0     -- VOSR_ABLATION_KAPPA=0.0   (pure safety-variance minimization)
  kappa_high  -- VOSR_ABLATION_KAPPA=1.5   (reward-dominated blend)
  eta_low     -- VOSR_ABLATION_ETA=0.1     (near-untempered, close to raw q*)
  eta_high    -- VOSR_ABLATION_ETA=5.0     (over-tempered, close to uniform b)

'deployed' is reused as the shared reference point for both ablations
(kappa=tier-tuned, eta=1.0 in both cases), so only 5 arms are actually run,
not 6. Each arm's logs land in ablation_logs/<arm>/ so filenames never
collide across arms. Per-run meta .json already records the effective
kappa/eta/margin (train.py change), so every run is independently
reproducible from its own metadata.
"""
import json
import os
import subprocess
import sys
import time

ENVS = ["SafetyPointGoal1-v0", "SafetyPointButton1-v0",
        "SafetyHalfCheetahVelocity-v1", "SafetyHopperVelocity-v1"]
SEEDS = [0, 1]
METHOD = "sac_lag_vosr"

TOTAL_STEPS = 18_000
EVAL_INTERVAL = 2_000
EVAL_EPISODES = 3
START_STEPS = 1_000
TRAIN_EVERY = 4

ARMS = {
    "deployed": {},
    "kappa_0": {"VOSR_ABLATION_KAPPA": "0.0"},
    "kappa_high": {"VOSR_ABLATION_KAPPA": "1.5"},
    "eta_low": {"VOSR_ABLATION_ETA": "0.1"},
    "eta_high": {"VOSR_ABLATION_ETA": "5.0"},
}

LOG_ROOT = "ablation_logs"
N_PARALLEL = int(os.environ.get("VOSR_N_PARALLEL", 20))
# generous flat per-run cap; 18k steps is small relative to prior full runs
TIME_BUDGET_SEC = 1800


def build_grid():
    grid = []
    for arm in ARMS:
        for env in ENVS:
            for seed in SEEDS:
                grid.append({"arm": arm, "env": env, "seed": seed})
    return grid


def launch(cfg):
    arm_dir = os.path.join(LOG_ROOT, cfg["arm"])
    os.makedirs(arm_dir, exist_ok=True)
    run_name = f"{cfg['env']}__{METHOD}__seed{cfg['seed']}"
    cmd = [sys.executable, "-m", "vosr.train",
           "--env", cfg["env"], "--method", METHOD, "--seed", str(cfg["seed"]),
           "--total_steps", str(TOTAL_STEPS),
           "--eval_interval", str(EVAL_INTERVAL),
           "--eval_episodes", str(EVAL_EPISODES),
           "--start_steps", str(START_STEPS),
           "--log_dir", arm_dir,
           "--device", "cpu",
           "--train_every", str(TRAIN_EVERY),
           "--time_budget_sec", str(TIME_BUDGET_SEC)]
    env_vars = os.environ.copy()
    env_vars["OMP_NUM_THREADS"] = "1"
    env_vars["MKL_NUM_THREADS"] = "1"
    env_vars.update(ARMS[cfg["arm"]])
    out_path = os.path.join(arm_dir, run_name + ".stdout.log")
    out_f = open(out_path, "w")
    p = subprocess.Popen(cmd, cwd=".", env=env_vars, stdout=out_f, stderr=subprocess.STDOUT)
    deadline = time.time() + TIME_BUDGET_SEC + 120
    return {"proc": p, "out_f": out_f, "name": f"{cfg['arm']}/{run_name}", "cfg": cfg,
            "start": time.time(), "deadline": deadline}


def main():
    os.makedirs(LOG_ROOT, exist_ok=True)
    grid = build_grid()
    print(f"Ablation campaign: {len(grid)} runs ({len(ENVS)} envs x {len(ARMS)} arms x {len(SEEDS)} seeds), "
          f"{TOTAL_STEPS} steps each, parallelism={N_PARALLEL}")

    manifest_f = open(os.path.join(LOG_ROOT, "manifest.jsonl"), "a")
    campaign_t0 = time.time()
    running = []
    pending = list(grid)

    while pending or running:
        while pending and len(running) < N_PARALLEL:
            running.append(launch(pending.pop(0)))
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
                print(f"[{elapsed/60:.1f} min] {r['name']} -> {status}  "
                      f"({len(pending)} pending, {len(running)-1} running)")
            else:
                still_running.append(r)
        running = still_running

    manifest_f.close()
    print(f"Ablation campaign complete. Total wall time {(time.time()-campaign_t0)/60:.1f} min")


if __name__ == "__main__":
    main()
