"""Extends Ablation 1 (kappa) to 2 more environments, chosen by a disclosed,
outcome-blind rule decided BEFORE running anything: one more environment
from each tier not yet doubly-covered by the original 4-env sweep
(PointGoal1, PointButton1 = severe; HalfCheetahVelocity = mild;
HopperVelocity = reward_max). AntVelocity is the only untested mild-tier
env. Walker2dVelocity is the first untested reward_max-tier env in
canonical environment order (HumanoidVelocity is the other candidate).

Same arms, same methodology, same 18k-step/2-seed scale as the original
ablation_campaign.py, so results are directly comparable. Writes into the
SAME ablation_logs/<arm>/ folders as Ablation 1 (filenames differ by env,
so no collision) -- this is additive, not a replacement.
"""
import json
import os
import subprocess
import sys
import time

ENVS = ["SafetyAntVelocity-v1", "SafetyWalker2dVelocity-v1"]
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
}

LOG_ROOT = "ablation_logs"
N_PARALLEL = int(os.environ.get("VOSR_N_PARALLEL", 12))
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
    print(f"Ablation-1-extend campaign: {len(grid)} runs ({len(ENVS)} envs x {len(ARMS)} arms x {len(SEEDS)} seeds), "
          f"{TOTAL_STEPS} steps each, parallelism={N_PARALLEL}")

    manifest_f = open(os.path.join(LOG_ROOT, "manifest_extend.jsonl"), "a")
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
    print(f"Ablation-1-extend campaign complete. Total wall time {(time.time()-campaign_t0)/60:.1f} min")


if __name__ == "__main__":
    main()
