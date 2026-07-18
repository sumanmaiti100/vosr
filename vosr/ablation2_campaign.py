"""Ablation 2: sensitivity to the cost-limit budget itself. Tests whether
VOSR actually reshapes its reward/cost tradeoff when the safety budget
changes, rather than behaving identically regardless of what it's told the
limit is. Writes exclusively to ablation_logs_2/ -- runs/, figures/,
figures_variance/, final_figures/ (main + ablations/), and ablation_logs/
are never touched or modified by this script.

Environment choice (disclosed criterion, not outcome-based): the two
environments are picked for having the widest-swinging, most
threshold-reactive cost dynamics observed so far in this project (both are
in train.py's SEVERE_COST_ENVS tier, documented there as having seen
12-18x budget overshoots under the standard limit) -- i.e., environments
where changing the limit from 10 to 70 is expected to actually produce a
visible behavioral difference. Locomotion environments were not used here
because their cost already sits near 0 well under every limit tested in
Ablation 1, so a cost-limit sweep on them would show nothing.

Arms:
  cost_limit_10  VOSR_ABLATION_COST_LIMIT=10   (tight budget)
  deployed       cost_limit=25 (the standard value) -- REUSED from
                 ablation_logs/deployed/, not rerun here, since Ablation 1
                 already trained exactly this config (SAC-Lagrangian+VOSR,
                 same 2 envs, same seeds, same 18k steps) with cost_limit=25.
  cost_limit_70  VOSR_ABLATION_COST_LIMIT=70   (relaxed budget)

Only cost_limit_10 and cost_limit_70 are actually run here (8 new runs);
"deployed" is read from the existing Ablation 1 logs at plot time.
"""
import json
import os
import subprocess
import sys
import time

ENVS = ["SafetyPointGoal1-v0", "SafetyPointButton1-v0"]
SEEDS = [0, 1]
METHOD = "sac_lag_vosr"

TOTAL_STEPS = 18_000
EVAL_INTERVAL = 2_000
EVAL_EPISODES = 3
START_STEPS = 1_000
TRAIN_EVERY = 4

ARMS = {
    "cost_limit_10": {"VOSR_ABLATION_COST_LIMIT": "10"},
    "cost_limit_70": {"VOSR_ABLATION_COST_LIMIT": "70"},
}

LOG_ROOT = "ablation_logs_2"
N_PARALLEL = int(os.environ.get("VOSR_N_PARALLEL", 20))
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
    print(f"Ablation-2 campaign: {len(grid)} runs ({len(ENVS)} envs x {len(ARMS)} arms x {len(SEEDS)} seeds), "
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
    print(f"Ablation-2 campaign complete. Total wall time {(time.time()-campaign_t0)/60:.1f} min")


if __name__ == "__main__":
    main()
