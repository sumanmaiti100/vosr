"""Orchestrates the full-scale variance diagnostic (vosr/variance_diagnostic.py)
across all envs/optimizers/seeds. WARNING: this writes to
variance_diagnostic_logs_full/, opening each target CSV in overwrite mode --
running this (or importing it, before this guard was added) will truncate
any existing data for the runs it launches. All top-level work is guarded
behind __main__ specifically so `import vosr.run_variance_diagnostic` is
safe and does not trigger execution.

By default only runs missing from LOG_DIR are launched (skip_existing=True)
so this is safe to re-run after a partial/interrupted campaign. Pass
grid=[...] to main() to target a specific subset instead of the full sweep.
"""
import subprocess
import sys
import os
import time

sys.path.insert(0, ".")
from vosr.full_campaign import TARGET_STEPS, EVAL_INTERVAL, RATES
from vosr.variance_diagnostic import LOG_DIR

ENVS = ["SafetyPointGoal1-v0", "SafetyPointCircle1-v0", "SafetyPointButton1-v0", "SafetyPointPush1-v0",
        "SafetyAntVelocity-v1", "SafetyHalfCheetahVelocity-v1", "SafetyHopperVelocity-v1",
        "SafetyWalker2dVelocity-v1", "SafetyHumanoidVelocity-v1", "SafetySwimmerVelocity-v1"]
SEEDS = [0, 1, 2]
OPTIMIZERS = ["sac_lag", "crpo", "pcrpo"]
N_PARALLEL = 24
# diagnostic-script rate measured directly (no shield/tiering overhead):
# ~166.7 s/1e4 steps, independent of environment in smoke testing so far.
DIAG_RATE = 166.7
SAFETY_TIME_MULT = 2.5


def estimated_seconds(env):
    return TARGET_STEPS[env] / 1e4 * DIAG_RATE


def already_done(env, seed, opt):
    # expected rows = 1 header + one per checkpoint; require >=95% of that
    # so a run truncated partway through (e.g. an interrupted process) is
    # correctly treated as incomplete, not mistaken for done.
    path = os.path.join(LOG_DIR, f"{env}__{opt}__seed{seed}.csv")
    if not os.path.exists(path):
        return False
    expected = TARGET_STEPS[env] // EVAL_INTERVAL[env] + 1
    with open(path) as f:
        actual = sum(1 for _ in f)
    return actual >= 0.95 * expected


def main(grid=None, skip_existing=True):
    env_vars = os.environ.copy()
    env_vars["OMP_NUM_THREADS"] = "1"
    env_vars["MKL_NUM_THREADS"] = "1"

    if grid is None:
        grid = [(e, s, o) for e in ENVS for s in SEEDS for o in OPTIMIZERS]
    if skip_existing:
        before = len(grid)
        grid = [(e, s, o) for (e, s, o) in grid if not already_done(e, s, o)]
        print(f"Skipping {before - len(grid)} runs already present in {LOG_DIR}/")
    grid.sort(key=lambda x: -estimated_seconds(x[0]))
    running = []
    pending = list(grid)
    t0 = time.time()

    total_est = sum(estimated_seconds(e) for e, s, o in grid)
    print(f"Total runs: {len(grid)}  |  est. total compute: {total_est/3600:.2f} run-hours  |  "
          f"parallelism: {N_PARALLEL}  |  est. wall time: {total_est/N_PARALLEL/60:.0f} min")

    while pending or running:
        while pending and len(running) < N_PARALLEL:
            env, seed, opt = pending.pop(0)
            cmd = [sys.executable, "-m", "vosr.variance_diagnostic", "--env", env, "--seed", str(seed), "--optimizer", opt]
            p = subprocess.Popen(cmd, cwd=".", env=env_vars, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            deadline = time.time() + estimated_seconds(env) * SAFETY_TIME_MULT + 120
            running.append((p, env, seed, opt, deadline))
        time.sleep(3)
        still = []
        for p, env, seed, opt, deadline in running:
            rc = p.poll()
            timed_out = time.time() > deadline and rc is None
            if rc is not None or timed_out:
                if timed_out:
                    p.kill()
                    p.wait()
                    status = "killed_timeout"
                else:
                    status = f"exit {rc}"
                print(f"[{time.time()-t0:.0f}s] {env} {opt} seed{seed} -> {status}  ({len(pending)} pending, {len(running)-1} running)")
            else:
                still.append((p, env, seed, opt, deadline))
        running = still

    print(f"Variance diagnostic (full scale) complete. Total wall time {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
