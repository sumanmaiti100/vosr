# VOSR: Variance-Optimal Sampled Replay

Code to train VOSR and its baselines (Uniform, TD-PER, Safety-PER, Uncertainty-PER replay,
wrapped around SAC-Lagrangian, CRPO, and PCRPO) across 10 Safety Gym environments, and to
reproduce every table and figure reported in the paper.

## Setup

```
pip install -r requirements.txt
```

Python 3.11. Training runs on CPU by default (`--device cpu` throughout) -- CPU was found
empirically faster than GPU for this workload's model/batch sizes. Pass `--device cuda` to
override.

## Quick check (NOT the paper's results -- just confirms the install works)

This is a ~30-second sanity check only, at a fraction of the real step budget. It exists to
confirm your environment is set up correctly before committing to a full run -- it does not
train to convergence and will not reproduce the paper's numbers by itself.

```
python -m vosr.train --env SafetyPointGoal1-v0 --method sac_lag_vosr --seed 0 \
    --total_steps 400 --eval_interval 200 --eval_episodes 1 --start_steps 100 \
    --log_dir runs --device cpu --train_every 4
```

This should finish without error and create `runs/SafetyPointGoal1-v0__sac_lag_vosr__seed0.csv`.

## Full training (reproduces the paper's reported results)

```
python -m vosr.reproduce_all
```

This is the actual full run: 15 methods x 10 environments x 3 seeds = 450 runs, each trained
to the real step budget below (not the 400-step quick check above), the same command used to
produce every number and figure in the paper. It prints an estimated compute/wall-clock time
before starting (parallelism defaults to 24 concurrent runs; set `VOSR_N_PARALLEL` to change
it), writes to `runs/` by default (`VOSR_LOG_DIR` to change it), and skips runs already
present so it's safe to resume after an interruption (`VOSR_SKIP_EXISTING=0` forces a full
rebuild).

Per-environment step budget and evaluation interval (from `vosr/reproduce_all.py`,
`TARGET_STEPS` / `EVAL_INTERVAL` -- also the values to pass to a single `vosr.train` run if
launching one environment/method by hand instead of the full grid):

| Environment | Total steps | Eval interval |
|---|---|---|
| SafetyPointGoal1-v0 | 77,500 | 2,500 |
| SafetyPointCircle1-v0 | 95,000 | 2,500 |
| SafetyPointButton1-v0 | 70,000 | 3,500 |
| SafetyPointPush1-v0 | 85,000 | 4,000 |
| SafetyAntVelocity-v1 | 132,000 | 6,000 |
| SafetyHalfCheetahVelocity-v1 | 138,000 | 6,000 |
| SafetyHopperVelocity-v1 | 144,000 | 6,000 |
| SafetyWalker2dVelocity-v1 | 138,000 | 6,000 |
| SafetyHumanoidVelocity-v1 | 120,000 | 6,000 |
| SafetySwimmerVelocity-v1 | 120,000 | 6,000 |

`--method` is any of the 15 keys in `vosr/train.py`'s `METHOD_TO_OPT_SAMPLER` (3 base
optimizers -- `sac_lag`, `crpo`, `pcrpo` -- x 5 samplers -- `uniform`, `td_per`, `safety_per`,
`uncertainty_per`, `vosr`). `--eval_episodes 3`, `--start_steps 1000`, `--train_every 4`,
`--device cpu` throughout, matching `vosr.reproduce_all`'s own launch settings exactly.

Full training-length runs take substantially longer than the quick check. `vosr.reproduce_all`
prints its own estimate at launch (from measured/estimated per-environment throughput in
`vosr/reproduce_all.py`'s `RATES` table); as a reference point, the complete 450-run grid is
roughly 310 total compute-hours, i.e. about 13 hours wall clock at the default 24-way
parallelism on a multi-core CPU machine (proportionally longer with less parallelism, e.g.
~39 hours at 8-way).

## Tables

```
python -m vosr.aggregate
```

IQM (interquartile mean) return/cost/violation tables with bootstrap 95% confidence
intervals, plus VOSR-vs-baseline deltas, from `runs/`.

## Figures

```
python -m vosr.make_figures                 # per-environment reward/cost, figures/
python vosr/make_paper_figures.py            # paper-ready combined figures, final_figures/
python vosr/make_paper_figures_split.py      # per-optimizer split figures, final_figures/
```

## Variance diagnostic (verifies the paper's variance-optimality claim empirically)

```
python -m vosr.run_variance_diagnostic       # writes to variance_diagnostic_logs_full/
python vosr/make_variance_figures.py         # builds the figures
```

## Ablation studies

```
python -m vosr.ablation_campaign             # kappa and eta ablations
python -m vosr.ablation1_extend_campaign     # kappa, 2 additional environments
python -m vosr.ablation2_campaign            # cost-limit sensitivity
python vosr/make_ablation_figures.py
python vosr/make_ablation2_figures.py
```

## Reproducibility notes

- Seeds (`torch.manual_seed`, `np.random.seed`) are set per run; CPU floating-point
  non-determinism across PyTorch versions/BLAS backends is not separately controlled for.
  Exact curves may vary slightly in the noise floor between machines; qualitative findings
  (which sampler wins where, whether cost stays under budget) were stable across reruns
  during development.
- `vosr/full_campaign.py` holds the per-environment step-budget/eval-interval constants used
  throughout the codebase; use `vosr/reproduce_all.py` to actually launch a full run from
  scratch.

## Citation

If you use this code, please cite the paper (see the paper PDF for the full citation).
