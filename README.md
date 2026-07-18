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

## Quick check

Confirm the install works with one small run (~30 seconds):

```
python -m vosr.train --env SafetyPointGoal1-v0 --method sac_lag_vosr --seed 0 \
    --total_steps 400 --eval_interval 200 --eval_episodes 1 --start_steps 100 \
    --log_dir runs --device cpu --train_every 4
```

This should finish without error and create `runs/SafetyPointGoal1-v0__sac_lag_vosr__seed0.csv`.

## Reproducing the main results (15 methods x 10 environments x 3 seeds)

```
python -m vosr.reproduce_all
```

Single entry point for the full 450-run grid. Prints an estimated compute/wall-clock time
before starting (parallelism defaults to 24 concurrent runs; set `VOSR_N_PARALLEL` to change
it). Writes to `runs/` by default (`VOSR_LOG_DIR` to change it); skips runs already present
so it's safe to resume after an interruption (`VOSR_SKIP_EXISTING=0` forces a full rebuild).

A single run can also be launched directly -- see the Quick check command above, with
`--total_steps` / `--eval_interval` set from the table in `vosr/reproduce_all.py`
(`TARGET_STEPS`, `EVAL_INTERVAL`). `--method` is any of the 15 keys in
`vosr/train.py`'s `METHOD_TO_OPT_SAMPLER` (3 base optimizers x 5 samplers).

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
