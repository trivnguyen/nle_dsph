# nle_dsph

Neural likelihood estimation for dwarf spheroidal stellar kinematics.

A conditional normalizing flow is trained on simulated galaxies to learn the
**per-star** likelihood

```
p( v_los | log10 R, theta, log r_star, sigma_v )
```

where `theta` are the dark-matter and orbital-anisotropy parameters,
`log r_star` is the measured half-light radius, `sigma_v` is the per-star
velocity uncertainty, and `R` is the projected radius. Given an observed
galaxy, the likelihood of its catalog is the sum of that density over its
stars, and sampling that against a prior gives a posterior on `theta`.

Learning the likelihood rather than the posterior directly has two
consequences worth knowing up front. The number of stars is not baked into
the model, so one trained flow serves catalogs of any size. And the prior is
not baked in either: it is supplied at inference, which means it can be
changed, narrowed, or examined without retraining.

## Why a per-star likelihood

Conditioning on where a star is, rather than modeling its position, removes
the survey's spatial selection function from the problem. The positions in a
catalog are the ones the survey chose to observe, so a model of
`p(x, y, v_los)` has to reproduce that selection. The likelihood of `v_los`
*given* where the star already is does not depend on any position-only
selection, so no selection augmentation is needed during training and no
correction at inference.

Conditioning on the projected radius rather than `(x, y)` goes one step
further: `p(v_los | R, theta)` is axisymmetric, so the model is not asked to
learn a symmetry that is known in advance, and the context no longer depends
on the sky orientation of the projection.

Both are configuration choices, not assumptions baked into the code. Two
flags decide what the flow models, and nothing else does:

| `condition_on_position` | `condition_on_radius` | the model is |
|---|---|---|
| `False` | (ignored) | `p(x, y, v_los \| theta, ...)` |
| `True` | `False` | `p(v_los \| x, y, theta, ...)` |
| `True` | `True` | `p(v_los \| log10 R, theta, ...)` |

`condition_on_radius` without `condition_on_position` raises: the radius
replaces the `(x, y)` context, it does not act on its own.

## Installation

No install step; put the repository root on `PYTHONPATH` or work from it.
Requires Python 3.10+ and

```
torch  pytorch-lightning  zuko  numpy  scipy  h5py  ml_collections
absl-py  tqdm  wandb  pocomc
```

`pocomc` is needed only for posterior sampling, `wandb` only for logging
(set `config.debug = True` to train without it). The notebooks additionally
use `matplotlib`, `pandas`, and `dsph_analysis` for reading observed
catalogs.

## Layout

```
nle_dsph/          the package
  nle.py             the NLE LightningModule and the flow it wraps
  transforms/        StarBatch, the observation pipeline, normalization
  datasets.py        HDF5 readers and per-star dataloaders
  training.py        data preparation, model construction, trainer, callbacks
  api.py             a small front end over all of the above
  profiles.py        analytic Zhao density and slope, in numpy
  utils.py           MLP, optimizers, LR schedules
configs/           configs of actual runs
scripts/           command-line entry points, and batch diagnostics
notebooks/         walkthroughs, the documented config template, CONFIG.md
slurm/             job scripts and a detached local launcher
tests/             framework-free checks, run with python tests/run_tests.py
```

## Training

A run is described entirely by a config file — an ordinary Python module
exposing `get_config()`. `notebooks/example_config.py` is the documented
template, carrying a comment on every field; copy it and strip what you do
not change. `configs/` holds the configs of actual runs.

```bash
python scripts/train_nle.py --config configs/nle_8p_v4.py
python scripts/train_nle.py --config configs/nle_8p_v4.py \
    --config.num_steps=50000
```

`--config.<field>=<value>` overrides any field, so a parameter sweep does not
need a config file edited per run. `slurm/train_nle.sh` wraps the same call
with logging and detaches by default; `slurm/train_nle.sbatch` is the
scheduler version.

The same thing from Python, which is what the notebooks use:

```python
from nle_dsph import api

config = api.load_config('configs/nle_8p_v4.py', num_steps=2000)
run_dir = api.train(config)
```

`scripts/train_nle_api.py` is a command-line entry point over `api.train`;
its docstring lists how it differs from `scripts/train_nle.py`.

### The run directory

Training writes `<workdir>/<project>/<run id>/`, containing the checkpoints,
`config_snapshot.json` and a verbatim copy of the config file. **That
directory is self-contained.** The normalization statistics live inside the
checkpoint as a Lightning hyperparameter, so a model can never be paired with
statistics from a different dataset, and the training data does not have to
still exist:

```python
model, config_snapshot = api.load_model(run_dir)
```

Never re-derive the normalization from a training file at inference time. If
that file is regenerated, every posterior produced afterwards is quietly
wrong.

## Inference

An observation is a plain dataclass. Build one from whatever catalog you are
working with:

```python
import numpy as np
from nle_dsph.api import Observation

obs = Observation(
    R_proj_kpc=R_proj_kpc,          # (N,) projected radius, kpc
    vlos_kms=vlos_kms,              # (N,) line-of-sight velocity, km/s
    vlos_err_kms=vlos_err_kms,      # (N,) per-star sigma, km/s, LINEAR
    log_rstar_kpc=np.log10(rhalf_kpc),
    log_rstar_err_dex=sigma_dex,    # photometric error on the above
    name='my_dwarf',
)
```

Sky geometry is supplied either as `pos_kpc` (tangent-plane `(X, Y)`) or as
`R_proj_kpc`, and neither is derived from the other — if a catalog reports
both, pass both and set `use` to say which the model is given. A radius
conditioned model takes `|pos|`, so handing it positions when a measured
radius exists is refused rather than silently preferred.

`vlos_err_kms` is **linear sigma in km/s, not log10**. The flow conditions on
sigma itself, so a 1 km/s error is passed as `1.0`. Pipelines that append
`log10(sigma_v)` to their parameter vector use the other convention, and
mixing them fails silently: every star reads as better measured than it is
and the posterior comes out too tight.

### The prior is yours to write

There is deliberately no helper that builds it. The prior has three groups of
dimensions, in this order, which `sample_posterior` checks:

```
[ labels... , v_sys , log_rstar ]
```

- **labels**, over the box the *training set* was drawn from. Never wider:
  outside it the model never saw a galaxy and the likelihood is
  extrapolating. Narrower is a legitimate choice, as long as it is stated.
- **`v_sys`**, the systemic velocity, fitted as a nuisance parameter and
  returned separately from the posterior.
- **`log_rstar`**, sampled over its photometric error. Being measured does
  not make it fixed: pinning it asserts the photometry is exact and makes
  every other posterior width an underestimate. It is truncated both at a
  chosen number of sigma and at `api.cond_box(model)`, the range of `r_star`
  the checkpoint was trained on, read back out of the checkpoint's own
  normalization.

```python
import pocomc as pc
from scipy.stats import truncnorm, uniform

dists = [uniform(lo, hi - lo) for lo, hi in zip(prior_min, prior_max)]
dists.append(uniform(-50.0, 100.0))                 # v_sys, km/s
dists.append(truncnorm(a, b, loc=mu, scale=sigma))  # log_rstar

result = api.sample_posterior(model, obs, pc.Prior(dists), labels=labels)
api.save_posterior(out_dir, obs, result)
```

Sampling is preconditioned Monte Carlo (`pocomc`): no burn-in to discard, no
mixing to judge by eye, and the evidence comes out as a by-product. The
result holds `posterior` as `(N, n_labels + 1)` with the conditioning column
last, `nuisance` holding `v_sys`, and `log_evidence`.

## Data format

Training reads one HDF5 file per shard, `<data_root>/<data_name>/data.<i>.h5`.
Stars are stored flat and indexed by `ptr`; per-galaxy quantities are named
1-D arrays of length `n_galaxies`. Three header attributes say which is
which: `node_features` (per star, `pos` and `vel` in kpc and km/s),
`graph_features` (per galaxy), and `all_features`.

Nothing is positional — quantities are requested by name through
`config.labels` and `config.cond_labels`.

`labels` and `cond_labels` both reach the flow as context. The difference is
where their prior comes from at inference: a `cond` is measured
independently, so it gets the measurement as its prior; a label can only get
the simulation box.

## Augmentation, not preprocessing

A random line-of-sight projection and a fresh per-star noise draw are applied
to **every batch of every epoch**, not once up front. This is what teaches
the flow that the likelihood depends on `sigma_v`, and it is why one set of
simulations can stand in for many observations. The pipeline is not applied
at inference, where real data arrives already projected and already noisy.

A consequence: the normalization statistics are min/max over a stochastic
pipeline, so no finite sample pins them down exactly. `config.norm_max_stars`
caps how many stars are measured, which trades a narrower range for a much
cheaper startup.

## Tests

```bash
python tests/run_tests.py            # all
python tests/run_tests.py profiles   # one module, by suffix
```

Plain asserts and a small runner, no framework. They cover the invariants
that fail silently: the galaxy-level train/val split, the normalization
statistics, the `Observation` geometry contract, and the analytic profile.

## Notebooks

`notebooks/00_nle_frontend.ipynb` trains a model and samples a posterior
through the API, end to end on a mock catalog.
`notebooks/01_nle_internals.ipynb` does the same work with every step written
out — the pipeline, the normalization, the flow, the training loop and the
sampler built by hand — and closes with a table mapping each step back onto
the API call that replaces it. `notebooks/example_config.py` and
`notebooks/CONFIG.md` document every config field, the first as comments and
the second as tables.
