# nle_dsph

**Neural likelihood estimation for dwarf spheroidal stellar kinematics.**

`nle_dsph` infers dark-matter density profiles from the kinematics of dwarf
spheroidal member stars, without writing down an analytic likelihood. A
conditional normalizing flow is trained on simulated galaxies to learn the
likelihood of a single star,

$$p\big(v_\mathrm{los} \mid \log_{10} R,\ \boldsymbol{\theta},\ \log r_\star,\ \sigma_v\big)$$

where $\boldsymbol{\theta}$ are the dark-matter and orbital-anisotropy
parameters, $R$ is the projected radius, $\log r_\star$ the measured
half-light radius and $\sigma_v$ the per-star velocity uncertainty. Which
observables the flow models and which it conditions on is a configuration
choice: modeling the sky positions alongside the velocity gives
$p(x, y, v_\mathrm{los} \mid \boldsymbol{\theta}, \ldots)$ instead, and where
proper motions are measured $v_\mathrm{los}$ is replaced by the full velocity
vector or any subset of it.

Stars are taken to be independent given $\boldsymbol{\theta}$, so the
log-likelihood of a catalog is the sum over its stars. Because the likelihood
is per star, one trained flow serves catalogs of any size, and the prior is
supplied at inference rather than baked in at training time.

## Installation

There is no install step yet. Put the repository root on `PYTHONPATH`, or work
from it. Python 3.10 or newer is required, along with
[PyTorch](https://pytorch.org),
[Lightning](https://lightning.ai/docs/pytorch/stable/),
[zuko](https://zuko.readthedocs.io), numpy, scipy, h5py,
[ml_collections](https://github.com/google/ml_collections), absl-py and tqdm.
[pocomc](https://pocomc.readthedocs.io) is needed for posterior sampling and
[wandb](https://docs.wandb.ai) for logging; set `config.debug = True` to train
without the latter.

```bash
git clone git@github.com:trivnguyen/nle_dsph.git
cd nle_dsph
python tests/run_tests.py
```

## Quick start

A run is described entirely by a config file — an ordinary Python module
exposing `get_config()`. Train one from the command line:

```bash
python scripts/train_nle.py --config configs/nle_8p_v4.py
```

Any field can be overridden with `--config.<field>=<value>`, so a parameter
sweep needs no per-run file edit.

The same thing from Python, through to a posterior:

```python
import numpy as np
import pocomc as pc
from scipy.stats import truncnorm, uniform

from nle_dsph import api

config = api.load_config('configs/nle_8p_v4.py', num_steps=2000)
run_dir = api.train(config)
model, snapshot = api.load_model(run_dir)

obs = api.Observation(
    R_proj_kpc=R_proj_kpc,        # (N,) projected radius, kpc
    vlos_kms=vlos_kms,            # (N,) line-of-sight velocity, km/s
    vlos_err_kms=vlos_err_kms,    # (N,) per-star sigma, km/s
    log_rstar_kpc=np.log10(rhalf_kpc),
    name='my_dwarf',
)

# [ labels..., v_sys, log_rstar ], in that order.
dists = [uniform(lo, hi - lo) for lo, hi in zip(prior_min, prior_max)]
dists.append(uniform(-50.0, 100.0))
dists.append(truncnorm(a, b, loc=mu, scale=sigma))

result = api.sample_posterior(model, obs, pc.Prior(dists), labels=labels)
api.save_posterior(out_dir, obs, result)
```

Training writes a self-contained run directory holding the checkpoints, a
config snapshot and a copy of the config file. The normalization statistics
live inside the checkpoint, so a model can never be paired with statistics
from a different dataset.

## Documentation

Two notebooks in `notebooks/` are the tutorials:
`00_nle_frontend.ipynb` trains a model and samples a posterior through the
API, end to end on a mock catalog; `01_nle_internals.ipynb` does the same work
with every step written out by hand, and closes with a table mapping each step
back onto the API call that replaces it.

Every config field is documented twice, as comments in
`notebooks/example_config.py` and as tables in `notebooks/CONFIG.md`.

## Attribution

A paper describing the method is in preparation. Please check back before
citing this code in published work.

## Contributing

Bug reports and feature requests go to the
[issue tracker](https://github.com/trivnguyen/nle_dsph/issues).
