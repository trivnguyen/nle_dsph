# The config, field by field

Every field of `example_config.py`, grouped. The config file itself carries
the same notes as comments, so read whichever you prefer.

`00_nle_frontend.ipynb` section 2 loads a config and overrides a handful of
fields; this file is the reference for what the rest of them do.

Fields read with `config.get(...)` may simply be left out of a config. Those
are the OPTIONAL ones in `example_config.py`.

---

### Every field in the config

`example_config.py` has a comment on each of these too, so you can read it
there instead. Prune whichever of this you find redundant.

**Seeds.** Two separate streams on purpose.

| field | what it does |
|---|---|
| `seed_data` | the train/val split, and the normalization measured from it |
| `seed_training` | weight initialization and batch shuffling |

Keep `seed_data` fixed and change `seed_training` to measure how much of a
loss difference is just optimiser noise. If you change `seed_data` the
held-out set changes too, so two val losses are no longer comparable.

**Data.** Shards are read from `<data_root>/<data_name>/data.<i>.h5` for
`i` in `range(init, init + num_datasets)`.

| field | what it does |
|---|---|
| `data_root`, `data_name`, `num_datasets`, `init` | which shards to read |
| `labels` | the inferred parameters, **in the order the flow sees them** |
| `cond_labels` | the measured per-galaxy quantities |
| `train_frac` | fraction of *galaxies* used for training |
| `num_workers` | DataLoader workers; 0 (main process) is usually fastest here |
| `data_on_device` | load the whole dataset onto the GPU and batch by indexing |

The `labels` order propagates into posterior columns, prior boxes and corner
plot axes, so changing it invalidates every posterior you have already saved.
Treat it as part of the model's identity.

`train_frac` splits galaxies, not stars. A per-star split would put ~100
near-duplicate stars of the same galaxy on both sides, and the val loss would
read far too low.

**Bookkeeping.** Checkpoints and the config snapshot go to
`<workdir>/<wandb_project>/<run id>/`.

| field | what it does |
|---|---|
| `workdir` | where run directories are created |
| `wandb_project`, `entity`, `name`, `tags` | wandb bookkeeping |
| `debug` | `True` turns wandb off entirely. Fine for a smoke test |
| `checkpoint`, `reset_optimizer` | resume from a checkpoint |
| `save_top_k` | how many best-val checkpoints to keep |
| `patience` | stop after this many validations with no improvement |

`reset_optimizer=True` loads the weights but starts the optimiser and the LR
schedule fresh. That is what you want when fine-tuning on new data. `False`
continues the run exactly where it stopped.

**NOTE**: `patience` counts validations, not epochs, so it interacts with
`val_check_interval` below.

**Model.** Two flags decide what the flow actually models, and nothing else
does:

| `condition_on_position` | `condition_on_radius` | the model is |
|---|---|---|
| `False` | (ignored) | `p(x, y, v \| theta, ...)` |
| `True` | `False` | `p(v \| x, y, theta, ...)` |
| `True` | `True` | `p(v \| log10 R, theta, ...)` |

`condition_on_radius=True` with `condition_on_position=False` raises. The
radius option replaces the `(x, y)` context, it does not act on its own.

*Why condition on position at all.* `p(x, y, v)` has to reproduce the spatial
selection function of the survey, because the positions you observe are the
ones the survey chose to observe. Conditioning removes that: the likelihood
of `v` *given where the star is* does not depend on any position-only
selection. So you need no selection augmentation when training and no
correction at inference.

*Why the radius instead of `(x, y)`.* `p(v_los | R, theta)` is axisymmetric.
The `(x, y)` version has to learn that symmetry from data; the `R` version
gets it for free. It also makes the context independent of the sky
orientation the random projection happened to pick.

**NOTE**: it is `log10 R`, not `R`. The model takes `|pos|` and then
`log10` (with a `+1` offset). Projected radii span roughly `1e-3` to 30 kpc,
so without the log almost every star piles into the first percent of the
range.

| flow field | what it does |
|---|---|
| `flows.flow_type` | `'spline'` (zuko NSF) or `'maf'` |
| `flows.num_transforms` | how many transforms are stacked |
| `flows.hidden_features` | hidden layers predicting each transform |
| `flows.num_bins` | spline knots per transform (`'spline'` only) |
| `flows.activation` | `tanh`, `relu`, `gelu`, `sigmoid`, `identity` |
| `context_embedding.*` | optional MLP over the context before the flow |

Splines are the default because the velocity distribution is sharply peaked
and bounded, which rational splines fit with far fewer transforms than affine
ones. `tanh` is the safe activation inside a spline flow: it is bounded, so
it will not push knot positions somewhere the inverse cannot be found.

The context embedding is worth having when the context is wide. Here it is
10 numbers (7 theta + 1 cond + 1 sigma_v + 1 log R), and without the MLP
every transform re-learns the same mapping from those 10 numbers
independently.

**The observation pipeline (`pre_transforms`).** This is applied to every
training batch, freshly each time, so the model sees a different projection
and a different noise draw of the same star on every epoch.

| field | what it does |
|---|---|
| `apply_projection` | draw a random line of sight per batch and project |
| `projection_args.use_proper_motions` | keep the two sky-plane velocities too |
| `apply_uncertainty` | draw a per-star sigma and add Gaussian noise |
| `uncertainty_args.distribution_type` | which distribution the sigma comes from |
| `uncertainty_args.vel_idx` | which velocity column to perturb (0 = `v_los`) |
| `uncertainty_args.low_range`, `width_range` | the sigma range, km/s |

This is augmentation, not preprocessing. It is what teaches the flow that the
likelihood depends on `sigma_v`, and it is why one set of simulations can
stand in for many different observations.

`distribution_type` can be `'uniform'`, `'gaussian'`, `'jeffreys'`,
`'gamma'`, `'uniform_varied'` or `'jeffreys_varied'`. The `_varied` forms
redraw the distribution's *own* range per galaxy, so different galaxies get
different error regimes instead of all sharing one. `'jeffreys'` is
log-uniform in sigma.

**NOTE**: the pipeline is *not* applied at inference. Real data arrives
already projected and already noisy.

**Optimizer, schedule and loop.**

| field | what it does |
|---|---|
| `optimizer.name`, `lr`, `weight_decay` | `'AdamW'` or `'Adam'` |
| `scheduler.name` | `'WarmUpCosineAnnealingLR'`, `'CosineAnnealingLR'`, `'ReduceLROnPlateau'` |
| `scheduler.decay_steps` | length of the cosine decay, in steps |
| `scheduler.warmup_steps` | linear ramp from 0 at the start |
| `scheduler.eta_min` | **a multiplier on `lr`, not an absolute LR** |
| `scheduler.interval` | `'step'` or `'epoch'` |
| `accelerator` | `'gpu'`, `'cpu'` or `'mps'` |
| `train_batch_size`, `eval_batch_size` | eval can be larger, no gradients |
| `num_epochs`, `num_steps` | set exactly one, the other is `-1` |
| `gradient_clip_val` | clip the global gradient norm |
| `val_check_interval` | validate every N steps instead of once per epoch |
| `precision` | `'32-true'` or `'16-mixed'` |

`eta_min` catches people out: `1e-3` with `lr=1e-3` means the LR floor is
`1e-6`, not `1e-3`.

Set `decay_steps` equal to `num_steps` so the LR reaches its floor exactly as
training ends. Stopping early leaves the LR high, and the last checkpoint is
then undertrained for its step count.

Steps are the honest unit here, because "one epoch" is however many batches
the shard count happens to give.

`gradient_clip_val` matters more for flows than for most models: they produce
occasional very large gradients when a batch lands in a badly-fit tail, and
one such batch can undo an hour of training.
