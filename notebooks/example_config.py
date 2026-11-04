"""Documented example config for per-star NLE.

This is the one config in the tree that carries prose. Copy it, strip
the comments, and keep only the fields you change -- every field below
has a default behavior described here, and a stripped copy is easier
to read than a copy that repeats this file.

Anything read with `config.get(...)` may simply be left out. Those are
marked OPTIONAL below; the rest are required.

The model this config builds is

    p( target | context )

and which observables land in `target` versus `context` is decided by
`model.condition_on_position` and `model.condition_on_radius` alone.
See the MODEL section.
"""

from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()

    # ---------------------------------------------------------------
    # SEEDS
    # ---------------------------------------------------------------
    # Two separate streams, deliberately. `seed_data` fixes the
    # train/val split (which galaxies land in which set) and the
    # normalization measured from it; `seed_training` fixes weight
    # initialization and batch shuffling. Re-running with the same
    # `seed_data` and a new `seed_training` measures optimiser noise
    # at a fixed dataset; changing `seed_data` changes the held-out
    # set, so the two val losses are no longer comparable.
    config.seed_data = 213189751
    config.seed_training = 792139710

    # ---------------------------------------------------------------
    # DATA
    # ---------------------------------------------------------------
    # Shards are read from `<data_root>/<data_name>/data.<i>.h5` for
    # i in range(init, init + num_datasets).
    config.data_root = (
        '/home/jupyter-tvnguyen/scratch/sbi_dsph/training_datasets')
    config.data_name = '8p_ZhaoPlumCOM_v4'
    config.num_datasets = 1
    config.init = 0                       # OPTIONAL, default 0

    # The inferred parameters, in the order the flow will see them.
    # This order is the order of everything downstream: posterior
    # columns, prior boxes, corner-plot axes. Changing it silently
    # invalidates every saved posterior, so treat it as part of the
    # model's identity rather than a formatting choice.
    config.labels = (
        'dm_alpha', 'dm_beta', 'dm_gamma', 'dm_log_rdm', 'dm_log_rho0',
        'df_beta0', 'df_log_ra',
    )

    # Measured per-galaxy quantities. OPTIONAL: omit for a model with
    # no conditioning axis at all.
    #
    # These reach the flow as context exactly like `labels` do. The
    # difference is where their prior comes from at inference: a
    # `cond` gets the measurement (photometry, with its error bar), a
    # label can only get the simulation box. Being measured does not
    # make a `cond` fixed -- it is still sampled, over that error bar.
    config.cond_labels = ('stellar_log_rstar_kpc',)

    # Fraction of *galaxies* (not stars) used for training. The split
    # is by galaxy so no galaxy contributes stars to both sets; with
    # ~100 stars each, a per-star split would put near-duplicates on
    # both sides and the val loss would read far too low.
    config.train_frac = 0.95

    # DataLoader worker processes. 0 means load in the main process,
    # which is usually fastest here: the per-star batches are small
    # tensor gathers, and workers cost more to fork than they save.
    config.num_workers = 0

    # OPTIONAL, default False. Load the whole dataset onto the
    # accelerator once and batch by indexing, instead of going through
    # a DataLoader. Much faster per step when the data fits in
    # (unified) memory; needs the dataset to actually fit.
    # config.data_on_device = False

    # OPTIONAL. Cap the number of val galaxies, and freeze one
    # projection + noise draw for them, so the val loss is not itself
    # noisy from epoch to epoch.
    # config.max_val_galaxies = 20_000
    # config.deterministic_val = True

    # ---------------------------------------------------------------
    # BOOKKEEPING AND LOGGING
    # ---------------------------------------------------------------
    # Checkpoints and the config snapshot land in
    # `<workdir>/<wandb_project>/<run id>/`. That directory is the
    # deliverable: it carries the weights, the normalization, and this
    # config, so it can be read back with nothing else.
    config.workdir = str(
        '/home/jupyter-tvnguyen/scratch/sbi_dsph/trained_models/nle')
    config.wandb_project = 'nle_8p_ZhaoPlumCOM_v4'   # OPTIONAL
    config.entity = 'sbi_dsph'            # OPTIONAL, wandb team
    config.name = None                    # OPTIONAL, wandb run name
    config.tags = ['nle']                 # OPTIONAL

    # True turns wandb off entirely (mode='disabled'). Good for a
    # notebook smoke test; leave False for anything you keep, or the
    # run has no logged history.
    config.debug = False                  # OPTIONAL

    # OPTIONAL. Force 'online' / 'offline' rather than letting the
    # launcher probe whether the wandb API is reachable.
    # config.wandb_mode = 'offline'

    # OPTIONAL. Resume: `checkpoint` is a file name inside the run's
    # `checkpoints/`, or an absolute path. `reset_optimizer=True`
    # loads the weights but starts the optimiser and LR schedule
    # fresh, which is what you want when fine-tuning on new data;
    # False continues the run exactly where it stopped.
    config.checkpoint = None
    config.reset_optimizer = False

    config.enable_progress_bar = True     # OPTIONAL
    config.save_top_k = 3                 # OPTIONAL, best-val ckpts kept

    # ---------------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------------
    config.model = model = ConfigDict()

    # These two flags decide what the flow models. Nothing else does.
    #
    #   condition_on_position  condition_on_radius   the model is
    #   ---------------------  -------------------   -------------------
    #   False                  (ignored)             p(x, y, v | theta, ...)
    #   True                   False                 p(v | x, y, theta, ...)
    #   True                   True                  p(v | log10 R, theta, ...)
    #
    # `condition_on_radius=True` without `condition_on_position=True`
    # raises; the radius option replaces the (x, y) context, it does
    # not act on its own.
    #
    # Why condition rather than model the positions: p(x, y, v) has to
    # reproduce the spatial selection function of the survey, because
    # the positions you observe are the ones the survey chose to
    # observe. Conditioning on position removes that -- the per-star
    # likelihood of v given where the star is does not depend on any
    # position-only selection -- so no selection augmentation is
    # needed at training time and no correction at inference.
    #
    # Why the radius rather than (x, y): p(v_los | R, theta) is
    # axisymmetric, so the (x, y) version has to learn a symmetry the
    # R version gets for free, and the context stops depending on the
    # sky orientation of the projection.
    #
    # Note it is log10 R, not R: the model takes |pos|, then
    # log10 (+1 offset). Projected radii span ~1e-3 to ~30 kpc, so the
    # log is what makes the context roughly uniform over its range
    # instead of piling every star into the first percent of it.
    model.condition_on_position = True
    model.condition_on_radius = True

    # The conditional normalizing flow itself.
    model.flows = ConfigDict()
    # 'spline' is a neural spline flow (zuko NSF), 'maf' a masked
    # autoregressive flow. Splines are the default here: the velocity
    # distribution is sharply peaked and bounded, which rational
    # splines fit with far fewer transforms than affine ones.
    model.flows.flow_type = 'spline'
    # Number of stacked transforms. More is more expressive and
    # proportionally slower; 3-4 is enough for a 1-D target.
    model.flows.num_transforms = 4
    # Hidden layers of the network that predicts each transform's
    # parameters from the context.
    model.flows.hidden_features = [128, 128]
    # Spline knots per transform. Only used by 'spline'. More knots
    # resolve sharper features and cost more to fit.
    model.flows.num_bins = 8
    # 'tanh', 'relu', 'gelu', 'sigmoid' or 'identity'. tanh is the
    # safe choice inside a spline flow: it is bounded, so it will not
    # push knot positions somewhere the inverse cannot be found.
    model.flows.activation = 'tanh'

    # OPTIONAL. One MLP over the whole context before the flow sees
    # it. Without it the flow is handed the raw concatenated context.
    # Worth having when the context is wide (theta + cond + sigma_v is
    # 10 numbers here): every transform would otherwise re-learn the
    # same mapping from those 10 numbers independently.
    model.context_embedding = ConfigDict()
    model.context_embedding.hidden_sizes = [128, 128]
    model.context_embedding.output_size = 64   # what the flow receives
    model.context_embedding.act_name = 'relu'
    model.context_embedding.batch_norm = False
    model.context_embedding.dropout = 0.0

    # ---------------------------------------------------------------
    # OBSERVATION PIPELINE (pre_transforms)
    # ---------------------------------------------------------------
    # Applied to every training batch, freshly each time, so the model
    # sees a different projection and a different noise draw of the
    # same star on every epoch. This is augmentation, not
    # preprocessing: it is what teaches the flow that the likelihood
    # depends on sigma_v, and it is why the same simulation can stand
    # in for many observations.
    #
    # At inference the pipeline is NOT applied -- real data arrives
    # already projected and already noisy.
    config.pre_transforms = pre_transforms = ConfigDict()

    # Draw a random line of sight per batch and project onto it.
    pre_transforms.apply_projection = True
    # use_proper_motions=False keeps only v_los. True also keeps the
    # two sky-plane velocity components.
    pre_transforms.projection_args = {'use_proper_motions': False}

    # Draw a per-star measurement sigma, add Gaussian noise to that
    # velocity component, and record the sigma so the flow can
    # condition on it. Setting this False trains p(v_true | ...), a
    # noiseless likelihood that then takes `vel_error=None`.
    pre_transforms.apply_uncertainty = True
    pre_transforms.uncertainty_args = {
        # 'uniform', 'gaussian', 'jeffreys', 'gamma', 'uniform_varied'
        # or 'jeffreys_varied'. The '_varied' forms redraw the
        # *distribution's own* range per galaxy, so different galaxies
        # get different error regimes rather than all sharing one.
        # 'jeffreys' is log-uniform in sigma, the scale-invariant
        # choice for a quantity known only by order of magnitude.
        'distribution_type': 'jeffreys_varied',
        # Which velocity column to perturb. 0 is v_los.
        'vel_idx': 0,
        # For '_varied': the galaxy's sigma floor is drawn from
        # `low_range`, and its width from `width_range`, both km/s.
        'low_range': (0.01, 0.1),
        'width_range': (5.0, 30.0),
    }

    # ---------------------------------------------------------------
    # OPTIMIZER AND SCHEDULE
    # ---------------------------------------------------------------
    config.optimizer = optimizer = ConfigDict()
    optimizer.name = 'AdamW'              # or 'Adam'
    optimizer.lr = 1e-3
    optimizer.weight_decay = 1e-4

    config.scheduler = scheduler = ConfigDict()
    # 'WarmUpCosineAnnealingLR', 'CosineAnnealingLR' or
    # 'ReduceLROnPlateau'.
    scheduler.name = 'WarmUpCosineAnnealingLR'
    # Length of the cosine decay, in steps. Normally set equal to
    # `num_steps` so the LR reaches its floor exactly as training
    # ends -- stopping early leaves the LR high and the last
    # checkpoint undertrained for its step count.
    scheduler.decay_steps = 100_000
    # Linear ramp from 0 at the start. Guards the first few hundred
    # steps, where a cold flow can take a step that it cannot undo.
    scheduler.warmup_steps = 2_000
    # NOTE: a *multiplier* on `lr`, not an absolute learning rate.
    # 1e-3 here with lr=1e-3 means the floor is 1e-6.
    scheduler.eta_min = 1e-3
    # 'step' steps the schedule every optimiser step; 'epoch' once per
    # epoch. One epoch here is thousands of steps, so 'step'.
    scheduler.interval = 'step'

    # ---------------------------------------------------------------
    # TRAINING LOOP
    # ---------------------------------------------------------------
    config.accelerator = 'gpu'            # 'gpu', 'cpu' or 'mps'
    config.train_batch_size = 4096
    config.eval_batch_size = 8192         # no gradients, so it can be larger

    # Set exactly one of these; the other is -1. Steps are the honest
    # unit for this model, because "one epoch" is however many batches
    # the shard count happens to give.
    config.num_epochs = -1
    config.num_steps = scheduler.decay_steps

    # Early stopping: stop after this many validations with no
    # improvement in val/loss. Counted in validations, not epochs, so
    # it interacts with `val_check_interval` below.
    config.patience = 20

    # Clip the global gradient norm. Flows produce occasional very
    # large gradients when a batch lands in a badly-fit tail; without
    # this one such batch can undo an hour of training.
    config.gradient_clip_val = 0.5

    # OPTIONAL. Validate every N steps instead of once per epoch.
    # Needed whenever `num_steps` is smaller than one epoch: the
    # checkpoint callbacks monitor val/loss, so with no validation
    # there is no monitored checkpoint. One epoch on 20 shards is
    # ~50k steps, so a 10k-step run that does not set this never
    # validates at all.
    # config.val_check_interval = 2_000

    # OPTIONAL, default '32-true'. '16-mixed' runs matmuls in fp16
    # under autocast and is ~1.4x faster at large batch; weights and
    # the validation step stay fp32. 'bf16-mixed' is deliberately not
    # offered -- its 8-bit mantissa costs far too much precision on
    # narrow velocity peaks.
    # config.precision = '32-true'

    # OPTIONAL, default 'highest'. 'high' enables TF32 matmuls. No
    # speed-up here (the step is launch-bound) and it loses spline
    # knot precision, so leave it alone.
    # config.matmul_precision = 'highest'

    return config
