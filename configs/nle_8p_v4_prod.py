"""Production per-star NLE on 8p_ZhaoPlumCOM_v4 (tara, single GB10).

Models p(v_los | log10 R, theta, log r_half, sigma_v) as in nle_8p_v4.py
(the 5-shard T2 run 37vbi58p), scaled up to the whole dataset and a
longer, larger-batch schedule. Design rationale and the measurements
behind every number: notes/nle_production_design.md.

Differences from nle_8p_v4.py
  data      5 -> 57 shards (5.48M galaxies, 540M stars), train_frac 0.98
  batch     4096 -> 16384 (GPU step time is launch-bound: ~flat up to 16k)
  steps     100k -> 500k (8.2B star-draws, ~15 epochs; 5-shard run: 0.41B)
  model     4x[128,128]/8 bins/ctx 64, unchanged: screening arm A tied
            B, C, D at -2.604 nats on 60k steps (2026-10-07), so the
            smallest architecture is kept
  val       deterministic held-out set capped at 50k galaxies, checked
            every 10k steps instead of once per epoch
`precision` and `matmul_precision` are implemented (training.py, nle.py).
Fields marked [needs code] are read only after the train_nle.py changes
listed in the design note (section 4); the current code ignores them, so
this file also runs unchanged on today's code (slower).
"""

from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()
    config.seed_data = 213189751
    config.seed_training = 792139710

    config.data_root = '/home/tnguyen/scratch/projects/sbi_dsph/datasets'
    config.data_name = '8p_ZhaoPlumCOM_v4'
    config.num_datasets = 57           # every shard present on tara
    config.init = 0
    config.labels = (
        'dm_alpha', 'dm_beta', 'dm_gamma', 'dm_log_rdm', 'dm_log_rho0',
        'df_beta0', 'df_log_ra',
    )
    config.cond_labels = ('stellar_log_rstar_kpc',)
    config.train_frac = 0.98
    config.max_val_galaxies = 50_000   # [needs code] cap the 2% val split
    config.deterministic_val = True    # [needs code] fixed projection+noise
    config.data_on_device = True       # [needs code] whole dataset on GPU,
                                       # batches gathered by index (no
                                       # per-star Python DataLoader)
    config.num_workers = 0

    config.workdir = ('/home/tnguyen/scratch/projects/sbi_dsph/'
                      'trained_models/nle')
    config.wandb_project = 'nle_8p_ZhaoPlumCOM_v4'
    config.entity = 'sbi_dsph'
    config.name = 'prod_57sh_A'
    config.tags = ['nle', 'production', '57shards']
    config.debug = False
    config.checkpoint = None
    config.id = None  # wandb run id to resume into (with checkpoint)
    config.reset_optimizer = False
    config.enable_progress_bar = True
    config.save_top_k = 3

    config.model = model = ConfigDict()
    model.condition_on_position = True
    model.condition_on_radius = True
    model.flows = ConfigDict()
    model.flows.flow_type = 'spline'
    model.flows.num_transforms = 4
    model.flows.hidden_features = [128, 128]
    model.flows.num_bins = 8
    model.flows.activation = 'tanh'
    model.context_embedding = ConfigDict()
    model.context_embedding.hidden_sizes = [128, 128]
    model.context_embedding.output_size = 64
    model.context_embedding.act_name = 'relu'
    model.context_embedding.batch_norm = False
    model.context_embedding.dropout = 0.0

    # Observation pipeline: unchanged from the T2 run. Error draws cover
    # 0.01-30 km/s, which brackets every Paper-4 catalogue.
    config.pre_transforms = pre_transforms = ConfigDict()
    pre_transforms.apply_projection = True
    pre_transforms.projection_args = {'use_proper_motions': False}
    pre_transforms.apply_uncertainty = True
    pre_transforms.uncertainty_args = {
        'distribution_type': 'jeffreys_varied',
        'vel_idx': 0,
        'low_range': (0.01, 0.1),
        'width_range': (5.0, 30.0),
    }

    config.optimizer = optimizer = ConfigDict()
    optimizer.name = 'AdamW'
    optimizer.lr = 1e-3
    optimizer.weight_decay = 1e-4

    config.scheduler = scheduler = ConfigDict()
    scheduler.name = 'WarmUpCosineAnnealingLR'
    scheduler.decay_steps = 500_000
    scheduler.warmup_steps = 5_000
    scheduler.eta_min = 1e-3           # multiplier: final lr = 1e-6
    scheduler.interval = 'step'

    config.accelerator = 'gpu'
    config.train_batch_size = 16384
    config.eval_batch_size = 65536
    config.num_epochs = -1
    config.num_steps = scheduler.decay_steps
    config.val_check_interval = 10_000  # [needs code] steps between val runs
    config.patience = 20               # val checks (= 200k steps) w/o gain
    config.gradient_clip_val = 0.5
    config.matmul_precision = 'highest'  # full-fp32 matmuls (TF32 off)
    # '32-true' or '16-mixed'. fp16 autocast is 1.35x faster at this batch
    # size with a 0.002-nat median per-star error against fp32 (bf16 is
    # not offered: 0.03-0.5 nats/star). Validation and inference always
    # run in fp32. Production default stays fp32; flip to '16-mixed' for
    # screening arms and extra seeds (design note, section 8).
    config.precision = '32-true'
    return config
