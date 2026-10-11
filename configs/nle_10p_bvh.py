"""Per-star NLE on 10p_ZhaoPlumJeansBVH_ktwin (BvH anisotropy).

Models p(v_los | log10 R, theta, log r_star, sigma_v) with the same
recipe as the production 8p run (`nle_8p_v4_prod.py`, arm A: NSF with 4
transforms of (128, 128), 8 bins, context MLP -> 64, batch 16384, 500k
steps, fp32), changed only where the dataset differs:

  labels   7 -> 9: the Zhao density parameters plus the four Baes & van
           Hese anisotropy parameters (df_betasym0, df_betasyminf,
           df_log_r0, df_n_aniso). The *symmetrized* beta is the label,
           as in the NPE config; the raw df_beta0/df_betainf reach -18
           and are not modeled.
  data     24 shards on tara = 2.32M galaxies, 226M stars (5.4 GB in
           float32), against 57 shards / 5.5M / 540M for 8p. 500k steps
           at batch 16384 is then ~36 epochs rather than ~15; the
           distinct-theta count, not the epoch count, is what limits
           what the model can learn.
  context  12-d (log R, 9 theta, log r_star, sigma_v) vs 10-d for 8p.

Known caveat, inherited from the dataset and shared with the NPE run on
it: the ktwin sampler clamps the kurtosis, and `frac_clamped` is
non-zero for 57% of galaxies (mean 0.23). Those galaxies are kept, so
the likelihood is learned for the simulator as it stands - filtering
here would train against a different prior from the NPE's.

    ./slurm/train_nle.sh configs/nle_10p_bvh.py
"""

from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()
    config.seed_data = 213189751
    config.seed_training = 792139710

    config.data_root = '/home/tnguyen/scratch/projects/sbi_dsph/datasets'
    config.data_name = '10p_ZhaoPlumJeansBVH_ktwin'
    config.num_datasets = 24           # every shard present on tara
    config.init = 0
    config.labels = (
        'dm_alpha', 'dm_beta', 'dm_gamma', 'dm_log_rdm', 'dm_log_rho0',
        'df_betasym0', 'df_betasyminf', 'df_log_r0', 'df_n_aniso',
    )
    config.cond_labels = ('stellar_log_rstar_kpc',)
    config.train_frac = 0.98
    config.max_val_galaxies = 50_000
    config.deterministic_val = True
    config.data_on_device = True
    config.num_workers = 0

    config.workdir = ('/home/tnguyen/scratch/projects/sbi_dsph/'
                      'trained_models/nle')
    config.wandb_project = 'nle_10p_ZhaoPlumJeansBVH_ktwin'
    config.entity = 'sbi_dsph'
    config.name = 'prod_24sh_A'
    config.tags = ['nle', 'production', 'bvh']
    config.debug = False
    config.checkpoint = ''
    config.id = ''
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
    scheduler.eta_min = 1e-3
    scheduler.interval = 'step'

    config.accelerator = 'gpu'
    config.train_batch_size = 16384
    config.eval_batch_size = 65536
    config.num_epochs = -1
    config.num_steps = scheduler.decay_steps
    config.val_check_interval = 10_000
    config.patience = 20
    config.gradient_clip_val = 0.5
    config.matmul_precision = 'highest'
    config.precision = '32-true'
    return config
