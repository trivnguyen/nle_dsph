"""Per-star NLE on 11p_ZhaoABGCOM_v2 (free alpha-beta-gamma tracer).

Models p(v_los | log10 R, theta, tracer shape, log r_star, sigma_v)
with the production 8p recipe (`nle_8p_v4_prod.py`, arm A: NSF with 4
transforms of (128, 128), 8 bins, context MLP -> 64, batch 16384, 500k
steps, fp32), changed only where the dataset differs:

  labels   the same seven Zhao + Osipkov-Merritt parameters as 8p. The
           four extra dataset parameters are the *tracer's* light
           profile, which is measured photometrically, not inferred.
  cond     1 -> 4: stellar_log_rstar_kpc, stellar_alpha, stellar_beta,
           stellar_gamma, in the NPE config's order
           (`chebconv_11params_abg_v2.py`) so both kinds carry the same
           conditioning vector.

           `stellar_log_rstar_kpc` is the tracer's Zhao **scale**
           radius, which is what the simulator drew and what the shape
           parameters refer to; `stellar_log_rhalf_kpc` (also in the
           file) is the derived half-light radius and is deliberately
           NOT used. Conditioning on the half-light radius instead
           would be conditioning on a function of the four, and the
           flow would have to invert it.

  data     53 shards on tara = 5.25M galaxies, 500M stars (12.0 GB in
           float32); 500k steps at batch 16384 is ~16 epochs, the same
           as the 8p production run.
  context  13-d (log R, 7 theta, 4 tracer, sigma_v) vs 10-d for 8p.

    ./slurm/train_nle.sh configs/nle_11p_abg.py
"""

from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()
    config.seed_data = 4768732
    config.seed_training = 687765413

    config.data_root = '/home/tnguyen/scratch/projects/sbi_dsph/datasets'
    config.data_name = '11p_ZhaoABGCOM_v2'
    config.num_datasets = 53           # every shard present on tara
    config.init = 0
    config.labels = (
        'dm_alpha', 'dm_beta', 'dm_gamma', 'dm_log_rdm', 'dm_log_rho0',
        'df_beta0', 'df_log_ra',
    )
    config.cond_labels = (
        'stellar_log_rstar_kpc', 'stellar_alpha',
        'stellar_beta', 'stellar_gamma',
    )
    config.train_frac = 0.98
    config.max_val_galaxies = 50_000
    config.deterministic_val = True
    config.data_on_device = True
    config.num_workers = 0

    config.workdir = ('/home/tnguyen/scratch/projects/sbi_dsph/'
                      'trained_models/nle')
    config.wandb_project = 'nle_11p_ZhaoABGCOM_v2'
    config.entity = 'sbi_dsph'
    config.name = 'prod_53sh_A'
    config.tags = ['nle', 'production', 'abg']
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
