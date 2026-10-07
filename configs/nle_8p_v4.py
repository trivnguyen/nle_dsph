"""Per-star NLE on the 8p_ZhaoPlumCOM_v4 dataset (gap test T2).

Models p(v_los | log10 R, theta, log r_star, sigma_v). Conditioning on
the projected radius makes the per-star likelihood independent of any
position-only selection function, so no selection augmentation is
needed. 20 shards = 2M galaxies ~ 200M stars.
"""

from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()
    config.seed_data = 213189751
    config.seed_training = 792139710

    config.data_root = '/home/tnguyen/scratch/projects/sbi_dsph/datasets'
    config.data_name = '8p_ZhaoPlumCOM_v4'
    config.num_datasets = 20
    config.init = 0
    config.labels = (
        'dm_alpha', 'dm_beta', 'dm_gamma', 'dm_log_rdm', 'dm_log_rho0',
        'df_beta0', 'df_log_ra',
    )
    config.cond_labels = ('stellar_log_rstar_kpc',)
    config.train_frac = 0.95
    config.num_workers = 0

    config.workdir = ('/home/tnguyen/scratch/projects/sbi_dsph/'
                      'trained_models/nle')
    config.wandb_project = 'nle_8p_ZhaoPlumCOM_v4'
    config.entity = 'sbi_dsph'
    config.name = None
    config.tags = ['nle', 'gap']
    config.debug = False
    config.checkpoint = None
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
    scheduler.decay_steps = 100_000
    scheduler.warmup_steps = 2_000
    scheduler.eta_min = 1e-3
    scheduler.interval = 'step'

    config.accelerator = 'gpu'
    config.train_batch_size = 4096
    config.eval_batch_size = 8192
    config.num_epochs = -1
    config.num_steps = scheduler.decay_steps
    config.patience = 20
    config.gradient_clip_val = 0.5
    return config
