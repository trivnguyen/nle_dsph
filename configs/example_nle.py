"""Example training config for per-star NLE (train_nle.py)."""

from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()

    # seeding
    config.seed_data = 42
    config.seed_training = 24

    ### DATA CONFIGURATION ###
    config.data_root = '/scratch/tvnguyen/datasets'
    config.data_name = '8p_ZhaoPlumCOM'
    config.num_datasets = 1
    config.init = 0
    # Inferred parameters (theta).
    config.labels = (
        'dm_alpha', 'dm_beta', 'dm_gamma', 'dm_log_rdm', 'dm_log_rho0',
        'df_beta0', 'df_log_ra',
    )
    # Measured conditioning variables (cond), fixed or MC-sampled at
    # inference; here the stellar half-light radius.
    config.cond_labels = ('stellar_log_rstar_kpc',)
    config.train_frac = 0.8
    config.num_workers = 0
    # Cap the stars used to measure the pos/vel/vel_error normalization.
    # Those statistics are min/max and the pipeline redraws projection
    # and noise each call, so no finite sample pins them down; a cap
    # narrows them, which is a real effect, not just a speed-up. Leave
    # unset to use every training star. theta/cond stay exact either way.
    # config.norm_max_stars = 5_000_000

    ### LOGGING AND WANDB CONFIGURATION ###
    config.workdir = './runs'
    config.wandb_project = 'nle_dsph_example'
    config.entity = None
    config.debug = False
    config.checkpoint = None  # Path to an NLE checkpoint to resume from
    config.reset_optimizer = True
    config.enable_progress_bar = True

    ### MODEL CONFIGURATION ###
    config.model = model = ConfigDict()

    # Velocity is always modeled. False: p(x, y, v | theta, err).
    # True: p(v | x, y, theta, err) (all sky position becomes context).
    model.condition_on_position = False

    # Conditional normalizing flow p(target | context).
    model.flows = ConfigDict()
    model.flows.flow_type = 'spline'
    model.flows.num_transforms = 6
    model.flows.hidden_features = [64, 64]
    model.flows.num_bins = 8
    model.flows.activation = 'tanh'

    # Optional single MLP over the combined context before the flow.
    # Uncomment to enable:
    # model.context_embedding = ConfigDict()
    # model.context_embedding.hidden_sizes = [64, 64]
    # model.context_embedding.output_size = 32
    # model.context_embedding.act_name = 'relu'
    # model.context_embedding.batch_norm = False
    # model.context_embedding.dropout = 0.0

    ### OBSERVATION PIPELINE (pre_transforms) ###
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

    ### OPTIMIZER AND SCHEDULER CONFIGURATION ###
    config.optimizer = optimizer = ConfigDict()
    optimizer.name = 'AdamW'
    optimizer.lr = 1e-3
    optimizer.weight_decay = 1e-4

    config.scheduler = scheduler = ConfigDict()
    scheduler.name = 'WarmUpCosineAnnealingLR'
    scheduler.decay_steps = 20_000
    scheduler.warmup_steps = int(0.05 * scheduler.decay_steps)
    scheduler.eta_min = 1e-6
    scheduler.interval = 'step'

    ### TRAINING CONFIGURATION ###
    config.accelerator = 'gpu'
    config.train_batch_size = 1024
    config.eval_batch_size = 1024
    config.num_epochs = 30
    config.num_steps = -1
    config.patience = 15
    config.gradient_clip_val = 0.5
    config.save_top_k = 3

    return config
