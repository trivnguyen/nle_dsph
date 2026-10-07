"""Error-convolution test: noiseless vs noisy per-star NLE on small data.

    ./slurm/train_nle.sh configs/nle_8p_v4_errtest.py:clean
    ./slurm/train_nle.sh configs/nle_8p_v4_errtest.py:noisy

`clean` models p(v_true | log R, theta, log r_half) with no measurement
noise and no sigma in the context; `noisy` is the usual
p(v_obs | log R, theta, log r_half, sigma_v); `joint` models
p(x, y, v_obs | theta, log r_half, sigma_v) with the sky position as a
target (the control for the selection-function test). Both: 2 shards (~200k
galaxies, 20M stars), the T2 architecture, batch 16384, 30k steps, same
data seed, same galaxy split. The check
(diagnostics/check_error_convolution.py):
convolving `clean` with N(0, sigma^2) should reproduce `noisy`.
"""

from configs.nle_8p_v4_prod import get_config as _prod


def get_config(arm='noisy'):
    arm = arm.lower()
    if arm not in ('clean', 'noisy', 'joint'):
        raise ValueError("arm must be 'clean', 'noisy' or 'joint'")
    config = _prod()
    config.name = f'errtest_{arm}'
    config.tags = ['nle', 'errtest', arm]
    config.wandb_project = 'nle_8p_ZhaoPlumCOM_v4'
    config.seed_training = 333
    config.num_datasets = 2
    config.train_frac = 0.95
    config.max_val_galaxies = 10_000

    config.model.flows.num_transforms = 4
    config.model.flows.hidden_features = [128, 128]
    config.model.flows.num_bins = 8
    config.model.context_embedding.hidden_sizes = [128, 128]
    config.model.context_embedding.output_size = 64

    if arm == 'clean':
        config.pre_transforms.apply_uncertainty = False
        config.pre_transforms.uncertainty_args = None
    if arm == 'joint':
        # p(x, y, v_los | theta, log r_half, sigma): position modeled, not
        # conditioned on. Control for the selection-function comparison.
        config.model.condition_on_position = False
        config.model.condition_on_radius = False

    config.scheduler.decay_steps = 30_000
    config.scheduler.warmup_steps = 1_000
    config.num_steps = config.scheduler.decay_steps
    config.val_check_interval = 2_500
    config.patience = 100
    return config
