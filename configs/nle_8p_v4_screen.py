"""Screening arms for the production NLE (tara).

Usage (one arm per process; two arms fit side by side on the GB10):
    ./slurm/train_nle.sh configs/nle_8p_v4_screen.py:A
    ./slurm/train_nle.sh configs/nle_8p_v4_screen.py:B

Every arm: 57 shards, batch 16384, 60k steps (~1B star-draws, 2.4x the
5-shard run's total), same data seed and the same deterministic val set,
so val/loss is comparable at the 0.001 nat level. Decision rule and the
reading of each arm: notes/nle_production_design.md, section 3.

    A  control: the T2 architecture (4x[128,128], 8 bins, ctx 64)
    B  wider:   5x[256,256], 16 bins, ctx [256,256]->128   (default)
    C  deeper:  8x[256,256], 16 bins, ctx 128
    D  B with SiLU conditioner activation instead of tanh
    E  B without the context MLP (raw 10-d context into the flow)
"""

from configs.nle_8p_v4_prod import get_config as _prod

ARMS = {
    'A': dict(num_transforms=4, hidden_features=[128, 128], num_bins=8,
              activation='tanh', ctx_hidden=[128, 128], ctx_out=64),
    'B': dict(num_transforms=5, hidden_features=[256, 256], num_bins=16,
              activation='tanh', ctx_hidden=[256, 256], ctx_out=128),
    'C': dict(num_transforms=8, hidden_features=[256, 256], num_bins=16,
              activation='tanh', ctx_hidden=[256, 256], ctx_out=128),
    'D': dict(num_transforms=5, hidden_features=[256, 256], num_bins=16,
              activation='silu', ctx_hidden=[256, 256], ctx_out=128),
    'E': dict(num_transforms=5, hidden_features=[256, 256], num_bins=16,
              activation='tanh', ctx_hidden=None, ctx_out=None),
}


def get_config(arm='B'):
    arm = arm.upper()
    if arm not in ARMS:
        raise ValueError(f'unknown arm {arm!r}; choose from {sorted(ARMS)}')
    a = ARMS[arm]
    config = _prod()
    config.name = f'screen_{arm}'
    config.tags = ['nle', 'screening', f'arm_{arm}']
    config.seed_training = 111

    config.model.flows.num_transforms = a['num_transforms']
    config.model.flows.hidden_features = a['hidden_features']
    config.model.flows.num_bins = a['num_bins']
    config.model.flows.activation = a['activation']
    if a['ctx_hidden'] is None:
        del config.model.context_embedding
    else:
        config.model.context_embedding.hidden_sizes = a['ctx_hidden']
        config.model.context_embedding.output_size = a['ctx_out']

    config.scheduler.decay_steps = 60_000
    config.scheduler.warmup_steps = 2_000
    config.num_steps = config.scheduler.decay_steps
    config.val_check_interval = 5_000
    config.patience = 100  # never stop a screening arm early
    config.precision = '32-true'  # user's call 2026-10-07: fp32 everywhere
    return config
