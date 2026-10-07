"""Mixed-completeness test: per-star NLEs for different velocity subsets.

    ./slurm/queue_nle.sh configs/nle_8p_v4_mixtest.py:pm \
                         configs/nle_8p_v4_mixtest.py:v3d

`pm`  models p(v_x, v_y | log R, theta, log r_half, sigma_x, sigma_y)
      (the two sky-plane velocity components, in km/s);
`v3d` models p(v_los, v_x, v_y | ...) with all three components.
The line-of-sight-only model is `nle_8p_v4_errtest.py:noisy`
(run 6xaw28uw). All three: 2 shards, the T2 architecture, batch 16384,
30k steps, same data seed and galaxy split, the same per-component
error distribution. Test: diagnostics/coverage_test.py --mode mixed.
"""

from configs.nle_8p_v4_errtest import get_config as _errtest

_ERR = {
    'distribution_type': 'jeffreys_varied',
    'low_range': (0.01, 0.1),
    'width_range': (5.0, 30.0),
}


def get_config(arm='v3d'):
    arm = arm.lower()
    if arm not in ('pm', 'v3d'):
        raise ValueError("arm must be 'pm' or 'v3d'")
    config = _errtest('noisy')
    config.name = f'mixtest_{arm}'
    config.tags = ['nle', 'mixtest', arm]
    components = (1, 2) if arm == 'pm' else (0, 1, 2)
    config.pre_transforms.projection_args = {'components': components}
    # one noise sampler per kept velocity column (the field is a dict in
    # the base config, so it has to be re-created to hold a list)
    del config.pre_transforms.uncertainty_args
    config.pre_transforms.uncertainty_args = [
        dict(_ERR, vel_idx=i) for i in range(len(components))]
    return config
