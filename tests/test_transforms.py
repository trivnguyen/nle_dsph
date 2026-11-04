"""Unit tests for the per-star container and the observation pipeline."""

import numpy as np
import torch

from nle_dsph.transforms import (
    RandomProjection, StarBatch, build_transformation, compute_field_norm,
)
from run_tests import close, raises

UNCERTAINTY_ARGS = {
    'distribution_type': 'jeffreys_varied',
    'vel_idx': 0,
    'low_range': (0.01, 0.1),
    'width_range': (5.0, 30.0),
}


def make_batch(n_stars=256, seed=0):
    """A raw Cartesian batch, as the dataloaders yield it."""
    generator = torch.Generator().manual_seed(seed)
    return StarBatch(
        pos=torch.randn(n_stars, 3, generator=generator),
        vel=10.0 * torch.randn(n_stars, 3, generator=generator),
        theta=torch.randn(n_stars, 7, generator=generator),
        cond=torch.randn(n_stars, 1, generator=generator),
    )


def test_clone_does_not_alias_the_original():
    """Transforms clone before mutating; a shared buffer would corrupt it."""
    batch = make_batch(8)
    copy = batch.clone()

    copy.pos[0, 0] = 999.0
    copy.vel[0, 0] = 999.0

    assert batch.pos[0, 0] != 999.0
    assert batch.vel[0, 0] != 999.0


def test_clone_preserves_none_fields():
    """vel_error is None before the pipeline runs and must stay None."""
    copy = StarBatch(pos=torch.zeros(4, 3), vel=torch.zeros(4, 3)).clone()

    assert copy.vel_error is None
    assert copy.theta is None
    assert copy.cond is None


def test_projection_does_not_mutate_the_input_batch():
    """The loader reuses its tensors across epochs."""
    batch = make_batch(32)
    before = batch.pos.clone()

    RandomProjection()(batch)

    assert torch.equal(batch.pos, before)


def test_projected_radius_never_exceeds_the_three_d_radius():
    """pos loses the line-of-sight axis, so R_proj <= r, never more."""
    batch = make_batch(256)

    out = RandomProjection()(batch)

    assert (out.pos.norm(dim=1) <= batch.pos.norm(dim=1) + 1e-4).all()


def test_projection_with_proper_motions_preserves_the_speed():
    """All three velocity components kept: a rotation preserves |v|."""
    batch = make_batch(256)

    out = RandomProjection(use_proper_motions=True)(batch)

    assert torch.allclose(
        out.vel.norm(dim=1), batch.vel.norm(dim=1), atol=1e-4)


def test_projection_drops_one_velocity_component_by_default():
    """v_los only, unless proper motions are asked for."""
    batch = make_batch(16)

    los_only = RandomProjection()(batch)
    with_pm = RandomProjection(use_proper_motions=True)(batch)

    assert los_only.pos.shape == (16, 2), los_only.pos.shape
    assert los_only.vel.shape == (16, 1), los_only.vel.shape
    assert with_pm.vel.shape == (16, 3), with_pm.vel.shape


def test_pipeline_sets_vel_error_and_perturbs_only_that_component():
    """The sampler writes a sigma and noises the column it names."""
    batch = make_batch(512)

    torch.manual_seed(0)
    projected = RandomProjection(use_proper_motions=True)(batch)
    torch.manual_seed(0)
    noised = build_transformation(
        apply_projection=True,
        projection_args={'use_proper_motions': True},
        apply_uncertainty=True,
        uncertainty_args=UNCERTAINTY_ARGS)(batch)

    assert noised.vel_error is not None
    assert (noised.vel_error[:, 0] > 0).all()
    # Columns 1 and 2 have no sampler of their own, so they are untouched.
    assert torch.allclose(noised.vel[:, 1:], projected.vel[:, 1:], atol=1e-4)
    assert not torch.allclose(noised.vel[:, 0], projected.vel[:, 0])


def test_uncertainty_requires_its_arguments():
    """Silently skipping the noise model would train the wrong likelihood."""
    assert raises(ValueError, 'uncertainty_args', build_transformation,
                  apply_uncertainty=True, uncertainty_args=None)


def test_norm_dict_has_a_loc_and_a_positive_scale_for_every_field():
    """The model divides by these; a zero scale is a silent inf."""
    norm = compute_field_norm(
        make_batch(1024), batch_size=256, apply_projection=True,
        apply_uncertainty=True, uncertainty_args=UNCERTAINTY_ARGS)

    for field in ('pos', 'vel', 'vel_error', 'theta', 'cond'):
        assert f'{field}_loc' in norm, field
        assert np.all(np.asarray(norm[f'{field}_scale']) > 0), field


def test_norm_dict_scale_is_floored_for_an_unperturbed_component():
    """Without a noise model vel_error is all zeros: scale must not be 0."""
    norm = compute_field_norm(
        make_batch(256), batch_size=128, apply_projection=True,
        apply_uncertainty=False)

    assert np.all(np.asarray(norm['vel_error_scale']) > 0)


def test_norm_dict_maps_theta_onto_minus_one_to_one():
    """theta uses midrange/half-range, so its box lands exactly on [-1, 1]."""
    batch = make_batch(2048)
    norm = compute_field_norm(
        batch, batch_size=512, apply_projection=True,
        apply_uncertainty=True, uncertainty_args=UNCERTAINTY_ARGS)

    scaled = ((batch.theta - torch.tensor(norm['theta_loc']))
              / torch.tensor(norm['theta_scale']))

    assert close(float(scaled.abs().max()), 1.0, abs_=1e-5)


def test_norm_dict_is_chunk_size_independent():
    """Stats are streamed in chunks; the answer must not depend on how."""
    batch = make_batch(1024)
    kwargs = dict(apply_projection=False, apply_uncertainty=False)

    small = compute_field_norm(batch, batch_size=64, **kwargs)
    large = compute_field_norm(batch, batch_size=1024, **kwargs)

    for key in small:
        assert np.allclose(small[key], large[key], atol=1e-6), key
