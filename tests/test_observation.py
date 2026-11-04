"""Unit tests for the Observation container's geometry contract."""

import numpy as np

from nle_dsph.api import Observation
from run_tests import raises

N = 12
VLOS = np.zeros(N)
VERR = np.ones(N)
POS = np.random.default_rng(0).normal(size=(N, 2))
RADII = np.hypot(POS[:, 0], POS[:, 1])
BASE = dict(vlos_kms=VLOS, vlos_err_kms=VERR, log_rstar_kpc=-0.5)


def test_positions_alone_select_position_mode():
    """One geometry given means no choice to make."""
    obs = Observation(pos_kpc=POS, **BASE)

    assert obs.use == 'position'
    assert obs.R_proj_kpc is None, 'R must not be derived from pos'


def test_radii_alone_select_radius_mode():
    """The mirror case: radii given, positions stay absent."""
    obs = Observation(R_proj_kpc=RADII, **BASE)

    assert obs.use == 'radius'
    assert obs.pos_kpc is None


def test_both_without_use_is_refused():
    """R_proj is a separate measurement, so the caller picks."""
    assert raises(ValueError, 'use=', Observation,
                  pos_kpc=POS, R_proj_kpc=RADII, **BASE)


def test_both_with_use_keeps_both_arrays_untouched():
    """Selecting one must not discard or overwrite the other."""
    obs = Observation(pos_kpc=POS, R_proj_kpc=RADII, use='radius', **BASE)

    assert obs.use == 'radius'
    assert np.array_equal(obs.pos_kpc, POS)
    assert np.array_equal(obs.R_proj_kpc, RADII)


def test_neither_geometry_is_refused():
    """Velocities alone are not an observation."""
    assert raises(ValueError, 'neither was set', Observation, **BASE)


def test_use_naming_an_absent_field_is_refused():
    """A typo must not silently fall back to the other field."""
    assert raises(ValueError, 'pos_kpc is None', Observation,
                  R_proj_kpc=RADII, use='position', **BASE)


def test_unknown_use_is_refused():
    assert raises(ValueError, "expected 'position' or 'radius'",
                  Observation, pos_kpc=POS, use='xy', **BASE)


def test_mismatched_lengths_are_refused():
    """A catalogue cut applied to one array and not another."""
    assert raises(ValueError, 'to match', Observation,
                  pos_kpc=POS[:-1], **BASE)
    assert raises(ValueError, 'to match', Observation,
                  pos_kpc=POS, R_proj_kpc=RADII[:-1], use='position', **BASE)


def test_model_positions_packs_radii_so_the_norm_is_R():
    """A radius-conditioned model reduces pos to |pos|."""
    packed = Observation(R_proj_kpc=RADII, **BASE).model_positions()

    assert packed.shape == (N, 2)
    assert np.allclose(np.hypot(packed[:, 0], packed[:, 1]), RADII)
    assert np.all(packed[:, 1] == 0.0)


def test_model_positions_passes_positions_through():
    packed = Observation(pos_kpc=POS, **BASE).model_positions()

    assert np.allclose(packed, POS.astype(np.float32))
