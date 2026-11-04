"""Unit tests for the analytic Zhao profile."""

import numpy as np

from nle_dsph.profiles import log_rho_150, zhao_logrho, zhao_slope
from run_tests import close


def test_density_equals_rho0_at_the_scale_radius():
    """At r = r_dm the Zhao form reduces to rho0 * 2^((gamma-beta)/alpha)."""
    alpha, beta, gamma, log_rdm, log_rho0 = 1.5, 4.0, 0.8, 0.3, 7.2
    theta = np.array([[alpha, beta, gamma, log_rdm, log_rho0]])

    got = zhao_logrho(theta, np.array([10 ** log_rdm]))[0, 0]

    expected = log_rho0 + (gamma - beta) / alpha * np.log10(2.0)
    assert close(got, expected, rel=1e-9), (got, expected)


def test_slope_runs_from_minus_gamma_to_minus_beta():
    """The logarithmic slope has the inner and outer limits it claims."""
    gamma, beta = 0.7, 4.5
    theta = np.array([[1.0, beta, gamma, 0.0]])

    inner, outer = zhao_slope(theta, np.array([1e-8, 1e8]))[0]

    assert close(inner, -gamma, abs_=1e-6), inner
    assert close(outer, -beta, abs_=1e-6), outer


def test_slope_matches_numerical_derivative_of_log_density():
    """zhao_slope is d ln rho / d ln r of zhao_logrho, not an approximation."""
    theta = np.array([[1.3, 3.6, 1.1, -0.2, 6.5]])
    r, h = 0.4, 1e-5

    log_rho = zhao_logrho(theta, np.array([r * np.exp(-h), r * np.exp(h)]))
    numerical = np.log(10) * (log_rho[0, 1] - log_rho[0, 0]) / (2 * h)
    analytic = zhao_slope(theta, np.array([r]))[0, 0]

    assert close(analytic, numerical, rel=1e-5), (analytic, numerical)


def test_log_rho_150_agrees_with_zhao_logrho_at_150_pc():
    """The summary statistic is the profile, with r_dm converted to kpc."""
    log_rstar = -0.55
    theta = np.array([[1.2, 3.9, 1.4, 0.6, 6.1, -0.3, 0.1]])

    theta_kpc = theta[:, :5].copy()
    theta_kpc[:, 3] += log_rstar
    want = zhao_logrho(theta_kpc, np.array([0.15]))[0, 0]

    assert close(log_rho_150(theta, log_rstar)[0], want, rel=1e-12)


def test_log_rho_150_ignores_columns_past_the_fifth():
    """Label vectors carry beta0 and log_ra; the density must not see them."""
    base = [1.2, 3.9, 1.4, 0.6, 6.1]
    with_df = np.array([base + [-0.3, 0.1]])
    without_df = np.array([base])

    assert close(log_rho_150(with_df, -0.5)[0],
                 log_rho_150(without_df, -0.5)[0], rel=1e-12)


def test_log_rho_150_does_not_mutate_its_input():
    """It shifts dm_log_rdm into kpc; doing that in place would corrupt
    a posterior chain on the second call."""
    theta = np.array([[1.2, 3.9, 1.4, 0.6, 6.1, -0.3, 0.1]])
    before = theta.copy()

    first = log_rho_150(theta, -0.5)[0]
    second = log_rho_150(theta, -0.5)[0]

    assert np.array_equal(theta, before), 'input was modified'
    assert close(first, second, rel=1e-12), (first, second)


def test_log_rho_150_is_vectorised_over_samples():
    """A posterior chain goes through in one call, row by row."""
    rng = np.random.default_rng(0)
    theta = rng.uniform([0.5, 2.5, -0.5, 0.0, 5.0],
                        [3.0, 10.0, 2.0, 2.5, 10.0], size=(32, 5))

    got = log_rho_150(theta, -0.5)

    assert got.shape == (32,), got.shape
    for i in range(32):
        assert close(got[i], log_rho_150(theta[i:i + 1], -0.5)[0], rel=1e-12)


def test_steeper_inner_slope_means_more_mass_well_inside_the_scale_radius():
    """Monotonicity in gamma, at a radius where gamma dominates."""
    shallow = np.array([[1.0, 3.0, 0.5, 0.0, 7.0]])
    steep = np.array([[1.0, 3.0, 1.5, 0.0, 7.0]])
    r = np.array([0.01])

    assert zhao_logrho(steep, r)[0, 0] > zhao_logrho(shallow, r)[0, 0]


def test_log_rho_150_accepts_one_rstar_per_row():
    """A sampled conditioning gives each posterior row its own r_star."""
    theta = np.array([[1.2, 3.9, 1.4, 0.6, 6.1],
                      [1.2, 3.9, 1.4, 0.6, 6.1]])
    log_rstar = np.array([-0.60, -0.50])

    got = log_rho_150(theta, log_rstar)

    assert got.shape == (2,), got.shape
    for i in range(2):
        want = log_rho_150(theta[i:i + 1], log_rstar[i])[0]
        assert close(got[i], want, rel=1e-12), (got[i], want)
    assert got[0] != got[1], 'different r_star must give different density'
