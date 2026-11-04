"""Analytic Zhao (generalized NFW) density profile, in numpy.

Pure functions of a label vector -- no torch, no trained model, no
external Jeans solver -- so they can be imported anywhere and tested
without data.

Two radius conventions appear in this project and the functions say
which they take. `theta` rows carry `dm_log_rdm` in units of the
stellar half-light radius, as the simulations are parameterized;
`zhao_logrho` and `zhao_slope` instead want it already converted to
kpc, because that is what the diagnostics plot against. `log_rho_150`
does the conversion itself from `log_rstar`.
"""

import numpy as np

__all__ = ['log_rho_150', 'zhao_logrho', 'zhao_slope']


def zhao_logrho(theta, r_kpc):
    """log10 of the Zhao density, Msun kpc^-3.

    Args:
        theta: (n_samples, >=5) array whose first five columns are
            alpha, beta, gamma, log10 r_dm [kpc], log10 rho0.
        r_kpc: (n_radii,) radii in kpc.

    Returns:
        (n_samples, n_radii) array of log10 rho.

    Example:
        >>> theta = np.array([[1.0, 3.0, 1.0, 0.0, 7.0]])
        >>> float(zhao_logrho(theta, np.array([1.0]))[0, 0])
        6.397940008672037
    """
    theta = np.atleast_2d(np.asarray(theta, dtype=float))
    alpha, beta, gamma, log_rdm, log_rho0 = (
        theta[:, i:i + 1] for i in range(5))
    x = np.asarray(r_kpc, dtype=float)[None, :] / 10 ** log_rdm
    return (log_rho0 - gamma * np.log10(x)
            + (gamma - beta) / alpha * np.log10(1 + x ** alpha))


def zhao_slope(theta, r_kpc):
    """Logarithmic density slope d ln rho / d ln r.

    Args:
        theta: (n_samples, >=4) array whose first four columns are
            alpha, beta, gamma, log10 r_dm [kpc].
        r_kpc: (n_radii,) radii in kpc.

    Returns:
        (n_samples, n_radii) array of slopes, running from -gamma at
        small radii to -beta at large radii.
    """
    theta = np.atleast_2d(np.asarray(theta, dtype=float))
    alpha, beta, gamma, log_rdm = (theta[:, i:i + 1] for i in range(4))
    x = np.asarray(r_kpc, dtype=float)[None, :] / 10 ** log_rdm
    return -gamma + (gamma - beta) * x ** alpha / (1 + x ** alpha)


def log_rho_150(theta, log_rstar):
    """log10 of the Zhao density at 150 pc, Msun kpc^-3.

    The summary statistic the dwarf posteriors are compared on: the
    dark-matter density at a radius where the kinematics constrain it,
    rather than an extrapolated central value.

    Args:
        theta: (n_samples, >=5) array in label space, whose first five
            columns are alpha, beta, gamma, `dm_log_rdm` (log10 of the
            scale radius in units of the half-light radius) and
            `dm_log_rho0`.
        log_rstar: log10 of the half-light radius in kpc, used to put
            `dm_log_rdm` into kpc.

    Returns:
        (n_samples,) array of log10 rho(150 pc).
    """
    theta = np.atleast_2d(np.asarray(theta, dtype=float))
    theta_kpc = theta[:, :5].copy()
    theta_kpc[:, 3] += log_rstar
    return zhao_logrho(theta_kpc, np.array([0.15]))[:, 0]
