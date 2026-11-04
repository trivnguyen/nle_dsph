"""Jeans-model helpers and plot bands for the NLE diagnostics.

The line-of-sight dispersion and kurtosis of the Jeans model via
dsph_analysis.sph_model.GeneralizedOMJeans (fourth-order Jeans with
beta' = beta, the GravSphere assumption), plus the 16-84% band helper.

The analytic Zhao density and slope used to live here too; they are
pure numpy with no external dependency, so they now live in
`nle_dsph.profiles` and are re-exported below for callers that still
import them from this module.
"""

import os
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import to_rgba

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, os.path.expanduser('~/my_modules'))
from dsph_analysis.sph_model import GeneralizedOMJeans  # noqa: E402

from nle_dsph.profiles import zhao_logrho, zhao_slope  # noqa: E402,F401

LOG_RHO_UNIT = 7.0  # sph_model's density unit, 1e7 Msun/kpc^3
R_GRID = np.logspace(-2, 0.7, 60)   # kpc
LABELS_KPC = ['dm_alpha', 'dm_beta', 'dm_gamma', 'log10 r_dm [kpc]',
              'log10 rho0', 'df_beta0', 'log10 r_a [kpc]']


def om_jeans_theta(theta, log_rstar):
    """8p labels (radii in r_star units) -> the 12-element Jeans vector."""
    alpha, beta, gamma, log_rdm, log_rho0, beta0, log_ra = theta
    return np.array([
        log_rho0 - LOG_RHO_UNIT, log_rdm + log_rstar, alpha, beta,
        gamma, log_ra + log_rstar, 2.0 ** beta0, 2.0,
        10 ** log_rstar, 0.0, 0.0, 0.0])


def moments(args):
    """(sigma2_los, kappa_los) of the Jeans model at radii R; NaN on failure.

    `args` is (theta in label space, log_rstar, R) so the function can go
    straight into multiprocessing.Pool.map.
    """
    theta, log_rstar, R = args
    try:
        r_star = 10 ** log_rstar
        model = GeneralizedOMJeans(
            om_jeans_theta(theta, log_rstar), min_rgrid=1e-3 * r_star,
            max_rgrid=1e3 * r_star, n_grid=300)
        s2 = np.asarray(model.sigma2_los(R))
        return s2, np.asarray(model.sigma4_los(R)) / s2 ** 2
    except Exception:
        return np.full(len(R), np.nan), np.full(len(R), np.nan)


def band(ax, x, y, style, label):
    """16-84% band plus median line in a method's colour and line style."""
    q = np.nanpercentile(y, [16, 50, 84], axis=0)
    ax.fill_between(x, q[0], q[2], facecolor=to_rgba(style['color'], 0.25),
                    edgecolor='none')
    ax.plot(x, q[1], color=style['color'], ls=style['ls'], lw=1.8, label=label)
