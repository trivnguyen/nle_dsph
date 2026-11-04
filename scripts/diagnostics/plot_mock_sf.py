"""Corner and profile figures for the velocity-only vs joint NLE mock test.

For each mock with saved chains from `mock_sf_compare.py`: a corner plot
of the seven labels (radii in kpc) for both models with the truth, and a
four-panel profile figure: (a) log10 of the dark-matter density rho(r)
in Msun/kpc^3 against 3-D radius r in kpc, (b) its logarithmic slope,
(c) the line-of-sight dispersion sigma_los(R) in km/s and (d) the
line-of-sight kurtosis kappa_los(R) (3 for a Gaussian) against projected
radius R in kpc. Bands are the 16-84% range over posterior draws; in (c)
and (d) the curves are the Jeans prediction at the median draw and the
points with bars the bin-averaged posterior predictive over the stars of
each data bin (sqrt(mean_i sigma2_i) and mean_i sigma4_i / (mean_i
sigma2_i)^2), offset slightly in R per model. Black points are the
error-deconvolved dispersion and kurtosis fits per bin (dsph_analysis),
black dashed lines the truth (fourth-order Jeans with beta' = beta).
Colours: Okabe-Ito blue for the velocity-only model, vermillion for the
joint model, no hatching.

    python diagnostics/plot_mock_sf.py [--chains <dir>] [--sets ...]
        [--profiles ...] [--n-draws 150]
Writes <chains>/figures/corner_<set>_<mock>.png and profiles_<set>_<mock>.png.
"""

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import corner
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.path.expanduser('~/my_modules'))
import dsph_analysis.kinematic_io as kio  # noqa: E402
from dsph_analysis import vdisp, vkurtosis  # noqa: E402
from scripts.diagnostics.profiles import (  # noqa: E402
    zhao_logrho, zhao_slope, band, moments, R_GRID, LABELS_KPC)
from scripts.diagnostics.mock_sf_compare import (  # noqa: E402
    MOCKS, SETS, PROFILES, OUT, load_mock, truth_labels)

# Okabe-Ito blue and vermillion; line style and marker as second channel.
STYLE = {'velocity-only p(v | R)': dict(color='#0072B2', ls='-', marker='o'),
         'joint p(x, y, v)': dict(color='#D55E00', ls='--', marker='s')}
TRUTH = dict(color='black', ls=(0, (6, 3)), lw=1.6)
DATA = dict(color='black')
FILE_KEY = {'velocity-only p(v | R)': 'los', 'joint p(x, y, v)': 'joint'}
X_OFFSET = {'velocity-only p(v | R)': 0.95, 'joint p(x, y, v)': 1.05}


def to_kpc(chain: np.ndarray, lrs: float) -> np.ndarray:
    """Convert a label-space chain (radii in r_half units) to radii in kpc.

    Args:
        chain: Posterior samples, shape (n, >= 7); columns 3 and 6 are
            log10 r_dm and log10 r_a in units of r_half.
        lrs: log10 of the half-light radius in kpc.

    Returns:
        The first seven columns with the two radii shifted by `lrs`.
    """
    kpc = chain[:, :7].copy()
    kpc[:, 3] += lrs
    kpc[:, 6] += lrs
    return kpc


def corner_figure(chains: dict, truth_kpc: np.ndarray, title: str,
                  out: Path) -> None:
    """Corner plot of the seven labels for every model, with the truth.

    Args:
        chains: Model name -> samples in kpc units (see `to_kpc`).
        truth_kpc: True labels in the same units.
        title: Figure title.
        out: Output image path.
    """
    fig = None
    for name, kpc in chains.items():
        st = STYLE[name]
        fig = corner.corner(
            kpc, labels=LABELS_KPC, fig=fig, color=st['color'],
            hist_kwargs=dict(density=True, ls=st['ls'], lw=1.6),
            contour_kwargs=dict(linestyles=st['ls']), plot_datapoints=False,
            plot_density=False, levels=(0.68, 0.95), smooth=1.0,
            truths=truth_kpc, truth_color='black')
    handles = [plt.Line2D([], [], color=s['color'], ls=s['ls'], label=n)
               for n, s in STYLE.items()]
    fig.legend(handles=handles, loc='upper right', fontsize=11)
    fig.suptitle(title, fontsize=12)
    fig.savefig(out, dpi=110)
    plt.close(fig)


def binned_data(R_stars: np.ndarray, v: np.ndarray, e: np.ndarray) -> tuple:
    """Error-deconvolved dispersion and kurtosis in four radial bins.

    Args:
        R_stars: Projected radii [kpc].
        v: Line-of-sight velocities [km/s].
        e: Their errors [km/s].

    Returns:
        Tuple `(disp, kurt, edges)`: the two `dsph_analysis` result dicts
        and the list of `(R_lo, R_hi)` bin edges in kpc.
    """
    nb = len(R_stars) // 4
    disp = vdisp.calc_vdisp_los_binned(
        R_stars, v, e, ntracer_per_bin=nb, nbins_min=4, nbins_max=4,
        nsteps=1500, verbose=False)
    kurt = vkurtosis.calc_kurtosis_los_binned(
        R_stars, v, e, ntracer_per_bin=nb, nbins_min=4, nbins_max=4,
        verbose=False, sampler_args={'nsteps': 1500})
    edges = list(zip(disp['R_mid'] - disp['R_em'],
                     disp['R_mid'] + disp['R_ep']))
    return disp, kurt, edges


def bin_averaged(s2: np.ndarray, s4: np.ndarray, R_stars: np.ndarray,
                 edges: list) -> tuple:
    """Bin-averaged posterior predictive dispersion and kurtosis.

    Per draw and bin: sqrt(mean_i sigma2_i) and
    mean_i sigma4_i / (mean_i sigma2_i)^2 over the stars in the bin.

    Args:
        s2: Jeans sigma2_los at the stars, shape (draws, n_stars).
        s4: Jeans sigma4_los at the stars, same shape.
        R_stars: Projected radii of the stars [kpc].
        edges: `(R_lo, R_hi)` per bin.

    Returns:
        Tuple `(sig_bin, kap_bin)`, each of shape (draws, n_bins).
    """
    sig_bin, kap_bin = [], []
    for lo, hi in edges:
        m = (R_stars >= lo) & (R_stars <= hi)
        m2 = s2[:, m].mean(1)
        sig_bin.append(np.sqrt(m2))
        kap_bin.append(s4[:, m].mean(1) / m2 ** 2)
    return np.array(sig_bin).T, np.array(kap_bin).T


def profile_figure(chains: dict, obs: dict, lrs: float, truth: np.ndarray,
                   title: str, out: Path, n_draws: int, pool: Pool) -> None:
    """Four-panel profile figure for every model against data and truth.

    Args:
        chains: Model name -> label-space samples (radii in r_half units).
        obs: Mock observation dict with `X`, `Y`, `v`, `err`.
        lrs: log10 r_half [kpc].
        truth: True labels in label space.
        title: Figure title.
        out: Output image path.
        n_draws: Posterior draws per model for the bands.
        pool: Worker pool for the Jeans moments.
    """
    rhalf = 10 ** lrs
    R_stars = np.hypot(obs['X'], obs['Y']).astype(float)
    disp, kurt, edges = binned_data(
        R_stars, obs['v'].astype(float), obs['err'].astype(float))
    ng = len(R_GRID)
    R_all = np.concatenate([R_GRID, R_stars])
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10))
    ax = axes.ravel()
    rng = np.random.default_rng(0)
    for name, chain in chains.items():
        st = STYLE[name]
        draws = chain[rng.choice(len(chain), n_draws, False)]
        kpc = to_kpc(draws, lrs)
        band(ax[0], R_GRID, zhao_logrho(kpc, R_GRID), st, name)
        band(ax[1], R_GRID, zhao_slope(kpc, R_GRID), st, name)
        res = pool.map(moments, [(th[:7], lrs, R_all) for th in draws])
        s2 = np.array([r[0] for r in res])
        s4 = np.array([r[1] for r in res]) * s2 ** 2
        sig_bin, kap_bin = bin_averaged(s2[:, ng:], s4[:, ng:], R_stars,
                                        edges)
        curves = ((ax[2], np.sqrt(s2[:, :ng]), sig_bin),
                  (ax[3], s4[:, :ng] / s2[:, :ng] ** 2, kap_bin))
        for a, curve, binned in curves:
            a.plot(R_GRID, np.nanmedian(curve, 0), color=st['color'],
                   ls=st['ls'], lw=1.2)
            q = np.nanpercentile(binned, [16, 50, 84], axis=0)
            a.errorbar(disp['R_mid'] * X_OFFSET[name], q[1],
                       yerr=[q[1] - q[0], q[2] - q[1]], fmt=st['marker'],
                       color=st['color'], ms=6, capsize=2, lw=1.3,
                       label=name, markeredgecolor='k', markeredgewidth=0.4,
                       zorder=4)

    tk = truth.copy()
    tk[3] += lrs
    tk[6] += lrs
    ax[0].plot(R_GRID, zhao_logrho(tk[None], R_GRID)[0], label='truth',
               **TRUTH)
    ax[1].plot(R_GRID, zhao_slope(tk[None], R_GRID)[0], **TRUTH)
    s2t, kt = moments((truth, lrs, R_GRID))
    ax[2].plot(R_GRID, np.sqrt(s2t),
               label="Jeans at the truth (beta' = beta)", **TRUTH)
    ax[3].plot(R_GRID, kt, **TRUTH)
    ax[2].errorbar(disp['R_mid'], disp['sigma'],
                   xerr=[disp['R_em'], disp['R_ep']],
                   yerr=[disp['sigma_em'], disp['sigma_ep']], fmt='o',
                   ms=7, capsize=3, lw=1.6, zorder=5,
                   label='data: deconvolved fit per bin', **DATA)
    ax[3].errorbar(kurt['R_mid'], kurt['kappa'],
                   xerr=[kurt['R_em'], kurt['R_ep']],
                   yerr=[kurt['kappa_em'], kurt['kappa_ep']], fmt='o',
                   ms=7, capsize=3, lw=1.6, zorder=5, **DATA)
    ax[3].axhline(3, color='0.6', lw=0.8)
    for a in ax:
        a.set_xscale('log')
        a.axvline(0.15, color='0.5', ls=':', lw=1)
        a.axvline(rhalf, color='0.5', ls='-.', lw=1)
        a.axvspan(R_stars.min(), R_stars.max(), color='0.93', zorder=0)
        a.grid(alpha=0.25)
    ax[0].set(xlabel='r [kpc]', ylabel='log10 rho(r) [Msun/kpc^3]',
              ylim=(5.5, 10), title='(a) dark-matter density')
    ax[1].set(xlabel='r [kpc]', ylabel='d ln rho / d ln r', ylim=(-4, 0.5),
              title='(b) logarithmic slope')
    ax[2].set(xlabel='R [kpc]', ylabel='sigma_los [km/s]', ylim=(0, 18),
              title='(c) sigma_los: curve = model, '
                    'points = bin-averaged predictive')
    ax[3].set(xlabel='R [kpc]', ylabel='kappa_los (Gaussian = 3)',
              ylim=(1.0, 5.5),
              title="(d) kappa_los, 4th-order Jeans with beta' = beta")
    ax[0].legend(fontsize=8, loc='lower left')
    ax[2].legend(fontsize=7.5, loc='upper left')
    fig.suptitle(f'{title}. Bands and bars: 16-84% over {n_draws} draws; '
                 f'dotted r = 150 pc, dash-dotted r_half = {rhalf:.3f} kpc; '
                 'grey: radial range of the stars', fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    """Make both figures for every mock whose chains exist."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--chains', default=str(OUT))
    ap.add_argument('--sets', nargs='+', default=list(SETS))
    ap.add_argument('--profiles', nargs='+', default=list(PROFILES))
    ap.add_argument('--n-draws', type=int, default=150)
    a = ap.parse_args()
    chains_dir = Path(a.chains)
    fig_dir = chains_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)
    meta = kio.load_meta('draco_1')
    rhalf = float(meta.rhalf_kpc.value)
    lrs = float(np.log10(rhalf))
    with Pool(8) as pool:
        for set_name in a.sets:
            params = pd.read_csv(
                MOCKS / SETS[set_name] / 'mock_catalog_params.csv')
            for prof in a.profiles:
                files = {n: chains_dir / f'{set_name}_{prof}_{k}.npy'
                         for n, k in FILE_KEY.items()}
                if not all(f.exists() for f in files.values()):
                    print('skip', set_name, prof, '(chains missing)')
                    continue
                chains = {n: np.load(f) for n, f in files.items()}
                truth = truth_labels(
                    params[params.profile == prof].iloc[0], rhalf)
                obs = load_mock(SETS[set_name], prof, meta)
                n = len(obs['v'])
                footprint = ('Walker 2023 footprint' if set_name == 'walker23'
                             else 'no selection')
                title = f'Mock {prof}, {set_name} ({n} stars, {footprint})'
                tk = to_kpc(truth[None], lrs)[0]
                corner_figure({k: to_kpc(c, lrs) for k, c in chains.items()},
                              tk, title,
                              fig_dir / f'corner_{set_name}_{prof}.png')
                profile_figure(chains, obs, lrs, truth, title,
                               fig_dir / f'profiles_{set_name}_{prof}.png',
                               a.n_draws, pool)
                print('wrote', set_name, prof, flush=True)
    print('->', fig_dir)


if __name__ == '__main__':
    main()
