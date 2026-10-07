"""Corner and profile figures for the velocity-only vs joint NLE mock test.

For each mock with saved chains from `mock_sf_compare.py`: a corner plot
of the seven labels (radii in kpc) for both models with the truth, and a
three-panel profile figure: (a) log10 of the dark-matter density rho(r)
in Msun/kpc^3 against 3-D radius r in kpc, (b) its logarithmic slope,
(c) the line-of-sight dispersion sigma_los(R) in km/s against projected
radius R in kpc. Bands are the 16-84% range over posterior draws; in (c)
the curves are the Jeans prediction at the median draw and the points
with bars the bin-averaged posterior predictive, sqrt(mean_i sigma2(R_i))
over the stars of each data bin, offset slightly in R per model. Black
points are the error-deconvolved dispersion fit per bin (dsph_analysis
vdisp), black dashed lines the truth. Reuses the density, slope, band
and Jeans-moment code of npe_gap/plot_profiles_compare.py and its
colour scheme (NLE = light blue for the velocity-only model; the joint
model takes the GNN's indigo since it also models positions).

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

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'npe_gap'))
sys.path.insert(0, os.path.expanduser('~/my_modules'))
import dsph_analysis.kinematic_io as kio  # noqa: E402
from dsph_analysis import vdisp  # noqa: E402
from palette import METHODS, TRUTH, DATA  # noqa: E402
from plot_profiles_compare import (  # noqa: E402
    zhao_logrho, zhao_slope, band, _moments, R_GRID, LABELS_KPC)
from diagnostics.mock_sf_compare import (  # noqa: E402
    MOCKS, SETS, PROFILES, OUT, load_mock, truth_labels)

STYLE = {'velocity-only p(v | R)': METHODS['NLE'],
         'joint p(x, y, v)': METHODS['8p_v4 (GNN)']}
FILE_KEY = {'velocity-only p(v | R)': 'los', 'joint p(x, y, v)': 'joint'}
X_OFFSET = {'velocity-only p(v | R)': 0.95, 'joint p(x, y, v)': 1.05}


def to_kpc(chain, lrs):
    """Label-space chain (radii in r_half units) -> radii in kpc."""
    kpc = chain[:, :7].copy()
    kpc[:, 3] += lrs
    kpc[:, 6] += lrs
    return kpc


def corner_figure(chains, truth_kpc, title, out):
    fig = None
    for name, kpc in chains.items():
        st = STYLE[name]
        fig = corner.corner(
            kpc, labels=LABELS_KPC, fig=fig, color=st['color'],
            hist_kwargs=dict(density=True, ls=st['ls'], lw=1.6),
            contour_kwargs=dict(linestyles=st['ls']), plot_datapoints=False,
            plot_density=False, levels=(0.68, 0.95), smooth=1.0,
            truths=truth_kpc, truth_color='black')
    fig.legend(handles=[plt.Line2D([], [], color=s['color'], ls=s['ls'], label=n)
                        for n, s in STYLE.items()],
               loc='upper right', fontsize=11)
    fig.suptitle(title, fontsize=12)
    fig.savefig(out, dpi=110)
    plt.close(fig)


def profile_figure(chains, obs, lrs, truth, title, out, n_draws, pool):
    rhalf = 10 ** lrs
    R_stars = np.hypot(obs['X'], obs['Y']).astype(float)
    data = vdisp.calc_vdisp_los_binned(
        R_stars, obs['v'].astype(float), obs['err'].astype(float),
        ntracer_per_bin=len(R_stars) // 4, nbins_min=4, nbins_max=4,
        nsteps=1500, verbose=False)
    edges = list(zip(data['R_mid'] - data['R_em'], data['R_mid'] + data['R_ep']))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for name, chain in chains.items():
        st = STYLE[name]
        draws = chain[np.random.default_rng(0).choice(len(chain), n_draws, False)]
        kpc = to_kpc(draws, lrs)
        band(ax[0], R_GRID, zhao_logrho(kpc, R_GRID), st, name)
        band(ax[1], R_GRID, zhao_slope(kpc, R_GRID), st, name)
        R_all = np.concatenate([R_GRID, R_stars])
        res = pool.map(_moments, [(th[:7], lrs, R_all) for th in draws])
        s2 = np.array([r[0] for r in res])
        ng = len(R_GRID)
        ax[2].plot(R_GRID, np.sqrt(np.nanmedian(s2[:, :ng], 0)), color=st['color'],
                   ls=st['ls'], lw=1.2)
        sig_bin = np.array([np.sqrt(s2[:, ng:][:, (R_stars >= lo) & (R_stars <= hi)].mean(1))
                            for lo, hi in edges]).T
        q = np.nanpercentile(sig_bin, [16, 50, 84], axis=0)
        ax[2].errorbar(data['R_mid'] * X_OFFSET[name], q[1],
                       yerr=[q[1] - q[0], q[2] - q[1]], fmt=st['marker'],
                       color=st['color'], ms=6, capsize=2, lw=1.3, label=name,
                       markeredgecolor='k', markeredgewidth=0.4, zorder=4)
    tk = truth.copy()
    tk[3] += lrs
    tk[6] += lrs
    ax[0].plot(R_GRID, zhao_logrho(tk[None], R_GRID)[0], label='truth', **TRUTH)
    ax[1].plot(R_GRID, zhao_slope(tk[None], R_GRID)[0], **TRUTH)
    s2t, _ = _moments((truth, lrs, R_GRID))
    ax[2].plot(R_GRID, np.sqrt(s2t), label='Jeans at the truth', **TRUTH)
    ax[2].errorbar(data['R_mid'], data['sigma'], xerr=[data['R_em'], data['R_ep']],
                   yerr=[data['sigma_em'], data['sigma_ep']], fmt='o', ms=7,
                   capsize=3, lw=1.6, zorder=5, label='data: deconvolved fit per bin',
                   **DATA)
    for a in ax:
        a.set_xscale('log')
        a.axvline(0.15, color='0.5', ls=':', lw=1)
        a.axvline(rhalf, color='0.5', ls='-.', lw=1)
        a.axvspan(R_stars.min(), R_stars.max(), color='0.93', zorder=0)
        a.grid(alpha=0.25)
    ax[0].set(xlabel='r [kpc]', ylabel='log10 rho(r) [Msun/kpc^3]', ylim=(5.5, 10),
              title='(a) dark-matter density')
    ax[1].set(xlabel='r [kpc]', ylabel='d ln rho / d ln r', ylim=(-4, 0.5),
              title='(b) logarithmic slope')
    ax[2].set(xlabel='R [kpc]', ylabel='sigma_los [km/s]', ylim=(0, 18),
              title='(c) sigma_los: curve = model, points = bin-averaged predictive')
    ax[0].legend(fontsize=8, loc='lower left')
    ax[2].legend(fontsize=7.5, loc='upper left')
    fig.suptitle(f'{title}. Bands and bars: 16-84% over {n_draws} draws; dotted r = 150 pc, '
                 f'dash-dotted r_half = {rhalf:.3f} kpc; grey: radial range of the stars',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
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
            params = pd.read_csv(MOCKS / SETS[set_name] / 'mock_catalog_params.csv')
            for prof in a.profiles:
                files = {n: chains_dir / f'{set_name}_{prof}_{k}.npy'
                         for n, k in FILE_KEY.items()}
                if not all(f.exists() for f in files.values()):
                    print('skip', set_name, prof, '(chains missing)')
                    continue
                chains = {n: np.load(f) for n, f in files.items()}
                truth = truth_labels(params[params.profile == prof].iloc[0], rhalf)
                obs = load_mock(SETS[set_name], prof, meta)
                n = len(obs['v'])
                title = (f'Mock {prof}, {set_name} ({n} stars'
                         + (', Walker 2023 footprint)' if set_name == 'walker23'
                            else ', no selection)'))
                tk = truth.copy()
                tk[3] += lrs
                tk[6] += lrs
                corner_figure({k: to_kpc(c, lrs) for k, c in chains.items()}, tk,
                              title, fig_dir / f'corner_{set_name}_{prof}.png')
                profile_figure(chains, obs, lrs, truth, title,
                               fig_dir / f'profiles_{set_name}_{prof}.png',
                               a.n_draws, pool)
                print('wrote', set_name, prof, flush=True)
    print('->', fig_dir)


if __name__ == '__main__':
    main()
