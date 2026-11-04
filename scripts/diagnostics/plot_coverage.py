"""Summary figure for a coverage_test run (selection or mixed mode).

Three panels over the galaxies of one run: (a) the 68% half-width of
log10 rho(150 pc) per galaxy for each case, galaxies ordered by the
first case's half-width; (b) z = (median - truth) / half-width per
galaxy, grey band |z| <= 1; (c) the empirical CDF of the PIT of
log10 rho(150 pc) for each case against the uniform diagonal, with the
Kolmogorov-Smirnov p-value in the legend.

    python diagnostics/plot_coverage.py --mode mixed [--tag shard41]
Reads <coverage dir>/<mode>_<case>_<tag>.csv, writes
<coverage dir>/figures/coverage_<mode>_<tag>.png.
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kstest  # noqa: E402

COV = Path(os.path.expanduser('~/scratch/projects/nle_dsph/coverage'))
# Okabe-Ito: blue, vermillion, bluish green.
CASES = {
    'selection': [('none', 'all stars'), ('inner', 'R < r_half only'),
                  ('outer', 'R > r_half only')],
    'mixed': [('los_all', 'v_los, all stars'),
              ('mixed', 'v_los on half, (v_x, v_y) on half'),
              ('v3d_all', '(v_x, v_y, v_los), all stars')],
}
STYLE = [dict(color='#0072B2', marker='o'), dict(color='#D55E00', marker='s'),
         dict(color='#009E73', marker='^')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=list(CASES), required=True)
    ap.add_argument('--tag', default='shard41')
    a = ap.parse_args()
    dfs = {c: pd.read_csv(COV / f'{a.mode}_{c}_{a.tag}.csv').set_index('galaxy')
           for c, _ in CASES[a.mode]}
    first = CASES[a.mode][0][0]
    order = dfs[first].halfwidth.sort_values().index
    x = np.arange(len(order))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for (c, label), st in zip(CASES[a.mode], STYLE):
        d = dfs[c].loc[order]
        ax[0].plot(x, d.halfwidth, ls='none', ms=4, alpha=0.8, label=label, **st)
        ax[1].plot(x, d.z, ls='none', ms=4, alpha=0.8, label=label, **st)
        p = np.sort(d.pit.values)
        ks = kstest(p, 'uniform').pvalue
        ax[2].step(np.concatenate([[0], p, [1]]),
                   np.concatenate([[0], np.arange(1, len(p) + 1) / len(p), [1]]),
                   color=st['color'], lw=1.6,
                   label=f'{label} (KS p = {ks:.2f})')
    ax[0].set_yscale('log')
    ax[0].set_ylabel('68% half-width of log10 rho(150 pc) [dex]')
    ax[0].set_xlabel(f'galaxy, ordered by the "{CASES[a.mode][0][1]}" half-width')
    ax[0].set_title('(a) posterior width per galaxy')
    ax[0].legend(fontsize=8)
    ax[1].axhspan(-1, 1, color='0.9', zorder=0)
    ax[1].axhline(0, color='0.5', lw=0.8)
    ax[1].set_ylim(-4, 4)
    ax[1].set_ylabel('z = (median - truth) / half-width')
    ax[1].set_xlabel('galaxy, same order')
    ax[1].set_title('(b) z per galaxy; grey: |z| <= 1')
    ax[2].plot([0, 1], [0, 1], color='k', ls='--', lw=1)
    ax[2].set_xlabel('PIT of log10 rho(150 pc) at the truth')
    ax[2].set_ylabel('empirical CDF')
    ax[2].set_title('(c) PIT against uniform')
    ax[2].legend(fontsize=8, loc='upper left')
    for s in ax:
        s.grid(alpha=0.25)
    n = len(order)
    fig.suptitle(f'coverage_test --mode {a.mode}, {n} held-out galaxies ({a.tag})',
                 fontsize=11)
    fig.tight_layout()
    out = COV / 'figures' / f'coverage_{a.mode}_{a.tag}.png'
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130)
    print('wrote', out)


if __name__ == '__main__':
    main()
