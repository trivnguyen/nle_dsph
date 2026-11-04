"""1-D log-likelihood scans for the marginalisation test.

For each held-out galaxy and each of the seven labels, the galaxy
log-likelihood along a scan of that label with the others held at the
truth, under the three likelihoods of `marginal_test.py` on the same
mixed data (v_los on a random half of the stars, (v_x, v_y) on the
other half): `mixed` (los + pm models), `marg` (3-D model with each
star's missing components integrated out), `v3d_all` (3-D model with
every star fully measured, for reference). Curves are shown as
log L - max log L. Mixed and marg should coincide up to the per-star
quadrature error (~1 nat per galaxy); v3d_all is narrower.

    python diagnostics/marginal_scan.py --los <run> --pm <run> --v3d <run> \
        --galaxies 5 12 15 [--n-points 41] [--n-grid 33]
Writes <marginal dir>/figures/scan_g<g>.png and scan_g<g>.npz.
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from infer_nle import PRIOR_MIN, PRIOR_MAX, LABELS  # noqa: E402
from scripts.diagnostics.pit_by_regime import load_shard  # noqa: E402
from scripts.diagnostics.marginal_test import (  # noqa: E402
    OUT, load_models, log_prob_rows, marginal_los, marginal_pm)
from scripts.diagnostics.plot_marginal import STYLE  # noqa: E402

PMIN, PMAX = PRIOR_MIN[:7], PRIOR_MAX[:7]


def likelihoods(models, obs, lrs, a):
    """Three callables theta (n, 7) -> galaxy log L (n,) on one galaxy."""
    R, v_los, v_pm, e_los, e_pm, half, scale = (
        obs[k] for k in ('R', 'v_los', 'v_pm', 'e_los', 'e_pm', 'half', 'scale'))
    ng = R.shape[0]
    m, mm = half, ~half

    def rep(x, nw):
        return x.repeat(nw, *([1] * (x.dim() - 1)))

    def ll_mixed(t):
        nw = t.shape[0]
        a1 = log_prob_rows(models['los'], rep(R[m], nw), rep(v_los[m, None], nw),
                           rep(e_los[m, None], nw),
                           t.repeat_interleave(int(m.sum()), 0), lrs)
        a2 = log_prob_rows(models['pm'], rep(R[mm], nw), rep(v_pm[mm], nw),
                           rep(e_pm[mm], nw),
                           t.repeat_interleave(int(mm.sum()), 0), lrs)
        return a1.view(nw, -1).sum(1) + a2.view(nw, -1).sum(1)

    def ll_marg(t):
        nw = t.shape[0]
        a1 = marginal_los(models['v3d'], rep(R[m], nw), rep(v_los[m], nw),
                          rep(e_los[m], nw), rep(scale[m], nw), a.sigma_miss,
                          t.repeat_interleave(int(m.sum()), 0), lrs,
                          a.n_grid, a.half_range)
        a2 = marginal_pm(models['v3d'], rep(R[mm], nw), rep(v_pm[mm], nw),
                         rep(e_pm[mm], nw), rep(scale[mm], nw), a.sigma_miss,
                         t.repeat_interleave(int(mm.sum()), 0), lrs,
                         a.n_grid, a.half_range)
        return a1.view(nw, -1).sum(1) + a2.view(nw, -1).sum(1)

    def ll_v3d(t):
        nw = t.shape[0]
        vel3 = torch.cat([v_los[:, None], v_pm], 1)
        err3 = torch.cat([e_los[:, None], e_pm], 1)
        return log_prob_rows(models['v3d'], rep(R, nw), rep(vel3, nw),
                             rep(err3, nw), t.repeat_interleave(ng, 0),
                             lrs).view(nw, ng).sum(1)

    return {'mixed': ll_mixed, 'marg': ll_marg, 'v3d_all': ll_v3d}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--los', required=True)
    ap.add_argument('--pm', required=True)
    ap.add_argument('--v3d', required=True)
    ap.add_argument('--galaxies', type=int, nargs='+', required=True)
    ap.add_argument('--shard', type=int, default=42)
    ap.add_argument('--n-gal', type=int, default=20)
    ap.add_argument('--err-los', type=float, default=2.0)
    ap.add_argument('--err-pm', type=float, default=5.0)
    ap.add_argument('--sigma-miss', type=float, default=3.0)
    ap.add_argument('--n-grid', type=int, default=33)
    ap.add_argument('--half-range', type=float, default=7.0)
    ap.add_argument('--n-points', type=int, default=41)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = load_models(a, dev)
    pos, vel, th, lrs, n = load_shard(a.shard, a.n_gal)
    ptr = np.concatenate([[0], np.cumsum(n.numpy())])
    (OUT / 'figures').mkdir(parents=True, exist_ok=True)
    for g in a.galaxies:
        f = OUT / f'obs_g{g}.npz'
        sl = slice(ptr[g], ptr[g + 1])
        ng = int(n[g])
        if f.exists():  # the observation marginal_test used
            o = np.load(f)
            obs = dict(R=torch.tensor(o['R']), v_los=torch.tensor(o['v_los']),
                       v_pm=torch.tensor(o['v_pm']), half=torch.tensor(o['half']),
                       sigma_true=float(o['sigma_true']))
        else:  # same recipe, fresh draw
            torch.manual_seed(a.seed + 1000 + g)
            rng = np.random.default_rng(a.seed + g)
            v3 = vel[sl]
            half = torch.zeros(ng, dtype=torch.bool)
            half[torch.from_numpy(rng.permutation(ng)[: ng // 2])] = True
            obs = dict(R=pos[sl][:, :2].norm(dim=1),
                       v_los=v3[:, 2] + a.err_los * torch.randn(ng),
                       v_pm=v3[:, :2] + a.err_pm * torch.randn(ng, 2), half=half,
                       sigma_true=float(v3[:, 2].pow(2).mean().sqrt()))
        obs = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in obs.items()}
        obs['e_los'] = torch.full((ng,), a.err_los, device=dev)
        obs['e_pm'] = torch.full((ng, 2), a.err_pm, device=dev)
        obs['scale'] = torch.full((ng,), obs['sigma_true'], device=dev)
        lls = likelihoods(models, obs, float(lrs[g]), a)
        truth = th[g].numpy()
        fig, axes = plt.subplots(1, 7, figsize=(22, 3.6))
        store = {}
        with torch.no_grad():
            for k, (ax, name) in enumerate(zip(axes, LABELS)):
                grid = np.linspace(PMIN[k], PMAX[k], a.n_points)
                thetas = np.repeat(truth[None], a.n_points, 0)
                thetas[:, k] = grid
                t = torch.tensor(thetas, dtype=torch.float32, device=dev)
                for case, fn in lls.items():
                    c = fn(t).cpu().numpy()
                    store[f'{name}_{case}'] = c
                    st = STYLE[case]
                    ax.plot(grid, c - c.max(), color=st['color'], ls=st['ls'],
                            lw=1.6, label=st['label'])
                store[f'{name}_grid'] = grid
                ax.axvline(truth[k], color='k', lw=1, ls=':')
                ax.set_ylim(-25, 1)
                ax.set_xlabel(name)
                ax.grid(alpha=0.25)
        axes[0].set_ylabel('log L - max (others at truth)')
        axes[0].legend(fontsize=7, loc='lower left')
        fig.suptitle(f'held-out galaxy {g} (sigma_true = {obs["sigma_true"]:.1f} km/s, '
                     f'{ng} stars): 1-D log-likelihood scans through the truth '
                     '(dotted); v_los on half the stars, (v_x, v_y) on the other half',
                     fontsize=10)
        fig.tight_layout()
        out = OUT / 'figures' / f'scan_g{g}.png'
        fig.savefig(out, dpi=120)
        plt.close(fig)
        np.savez(OUT / f'scan_g{g}.npz', truth=truth, **store)
        # Reason: the quadrature grid is sized by the galaxy's true
        # velocity scale, so far from the truth (hundreds of nats down,
        # where the model's own scale differs by orders of magnitude) the
        # integral is under-resolved. Compare only within the plotted
        # window, log L > max - 25, which is all a posterior ever sees.
        devs = []
        for nm in LABELS:
            cm, cg = store[f'{nm}_mixed'], store[f'{nm}_marg']
            w = cm > cm.max() - 25
            devs.append((cm - cg)[w])
        d = np.concatenate(devs)
        d -= d.mean()
        print(f'wrote {out} | mixed - marg within 25 nats of the peak '
              f'({len(d)} scan points): std {d.std():.2f} nats, max |dev| '
              f'{np.abs(d).max():.2f}', flush=True)


if __name__ == '__main__':
    main()
