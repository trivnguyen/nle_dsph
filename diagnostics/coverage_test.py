"""Coverage of per-star NLE posteriors on held-out simulated galaxies.

One galaxy at a time: build its observation (projection along z, fixed
per-component errors, optional position selection, optional split of the
stars into measured subsets), sample the 7 labels with emcee on the GPU
(all walkers in one batch), and record z = (median - truth) / half-width
and the PIT of log10 rho(150 pc). Systemic velocities are fixed at 0,
which is the simulated galaxies' rest frame.

Modes:
  selection  the velocity-only model p(v_los | log R, ...) on the same
             galaxies with no selection, with only R < r_half kept, and
             with only R > r_half kept. A position-only selection cancels
             from a likelihood conditioned on R, so the posteriors should
             stay calibrated in all three (the widths change with the
             star count and the radii, not the coverage).
  mixed      three likelihoods on the same galaxies: v_los for all stars
             (model `los`), the mixed likelihood with v_los for a random
             half of the stars and (v_x, v_y) for the other half (models
             `los` and `pm`), and all three components for all stars
             (model `v3d`). Mixed should be calibrated and sit between the
             other two in width.

    python diagnostics/coverage_test.py --mode selection --los <run dir>
    python diagnostics/coverage_test.py --mode mixed --los <run dir> \
        --pm <run dir> --v3d <run dir>
Writes <out>/<mode>_<case>.csv (one row per galaxy) and prints a summary.
"""

import argparse
import math
import os
import sys
from pathlib import Path

import emcee
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nle import NLE  # noqa: E402
from transforms import StarBatch  # noqa: E402
from infer_nle import PRIOR_MIN, PRIOR_MAX, LABELS, log_rho_150  # noqa: E402
from diagnostics.pit_by_regime import best_checkpoint, load_shard  # noqa: E402

PMIN, PMAX = PRIOR_MIN[:7], PRIOR_MAX[:7]
OUT = Path(os.path.expanduser('~/scratch/projects/sbi_dsph/npe_gap/nle/coverage'))


class Term:
    """One additive term of the galaxy log-likelihood: a model on a star subset."""

    def __init__(self, model, R, vel, err, lrs):
        self.model = model
        self.R, self.vel, self.err = R, vel, err
        self.lrs = lrs
        self.n = R.shape[0]

    def log_like(self, theta):
        """Sum of per-star log p for each row of theta, shape (n_walkers,)."""
        nw = theta.shape[0]
        n = self.n
        pos = torch.zeros((nw * n, 2), device=theta.device)
        pos[:, 0] = self.R.repeat(nw)
        b = StarBatch(
            pos=pos, vel=self.vel.repeat(nw, 1), vel_error=self.err.repeat(nw, 1),
            theta=theta.repeat_interleave(n, 0),
            cond=torch.full((nw * n, 1), self.lrs, device=theta.device))
        lp = self.model.log_prob(b).view(nw, n)
        lp = torch.where(torch.isfinite(lp), lp, torch.full_like(lp, -1e30))
        return lp.sum(1)


def sample(terms, n_walkers, n_steps, seed, device):
    """emcee over the 7 labels with the box prior; returns the flat chain."""
    rng = np.random.default_rng(seed)

    def log_post(theta):
        out = np.full(theta.shape[0], -np.inf)
        ok = np.all((theta >= PMIN) & (theta <= PMAX), axis=1)
        if ok.any():
            t = torch.tensor(theta[ok], dtype=torch.float32, device=device)
            with torch.no_grad():
                ll = sum(term.log_like(t) for term in terms)
            out[ok] = ll.cpu().numpy()
        return out

    p0 = rng.uniform(PMIN, PMAX, size=(n_walkers, 7))
    moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
    s = emcee.EnsembleSampler(n_walkers, 7, log_post, moves=moves,
                              vectorize=True)
    s.run_mcmc(p0, n_steps, progress=False)
    return s.get_chain(discard=n_steps // 2, flat=True)


def summarize(chain, truth, lrs):
    lr = log_rho_150(chain, lrs)
    tr = float(log_rho_150(truth[None], lrs)[0])
    lo, med, hi = np.percentile(lr, [16, 50, 84])
    hw = 0.5 * (hi - lo)
    return dict(truth=tr, median=med, halfwidth=hw, z=(med - tr) / hw,
                pit=float(np.mean(lr < tr)))


def build_cases(mode, pos, vel, n, errs, models, rng):
    """Yield (case name, [Term]) for one galaxy, given its z-projection."""
    R = pos[:, :2].norm(dim=1)
    v_los = vel[:, 2:3] + errs['los'] * torch.randn(n, 1)
    e_los = torch.full((n, 1), errs['los'])
    if mode == 'selection':
        yield 'none', [('los', torch.ones(n, dtype=torch.bool), v_los, e_los)]
        yield 'inner', [('los', R < errs['rhalf'], v_los, e_los)]
        yield 'outer', [('los', R >= errs['rhalf'], v_los, e_los)]
    else:
        v_pm = vel[:, :2] + errs['pm'] * torch.randn(n, 2)
        e_pm = torch.full((n, 2), errs['pm'])
        half = torch.zeros(n, dtype=torch.bool)
        half[torch.from_numpy(rng.permutation(n)[: n // 2])] = True
        yield 'los_all', [('los', torch.ones(n, dtype=torch.bool), v_los, e_los)]
        yield 'mixed', [('los', half, v_los, e_los), ('pm', ~half, v_pm, e_pm)]
        yield 'v3d_all', [('v3d', torch.ones(n, dtype=torch.bool),
                           torch.cat([v_los, v_pm], 1),
                           torch.cat([e_los, e_pm], 1))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['selection', 'mixed'], required=True)
    ap.add_argument('--los', required=True, help='run dir of the v_los model')
    ap.add_argument('--pm', help='run dir of the (v_x, v_y) model')
    ap.add_argument('--v3d', help='run dir of the 3-component model')
    ap.add_argument('--shard', type=int, default=41)
    ap.add_argument('--n-gal', type=int, default=150)
    ap.add_argument('--err-los', type=float, default=2.0)
    ap.add_argument('--err-pm', type=float, default=5.0)
    ap.add_argument('--n-walkers', type=int, default=48)
    ap.add_argument('--n-steps', type=int, default=1200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = {}
    for key in ('los', 'pm', 'v3d'):
        rd = getattr(a, key)
        if rd:
            models[key] = NLE.load_from_checkpoint(
                best_checkpoint(rd), map_location=dev, pre_transforms=None).eval()
    if a.mode == 'mixed' and not {'los', 'pm', 'v3d'} <= set(models):
        ap.error('--mode mixed needs --los, --pm and --v3d')

    pos, vel, th, lrs, n = load_shard(a.shard, a.n_gal)
    ptr = np.concatenate([[0], np.cumsum(n.numpy())])
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    rows = {}
    for g in range(len(n)):
        sl = slice(ptr[g], ptr[g + 1])
        errs = dict(los=a.err_los, pm=a.err_pm, rhalf=10 ** float(lrs[g]))
        for case, parts in build_cases(a.mode, pos[sl], vel[sl], int(n[g]),
                                       errs, models, rng):
            terms = []
            n_used = 0
            for key, mask, v, e in parts:
                if mask.sum() == 0:
                    continue
                R = pos[sl][:, :2].norm(dim=1)[mask]
                terms.append(Term(models[key], R.to(dev), v[mask].to(dev),
                                  e[mask].to(dev), float(lrs[g])))
                n_used += int(mask.sum())
            chain = sample(terms, a.n_walkers, a.n_steps, a.seed + g, dev)
            row = dict(galaxy=g, n_stars=n_used, log_rstar=float(lrs[g]),
                       **summarize(chain, th[g].numpy(), float(lrs[g])))
            rows.setdefault(case, []).append(row)
        if (g + 1) % 10 == 0:
            print(f'[{g + 1}/{len(n)}] ' + ' | '.join(
                f"{c}: rms z {np.sqrt(np.mean([r['z'] ** 2 for r in rs])):.2f}"
                for c, rs in rows.items()), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    tag = a.tag or f'shard{a.shard}'
    print('\ncase | galaxies | median half-width of log10 rho(150 pc) (dex) | '
          'rms z | frac |z|<=1 (0.68) | mean z | PIT KS p')
    from scipy.stats import kstest
    for case, rs in rows.items():
        df = pd.DataFrame(rs)
        df.to_csv(OUT / f'{a.mode}_{case}_{tag}.csv', index=False)
        ks = kstest(df.pit, 'uniform').pvalue
        print(f'{case:8s} | {len(df):4d} | {df.halfwidth.median():.3f} | '
              f'{np.sqrt((df.z ** 2).mean()):.2f} | '
              f'{(df.z.abs() <= 1).mean():.2f} | {df.z.mean():+.2f} | {ks:.3f}')
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
