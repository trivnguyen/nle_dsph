"""Mixed likelihood against the 3-D model with the missing components
integrated out.

For a star with only v_los measured, the 3-D model gives
    p(v_los | ctx) = Int p(v_los, v_x, v_y | ctx) dv_x dv_y,
and for a star with only (v_x, v_y),
    p(v_x, v_y | ctx) = Int p(v_los, v_x, v_y | ctx) dv_los.
If the single-component models (`los`, `pm`) and the 3-D model (`v3d`)
are all exact, these marginals equal the single-component densities and
the two galaxy likelihoods on mixed data coincide. The integrals are
trapezoid quadrature on +-`--half-range` times the star's expected
velocity scale sqrt(sigma_true^2 + sigma_miss^2); `sigma_miss` is the
error assigned to the missing component, which the marginal must not
depend on (checked by `--sigma-miss-alt`).

Stage 1 (`--per-star`): on held-out stars at the true theta, the
difference log p_model - log p_marginal per star and per galaxy, by
sigma_true bin, for the los and pm cases, at two grid sizes and two
sigma_miss values.
Stage 2 (default): for `--n-gal` galaxies, emcee posteriors under three
likelihoods on the same mixed data (v_los on a random half, (v_x, v_y)
on the other half): `mixed` (los + pm models), `marg` (v3d with the
missing components integrated out), and `v3d_all` (every star with all
three components, the information ceiling). Chains are saved for corner
plots; the per-galaxy table matches coverage_test's.

    python diagnostics/marginal_test.py --los <run> --pm <run> --v3d <run> --per-star
    python diagnostics/marginal_test.py --los <run> --pm <run> --v3d <run> --n-gal 20
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
from infer_nle import PRIOR_MIN, PRIOR_MAX, log_rho_150  # noqa: E402
from diagnostics.pit_by_regime import best_checkpoint, load_shard  # noqa: E402

PMIN, PMAX = PRIOR_MIN[:7], PRIOR_MAX[:7]
OUT = Path(os.path.expanduser('~/scratch/projects/nle_dsph/marginal'))


def _trap_logw(n, device):
    """log trapezoid weights on n nodes (spacing factored out)."""
    w = torch.zeros(n, device=device)
    w[[0, -1]] = math.log(0.5)
    return w


def log_prob_rows(model, R, vel, err, theta, lrs):
    """Per-row log p for aligned (rows, k) velocity/error tensors."""
    n = R.shape[0]
    pos = torch.zeros((n, 2), device=R.device)
    pos[:, 0] = R
    b = StarBatch(pos=pos, vel=vel, vel_error=err, theta=theta,
                  cond=torch.full((n, 1), lrs, device=R.device))
    # Reason: log_prob is per normalised velocity unit; convert to per
    # (km/s)^k so densities of models with different k and vel_scale
    # can be compared and integrated in physical units.
    lp = model.log_prob(b) - torch.log(model.vel_scale).sum()
    return torch.where(torch.isfinite(lp), lp, torch.full_like(lp, -1e30))


def marginal_los(v3d, R, v_los, err_los, scale, sigma_miss, theta, lrs,
                 n_grid, half_range):
    """log Int p_v3d(v_los, v_x, v_y) dv_x dv_y per star, theta per star."""
    dev = R.device
    n = R.shape[0]
    t = torch.linspace(-1, 1, n_grid, device=dev)
    s = torch.sqrt(scale ** 2 + sigma_miss ** 2) * half_range      # (n,)
    gx = s[:, None] * t[None, :]                                    # (n, G)
    G2 = n_grid * n_grid
    vx = gx[:, :, None].expand(n, n_grid, n_grid).reshape(n, G2)
    vy = gx[:, None, :].expand(n, n_grid, n_grid).reshape(n, G2)
    vel = torch.stack([v_los[:, None].expand(n, G2), vx, vy], 2).reshape(-1, 3)
    err = torch.stack([err_los[:, None].expand(n, G2),
                       torch.full((n, G2), sigma_miss, device=dev),
                       torch.full((n, G2), sigma_miss, device=dev)], 2
                      ).reshape(-1, 3)
    lp = log_prob_rows(v3d, R.repeat_interleave(G2), vel, err,
                       theta.repeat_interleave(G2, 0), lrs).view(n, G2)
    w1 = _trap_logw(n_grid, dev)
    w = (w1[:, None] + w1[None, :]).reshape(G2)
    d = 2 * s / (n_grid - 1)
    return torch.logsumexp(lp + w[None, :], 1) + 2 * torch.log(d)


def marginal_pm(v3d, R, v_pm, err_pm, scale, sigma_miss, theta, lrs,
                n_grid, half_range):
    """log Int p_v3d(v_los, v_x, v_y) dv_los per star."""
    dev = R.device
    n = R.shape[0]
    t = torch.linspace(-1, 1, n_grid, device=dev)
    s = torch.sqrt(scale ** 2 + sigma_miss ** 2) * half_range
    g = s[:, None] * t[None, :]
    vel = torch.cat([g.reshape(-1, 1),
                     v_pm.repeat_interleave(n_grid, 0)], 1)
    err = torch.cat([torch.full((n * n_grid, 1), sigma_miss, device=dev),
                     err_pm.repeat_interleave(n_grid, 0)], 1)
    lp = log_prob_rows(v3d, R.repeat_interleave(n_grid), vel, err,
                       theta.repeat_interleave(n_grid, 0), lrs).view(n, n_grid)
    d = 2 * s / (n_grid - 1)
    return torch.logsumexp(lp + _trap_logw(n_grid, dev)[None, :], 1) + torch.log(d)


def load_models(a, dev):
    return {k: NLE.load_from_checkpoint(best_checkpoint(getattr(a, k)),
                                        map_location=dev,
                                        pre_transforms=None).eval()
            for k in ('los', 'pm', 'v3d')}


def per_star(a, models, dev):
    pos, vel, th, lrs, n = load_shard(a.shard, a.n_gal_star)
    G = len(n)
    gid = torch.repeat_interleave(torch.arange(G), n)
    st = (torch.zeros(G).index_add_(0, gid, vel[:, 2] ** 2) / n).sqrt()
    R = pos[:, :2].norm(dim=1).to(dev)
    torch.manual_seed(a.seed)
    v_los = (vel[:, 2] + a.err_los * torch.randn(len(gid))).to(dev)
    v_pm = (vel[:, :2] + a.err_pm * torch.randn(len(gid), 2)).to(dev)
    e_los = torch.full_like(v_los, a.err_los)
    e_pm = torch.full_like(v_pm, a.err_pm)
    theta = th[gid].to(dev)
    scale = st[gid].to(dev)
    gid_d = gid.to(dev)
    rows = []
    with torch.no_grad():
        ref_los = log_prob_rows(models['los'], R, v_los[:, None], e_los[:, None],
                                theta, float(lrs[0]))  # lrs per galaxy below
        # lrs differs per galaxy: recompute per-galaxy in chunks
        def per_gal(fn):
            out = torch.empty(len(gid), device=dev)
            for g in range(G):
                m = gid_d == g
                out[m] = fn(m, float(lrs[g]))
            return out
        ref_los = per_gal(lambda m, l: log_prob_rows(
            models['los'], R[m], v_los[m, None], e_los[m, None], theta[m], l))
        ref_pm = per_gal(lambda m, l: log_prob_rows(
            models['pm'], R[m], v_pm[m], e_pm[m], theta[m], l))
        for n_grid in a.grids:
            for sm in (a.sigma_miss, a.sigma_miss_alt):
                m_los = per_gal(lambda m, l: marginal_los(
                    models['v3d'], R[m], v_los[m], e_los[m], scale[m], sm,
                    theta[m], l, n_grid, a.half_range))
                m_pm = per_gal(lambda m, l: marginal_pm(
                    models['v3d'], R[m], v_pm[m], e_pm[m], scale[m], sm,
                    theta[m], l, n_grid, a.half_range))
                for case, d in (('los', ref_los - m_los), ('pm', ref_pm - m_pm)):
                    dg = torch.zeros(G, device=dev).index_add_(0, gid_d, d)
                    for lo, hi in ((0, 1), (1, 3), (3, 10), (10, 40)):
                        mk = (scale >= lo) & (scale < hi)
                        mg = (st.to(dev) >= lo) & (st.to(dev) < hi)
                        if mk.sum() < 300:
                            continue
                        rows.append(dict(
                            case=case, n_grid=n_grid, sigma_miss=sm,
                            sig_lo=lo, sig_hi=hi, n_stars=int(mk.sum()),
                            star_mean=float(d[mk].mean()),
                            star_rms=float(d[mk].pow(2).mean().sqrt()),
                            gal_mean=float(dg[mg].mean()),
                            gal_rms=float(dg[mg].pow(2).mean().sqrt())))
                print(f'grid {n_grid} sigma_miss {sm} done', flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f'per_star_shard{a.shard}.csv', index=False)
    pd.set_option('display.width', 200)
    print('d = log p_single-component - log marginal of p_v3d (nats); '
          'star: per star, gal: summed per galaxy')
    print(df.round(4).to_string(index=False))


class Term:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, t):
        return self.fn(t)


def posterior(a, models, dev):
    pos, vel, th, lrs, n = load_shard(a.shard, a.n_gal)
    ptr = np.concatenate([[0], np.cumsum(n.numpy())])
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    rows = []
    OUT.mkdir(parents=True, exist_ok=True)
    for g in range(len(n)):
        sl = slice(ptr[g], ptr[g + 1])
        ng = int(n[g])
        R = pos[sl][:, :2].norm(dim=1).to(dev)
        v3 = vel[sl].to(dev)
        st = float(v3[:, 2].pow(2).mean().sqrt())
        v_los = v3[:, 2] + a.err_los * torch.randn(ng, device=dev)
        v_pm = v3[:, :2] + a.err_pm * torch.randn(ng, 2, device=dev)
        e_los = torch.full((ng,), a.err_los, device=dev)
        e_pm = torch.full((ng, 2), a.err_pm, device=dev)
        half = torch.zeros(ng, dtype=torch.bool, device=dev)
        half[torch.from_numpy(rng.permutation(ng)[: ng // 2]).to(dev)] = True
        scale = torch.full((ng,), st, device=dev)
        l = float(lrs[g])

        def rep(x, nw):
            return x.repeat(nw, *([1] * (x.dim() - 1)))

        def ll_mixed(t):
            nw = t.shape[0]
            m, mm = half, ~half
            th_los = t.repeat_interleave(int(m.sum()), 0)
            th_pm = t.repeat_interleave(int(mm.sum()), 0)
            a1 = log_prob_rows(models['los'], rep(R[m], nw), rep(v_los[m, None], nw),
                               rep(e_los[m, None], nw), th_los, l).view(nw, -1).sum(1)
            a2 = log_prob_rows(models['pm'], rep(R[mm], nw), rep(v_pm[mm], nw),
                               rep(e_pm[mm], nw), th_pm, l).view(nw, -1).sum(1)
            return a1 + a2

        def ll_marg(t):
            nw = t.shape[0]
            m, mm = half, ~half
            th_los = t.repeat_interleave(int(m.sum()), 0)
            th_pm = t.repeat_interleave(int(mm.sum()), 0)
            a1 = marginal_los(models['v3d'], rep(R[m], nw), rep(v_los[m], nw),
                              rep(e_los[m], nw), rep(scale[m], nw), a.sigma_miss,
                              th_los, l, a.n_grid, a.half_range).view(nw, -1).sum(1)
            a2 = marginal_pm(models['v3d'], rep(R[mm], nw), rep(v_pm[mm], nw),
                             rep(e_pm[mm], nw), rep(scale[mm], nw), a.sigma_miss,
                             th_pm, l, a.n_grid, a.half_range).view(nw, -1).sum(1)
            return a1 + a2

        def ll_v3d(t):
            nw = t.shape[0]
            vel3 = torch.cat([v_los[:, None], v_pm], 1)
            err3 = torch.cat([e_los[:, None], e_pm], 1)
            return log_prob_rows(models['v3d'], rep(R, nw), rep(vel3, nw),
                                 rep(err3, nw), t.repeat_interleave(ng, 0),
                                 l).view(nw, ng).sum(1)

        truth = th[g].numpy()
        tr = float(log_rho_150(truth[None], l)[0])
        for case, fn in (('mixed', ll_mixed), ('marg', ll_marg), ('v3d_all', ll_v3d)):
            def log_post(theta):
                out = np.full(theta.shape[0], -np.inf)
                ok = np.all((theta >= PMIN) & (theta <= PMAX), axis=1)
                if ok.any():
                    t = torch.tensor(theta[ok], dtype=torch.float32, device=dev)
                    with torch.no_grad():
                        out[ok] = fn(t).cpu().numpy()
                return out
            rng_w = np.random.default_rng(a.seed + g)
            p0 = rng_w.uniform(PMIN, PMAX, size=(a.n_walkers, 7))
            moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
            s = emcee.EnsembleSampler(a.n_walkers, 7, log_post, moves=moves,
                                      vectorize=True)
            s.run_mcmc(p0, a.n_steps, progress=False)
            chain = s.get_chain(discard=a.n_steps // 2, flat=True)
            np.save(OUT / f'chain_g{g}_{case}.npy', chain)
            lr = log_rho_150(chain, l)
            lo, med, hi = np.percentile(lr, [16, 50, 84])
            hw = 0.5 * (hi - lo)
            rows.append(dict(galaxy=g, case=case, n_stars=ng, sigma_true=st,
                             log_rstar=l, truth=tr, median=med, halfwidth=hw,
                             z=(med - tr) / hw, pit=float(np.mean(lr < tr))))
        r = {x['case']: x for x in rows[-3:]}
        print(f"[{g + 1}/{len(n)}] sigma_true {st:.1f}: " + ' | '.join(
            f"{c} {r[c]['median']:.2f}+-{r[c]['halfwidth']:.3f} z{r[c]['z']:+.2f}"
            for c in r), flush=True)
        np.save(OUT / f'truth_g{g}.npy', np.append(truth, l))
        np.savez(OUT / f'obs_g{g}.npz', R=R.cpu().numpy(), v_los=v_los.cpu().numpy(),
                 v_pm=v_pm.cpu().numpy(), err_los=a.err_los, err_pm=a.err_pm,
                 half=half.cpu().numpy(), sigma_true=st)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f'posterior_shard{a.shard}.csv', index=False)
    print('\ncase | galaxies | median half-width (dex) | rms z | frac |z|<=1 | mean z')
    for case in ('mixed', 'marg', 'v3d_all'):
        d = df[df.case == case]
        print(f'{case:8s} | {len(d):3d} | {d.halfwidth.median():.3f} | '
              f'{np.sqrt((d.z ** 2).mean()):.2f} | {(d.z.abs() <= 1).mean():.2f} | '
              f'{d.z.mean():+.2f}')
    m = df[df.case == 'mixed'].set_index('galaxy')
    mg = df[df.case == 'marg'].set_index('galaxy')
    dz = (m['median'] - mg['median']) / m.halfwidth
    print(f'mixed vs marg: |median difference| / mixed half-width: median '
          f'{dz.abs().median():.2f}, max {dz.abs().max():.2f}; half-width ratio '
          f'marg/mixed median {(mg.halfwidth / m.halfwidth).median():.2f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--los', required=True)
    ap.add_argument('--pm', required=True)
    ap.add_argument('--v3d', required=True)
    ap.add_argument('--per-star', action='store_true')
    ap.add_argument('--shard', type=int, default=42)
    ap.add_argument('--n-gal-star', type=int, default=400)
    ap.add_argument('--n-gal', type=int, default=20)
    ap.add_argument('--err-los', type=float, default=2.0)
    ap.add_argument('--err-pm', type=float, default=5.0)
    ap.add_argument('--sigma-miss', type=float, default=3.0)
    ap.add_argument('--sigma-miss-alt', type=float, default=10.0)
    ap.add_argument('--grids', type=int, nargs='+', default=[17, 33])
    ap.add_argument('--n-grid', type=int, default=17)
    ap.add_argument('--half-range', type=float, default=7.0)
    ap.add_argument('--n-walkers', type=int, default=48)
    ap.add_argument('--n-steps', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = load_models(a, dev)
    if a.per_star:
        per_star(a, models, dev)
    else:
        posterior(a, models, dev)


if __name__ == '__main__':
    main()
