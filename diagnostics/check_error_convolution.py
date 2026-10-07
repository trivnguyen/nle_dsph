"""Does a noise-trained NLE equal the noiseless NLE convolved with the errors?

If the per-star likelihoods are learned exactly,
    p_noisy(v_obs | R, theta, sigma)
        = Int p_clean(v | R, theta) N(v_obs; v, sigma) dv
for every sigma. This script evaluates both sides at the TRUE theta on
held-out galaxies, for a few fixed sigma values, and reports the
difference per star and summed per galaxy (the quantity that drives a
posterior). It also compares the two log-likelihoods along a 1-D scan of
one parameter for a few galaxies, which is the posterior-level version of
the same question.

The convolution is done by trapezoid quadrature on a per-star grid of
`--n-grid` points covering the intersection of v_obs +- 8 sigma and the
intrinsic range +- 12 sigma_true, so the narrower of the two scales is
resolved (a 0.1 km/s galaxy under 10 km/s errors, or the reverse). Both
log-densities are converted to per-km/s before differencing.

    python diagnostics/check_error_convolution.py \
        --clean <run dir of the noiseless model> \
        --noisy <run dir of the noisy model> \
        [--noisy2 <another noisy run dir>] \
        [--shard 40] [--n-gal 3000] [--sigmas 0.3 1 3 10]
Writes <clean run dir>/error_convolution_<tag>.csv and .png.
"""

import argparse, math, os, sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nle import NLE  # noqa: E402
from transforms import StarBatch  # noqa: E402
from diagnostics.pit_by_regime import (  # noqa: E402
    best_checkpoint, load_shard, LABELS)


def log_prob(m, R, v, err, theta, lrs):
    """Per-star log p for a model; err=None for the noiseless model."""
    n = R.shape[0]
    pos = torch.zeros((n, 2), device=R.device)
    pos[:, 0] = R  # only |pos| enters with condition_on_radius
    b = StarBatch(pos=pos, vel=v.view(n, 1),
                  vel_error=None if err is None else err.view(n, 1),
                  theta=theta, cond=lrs.view(n, 1))
    # Reason: NLE.log_prob is a density per normalized velocity unit and
    # each model has its own vel_scale; compare per km/s.
    return m.log_prob(b) - torch.log(m.vel_scale[0])


def convolved_log_prob(m_clean, R, v_obs, sigma, sig_true, theta, lrs,
                       n_grid=1025, chunk=2048):
    """log Int p_clean(v | ctx) N(v_obs; v, sigma) dv by trapezoid rule."""
    out = torch.empty_like(v_obs)
    for a in range(0, len(v_obs), chunk):
        sl = slice(a, a + chunk)
        vo, st = v_obs[sl], sig_true[sl]
        # Reason: the integrand is non-negligible only where BOTH factors
        # are, so the grid covers the intersection of the clean density's
        # support (|v| < 12 sigma_true) and the kernel's (v_obs +- 8 sigma);
        # a union grid under-resolves a 0.1 km/s peak under 10 km/s errors.
        lo = torch.maximum(vo - 8 * sigma, -12 * st)
        hi = torch.minimum(vo + 8 * sigma, 12 * st)
        hi = torch.maximum(hi, lo + 1e-3)
        t = torch.linspace(0, 1, n_grid, device=vo.device)
        grid = lo[:, None] + (hi - lo)[:, None] * t[None, :]          # (B, G)
        B = grid.shape[0]
        lp = log_prob(
            m_clean, R[sl].repeat_interleave(n_grid), grid.reshape(-1), None,
            theta[sl].repeat_interleave(n_grid, 0),
            lrs[sl].repeat_interleave(n_grid)).view(B, n_grid)
        lk = -0.5 * ((vo[:, None] - grid) / sigma) ** 2 - math.log(
            sigma * math.sqrt(2 * math.pi))
        f = lp + lk
        dv = ((hi - lo) / (n_grid - 1))[:, None]
        w = torch.full_like(f, 0.0)
        w[:, 1:-1] = 0.0
        w[:, [0, -1]] = math.log(0.5)
        out[sl] = torch.logsumexp(f + w, dim=1) + torch.log(dv[:, 0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clean', required=True)
    ap.add_argument('--noisy', required=True)
    ap.add_argument('--noisy2', default=None,
                    help='second noisy model (e.g. the 5-shard T2 run)')
    ap.add_argument('--shard', type=int, default=40)
    ap.add_argument('--n-gal', type=int, default=3000)
    ap.add_argument('--sigmas', type=float, nargs='+',
                    default=[0.3, 1.0, 3.0, 10.0])
    ap.add_argument('--n-grid', type=int, default=1025)
    ap.add_argument('--tag', default='')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    mc = NLE.load_from_checkpoint(best_checkpoint(a.clean), map_location=dev,
                                  pre_transforms=None).eval()
    noisy = {'noisy': NLE.load_from_checkpoint(
        best_checkpoint(a.noisy), map_location=dev,
        pre_transforms=None).eval()}
    if a.noisy2:
        noisy['noisy2'] = NLE.load_from_checkpoint(
            best_checkpoint(a.noisy2), map_location=dev,
            pre_transforms=None).eval()
    assert not mc.condition_on_error, 'clean model must not condition on sigma'

    pos, vel, th, lrs, n = load_shard(a.shard, a.n_gal)
    G = len(n)
    gid = torch.repeat_interleave(torch.arange(G), n)
    sig_true = torch.zeros(G).index_add_(0, gid, vel[:, 2] ** 2)
    sig_true = (sig_true / n).sqrt()
    R = pos[:, :2].norm(dim=1).to(dev)
    vt = vel[:, 2].to(dev)
    theta = th[gid].to(dev)
    lr = lrs[gid].to(dev)
    st = sig_true[gid].to(dev)
    gid_d = gid.to(dev)
    torch.manual_seed(a.seed)

    rows, per_star = [], {}
    with torch.no_grad():
        for sigma in a.sigmas:
            v_obs = vt + sigma * torch.randn_like(vt)
            err = torch.full_like(vt, sigma)
            lconv = convolved_log_prob(mc, R, v_obs, sigma, st, theta, lr,
                                       n_grid=a.n_grid)
            for name, mn in noisy.items():
                ln = log_prob(mn, R, v_obs, err, theta, lr)
                d = ln - lconv
                dg = torch.zeros(G, device=dev).index_add_(0, gid_d, d)
                per_star[(name, sigma)] = (d.cpu(), dg.cpu(), st.cpu())
                for lo, hi in ((0, 1), (1, 3), (3, 10), (10, 40)):
                    mk = (st >= lo) & (st < hi)
                    mg = (sig_true.to(dev) >= lo) & (sig_true.to(dev) < hi)
                    if mk.sum() < 500:
                        continue
                    rows.append(dict(
                        model=name, sigma=sigma, sig_lo=lo, sig_hi=hi,
                        n_stars=int(mk.sum()), n_gal=int(mg.sum()),
                        star_mean=float(d[mk].mean()),
                        star_rms=float(d[mk].pow(2).mean().sqrt()),
                        star_p99=float(d[mk].abs().quantile(0.99)),
                        gal_mean=float(dg[mg].mean()),
                        gal_rms=float(dg[mg].pow(2).mean().sqrt()),
                        gal_p99=float(dg[mg].abs().quantile(0.99)),
                        nll_noisy=float(-ln[mk].mean()),
                        nll_conv=float(-lconv[mk].mean())))
    df = pd.DataFrame(rows)
    tag = a.tag or f'shard{a.shard}'
    out = Path(a.clean) / f'error_convolution_{tag}.csv'
    df.to_csv(out, index=False)
    pd.set_option('display.width', 250)
    print('d = log p_noisy - log Int p_clean N  (nats); '
          'star = per star, gal = summed per galaxy')
    print(df.round(4).to_string(index=False))
    print(f'-> {out}')

    # posterior-level: scan dm_log_rho0 for 6 galaxies at sigma = 1 and 3
    try:
        k = LABELS.index('dm_log_rho0')
        gsel = torch.arange(G)[(sig_true > 2) & (sig_true < 15)][:6]
        fig, axes = plt.subplots(2, len(gsel), figsize=(3.2 * len(gsel), 6),
                                 sharex=False)
        scan = torch.linspace(5.0, 10.0, 81, device=dev)
        with torch.no_grad():
            for col, g in enumerate(gsel):
                mk = (gid_d == int(g))
                for row, sigma in enumerate((1.0, 3.0)):
                    v_obs = vt[mk] + sigma * torch.randn(int(mk.sum()),
                                                         device=dev)
                    err = torch.full_like(v_obs, sigma)
                    curves = {name: [] for name in noisy}
                    curves['conv'] = []
                    for val in scan:
                        tt = theta[mk].clone(); tt[:, k] = val
                        curves['conv'].append(float(convolved_log_prob(
                            mc, R[mk], v_obs, sigma, st[mk], tt, lr[mk],
                            n_grid=a.n_grid).sum()))
                        for name, mn in noisy.items():
                            curves[name].append(float(log_prob(
                                mn, R[mk], v_obs, err, tt, lr[mk]).sum()))
                    ax = axes[row, col]
                    for name, c in curves.items():
                        c = np.array(c)
                        ax.plot(scan.cpu(), c - c.max(), label=name)
                    ax.axvline(float(th[g, k]), color='k', ls=':', lw=1)
                    ax.set_ylim(-30, 1)
                    ax.set_title(f'gal {int(g)}, '
                                 f'sigma_true={sig_true[g]:.1f}, '
                                 f'err={sigma}', fontsize=8)
                    if col == 0:
                        ax.set_ylabel('log L - max')
                    ax.set_xlabel('dm_log_rho0')
        axes[0, 0].legend(fontsize=7)
        fig.tight_layout()
        png = Path(a.clean) / f'error_convolution_{tag}.png'
        fig.savefig(png, dpi=120)
        print(f'-> {png}')
    except Exception as ex:  # plotting is a convenience, not the test
        print('scan figure skipped:', ex)


if __name__ == '__main__':
    main()
