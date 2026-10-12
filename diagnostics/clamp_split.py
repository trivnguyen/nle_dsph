"""Does the BvH clamp explain its unconverged validation loss?

The ktwin sampler behind `10p_ZhaoPlumJeansBVH_ktwin` clamps the
kurtosis when a galaxy's fourth moment leaves the range it can sample,
and records the fraction of stars affected as `frac_clamped`. Where the
clamp is active the velocities are not draws from the distribution
function the labels specify, so the map from theta to the velocity
density has a discontinuity - something a flow chases without settling.
That is the hypothesis for the 0.038-nat scatter this run's validation
loss shows after step 334k, against 0.007 for the ABG run.

This splits a held-out shard by `frac_clamped` and reports, per bin,
the per-star NLL and the calibration of the model at the true labels.
A clamp-driven failure looks like: clean galaxies calibrated and tight,
clamped ones mis-calibrated, and the gap growing with the clamped
fraction. A uniformly poor table would mean something else is wrong.

    python diagnostics/clamp_split.py --run-dir <run dir> [--shards 22 23]
"""

import argparse
import math
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nle import NLE  # noqa: E402
from transforms import StarBatch  # noqa: E402
from diagnostics.pit_by_regime import best_checkpoint  # noqa: E402

DATA = Path(os.path.expanduser(
    '~/scratch/projects/sbi_dsph/datasets/10p_ZhaoPlumJeansBVH_ktwin'))
LABELS = ('dm_alpha', 'dm_beta', 'dm_gamma', 'dm_log_rdm', 'dm_log_rho0',
          'df_betasym0', 'df_betasyminf', 'df_log_r0', 'df_n_aniso')
EDGES = (-1e-9, 1e-9, 0.05, 0.2, 0.5, 1.0001)
NAMES = ('0 (clean)', '(0, 0.05]', '(0.05, 0.2]', '(0.2, 0.5]', '(0.5, 1]')


def load(shard: int, n_gal: int):
    """Read one shard's stars, labels and clamp fraction."""
    with h5py.File(DATA / f'data.{shard}.h5', 'r') as h:
        ptr = h['ptr'][:n_gal + 1]
        n = h['num_stars'][:n_gal]
        stars = int(ptr[-1])
        pos = torch.tensor(h['pos'][:stars], dtype=torch.float32)
        vel = torch.tensor(h['vel'][:stars], dtype=torch.float32)
        theta = torch.tensor(
            np.column_stack([h[k][:n_gal] for k in LABELS]),
            dtype=torch.float32)
        lrs = torch.tensor(h['stellar_log_rstar_kpc'][:n_gal],
                           dtype=torch.float32)
        frac = np.asarray(h['frac_clamped'][:n_gal], dtype=float)
    return pos, vel, theta, lrs, torch.tensor(n), frac


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--shards', type=int, nargs='+', default=[22, 23])
    ap.add_argument('--n-gal', type=int, default=20000)
    ap.add_argument('--errors', type=float, nargs='+', default=[1.0, 3.0])
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = best_checkpoint(a.run_dir)
    model = NLE.load_from_checkpoint(
        ckpt, map_location=device, pre_transforms=None).eval()
    torch.manual_seed(a.seed)

    rows = []
    for shard in a.shards:
        pos, vel, theta, lrs, n, frac = load(shard, a.n_gal)
        gid = torch.repeat_interleave(torch.arange(len(n)), n)
        sigma = (torch.zeros(len(n)).index_add_(
            0, gid, vel[:, 2] ** 2) / n).sqrt()
        err_g = torch.tensor(a.errors)[torch.arange(len(n)) % len(a.errors)]
        err = err_g[gid]
        v = vel[:, 2] + err * torch.randn(len(err))
        scale = float(model.vel_scale[0])
        lp, z = [], []
        with torch.no_grad():
            for s in range(0, len(v), 262144):
                sl = slice(s, s + 262144)
                batch = StarBatch(
                    pos=pos[sl, :2], vel=v[sl].view(-1, 1),
                    vel_error=err[sl].view(-1, 1), theta=theta[gid[sl]],
                    cond=lrs[gid[sl]].view(-1, 1)).to(device)
                norm = model._normalize(batch)
                target = model._assemble(norm, model.target_fields)
                context = model.context_embedding(
                    model._assemble(norm, model.context_fields))
                dist = model.flow(context)
                lp.append(dist.log_prob(target).cpu())
                z.append(dist.transform(target).cpu().squeeze(1))
        lp, z = torch.cat(lp), torch.cat(z)
        pit = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
        # A Gaussian at the galaxy's true dispersion: the floor any
        # Gaussian-LOSVD model could reach, so the columns stay
        # comparable between bins of different dispersion.
        s2 = (sigma[gid] ** 2 + err ** 2) / scale ** 2
        lp_gauss = (-0.5 * torch.log(2 * math.pi * s2)
                    - 0.5 * (v / scale) ** 2 / s2)
        fs = frac[gid.numpy()]
        for lo, hi, name in zip(EDGES[:-1], EDGES[1:], NAMES):
            mask = (fs > lo) & (fs <= hi)
            if mask.sum() < 2000:
                continue
            m = torch.from_numpy(mask)
            rows.append(dict(
                shard=shard, bin=name, n_gal=int(((frac > lo) &
                                                  (frac <= hi)).sum()),
                n_stars=int(mask.sum()),
                nll=float(-lp[m].mean()),
                nll_gauss=float(-lp_gauss[m].mean()),
                pit_mean=float(pit[m].mean()),
                pit_std=float(pit[m].std()),
                tail3=float((z[m].abs() > 3).float().mean())))
    df = pd.DataFrame(rows)
    out = Path(a.run_dir) / 'clamp_split.csv'
    df.to_csv(out, index=False)
    pd.set_option('display.width', 200)
    print(f'checkpoint: {ckpt}')
    print('per-star NLL and calibration at the true labels, by the '
          'galaxy\'s clamped fraction')
    print('(calibrated: pit_mean 0.5, pit_std 0.2887, tail3 0.0027)')
    print(df.round(4).to_string(index=False))
    agg = df.groupby('bin', sort=False).apply(
        lambda g: pd.Series({
            'n_stars': g.n_stars.sum(),
            'nll': np.average(g.nll, weights=g.n_stars),
            'nll_gauss': np.average(g.nll_gauss, weights=g.n_stars),
            'pit_mean': np.average(g.pit_mean, weights=g.n_stars),
            'pit_std': np.average(g.pit_std, weights=g.n_stars)}),
        include_groups=False)
    print('\nboth shards together:')
    print(agg.round(4).to_string())
    print(f'-> {out}')


if __name__ == '__main__':
    main()
