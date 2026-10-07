"""Per-regime calibration of a per-star NLE on held-out simulated shards.

For every star of the held-out galaxies, at the TRUE theta: the per-star
log p, and the probability integral transform PIT = Phi(z) of the flow's
base coordinate. Binned by the galaxy's true sigma_los and by the assigned
error. A calibrated density has PIT mean 0.5, std 0.2887 and 0.27% of
|z| > 3 in every bin. The Gaussian column is the NLL of a Gaussian with the
TRUE intrinsic dispersion convolved with the error -- the floor for any
Gaussian-LOSVD model.

    python diagnostics/pit_by_regime.py --run-dir <run dir> [--ckpt best]
        [--shards 31 32] [--n-gal 40000] [--errors 0.3 1 3]
Writes <run dir>/pit_by_regime.csv and prints the table.
"""

import argparse, glob, math, os, sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nle import NLE  # noqa: E402
from transforms import StarBatch  # noqa: E402

LABELS = ('dm_alpha', 'dm_beta', 'dm_gamma', 'dm_log_rdm', 'dm_log_rho0',
          'df_beta0', 'df_log_ra')
DATA = Path(os.path.expanduser(
    '~/scratch/projects/sbi_dsph/datasets/8p_ZhaoPlumCOM_v4'))


def best_checkpoint(run_dir) -> str:
    """Return the checkpoint with the lowest validation loss in its name.

    Args:
        run_dir: Run directory holding a `checkpoints/` folder.

    Returns:
        Path of the `*-loss=<val>.ckpt` file with the smallest `<val>`.
    """
    cks = [c for c in glob.glob(f'{run_dir}/checkpoints/*.ckpt')
           if 'loss=' in c]
    return min(cks, key=lambda c: float(c.split('loss=')[1][:-5]))


def load_shard(i: int, n_gal: int) -> tuple:
    """Read the first `n_gal` galaxies of dataset shard `i` as float32.

    Args:
        i: Shard index (`data.<i>.h5`).
        n_gal: Number of galaxies to read from the start of the shard.

    Returns:
        Tuple `(pos, vel, theta, log_rstar, num_stars)`: per-star 3-D
        positions and velocities, and per-galaxy labels, log10 r_half
        [kpc] and star counts.
    """
    with h5py.File(DATA / f'data.{i}.h5', 'r') as h:
        ptr = h['ptr'][:n_gal + 1]
        n = h['num_stars'][:n_gal]
        S = int(ptr[-1])
        pos = torch.tensor(h['pos'][:S], dtype=torch.float32)
        vel = torch.tensor(h['vel'][:S], dtype=torch.float32)
        th = torch.tensor(np.column_stack([h[k][:n_gal] for k in LABELS]),
                          dtype=torch.float32)
        lrs = torch.tensor(h['stellar_log_rstar_kpc'][:n_gal],
                           dtype=torch.float32)
    return pos, vel, th, lrs, torch.tensor(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--ckpt', default='best')
    ap.add_argument('--shards', type=int, nargs='+', default=[31, 32])
    ap.add_argument('--n-gal', type=int, default=40000)
    ap.add_argument('--errors', type=float, nargs='+', default=[0.3, 1.0, 3.0])
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = best_checkpoint(a.run_dir) if a.ckpt == 'best' else a.ckpt
    m = NLE.load_from_checkpoint(ck, map_location=dev,
                                 pre_transforms=None).eval()
    torch.manual_seed(a.seed)
    rows = []
    for sh in a.shards:
        pos, vel, th, lrs, n = load_shard(sh, a.n_gal)
        G = len(n)
        gid = torch.repeat_interleave(torch.arange(G), n)
        sig = torch.zeros(G).index_add_(0, gid, vel[:, 2] ** 2)
        sig = (sig / n).sqrt()
        err_g = torch.tensor(a.errors)[torch.arange(G) % len(a.errors)]
        err = err_g[gid]
        v = vel[:, 2] + torch.randn(len(err)) * err
        vs = float(m.vel_scale[0])
        lp, z = [], []
        with torch.no_grad():
            for s in range(0, len(v), 262144):
                sl = slice(s, s + 262144)
                b = StarBatch(pos=pos[sl, :2], vel=v[sl].view(-1, 1),
                              vel_error=err[sl].view(-1, 1),
                              theta=th[gid[sl]], cond=lrs[gid[sl]].view(-1, 1)
                              ).to(dev)
                nb = m._normalize(b)
                tgt = m._assemble(nb, m.target_fields)
                ctx = m.context_embedding(m._assemble(nb, m.context_fields))
                d = m.flow(ctx)
                lp.append(d.log_prob(tgt).cpu())
                z.append(d.transform(tgt).cpu().squeeze(1))
        lp, z = torch.cat(lp), torch.cat(z)
        s2 = (sig[gid] ** 2 + err ** 2) / vs ** 2
        lp_g = -0.5 * torch.log(2 * math.pi * s2) - 0.5 * (v / vs) ** 2 / s2
        pit = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
        edges = [0, 0.3, 1, 3, 10, 40]
        for e in a.errors:
            for lo, hi in zip(edges[:-1], edges[1:]):
                mk = (sig[gid] >= lo) & (sig[gid] < hi) & (err == e)
                if mk.sum() < 2000:
                    continue
                rows.append(dict(
                    shard=sh, sigma_lo=lo, sigma_hi=hi, err=e,
                    n=int(mk.sum()), nll=float(-lp[mk].mean()),
                    nll_gauss=float(-lp_g[mk].mean()),
                    pit_mean=float(pit[mk].mean()),
                    pit_std=float(pit[mk].std()),
                    tail3=float((z[mk].abs() > 3).float().mean())))
        rows.append(dict(shard=sh, sigma_lo=0, sigma_hi=40, err=-1,
                         n=len(lp), nll=float(-lp.mean()),
                         nll_gauss=float(-lp_g.mean()),
                         pit_mean=float(pit.mean()), pit_std=float(pit.std()),
                         tail3=float((z.abs() > 3).float().mean())))
    df = pd.DataFrame(rows)
    out = Path(a.run_dir) / 'pit_by_regime.csv'
    df.to_csv(out, index=False)
    print(f'checkpoint: {ck}')
    print(df.round(4).to_string(index=False))
    print(f'-> {out}')


if __name__ == '__main__':
    main()
