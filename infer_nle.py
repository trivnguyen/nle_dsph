"""Posterior from a trained per-star NLE at one target's observation.

Reads the exact stars the NPE pipeline used (`target.npz` written by
npe_inference under <results>/<model>/<target>/), so NLE and NPE are
compared on identical data. The galaxy log-likelihood is the i.i.d.
sum of per-star log p(v_los | log10 R, theta, log r_half, sigma_v)
with a systemic velocity v_sys as an eighth, nuisance parameter;
the prior is the 8p_v4 box. emcee does the sampling.

Usage:
    python infer_nle.py --run-dir <nle run dir> \
        --targets dra_mockicrs_walker23_CuspIso dra_walker23_memtli_v250401
Outputs <OUT>/<tag>/<target>.npz with posterior samples (label space
plus log rho at 150 pc) and a summary.csv across targets.
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from nle import NLE
from transforms import StarBatch

RESULTS = Path('/scratch/tvnguyen/npe_inference')
OUT = Path(os.path.expanduser('~/scratch/projects/nle_dsph/inference'))
LABELS = ('dm_alpha', 'dm_beta', 'dm_gamma', 'dm_log_rdm', 'dm_log_rho0',
          'df_beta0', 'df_log_ra')
PRIOR_MIN = np.array([0.5, 2.5, -0.5, 0.0, 5.0, -0.499, -0.5, -50.0])
PRIOR_MAX = np.array([3.0, 10.0, 2.0, 2.5, 10.0, 0.0, 3.0, 50.0])

_G = {}


def log_rho_150(theta: np.ndarray, log_rstar: float) -> np.ndarray:
    """log10 of the Zhao density at 150 pc, theta rows in label space."""
    a, b, g, lrdm, lrho0 = (theta[:, i] for i in range(5))
    x = 0.15 / 10 ** (lrdm + log_rstar)
    return (lrho0 - g * np.log10(x)
            + (g - b) / a * np.log10(1 + x ** a))


def _init(model_path, obs):
    torch.set_num_threads(1)
    _G['model'] = NLE.load_from_checkpoint(model_path, map_location='cpu',
                                           pre_transforms=None).eval()
    _G.update(obs)


def log_post(theta: np.ndarray) -> float:
    if np.any(theta < PRIOR_MIN) or np.any(theta > PRIOR_MAX):
        return -np.inf
    R, v, err, lrs = _G['R'], _G['v'], _G['err'], _G['lrs']
    n = len(R)
    pos = np.zeros((n, 2), dtype=np.float32)
    pos[:, 0] = R  # Reason: only |pos| enters with condition_on_radius.
    batch = StarBatch(
        pos=torch.from_numpy(pos),
        vel=torch.tensor(v - theta[7], dtype=torch.float32).view(n, 1),
        vel_error=torch.tensor(err, dtype=torch.float32).view(n, 1),
        theta=torch.tensor(theta[:7], dtype=torch.float32).expand(n, 7),
        cond=torch.tensor([[lrs]], dtype=torch.float32).expand(n, 1))
    lp = _G['model'].log_prob(batch)
    return float(lp.sum()) if torch.isfinite(lp).all() else -np.inf


def run_target(model_path, npe_model, target, n_workers, n_walkers=64,
               n_steps=5000, seed=0):
    t = np.load(RESULTS / npe_model / target / 'target.npz',
                allow_pickle=True)
    obs = dict(R=t['R_proj_kpc'].astype(np.float32),
               v=t['vlos_kms'].astype(np.float64),
               err=t['vlos_err_kms'].astype(np.float32),
               lrs=float(np.log10(float(t['rhalf_kpc']))))
    import emcee
    rng = np.random.default_rng(seed)
    p0 = rng.uniform(PRIOR_MIN, PRIOR_MAX, size=(n_walkers, 8))
    p0[:, 7] = np.median(obs['v']) + rng.normal(0, 1, n_walkers)
    moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
    with Pool(n_workers, initializer=_init,
              initargs=(model_path, obs)) as pool:
        sampler = emcee.EnsembleSampler(n_walkers, 8, log_post, pool=pool,
                                        moves=moves)
        sampler.run_mcmc(p0, n_steps, progress=False)
    chain = sampler.get_chain(discard=n_steps // 2, flat=True)
    tau = float(np.max(sampler.get_autocorr_time(tol=0)))
    return chain, obs, tau, float(np.mean(sampler.acceptance_fraction))


def main(args):
    run_dir = Path(args.run_dir)
    model_path = str(run_dir / 'checkpoints' / args.checkpoint)
    out = OUT / args.tag
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for target in args.targets:
        chain, obs, tau, acc = run_target(
            model_path, args.npe_model, target, args.n_workers,
            n_steps=args.n_steps)
        lr150 = log_rho_150(chain, obs['lrs'])
        np.savez(out / f'{target}.npz', posterior=chain, log_rho_150=lr150,
                 log_rstar=obs['lrs'], n_stars=len(obs['R']), tau=tau, acc=acc)
        npe = np.load(
            RESULTS / args.npe_model / target / 'posterior_model_units.npy')
        npe_lr150 = log_rho_150(npe, obs['lrs'])
        for j, name in enumerate(list(LABELS) + ['v_sys', 'log_rho_150']):
            col = chain[:, j] if j < 8 else lr150
            a = np.percentile(col, [16, 50, 84])
            if j < 7:
                b = np.percentile(npe[:, j], [16, 50, 84])
            elif j == 8:
                b = np.percentile(npe_lr150, [16, 50, 84])
            else:
                b = np.array([np.nan] * 3)
            rows.append(dict(target=target, n_stars=len(obs['R']),
                             param=name,
                             nle_med=a[1], nle_w68=a[2] - a[0],
                             npe_med=b[1], npe_w68=b[2] - b[0],
                             ratio=(a[2] - a[0]) / (b[2] - b[0])))
        q = np.percentile(lr150, [16, 50, 84])
        qn = np.percentile(npe_lr150, [16, 50, 84])
        print(f'{target}: N={len(obs["R"])} tau={tau:.0f} acc={acc:.2f} '
              f'log rho150 NLE {q[1]:.2f} +- {(q[2] - q[0]) / 2:.2f} | '
              f'NPE {qn[1]:.2f} +- {(qn[2] - qn[0]) / 2:.2f}', flush=True)
    import pandas as pd
    df = pd.DataFrame(rows); df.to_csv(out / 'summary.csv', index=False)
    print('\n68% width ratio NLE / NPE, median over targets:')
    print(df.groupby('param')['ratio'].median().round(3).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--checkpoint', default='last.ckpt')
    ap.add_argument('--npe-model', default='8p_v4')
    ap.add_argument('--targets', nargs='+', required=True)
    ap.add_argument('--tag', default='nle_8p_v4')
    ap.add_argument('--n-workers', type=int, default=12)
    ap.add_argument('--n-steps', type=int, default=5000)
    main(ap.parse_args())
