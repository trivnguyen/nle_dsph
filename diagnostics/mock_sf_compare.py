"""Velocity-only against joint-position NLE on the Draco mocks.

Compares p(v_los | log R, theta, log r_half, sigma) (`--los`) with
p(x, y, v_los | theta, log r_half, sigma) (`--joint`) on the twelve mocks
in vr_catalogs/mock/icrs: six with the Walker 2023 footprint applied
(draco1_walker23_icrs, 510 stars) and six without selection
(draco1_walker23_noSF). The mocks are read with the pipeline's own loader
(perspective correction, |v_los| < 50 km/s cut, Draco's LVDB half-light
radius), so both models see the stars the NPE and T2 runs saw. emcee on
the GPU over the seven labels and the systemic velocity under the 8p_v4
box; output per mock and model: median and 68% half-width of
log10 rho(150 pc), z = (median - truth) / half-width, and the ratio of
the joint model's half-width to the velocity-only one.

    python diagnostics/mock_sf_compare.py --los <run dir> --joint <run dir>
Writes <out>/mock_sf_compare.csv and the chains as <out>/<set>_<mock>_<model>.npy.
"""

import argparse
import inspect
import os
import sys
from pathlib import Path

import emcee
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, os.path.expanduser('~/my_modules'))
import dsph_analysis.kinematic_io as kio  # noqa: E402
from nle import NLE  # noqa: E402
from transforms import StarBatch  # noqa: E402
from infer_nle import PRIOR_MIN, PRIOR_MAX, LABELS, log_rho_150  # noqa: E402
from diagnostics.pit_by_regime import best_checkpoint  # noqa: E402

MOCKS = Path(os.path.expanduser('~/scratch/vr_catalogs/vr_catalogs/mock/icrs'))
SETS = {'walker23': 'draco1_walker23_icrs', 'noSF': 'draco1_walker23_noSF'}
PROFILES = ('CuspIso', 'CoreIso', 'CuspOM', 'CoreOM', 'CuspTan', 'CoreTan')
OUT = Path(os.path.expanduser('~/scratch/projects/sbi_dsph/npe_gap/nle/mock_sf'))


def truth_labels(row: pd.Series, rhalf_kpc: float) -> np.ndarray:
    """Injected parameters in label space (radii in units of r_half)."""
    log_ra = np.log10(row.r_a / rhalf_kpc) if np.isfinite(row.r_a) else PRIOR_MAX[6]
    return np.array([row.alp, row.bet, row.gam, np.log10(row.r_s / rhalf_kpc),
                     np.log10(row.rho_s), row.beta0, log_ra])


def load_mock(set_dir: str, profile: str, meta):
    loader = kio.LOADERS['mock_icrs']
    kw = {k: v for k, v in dict(apply_perspective_corr=True,
                                vlos_abs_max=50.0).items()
          if k in inspect.signature(loader).parameters}
    d = kio.load_kinematic_data(str(MOCKS / set_dir / f'{profile}_mock_catalog.csv'),
                                meta, source='mock_icrs', **kw)
    return dict(X=d.X_proj.value.astype(np.float32),
                Y=d.Y_proj.value.astype(np.float32),
                v=d.vlos.value.astype(np.float32),
                err=d.vlos_err.value.astype(np.float32))


def galaxy_log_like(model, obs, lrs, theta, device):
    """Summed per-star log p for each row of theta (7 labels + v_sys)."""
    n = len(obs['v'])
    X = torch.tensor(obs['X'], device=device)
    Y = torch.tensor(obs['Y'], device=device)
    v = torch.tensor(obs['v'], device=device)
    err = torch.tensor(obs['err'], device=device).view(n, 1)
    t = torch.tensor(theta, dtype=torch.float32, device=device)
    nw = t.shape[0]
    pos = torch.stack([X, Y], 1).repeat(nw, 1)
    vel = (v.repeat(nw) - t[:, 7].repeat_interleave(n)).view(-1, 1)
    b = StarBatch(pos=pos, vel=vel, vel_error=err.repeat(nw, 1),
                  theta=t[:, :7].repeat_interleave(n, 0),
                  cond=torch.full((nw * n, 1), lrs, device=device))
    with torch.no_grad():
        lp = model.log_prob(b).view(nw, n)
    lp = torch.where(torch.isfinite(lp), lp, torch.full_like(lp, -1e30))
    return lp.sum(1).cpu().numpy()


def position_factor_leak(models, obs, lrs, chain, device, n_samples=400):
    """Theta-dependence of the joint model's implied position factor.

    log L_joint(theta) - log L_los(theta) = sum_i log p(x_i, y_i | theta,
    r_half) is a constant in theta when the position factor is the
    theta-independent Plummer profile. Returns the std of that difference
    over posterior samples (nats) and its correlation with log rho(150 pc).
    """
    idx = np.random.default_rng(1).choice(len(chain), n_samples, replace=False)
    th = chain[idx]
    d = (galaxy_log_like(models['joint'], obs, lrs, th, device)
         - galaxy_log_like(models['los'], obs, lrs, th, device))
    lr = log_rho_150(th[:, :7], lrs)
    return float(np.std(d)), float(np.corrcoef(d, lr)[0, 1])


def sample(model, obs, lrs, n_walkers, n_steps, seed, device):
    """emcee over 7 labels + v_sys; all walkers evaluated in one GPU batch."""

    def log_post(theta):
        out = np.full(theta.shape[0], -np.inf)
        ok = np.all((theta >= PRIOR_MIN) & (theta <= PRIOR_MAX), axis=1)
        if ok.any():
            out[ok] = galaxy_log_like(model, obs, lrs, theta[ok], device)
        return out

    rng = np.random.default_rng(seed)
    p0 = rng.uniform(PRIOR_MIN, PRIOR_MAX, size=(n_walkers, 8))
    p0[:, 7] = np.median(obs['v']) + rng.normal(0, 1, n_walkers)
    moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
    s = emcee.EnsembleSampler(n_walkers, 8, log_post, moves=moves,
                              vectorize=True)
    s.run_mcmc(p0, n_steps, progress=False)
    return s.get_chain(discard=n_steps // 2, flat=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--los', required=True)
    ap.add_argument('--joint', default=None)
    ap.add_argument('--sets', nargs='+', default=list(SETS))
    ap.add_argument('--profiles', nargs='+', default=list(PROFILES))
    ap.add_argument('--n-walkers', type=int, default=64)
    ap.add_argument('--n-steps', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = {'los': NLE.load_from_checkpoint(
        best_checkpoint(a.los), map_location=dev, pre_transforms=None).eval()}
    if a.joint:
        models['joint'] = NLE.load_from_checkpoint(
            best_checkpoint(a.joint), map_location=dev, pre_transforms=None).eval()
        assert not models['joint'].condition_on_position
    assert models['los'].condition_on_position

    meta = kio.load_meta('draco_1')
    rhalf = float(meta.rhalf_kpc.value)
    lrs = float(np.log10(rhalf))
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for set_name in a.sets:
        params = pd.read_csv(MOCKS / SETS[set_name] / 'mock_catalog_params.csv')
        for profile in a.profiles:
            obs = load_mock(SETS[set_name], profile, meta)
            truth = truth_labels(params[params.profile == profile].iloc[0], rhalf)
            tr150 = float(log_rho_150(truth[None], lrs)[0])
            hw, chains = {}, {}
            for name, m in models.items():
                chain = sample(m, obs, lrs, a.n_walkers, a.n_steps, a.seed, dev)
                chains[name] = chain
                np.save(OUT / f'{set_name}_{profile}_{name}.npy', chain)
                lr = log_rho_150(chain[:, :7], lrs)
                lo, med, hi = np.percentile(lr, [16, 50, 84])
                hw[name] = 0.5 * (hi - lo)
                rows.append(dict(set=set_name, mock=profile, model=name,
                                 n_stars=len(obs['v']), truth=tr150, median=med,
                                 halfwidth=hw[name], z=(med - tr150) / hw[name]))
                print(f'{set_name:8s} {profile:8s} {name:5s} n={len(obs["v"]):3d} '
                      f'log rho150 = {med:.3f} +- {hw[name]:.3f} (truth {tr150:.3f}, '
                      f'z {(med - tr150) / hw[name]:+.2f})', flush=True)
            if 'joint' in hw:
                leak_std, leak_corr = position_factor_leak(
                    models, obs, lrs, chains['los'], dev)
                for r in rows[-2:]:
                    r['leak_std_nats'] = leak_std
                    r['leak_corr_rho150'] = leak_corr
                print(f'{"":17s} half-width ratio joint/los = '
                      f'{hw["joint"] / hw["los"]:.2f} | position-factor leak: '
                      f'std {leak_std:.2f} nats, corr with log rho150 '
                      f'{leak_corr:+.2f}')
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'mock_sf_compare.csv', index=False)
    print(f'-> {OUT}/mock_sf_compare.csv')


if __name__ == '__main__':
    main()
