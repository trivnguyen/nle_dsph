"""Corner and profile figures for marginal_test.py chains: mixed vs
marginalised 3-D vs full 3-D on the same galaxy.

    python diagnostics/plot_marginal.py [--galaxies 0 1 2] [--shard 42]
Reads <marginal dir>/chain_g<g>_<case>.npy and truth_g<g>.npy, writes
<marginal dir>/figures/{corner,profiles}_g<g>.png. Default: every galaxy
with a complete set of chains. The profile figure's data panels use the
v_los of every star (what the fully measured case sees); the mixed and
marginalised likelihoods saw v_los on half of them. Observations come
from obs_g<g>.npz when marginal_test saved it, else they are regenerated
by replaying marginal_test's seeded draws on the same device.
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path

from multiprocessing import Pool

import corner
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import scripts.diagnostics.plot_mock_sf as pms  # noqa: E402
from scripts.diagnostics.plot_mock_sf import LABELS_KPC, to_kpc  # noqa: E402
from scripts.diagnostics.pit_by_regime import load_shard  # noqa: E402

MARG = Path(os.path.expanduser('~/scratch/projects/nle_dsph/marginal'))
# Okabe-Ito blue / vermillion / bluish green.
STYLE = {
    'mixed': dict(color='#0072B2', ls='-', marker='o',
                  label='mixed: los + pm models'),
    'marg': dict(color='#D55E00', ls='--', marker='s',
                 label='3-D model, missing components integrated'),
    'v3d_all': dict(color='#009E73', ls=':', marker='^',
                    label='3-D model, all components measured'),
}
# profile_figure looks its styles up by chain name in plot_mock_sf's dicts
pms.STYLE.update({v['label']: v for v in STYLE.values()})
pms.X_OFFSET.update({STYLE['mixed']['label']: 0.92,
                     STYLE['marg']['label']: 1.0,
                     STYLE['v3d_all']['label']: 1.08})


def replay_obs(shard, n_gal, g, err_los, err_pm, seed, device):
    """Regenerate marginal_test's observation for galaxy g by replaying
    its torch draws in order (same seed, same device)."""
    pos, vel, th, lrs, n = load_shard(shard, n_gal)
    ptr = np.concatenate([[0], np.cumsum(n.numpy())])
    torch.manual_seed(seed)
    for k in range(g + 1):
        ng = int(n[k])
        a = torch.randn(ng, device=device)
        b = torch.randn(ng, 2, device=device)
    sl = slice(ptr[g], ptr[g + 1])
    v3 = vel[sl].to(device)
    return dict(X=pos[sl][:, 0].numpy(), Y=pos[sl][:, 1].numpy(),
                v=(v3[:, 2] + err_los * a).cpu().numpy(),
                err=np.full(ng, err_los, dtype=np.float32),
                sigma_true=float(v3[:, 2].pow(2).mean().sqrt()))


def load_obs(g, a, device):
    f = MARG / f'obs_g{g}.npz'
    if f.exists():
        o = np.load(f)
        R = o['R']
        return dict(X=R, Y=np.zeros_like(R), v=o['v_los'],
                    err=np.full(len(R), float(o['err_los']), dtype=np.float32),
                    sigma_true=float(o['sigma_true']))
    return replay_obs(a.shard, a.n_gal, g, a.err_los, a.err_pm, a.seed, device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--galaxies', type=int, nargs='*', default=None)
    ap.add_argument('--shard', type=int, default=42)
    ap.add_argument('--n-gal', type=int, default=20)
    ap.add_argument('--err-los', type=float, default=2.0)
    ap.add_argument('--err-pm', type=float, default=5.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-draws', type=int, default=100)
    ap.add_argument('--no-profiles', action='store_true')
    a = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gal = a.galaxies
    if gal is None:
        # Reason: the galaxy id was re-extracted inside an f-string, which
        # is a SyntaxError before Python 3.12 (backslash in the expression).
        def galaxy_id(path):
            return re.search(r'truth_g(\d+)', path).group(1)

        gal = sorted(
            int(galaxy_id(f))
            for f in glob.glob(str(MARG / 'truth_g*.npy'))
            if all((MARG / f'chain_g{galaxy_id(f)}_{c}.npy').exists()
                   for c in STYLE))
    (MARG / 'figures').mkdir(exist_ok=True)
    pool = Pool(8)
    for g in gal:
        t = np.load(MARG / f'truth_g{g}.npy')
        truth, lrs = t[:7], float(t[7])
        chains = {st['label']: np.load(MARG / f'chain_g{g}_{case}.npy')
                  for case, st in STYLE.items()}
        if not a.no_profiles:
            obs = load_obs(g, a, device)
            pms.profile_figure(
                chains, obs, lrs, truth,
                f'held-out galaxy {g} (sigma_true = {obs["sigma_true"]:.1f} km/s, '
                f'{len(obs["v"])} stars; mixed/marg saw v_los on half of them)',
                MARG / 'figures' / f'profiles_g{g}.png', a.n_draws, pool)
        fig = None
        for case, st in STYLE.items():
            f = MARG / f'chain_g{g}_{case}.npy'
            if not f.exists():
                continue
            fig = corner.corner(
                to_kpc(np.load(f), lrs), labels=LABELS_KPC, fig=fig,
                color=st['color'], hist_kwargs=dict(density=True, ls=st['ls'], lw=1.6),
                contour_kwargs=dict(linestyles=st['ls']), plot_datapoints=False,
                plot_density=False, levels=(0.68, 0.95), smooth=1.0,
                truths=to_kpc(truth[None], lrs)[0], truth_color='black')
        handles = [plt.Line2D([], [], color=s['color'], ls=s['ls'], label=s['label'])
                   for s in STYLE.values()]
        fig.legend(handles=handles, loc='upper right', fontsize=11)
        fig.suptitle(f'held-out galaxy {g}: v_los on half the stars, (v_x, v_y) '
                     'on the other half; truth in black', fontsize=12)
        out = MARG / 'figures' / f'corner_g{g}.png'
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print('wrote', out)


if __name__ == '__main__':
    main()
