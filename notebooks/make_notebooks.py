"""Generate the interactive notebooks from the diagnostics modules.

Each notebook imports the corresponding `diagnostics/*.py` module and
calls its functions, so the logic lives in one place; the notebook holds
the parameters, the run, and the inline figures.  Re-run this script
after changing a notebook's cell list:  python notebooks/make_notebooks.py
"""

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
SETUP = '''import os, sys
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT)); sys.path.insert(0, os.path.expanduser('~/my_modules'))
os.chdir(ROOT)
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import Image, display
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)
MODELS = Path(os.path.expanduser('~/scratch/projects/sbi_dsph/trained_models/nle/nle_8p_ZhaoPlumCOM_v4'))
OUT = Path(os.path.expanduser('~/scratch/projects/nle_dsph'))'''


def nb(title, cells):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(f'# {title}')]
    for kind, src in cells:
        n.cells.append(nbf.v4.new_markdown_cell(src) if kind == 'md'
                       else nbf.v4.new_code_cell(src))
    n.metadata['kernelspec'] = dict(name='python3', display_name='Python 3',
                                    language='python')
    return n


BOOKS = {
    '01_screening.ipynb': nb('Screening arms: best validation loss per architecture', [
        ('md', 'Table: arm, architecture, wandb run, parameter count, best `val/loss` '
               '(nats per star on the deterministic 50k-galaxy validation set), the step '
               'of that best, and wall time. Reads the run dirs named in '
               '`~/scratch/local_logs/nle/*screen_<arm>_*`.'),
        ('code', SETUP),
        ('code', 'from diagnostics import screening_table\nscreening_table.main()'),
    ]),
    '02_pit_by_regime.ipynb': nb('Per-regime calibration (PIT) of a trained NLE', [
        ('md', 'For every star of held-out simulated galaxies, at the true theta: per-star '
               'log p and the PIT of the flow base coordinate, binned by the galaxy\'s true '
               'sigma_los and the assigned error. A calibrated density has PIT mean 0.5, '
               'std 0.2887 and 0.27% of |z| > 3 in every bin. `nll_gauss` is the NLL of a '
               'Gaussian with the TRUE intrinsic dispersion convolved with the error.'),
        ('code', SETUP),
        ('code', "RUN = MODELS / '37vbi58p'   # the model to test\nSHARDS = [31]\nN_GAL = 20000\nERRORS = [0.3, 1.0, 3.0]"),
        ('code', "from diagnostics import pit_by_regime\n"
                 "sys.argv = ['pit', '--run-dir', str(RUN), '--shards', *map(str, SHARDS), '--n-gal', str(N_GAL), '--errors', *map(str, ERRORS)]\n"
                 "pit_by_regime.main()"),
        ('code', "df = pd.read_csv(RUN / 'pit_by_regime.csv')\n"
                 "df = df[df.err > 0]\n"
                 "fig, ax = plt.subplots(1, 3, figsize=(14, 4))\n"
                 "for e, g in df.groupby('err'):\n"
                 "    x = np.sqrt(g.sigma_lo.clip(lower=0.1) * g.sigma_hi)\n"
                 "    ax[0].plot(x, g.pit_mean, 'o-', label=f'err {e} km/s'); ax[1].plot(x, g.pit_std, 'o-'); ax[2].plot(x, g.nll - g.nll_gauss, 'o-')\n"
                 "ax[0].axhline(0.5, color='k', lw=0.8); ax[1].axhline(0.2887, color='k', lw=0.8); ax[2].axhline(0, color='k', lw=0.8)\n"
                 "for a, t in zip(ax, ['PIT mean (0.5)', 'PIT std (0.2887)', 'NLL - Gaussian(true sigma) [nats/star]']):\n"
                 "    a.set_xscale('log'); a.set_xlabel('sigma_true bin centre [km/s]'); a.set_title(t); a.grid(alpha=.3)\n"
                 "ax[0].legend(); plt.show()"),
    ]),
    '03_error_convolution.ipynb': nb('Noise-trained NLE vs noiseless NLE convolved with the error kernel', [
        ('md', 'd = log p_noisy(v_obs | sigma) - log Int p_clean(v) N(v_obs; v, sigma) dv at the '
               'true theta, per star and summed per galaxy, by error sigma and by the galaxy\'s '
               'intrinsic dispersion bin. Both densities are per km/s; the quadrature grid '
               'covers the overlap of the two supports. Agreement means the noise augmentation '
               'is learned consistently with the physics.'),
        ('code', SETUP),
        ('code', "CLEAN = MODELS / 'voj63drr'      # p(v_true | R, theta, r_half)\nNOISY = MODELS / '6xaw28uw'      # p(v_obs | R, theta, r_half, sigma), 2 shards\nNOISY2 = MODELS / '37vbi58p'     # the 5-shard T2 model\nSHARD, N_GAL = 40, 1000\nSIGMAS = [0.3, 1.0, 3.0, 10.0]"),
        ('code', "from diagnostics import check_error_convolution as cec\n"
                 "sys.argv = ['cec', '--clean', str(CLEAN), '--noisy', str(NOISY), '--noisy2', str(NOISY2), '--shard', str(SHARD), '--n-gal', str(N_GAL), '--sigmas', *map(str, SIGMAS), '--tag', 'nb']\n"
                 "cec.main()"),
        ('code', "df = pd.read_csv(CLEAN / 'error_convolution_nb.csv'); display(df.round(4))\ndisplay(Image(str(CLEAN / 'error_convolution_nb.png')))"),
    ]),
    '04_mock_sf.ipynb': nb('Velocity-only p(v | R) vs joint p(x, y, v) on the Draco mocks', [
        ('md', 'Chains from `diagnostics/mock_sf_compare.py` for the twelve mocks (six with the '
               'Walker 2023 footprint, six without selection). Summary columns: set, mock, model, '
               'star count, true and posterior median log10 rho(150 pc) in Msun/kpc^3, 68% '
               'half-width (dex), z = (median - truth)/half-width, and the position-factor leak: '
               'std over the velocity-only posterior of log L_joint - log L_los (nats; zero for an '
               'exact theta-independent position factor) and its correlation with log rho(150). '
               'Figures: corner plots (seven labels, radii in kpc) and four-panel profiles '
               '(density, slope, sigma_los and kappa_los against the binned data and the truth).'),
        ('code', SETUP),
        ('code', "CHAINS = OUT / 'mock_sf'\ndf = pd.read_csv(CHAINS / 'mock_sf_compare.csv')\n"
                 "wide = df.pivot_table(index=['set', 'mock'], columns='model', values=['median', 'halfwidth', 'z'])\n"
                 "wide['hw_ratio'] = wide[('halfwidth', 'joint')] / wide[('halfwidth', 'los')]\n"
                 "display(wide.round(3)); display(df[df.model == 'joint'][['set', 'mock', 'leak_std_nats', 'leak_corr_rho150']].round(2))"),
        ('code', "for f in sorted((CHAINS / 'figures').glob('profiles_*.png')):\n    display(Image(str(f), width=1100))"),
        ('code', "for f in sorted((CHAINS / 'figures').glob('corner_*.png')):\n    display(Image(str(f), width=900))"),
        ('md', 'Re-run one mock interactively (both models, short chain) and the leak diagnostic:'),
        ('code', "import torch\nfrom nle import NLE\nfrom diagnostics import mock_sf_compare as msc\nfrom diagnostics.pit_by_regime import best_checkpoint\nimport dsph_analysis.kinematic_io as kio\n"
                 "dev = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
                 "models = {'los': NLE.load_from_checkpoint(best_checkpoint(MODELS / '6xaw28uw'), map_location=dev, pre_transforms=None).eval(),\n"
                 "          'joint': NLE.load_from_checkpoint(best_checkpoint(sorted(MODELS.glob('*'))[0]), map_location=dev, pre_transforms=None).eval()}  # <- set the joint run id\n"
                 "meta = kio.load_meta('draco_1'); rhalf = float(meta.rhalf_kpc.value); lrs = float(np.log10(rhalf))\n"
                 "SET, PROF = 'noSF', 'CoreOM'\n"
                 "obs = msc.load_mock(msc.SETS[SET], PROF, meta)\n"
                 "params = pd.read_csv(msc.MOCKS / msc.SETS[SET] / 'mock_catalog_params.csv')\n"
                 "truth = msc.truth_labels(params[params.profile == PROF].iloc[0], rhalf)\n"
                 "chains = {k: msc.sample(m, obs, lrs, 48, 800, 0, dev) for k, m in models.items()}\n"
                 "for k, c in chains.items():\n"
                 "    lr = msc.log_rho_150(c[:, :7], lrs); print(k, np.percentile(lr, [16, 50, 84]).round(3), 'truth', msc.log_rho_150(truth[None], lrs).round(3))\n"
                 "print('leak std [nats], corr with log rho150:', msc.position_factor_leak(models, obs, lrs, chains['los'], dev))"),
    ]),
    '05_coverage.ipynb': nb('Coverage of the NLE posteriors on held-out simulated galaxies', [
        ('md', 'Rows of `coverage/<mode>_<case>_<tag>.csv`, one per galaxy: star count used, '
               'log10 r_half, true and median log10 rho(150 pc), 68% half-width (dex), z and '
               'PIT of the truth. Selection mode: the same galaxies with no selection, only '
               'R < r_half, only R > r_half (a position-only selection cancels from a likelihood '
               'conditioned on R, so coverage should hold in all three). Mixed mode: v_los on '
               'all stars, v_los on half + (v_x, v_y) on the other half, all three components.'),
        ('code', SETUP),
        ('code', "from scipy.stats import kstest\nMODE, TAG = 'selection', 'shard41'   # or 'mixed'\n"
                 "files = sorted((OUT / 'coverage').glob(f'{MODE}_*_{TAG}.csv'))\n"
                 "rows = []\nfor f in files:\n    d = pd.read_csv(f); case = f.stem[len(MODE) + 1:-len(TAG) - 1]\n"
                 "    rows.append(dict(case=case, galaxies=len(d), median_halfwidth=d.halfwidth.median(), rms_z=np.sqrt((d.z**2).mean()), frac_abs_z_le_1=(d.z.abs() <= 1).mean(), mean_z=d.z.mean(), pit_ks_p=kstest(d.pit, 'uniform').pvalue))\n"
                 "display(pd.DataFrame(rows).round(3))"),
        ('code', "fig, ax = plt.subplots(1, 2, figsize=(11, 4))\n"
                 "for f in files:\n    d = pd.read_csv(f); case = f.stem[len(MODE) + 1:-len(TAG) - 1]\n"
                 "    ax[0].hist(d.z, bins=np.linspace(-4, 4, 25), histtype='step', lw=1.6, label=case)\n"
                 "    ax[1].hist(d.pit, bins=np.linspace(0, 1, 11), histtype='step', lw=1.6, label=case)\n"
                 "ax[0].set_xlabel('z = (median - truth) / half-width'); ax[1].set_xlabel('PIT of the truth (flat if calibrated)')\n"
                 "ax[0].legend(); plt.show()"),
    ]),
}

if __name__ == '__main__':
    for name, book in BOOKS.items():
        nbf.validate(book)
        nbf.write(book, HERE / name)
        print('wrote', name)
