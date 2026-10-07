"""Generate the two walkthrough notebooks (03 error convolution, 04 mock
selection-function comparison).

Each notebook re-derives its test step by step with the actual operations
in the cells; `diagnostics/check_error_convolution.py` and
`diagnostics/mock_sf_compare.py` are the batch versions of the same steps.
Re-run after editing:  python notebooks/make_notebooks.py
"""

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent

SETUP = '''# --- paths and imports -------------------------------------------------
import os, sys, math, time
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT))                           # nle_dsph package
sys.path.insert(0, os.path.expanduser('~/my_modules'))  # dsph_analysis (catalogue loader)
os.chdir(ROOT)

import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from IPython.display import Image, display

pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

MODELS = Path(os.path.expanduser('~/scratch/projects/sbi_dsph/trained_models/nle/nle_8p_ZhaoPlumCOM_v4'))
OUT = Path(os.path.expanduser('~/scratch/projects/nle_dsph'))
LABELS = ('dm_alpha', 'dm_beta', 'dm_gamma', 'dm_log_rdm', 'dm_log_rho0', 'df_beta0', 'df_log_ra')

from nle import NLE                     # the Lightning module: flow + normalisation
from transforms import StarBatch        # the per-star container the model consumes
from diagnostics.pit_by_regime import best_checkpoint, load_shard'''


def md(s):
    return ('md', s)


def code(s):
    return ('code', s)


def nb(title, cells):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(f'# {title}')]
    for kind, src in cells:
        n.cells.append(nbf.v4.new_markdown_cell(src) if kind == 'md'
                       else nbf.v4.new_code_cell(src))
    n.metadata['kernelspec'] = dict(name='python3', display_name='Python 3',
                                    language='python')
    return n


# ==========================================================================
B03 = nb('Does a noise-trained NLE equal a noiseless NLE convolved with the errors?', [
    md('''## Why this test

The production NLE is trained with measurement noise injected: every training
star gets a random $\\sigma_v$, Gaussian noise of that width is added to its
$v_{\\rm los}$, and $\\sigma_v$ is appended to the flow's context. The model
therefore learns

$$p_{\\rm noisy}(v_{\\rm obs}\\mid \\log R,\\, \\theta,\\, \\log r_{1/2},\\, \\sigma_v).$$

A second model is trained on the same simulations with no noise and no
$\\sigma_v$ in its context:

$$p_{\\rm clean}(v \\mid \\log R,\\, \\theta,\\, \\log r_{1/2}).$$

If both flows are faithful, the first is the second convolved with the error
kernel,

$$p_{\\rm noisy}(v_{\\rm obs}\\mid\\sigma_v) \;=\; \\int p_{\\rm clean}(v)\;
\\mathcal N(v_{\\rm obs};\\, v,\\, \\sigma_v^2)\\, {\\rm d}v ,$$

for every $\\sigma_v$ and every context. Two reasons to check it:

1. It is the paper's demonstration that the noise augmentation is learned
   consistently with the physics (the model does not use $\\sigma_v$ as a
   shortcut).
2. Real errors need not be Gaussian. The noiseless model is the one that lets
   an arbitrary per-star kernel be applied at inference; the convolution
   below is that machinery.

**Models used.** Both trained on shards 0–1 (190k galaxies), T2 architecture
(NSF 4×[128,128], 8 bins, context MLP → 64), 30k steps at batch 16384:
`errtest_clean` = run `voj63drr`, `errtest_noisy` = run `6xaw28uw`
(`configs/nle_8p_v4_errtest.py`).'''),
    code(SETUP),
    code('''clean = NLE.load_from_checkpoint(best_checkpoint(MODELS / 'voj63drr'), map_location=dev, pre_transforms=None).eval()
noisy = NLE.load_from_checkpoint(best_checkpoint(MODELS / '6xaw28uw'), map_location=dev, pre_transforms=None).eval()

# The context lists show the one structural difference: `vel_error` is in
# the noisy model's context and absent from the clean one's.
print('clean  target', clean.target_fields, ' context', clean.context_fields)
print('noisy  target', noisy.target_fields, ' context', noisy.context_fields)'''),
    md('''## 1. Units: the flow's density is per *normalised* velocity

`NLE.log_prob` returns $\\log p$ per unit of the normalised velocity
$\\tilde v = (v - v_{\\rm loc}) / v_{\\rm scale}$, and each model has its own
`vel_scale` (the half-range of the training velocities after noise). To
compare the two models, or to feed either into a sampler alongside anything
in km/s, subtract $\\log v_{\\rm scale}$:

$$\\log p(v)\\,[{\\rm per\\ km/s}] = \\log p(\\tilde v) - \\log v_{\\rm scale}.$$

Forgetting this gives a constant offset of $\\log(v_{\\rm scale,noisy} /
v_{\\rm scale,clean})$ nats per star that looks like a model disagreement
(it was the first thing this test "found").'''),
    code('''print('vel_scale  clean %.2f  noisy %.2f km/s  ->  log ratio %.4f nats/star' % (
    clean.vel_scale[0], noisy.vel_scale[0], math.log(float(noisy.vel_scale[0]) / float(clean.vel_scale[0]))))


def logp_kms(model, R, v, err, theta, lrs):
    """Per-star log p in units of 1/(km/s).

    R      (n,)   projected radius [kpc]; only |pos| enters with condition_on_radius
    v      (n,)   velocity to evaluate at [km/s]
    err    (n,)   sigma_v [km/s], or None for the clean model
    theta  (n,7)  labels in the training parametrisation (radii in units of r_half)
    lrs    float  log10 r_half [kpc]
    """
    n = len(R)
    pos = torch.zeros((n, 2), device=dev)
    pos[:, 0] = R
    b = StarBatch(pos=pos, vel=v.view(n, 1),
                  vel_error=None if err is None else err.view(n, 1),
                  theta=theta, cond=torch.full((n, 1), lrs, device=dev))
    with torch.no_grad():
        return model.log_prob(b) - torch.log(model.vel_scale[0])'''),
    md('''## 2. One star, by eye

Take one held-out galaxy (shard 40 was never used in training), one of its
stars, and a chosen $\\sigma_v$. Plot $p_{\\rm clean}(v)$, its numerical
convolution with $\\mathcal N(0, \\sigma_v^2)$ on a fine grid, and
$p_{\\rm noisy}(v_{\\rm obs} \\mid \\sigma_v)$ evaluated directly. The
dashed curve should sit on the shaded one.'''),
    code('''pos, vel, th, lrs_all, n = load_shard(40, n_gal=50)   # (pos, vel) per star; (th, lrs, n) per galaxy

g = 3                                       # which galaxy
sl = slice(int(n[:g].sum()), int(n[:g + 1].sum()))
R = pos[sl, :2].norm(dim=1).to(dev)         # projection along z: sky plane = (x, y)
v_true = vel[sl, 2].to(dev)                 # true line-of-sight velocity
theta = th[g].expand(len(R), 7).to(dev)     # every star of a galaxy shares its theta
lrs = float(lrs_all[g])
sig_true = float(v_true.std())

i, sigma = 0, 2.0                           # star index, assumed error [km/s]
vg = torch.linspace(-30, 30, 1201, device=dev)
same = lambda k: (R[i].expand(k), theta[:k], lrs)

p_clean = torch.exp(logp_kms(clean, R[i].expand(1201), vg, None, theta[:1].expand(1201, 7), lrs)).cpu()
p_noisy = torch.exp(logp_kms(noisy, R[i].expand(1201), vg, torch.full((1201,), sigma, device=dev),
                             theta[:1].expand(1201, 7), lrs)).cpu()

# direct convolution on the grid: p_conv(v_obs) = sum_v p_clean(v) N(v_obs; v, sigma) dv
vgc = vg.cpu(); dv = float(vgc[1] - vgc[0])
kern = torch.exp(-0.5 * ((vgc[:, None] - vgc[None, :]) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
p_conv = (kern * p_clean[None, :]).sum(1) * dv

plt.figure(figsize=(7, 4))
plt.plot(vgc, p_clean, color='0.6', label='$p_{\\\\rm clean}(v)$, no noise')
plt.fill_between(vgc, p_conv, alpha=.35, label=f'$p_{{\\\\rm clean}} * \\\\mathcal{{N}}(0,{sigma}^2)$')
plt.plot(vgc, p_noisy, 'k--', label=f'$p_{{\\\\rm noisy}}(v_{{\\\\rm obs}}\\\\mid\\\\sigma_v={sigma})$')
plt.xlabel('v [km/s]'); plt.ylabel('density [1/(km/s)]'); plt.legend()
plt.title(f'galaxy {g}: $\\\\sigma_{{\\\\rm true}}$ = {sig_true:.1f} km/s;  star {i} at R = {float(R[i]):.3f} kpc')
plt.show()'''),
    md('''## 3. The convolution for many stars at once

For a quantitative comparison we need the convolved density at every star's
observed velocity. The integrand $p_{\\rm clean}(v)\\,\\mathcal N(v_{\\rm obs};
v, \\sigma_v)$ is only non-negligible where *both* factors are:

* the intrinsic peak lives in $|v| \\lesssim 12\\,\\sigma_{\\rm true}$,
* the kernel in $v_{\\rm obs} \\pm 8\\sigma_v$.

The grid spans the **intersection** of the two, so it resolves whichever is
narrower: a 0.1 km/s galaxy under 10 km/s errors, or a 20 km/s galaxy with
0.3 km/s errors. (Gridding the union instead left 0.4 nats/star of quadrature
error in the first case.) The sum is a trapezoid rule done in log space with
`logsumexp` so very small densities do not underflow.'''),
    code('''def convolved_logp(R, v_obs, sigma, sig_true, theta, lrs, n_grid=257):
    """log Int p_clean(v) N(v_obs; v, sigma) dv, per star, in 1/(km/s).

    v_obs, sig_true (n,) tensors; sigma a float; theta (n,7); lrs float.
    n_grid trapezoid nodes per star (accuracy/cost trade-off, section 5).
    """
    lo = torch.maximum(v_obs - 8 * sigma, -12 * sig_true)
    hi = torch.minimum(v_obs + 8 * sigma, 12 * sig_true)
    hi = torch.maximum(hi, lo + 1e-3)                       # guard against an empty overlap
    t = torch.linspace(0, 1, n_grid, device=dev)
    grid = lo[:, None] + (hi - lo)[:, None] * t[None, :]    # (n, n_grid)
    B = len(v_obs)
    # one batched forward pass over all (star, node) pairs
    lp = logp_kms(clean, R.repeat_interleave(n_grid), grid.reshape(-1), None,
                  theta.repeat_interleave(n_grid, 0), lrs).view(B, n_grid)
    lk = -0.5 * ((v_obs[:, None] - grid) / sigma) ** 2 - math.log(sigma * math.sqrt(2 * math.pi))
    w = torch.zeros_like(lp); w[:, [0, -1]] = math.log(0.5)  # trapezoid end weights
    return torch.logsumexp(lp + lk + w, dim=1) + torch.log((hi - lo) / (n_grid - 1))


# d = log p_noisy - log(convolved p_clean), per star, at this galaxy's true theta
st = torch.full_like(v_true, sig_true)
for sigma in (0.3, 1.0, 3.0, 10.0):
    torch.manual_seed(0)
    v_obs = v_true + sigma * torch.randn_like(v_true)
    d = (logp_kms(noisy, R, v_obs, torch.full_like(v_obs, sigma), theta, lrs)
         - convolved_logp(R, v_obs, sigma, st, theta, lrs))
    print(f'sigma_v = {sigma:5.1f} km/s:  per star  mean {d.mean():+.4f}  rms {d.pow(2).mean().sqrt():.4f}'
          f'   |  summed over the galaxy ({len(d)} stars) {d.sum():+.2f} nats')'''),
    md('''**Reading it.** The per-star mean of $d$ is at the $10^{-3}$ level and the
galaxy sum below a nat. The rms per star (a few $\\times 10^{-2}$) is the
*combined* approximation error of two independently trained flows plus the
quadrature; it adds up incoherently over a galaxy ($\\sqrt{N}$), which is why
the galaxy sum is small. For the posterior what matters is the galaxy-level
error, since every star is evaluated at the same $\\theta$.

## 4. The batch result (3000 galaxies, four $\\sigma_v$)

`diagnostics/check_error_convolution.py` runs the same comparison on 3000
galaxies of shard 40 and also compares against the 5-shard T2 model
(`noisy2`). Columns: `sigma` the assumed error (km/s); `sig_lo`–`sig_hi` the
bin of the galaxy's intrinsic dispersion (km/s); `star_*` statistics of $d$
per star; `gal_*` of $d$ summed over a galaxy; `nll_*` the mean negative log
density of each side. The figure scans $\\log L$ in `dm_log_rho0` for six
galaxies with the other labels at the truth: the three curves should
coincide.'''),
    code('''csv = sorted((MODELS / 'voj63drr').glob('error_convolution_*v2.csv'))
if csv:
    display(pd.read_csv(csv[-1]).round(4))
    display(Image(str(csv[-1]).replace('.csv', '.png')))
else:
    print('batch run not found; run  python diagnostics/check_error_convolution.py --clean', MODELS / 'voj63drr', '--noisy', MODELS / '6xaw28uw')'''),
    md('''## 5. What the noiseless route costs

Each convolved evaluation is `n_grid` forward passes of the clean model, so
its cost relative to one noisy-model evaluation is about `1.1 × n_grid`. The
accuracy floor worth reaching is the model-to-model scatter above (~1 nat per
galaxy): 65 nodes get there, 129 are well below it. The kernel shape does not
change this; a heavy-tailed kernel only widens the grid.'''),
    code('''sigma = 1.0
torch.manual_seed(0); v_obs = v_true + sigma * torch.randn_like(v_true); e = torch.full_like(v_obs, sigma)
ref = convolved_logp(R, v_obs, sigma, st, theta, lrs, n_grid=1025)


def timeit(f, reps=5):
    f(); torch.cuda.synchronize(); t0 = time.time()
    for _ in range(reps): f()
    torch.cuda.synchronize(); return (time.time() - t0) / reps


t_noisy = timeit(lambda: logp_kms(noisy, R, v_obs, e, theta, lrs))
print(f'{"n_grid":>7} {"time / noisy":>13} {"rms error vs 1025 nodes [nats/star]":>38}')
for ng in (17, 33, 65, 129, 257):
    t = timeit(lambda: convolved_logp(R, v_obs, sigma, st, theta, lrs, n_grid=ng), reps=2)
    err_ng = (convolved_logp(R, v_obs, sigma, st, theta, lrs, n_grid=ng) - ref).pow(2).mean().sqrt()
    print(f'{ng:>7} {t / t_noisy:>12.1f}x {float(err_ng):>38.4f}')'''),
])

# ==========================================================================
B04 = nb('Velocity-only p(v | R) versus joint p(x, y, v): does modelling the positions change the posterior?', [
    md('''## The question

The production NLE conditions on the projected radius and models only the
velocity,

$$\\mathcal L_{\\rm los}(\\theta) = \\prod_i p(v_i \\mid \\log R_i,\\, \\theta,\\,
\\log r_{1/2},\\, \\sigma_i).$$

The alternative (`condition_on_position = False`) models the sky position too:

$$\\mathcal L_{\\rm joint}(\\theta) = \\prod_i p(x_i, y_i, v_i \\mid \\theta,\\,
\\log r_{1/2},\\, \\sigma_i).$$

With $r_{1/2}$ fixed at its photometric value, the joint density factorises
exactly as $p(v \\mid R, \\theta, r_{1/2})\\, p(x, y \\mid r_{1/2})$, and the
second factor is the projected Plummer profile, which carries **no**
$\\theta$. So $\\log\\mathcal L_{\\rm joint} = \\log\\mathcal L_{\\rm los} +
{\\rm const}$ and the two posteriors are identical *if both flows are exact*.
A position-only selection function multiplies the velocity-only likelihood by
a $\\theta$-independent factor as well, which is the Jeans-style argument for
conditioning on $R$ in the first place.

What can differ is the approximation. The joint flow has to *learn* that its
position marginal does not depend on $\\theta$; nothing in the training
objective enforces it. Any residual dependence is a per-star error that is
coherent across a galaxy's stars, so $\\epsilon$ nats per star becomes
$N\\epsilon$ nats in the galaxy likelihood. This notebook:

1. builds both galaxy likelihoods on a Draco mock,
2. scans and samples them,
3. measures the "leak" $\\log\\mathcal L_{\\rm joint} - \\log\\mathcal L_{\\rm
   los}$ over the posterior, which is a constant for an exact position factor,
4. shows the batch result on all twelve mocks with and without the Walker
   2023 selection.

**Models.** Both on shards 0–1, T2 architecture, 30k steps: the velocity-only
model `errtest_noisy` (run `6xaw28uw`) and the joint model `errtest_joint`
(`configs/nle_8p_v4_errtest.py:joint`), which differs only in
`condition_on_position = condition_on_radius = False`.'''),
    code(SETUP),
    code('''import dsph_analysis.kinematic_io as kio
from diagnostics.mock_sf_compare import MOCKS, SETS, load_mock, truth_labels, PRIOR_MIN, PRIOR_MAX, log_rho_150

los = NLE.load_from_checkpoint(best_checkpoint(MODELS / '6xaw28uw'), map_location=dev, pre_transforms=None).eval()
# the joint run: the newest run dir whose config snapshot carries the name errtest_joint
JOINT_RUN = sorted(d for d in MODELS.iterdir()
                   if (d / 'config_snapshot.json').exists() and '"errtest_joint"' in (d / 'config_snapshot.json').read_text())[-1]
joint = NLE.load_from_checkpoint(best_checkpoint(JOINT_RUN), map_location=dev, pre_transforms=None).eval()

print('velocity-only  target', los.target_fields, '  context', los.context_fields)
print('joint          target', joint.target_fields, '  context', joint.context_fields, '  (run', JOINT_RUN.name + ')')
print('both condition on log r_half (`cond`); neither infers it')'''),
    md('''## 1. A mock through the pipeline's loader

The mocks live in `vr_catalogs/mock/icrs`: six profiles × two sets,
`draco1_walker23_icrs` (510 stars, the Walker 2023 footprint applied) and
`draco1_walker23_noSF` (200 stars, no selection). The `mock_icrs` loader does
what the NPE pipeline does: perspective correction, the $|v_{\\rm los}| < 50$
km/s cut, and the RA/Dec → kpc projection at Draco's LVDB distance. The truth
file stores $r_s$ and $r_a$ in kpc; the labels use units of $r_{1/2}$, so
`truth_labels` divides them out.'''),
    code('''meta = kio.load_meta('draco_1')
rhalf = float(meta.rhalf_kpc.value); lrs = float(np.log10(rhalf))

SET, PROF = 'walker23', 'CuspIso'           # <- change these to look at another mock
obs = load_mock(SETS[SET], PROF, meta)       # dict of float32 arrays: X, Y [kpc], v, err [km/s]
params = pd.read_csv(MOCKS / SETS[SET] / 'mock_catalog_params.csv')
truth = truth_labels(params[params.profile == PROF].iloc[0], rhalf)
tr150 = float(log_rho_150(truth[None], lrs)[0])

print(f'{len(obs["v"])} stars | r_half = {rhalf} kpc | median sigma_v = {np.median(obs["err"]):.2f} km/s')
print('truth (label space):', dict(zip(LABELS, truth.round(3))))
print(f'true log10 rho(150 pc) = {tr150:.3f}')

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].scatter(obs['X'], obs['Y'], s=4); ax[0].set_aspect('equal'); ax[0].set_xlabel('X [kpc]'); ax[0].set_ylabel('Y [kpc]')
ax[1].errorbar(np.hypot(obs['X'], obs['Y']), obs['v'], obs['err'], fmt='.', ms=3, lw=.5)
ax[1].set_xscale('log'); ax[1].set_xlabel('R [kpc]'); ax[1].set_ylabel('v_los [km/s]'); plt.show()'''),
    md('''## 2. The galaxy log-likelihood

Eight parameters: the seven labels and the systemic velocity $v_{\\rm sys}$.
For each parameter vector, every star gets the same $\\theta$
(`repeat_interleave`), the observed velocities are shifted by $v_{\\rm sys}$,
and the per-star $\\log p$ are summed. The same function serves both models:
the joint one reads `pos` as part of its target, the velocity-only one as
context. Several parameter vectors are evaluated in one batch, which is what
makes the GPU sampler below fast.'''),
    code('''X, Y, v = (torch.tensor(obs[k], device=dev) for k in ('X', 'Y', 'v'))
err = torch.tensor(obs['err'], device=dev).view(-1, 1)
n = len(v)


def log_like(model, theta):
    """Galaxy log-likelihood for each row of theta (shape (k, 8)); returns (k,)."""
    t = torch.tensor(np.atleast_2d(theta), dtype=torch.float32, device=dev)
    nw = len(t)
    b = StarBatch(pos=torch.stack([X, Y], 1).repeat(nw, 1),
                  vel=(v.repeat(nw) - t[:, 7].repeat_interleave(n)).view(-1, 1),
                  vel_error=err.repeat(nw, 1),
                  theta=t[:, :7].repeat_interleave(n, 0),
                  cond=torch.full((nw * n, 1), lrs, device=dev))
    with torch.no_grad():
        lp = model.log_prob(b).view(nw, n)
    lp = torch.where(torch.isfinite(lp), lp, torch.full_like(lp, -1e30))
    return lp.sum(1).cpu().numpy()


theta_true = np.append(truth, 0.0)           # v_sys = 0: the mock is in Draco's rest frame after the loader
print('log L at the truth:  velocity-only %.1f   joint %.1f' % (log_like(los, theta_true)[0], log_like(joint, theta_true)[0]))
print('(the difference is sum_i log p(x_i, y_i | r_half) plus the log vel_scale Jacobian: a constant in theta if the joint model is exact)')'''),
    md('''## 3. A one-dimensional scan

Vary one label with the others held at the truth. Up to a vertical shift the
two curves should coincide; a difference in **shape** is $\\theta$ leaking
into the learned position factor.'''),
    code('''fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, name in zip(axes, ('dm_log_rho0', 'dm_gamma', 'dm_log_rdm')):
    k = LABELS.index(name)
    grid = np.linspace(PRIOR_MIN[k], PRIOR_MAX[k], 101)
    scan = np.tile(theta_true, (101, 1)); scan[:, k] = grid
    for model, lab, st in ((los, 'velocity-only p(v | R)', '-'), (joint, 'joint p(x, y, v)', '--')):
        ll = log_like(model, scan); ax.plot(grid, ll - ll.max(), st, label=lab)
    ax.axvline(truth[k], color='k', ls=':', lw=1); ax.set_ylim(-40, 2); ax.set_xlabel(name); ax.set_ylabel('log L - max')
axes[0].legend(); plt.show()'''),
    md('''## 4. Sampling the two posteriors

`emcee` over the eight parameters with the 8p_v4 box prior; all walkers are
evaluated in one GPU batch (`vectorize=True`). Differential-evolution moves
as in the T2 run. 48 walkers × 1000 steps is enough to see the posterior
here; the batch script uses 64 × 4000.'''),
    code('''import emcee


def sample(model, n_walkers=48, n_steps=1000, seed=0):
    def log_post(th):
        out = np.full(len(th), -np.inf)
        ok = np.all((th >= PRIOR_MIN) & (th <= PRIOR_MAX), axis=1)   # flat box prior
        if ok.any():
            out[ok] = log_like(model, th[ok])
        return out
    rng = np.random.default_rng(seed)
    p0 = rng.uniform(PRIOR_MIN, PRIOR_MAX, (n_walkers, 8))
    p0[:, 7] = np.median(obs['v']) + rng.normal(0, 1, n_walkers)
    s = emcee.EnsembleSampler(n_walkers, 8, log_post, vectorize=True,
                              moves=[(emcee.moves.DEMove(), .8), (emcee.moves.DESnookerMove(), .2)])
    s.run_mcmc(p0, n_steps, progress=False)
    return s.get_chain(discard=n_steps // 2, flat=True)


chains = {'velocity-only': sample(los), 'joint': sample(joint)}
for name, c in chains.items():
    lo_, med, hi = np.percentile(log_rho_150(c[:, :7], lrs), [16, 50, 84]); hw = (hi - lo_) / 2
    print(f'{name:14s} log10 rho(150 pc) = {med:.3f} +- {hw:.3f}    z = {(med - tr150) / hw:+.2f}    (truth {tr150:.3f})')'''),
    code('''# marginals of log rho(150 pc) and of the three labels that set it
fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
for name, c, col in (('velocity-only p(v | R)', chains['velocity-only'], '#0072B2'), ('joint p(x, y, v)', chains['joint'], '#D55E00')):
    axes[0].hist(log_rho_150(c[:, :7], lrs), bins=40, histtype='step', lw=1.8, color=col, density=True, label=name)
    for ax, k in zip(axes[1:], (LABELS.index('dm_gamma'), LABELS.index('dm_log_rdm'), LABELS.index('dm_log_rho0'))):
        ax.hist(c[:, k], bins=40, histtype='step', lw=1.8, color=col, density=True); ax.set_xlabel(LABELS[k]); ax.axvline(truth[k], color='k', ls=':')
axes[0].axvline(tr150, color='k', ls=':'); axes[0].set_xlabel('log10 rho(150 pc)'); axes[0].legend(fontsize=8); plt.show()'''),
    md('''## 5. The leak diagnostic

$$\\Delta(\\theta) \\equiv \\log\\mathcal L_{\\rm joint}(\\theta) - \\log\\mathcal
L_{\\rm los}(\\theta) = \\sum_i \\log p(x_i, y_i \\mid \\theta, r_{1/2}).$$

For an exact position factor this is the same number at every $\\theta$.
Evaluate it on samples of the velocity-only posterior: its standard
deviation is the size of the leak in nats at the galaxy level, and its
correlation with $\\log\\rho(150\\,{\\rm pc})$ says whether the leak pulls the
density posterior in a definite direction (it did, at the −0.5 level, for a
half-trained joint model).'''),
    code('''idx = np.random.default_rng(1).choice(len(chains['velocity-only']), 400, replace=False)
th = chains['velocity-only'][idx]
delta = log_like(joint, th) - log_like(los, th)
lr = log_rho_150(th[:, :7], lrs)
print(f'leak: std {delta.std():.2f} nats over the posterior; corr with log rho(150 pc) {np.corrcoef(delta, lr)[0, 1]:+.2f}   (exact model: 0, 0)')
print(f'per star that is {delta.std() / n:.4f} nats, which is below what a per-star PIT test can see')

plt.figure(figsize=(5, 4)); plt.scatter(lr, delta - delta.mean(), s=6)
plt.xlabel('log10 rho(150 pc) of the sample'); plt.ylabel('$\\\\Delta$ - mean  [nats]'); plt.show()'''),
    md('''## 6. All twelve mocks

`diagnostics/mock_sf_compare.py` runs both models on every mock with
64 × 4000 steps and records the leak diagnostic; `diagnostics/plot_mock_sf.py`
makes the figures. Summary columns: `median` and `halfwidth` of
$\\log_{10}\\rho(150\\,{\\rm pc})$ per model (dex), `z` against the truth,
`hw_ratio` = joint half-width / velocity-only half-width, and the leak std and
correlation. The profile figures show the dark-matter density and slope
against 3-D radius, and $\\sigma_{\\rm los}$ and $\\kappa_{\\rm los}$ against
projected radius (curves: Jeans model at the median draw; points: bin-averaged
posterior predictive; black: error-deconvolved binned data; dashed black:
truth). The corner plots show the seven labels with radii in kpc.

What to look for: whether the joint model's $z$ or half-width differs
between the `walker23` and `noSF` sets in a way the velocity-only model's
does not; that would be the selection acting through the learned position
factor.'''),
    code('''CH = OUT / 'mock_sf'
if (CH / 'mock_sf_compare.csv').exists():
    df = pd.read_csv(CH / 'mock_sf_compare.csv')
    wide = df.pivot_table(index=['set', 'mock'], columns='model', values=['median', 'halfwidth', 'z'])
    wide['hw_ratio'] = wide[('halfwidth', 'joint')] / wide[('halfwidth', 'los')]
    display(wide.round(3))
    if 'leak_std_nats' in df:
        display(df[df.model == 'joint'][['set', 'mock', 'n_stars', 'leak_std_nats', 'leak_corr_rho150']].round(2))
    for f in sorted((CH / 'figures').glob('profiles_*.png')):
        display(Image(str(f), width=1100))
    for f in sorted((CH / 'figures').glob('corner_*.png')):
        display(Image(str(f), width=900))
else:
    print('batch run not finished yet; see ~/scratch/local_logs/nle/mock_sf_compare.log')'''),
])

BOOKS = {'03_error_convolution.ipynb': B03, '04_mock_sf.ipynb': B04}

if __name__ == '__main__':
    for name, book in BOOKS.items():
        nbf.validate(book)
        nbf.write(book, HERE / name)
        print('wrote', name)
