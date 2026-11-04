"""Notebook front end for per-star NLE: train, load, observe, sample.

A run is reproducible from its directory alone. Everything needed is
written at training time: `train` snapshots the config beside the
checkpoints it describes, and the Lightning checkpoint already carries
`norm_dict` in its `hyper_parameters`. `load_model` therefore needs
nothing but the run directory -- never the training data, and never a
separately saved mean and standard deviation.

`scripts/train_nle.py` is the batch twin of `train`: both call the
same helpers in `nle_dsph.training` and write the same run directory.
"""

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pocomc as pc
import pytorch_lightning as pl
import torch

from nle_dsph import training
from nle_dsph.nle import NLE
from nle_dsph.profiles import log_rho_150
from nle_dsph.transforms import StarBatch, build_transformation

__all__ = [
    'Observation',
    'cond_box',
    'galaxy_log_prob',
    'load_config',
    'load_model',
    'log_rho_150',
    'sample_posterior',
    'save_posterior',
    'train',
]

# Stand-in for an impossible particle. Far below any attainable
# log-likelihood (a 1000-star galaxy reaches order 1e3 nats), and finite,
# which is what pocomc's importance weights need.
LOG_LIKE_FLOOR = -1e10


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Fields the pipeline reads with `config.get(...)`, so a config may leave
# them out and still be overridden with them. Regenerate after adding a
# new optional field upstream:
#   grep -ohE "config\.get\((\"|')[a-zA-Z_]+(\"|')" *.py | \
#       sed -E "s/config.get\(.//;s/.$//" | sort -u
OPTIONAL_FIELDS = frozenset({
    'checkpoint', 'cond_labels', 'data_on_device', 'debug',
    'deterministic_val', 'enable_progress_bar', 'entity',
    'gradient_clip_val', 'id', 'init', 'matmul_precision',
    'max_val_galaxies', 'name', 'precision', 'reset_optimizer',
    'save_top_k', 'tags', 'val_check_interval', 'wandb_mode',
    'wandb_project',
})


def load_config(path, **overrides):
    """Load a `configs/*.py` `get_config()` and apply overrides.

    The notebook twin of `--config <path> --config.<field>=<value>`.

    Args:
        path: Path to a config module defining `get_config()`.
        **overrides: Top-level fields to replace, e.g. `num_steps=2000`.
            Nested fields are edited on the returned object instead:
            `config.model.flows.num_bins = 6`.

    Returns:
        The `ConfigDict`, editable in place.

    Raises:
        KeyError: If an override names neither a declared field nor one
            of `OPTIONAL_FIELDS`. Without the check a typo would add a
            field nothing reads, and training would quietly run with the
            value the override was meant to replace.

    Example:
        >>> config = load_config('configs/nle_8p_v4.py', num_steps=2000)
    """
    spec = importlib.util.spec_from_file_location('nle_config', str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.get_config()

    for key, value in overrides.items():
        if key not in config and key not in OPTIONAL_FIELDS:
            raise KeyError(
                f'{key!r} is neither a field of {path} nor a known '
                'optional field (api.OPTIONAL_FIELDS)')
        config[key] = value
    return config


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config, config_path=None):
    """Train an NLE from a config and return its run directory.

    The in-process twin of `python scripts/train_nle.py --config <file>`: same
    dataloaders, same callbacks, same run-directory layout. Use it for
    short runs from a notebook, and the script (or
    `slurm/train_nle.sh`) for anything that must survive a logout.

    Args:
        config: Config as returned by `load_config`.
        config_path: Optional path to the config file, for the snapshot.

    Returns:
        The run directory, holding `checkpoints/` and
        `config_snapshot.json`.

    Raises:
        RuntimeError: If training finished without writing a checkpoint,
            which happens when `num_steps` is shorter than one epoch and
            `val_check_interval` is unset: the checkpoints monitor
            `val/loss`, and validation only runs at epoch end.
    """
    logger, project_dir = training.create_wandb_logger(config, tag='nle')

    train_loader, val_loader, norm_dict = training.prepare_data(config)
    pre_transforms = build_transformation(**config.pre_transforms.to_dict())
    model = training.create_model(config, pre_transforms, norm_dict)

    trainer_kwargs = dict(num_sanity_val_steps=0)
    val_check_interval = config.get('val_check_interval', None)
    if val_check_interval is not None:
        # Lightning requires the interval to fit inside one epoch.
        val_check_interval = min(int(val_check_interval), len(train_loader))
        trainer_kwargs['val_check_interval'] = val_check_interval
        print(f'[Trainer] Validation every {val_check_interval} steps')

    trainer = training.build_trainer(
        config, project_dir, training.create_base_callbacks(config), logger,
        **trainer_kwargs)

    # Snapshot before fitting, so a run that crashes still records what
    # it was asked to do.
    training.save_config_snapshot(config, project_dir, config_path)

    pl.seed_everything(config.seed_training, workers=True)
    training.fit(trainer, model, train_loader, val_loader,
                 checkpoint_path=None, reset_optimizer=False)

    # Reason: Lightning resolves the checkpoint directory from the logger
    # during `fit`, and it does not always agree with
    # create_wandb_logger's project_dir -- with debug=True the
    # disabled-wandb stub reports project 'dummy', which sends the
    # snapshot and the checkpoints to different directories. The
    # callback's own dirpath is authoritative, so repeat the snapshot
    # beside the checkpoints it actually describes.
    run_dir = Path(trainer.checkpoint_callback.dirpath).parent
    if run_dir != project_dir:
        print(f'[Setup] Checkpoints went to {run_dir}, not {project_dir}; '
              'writing the snapshot there too')
        training.save_config_snapshot(config, run_dir, config_path)

    if not list((run_dir / 'checkpoints').glob('*.ckpt')):
        raise RuntimeError(
            f'training wrote no checkpoint to {run_dir}/checkpoints. The '
            'checkpoint callbacks monitor val/loss, which is only logged '
            'when validation runs; set config.val_check_interval so it '
            'runs before the step budget is spent.')
    return run_dir


def load_model(run_dir, checkpoint='last.ckpt'):
    """Load a trained NLE and its config snapshot from a run directory.

    `norm_dict` comes out of the checkpoint, so the model normalizes
    exactly as it did during training no matter what has happened to
    the training data since.

    Args:
        run_dir: Run directory written by `train` or `scripts/train_nle.py`.
        checkpoint: File name inside `<run_dir>/checkpoints`.

    Returns:
        Tuple of (NLE in eval mode on the CPU, config snapshot dict).
    """
    run_dir = Path(run_dir)
    model = NLE.load_from_checkpoint(
        str(run_dir / 'checkpoints' / checkpoint), map_location='cpu',
        pre_transforms=None).eval()
    with open(run_dir / 'config_snapshot.json') as f:
        config = json.load(f)
    return model, config


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """One galaxy's observed stars, frozen in physical units.

    Sky geometry is supplied one of two ways, and never derived from
    the other: `pos_kpc` for tangent-plane positions, `R_proj_kpc` for
    projected radii. Give whichever your catalogue actually measured.
    A loader that reports both may not reduce one to the other exactly
    -- `dsph_analysis.kinematic_io` returns `X_proj`, `Y_proj` and
    `R_proj` as three separate columns -- so recomputing one here would
    quietly overwrite what you measured.

    Attributes:
        vlos_kms: Line-of-sight velocity, (n_stars,) in km/s. Either
            frame works as long as you know which: `sample_posterior`
            fits a systemic velocity and subtracts it, so `v_sys` comes
            back near zero for a systemic-subtracted catalogue and near
            the systemic velocity for a heliocentric one.
        vlos_err_kms: Per-star measurement sigma, (n_stars,) in km/s,
            **linear, not log10**. The flow conditions on sigma itself:
            `UncertaintySampler` records the sampled standard deviation
            and `NLE._normalize` applies a plain affine rescaling to it,
            so a 1 km/s error is fed in as 1.0. The `jeffreys` and
            `jeffreys_varied` options name the distribution sigma is
            *drawn* from during training, not how it is represented.

            Worth checking against whatever produced your catalog: an
            sbi-style pipeline that appends `log10(sigma_v)` to theta
            will hand you logs. Passing those here is silent -- a 1 km/s
            error arrives as 0.0, normalizes to the bottom of the
            trained range, and every star reads as near-noiseless, so
            the posterior comes out tighter than the data supports.
        log_rstar_kpc: Measured half-light radius, log10 of kpc. The
            model's `cond`. Measured, but not therefore fixed: by
            default it is sampled as an extra posterior dimension over
            its photometric error, so the posterior carries it as a
            column like any label.
        pos_kpc: Tangent-plane position, (n_stars, 2) in kpc. None when
            only radii are available.
        R_proj_kpc: Projected radius, (n_stars,) in kpc. None when only
            positions are available.
        use: Which of the two the model is given, 'position' or
            'radius'. Inferred when only one is set; required when both
            are, because the choice is yours to make rather than this
            container's. `galaxy_log_prob` additionally refuses
            `use='position'` under a radius-conditioned model when
            `R_proj_kpc` is set: that model derives the radius as
            `|pos|`, so the measured column would go unread.
        log_rstar_err_dex: Photometric uncertainty on `log_rstar_kpc`,
            in dex. Required unless the conditioning is pinned. For
            LVDB dwarfs this runs from 0.008 dex (Sculptor, Draco) to
            0.065 dex (Segue 1).
        name: Used for the output file name.
        theta_true: True labels for a simulated galaxy, (n_labels,), in
            model units -- radii relative to `r_star`, as the simulator
            drew them, not kpc and not `norm_dict`-scaled. None for
            real data.

    Raises:
        ValueError: If neither `pos_kpc` nor `R_proj_kpc` is given, if
            both are given without `use`, if `use` names a field that
            is None, or if any array disagrees with `vlos_kms` on the
            number of stars.
    """

    vlos_kms: np.ndarray
    vlos_err_kms: np.ndarray
    log_rstar_kpc: float
    pos_kpc: np.ndarray = None
    R_proj_kpc: np.ndarray = None
    use: str = None
    log_rstar_err_dex: float = None
    name: str = 'unnamed'
    theta_true: np.ndarray = None

    def __post_init__(self):
        has_pos = self.pos_kpc is not None
        has_radius = self.R_proj_kpc is not None

        if not (has_pos or has_radius):
            raise ValueError(
                'give the sky geometry as either pos_kpc (n_stars, 2) '
                'or R_proj_kpc (n_stars,); neither was set')

        if self.use is None:
            if has_pos and has_radius:
                raise ValueError(
                    'both pos_kpc and R_proj_kpc were given, so the '
                    "choice is yours: set use='position' or "
                    "use='radius'. They are separate measurements and "
                    'this container will not pick for you')
            self.use = 'position' if has_pos else 'radius'

        if self.use not in ('position', 'radius'):
            raise ValueError(
                f"use={self.use!r}; expected 'position' or 'radius'")
        if self.use == 'position' and not has_pos:
            raise ValueError("use='position' but pos_kpc is None")
        if self.use == 'radius' and not has_radius:
            raise ValueError("use='radius' but R_proj_kpc is None")

        n = len(self.vlos_kms)
        for field, value, ndim in (('vlos_err_kms', self.vlos_err_kms, 1),
                                   ('pos_kpc', self.pos_kpc, 2),
                                   ('R_proj_kpc', self.R_proj_kpc, 1)):
            if value is None:
                continue
            if np.ndim(value) != ndim or len(value) != n:
                raise ValueError(
                    f'{field} has shape {np.shape(value)}, expected '
                    f'{(n, 2) if ndim == 2 else (n,)} to match '
                    f'{n} velocities')

    @property
    def n_stars(self) -> int:
        """Number of stars in the catalogue."""
        return len(self.vlos_kms)

    def model_positions(self) -> np.ndarray:
        """The (n_stars, 2) array the NLE's `pos` field expects.

        A radius-conditioned model reduces `pos` to `|pos|`, so a
        radius-only observation is packed as (R, 0): the norm is R and
        the second column never reaches the flow. Positions pass
        through unchanged.
        """
        if self.use == 'position':
            return np.asarray(self.pos_kpc, dtype=np.float32)
        packed = np.zeros((self.n_stars, 2), dtype=np.float32)
        packed[:, 0] = np.asarray(self.R_proj_kpc, dtype=np.float32)
        return packed

    @classmethod
    def from_dataset(cls, node_feats, graph_feats, index, pre_transforms,
                     labels, cond_label='stellar_log_rstar_kpc',
                     log_rstar_err_dex=0.02, seed=0):
        """Freeze one galaxy of a training dataset as an observation.

        For sanity-checking a model against its own training
        distribution, not for analysis: a galaxy drawn from the set the
        model was fitted to says nothing about how it generalises.
        Build an `Observation` from a catalogue instead.

        Draws a single projection and a single noise realization from
        `pre_transforms`, so the result is one catalogue rather than the
        augmented stream the model trains on.

        Args:
            node_feats: `pos`/`vel` arrays from `datasets.read_datasets`
                with `concat=True`.
            graph_feats: Per-galaxy arrays from the same call.
            index: Index of the galaxy to freeze.
            pre_transforms: Pipeline from `build_transformation`, built
                with the same arguments the model was trained with.
            labels: Inferred parameter names, in the model's order.
            cond_label: Measured conditioning variable.
            log_rstar_err_dex: Photometric error to attach, in dex. A
                simulated galaxy has an exact r_star, so this stands in
                for the measurement a real dwarf would carry; the LVDB
                range is 0.008 to 0.065 dex.
            seed: Seeds the projection and noise draw, so re-running the
                cell gives the same catalogue.

        Returns:
            The frozen `Observation`.
        """
        ptr = np.concatenate(
            [[0], np.cumsum(graph_feats['num_stars'])]).astype(int)
        rows = slice(ptr[index], ptr[index + 1])

        torch.manual_seed(seed)
        batch = pre_transforms(StarBatch(
            pos=torch.tensor(node_feats['pos'][rows], dtype=torch.float32),
            vel=torch.tensor(node_feats['vel'][rows], dtype=torch.float32)))

        return cls(
            pos_kpc=batch.pos.numpy(),
            vlos_kms=batch.vel[:, 0].numpy().astype(np.float64),
            vlos_err_kms=batch.vel_error[:, 0].numpy(),
            log_rstar_kpc=float(graph_feats[cond_label][index]),
            log_rstar_err_dex=log_rstar_err_dex,
            name=f'galaxy{index:05d}',
            theta_true=np.array([graph_feats[k][index] for k in labels],
                                dtype=np.float32))

    @classmethod
    def from_npz(cls, path, name=None):
        """Read a `target.npz` written by the NPE inference pipeline.

        That file records projected radii rather than sky positions, so
        the result has `R_proj_kpc` set and `pos_kpc` None, and is
        usable only with a radius-conditioned model.

        Args:
            path: Path to the `target.npz`.
            name: Output name; defaults to the parent directory's name.

        Returns:
            The `Observation`.
        """
        path = Path(path)
        target = np.load(path, allow_pickle=True)

        return cls(
            R_proj_kpc=target['R_proj_kpc'].astype(np.float32),
            vlos_kms=target['vlos_kms'].astype(np.float64),
            vlos_err_kms=target['vlos_err_kms'].astype(np.float32),
            log_rstar_kpc=float(np.log10(float(target['rhalf_kpc']))),
            log_rstar_err_dex=_rhalf_err_dex(target),
            name=name or path.parent.name)


def _rhalf_err_dex(target):
    """Symmetrized r_half error in dex, or None when absent.

    LVDB stores asymmetric `rhalf_kpc_em`/`_ep`; the conditioning axis
    is log10(r_star / kpc), so they are averaged and propagated through
    the logarithm: sigma = (em + ep) / 2 / (r_half ln 10). Matches
    `sbi_infer.sampling.conditioning_sigma_dex`.
    """
    keys = set(getattr(target, 'files', ()) or target.keys())
    if not {'rhalf_kpc_em', 'rhalf_kpc_ep'} <= keys:
        return None
    rhalf = float(target['rhalf_kpc'])
    err = 0.5 * (float(target['rhalf_kpc_em']) + float(target['rhalf_kpc_ep']))
    if not np.isfinite(err) or err <= 0 or rhalf <= 0:
        return None
    return err / (rhalf * np.log(10.0))


@torch.no_grad()
def galaxy_log_prob(model, obs, theta, v_sys=0.0, log_rstar=None):
    """Galaxy log-likelihood: the i.i.d. sum of per-star log densities.

    The model itself only ever evaluates one star. Treating the stars of
    a galaxy as independent given theta is an assumption made here, not
    by the network.

    Args:
        model: Trained NLE.
        obs: The observed galaxy.
        theta: Candidate labels in physical units, (n_labels,) or
            (n_candidates, n_labels). The model normalizes internally.
        v_sys: Systemic velocity in km/s, scalar or (n_candidates,),
            subtracted from `obs.vlos_kms` before evaluation.
        log_rstar: Conditioning value per candidate, scalar or
            (n_candidates,). None uses the measured
            `obs.log_rstar_kpc`, which is the pinned case.

    Returns:
        (n_candidates,) array of summed log-likelihoods. A candidate
        with any non-finite per-star density returns -inf.

    Raises:
        ValueError: If the observation's geometry does not match what
            the model conditions on. Both directions are refused: a
            radius-only observation under a model that needs sky
            positions, and `use='position'` under a radius-conditioned
            model that was handed a measured `R_proj_kpc`.
    """
    if obs.use == 'radius' and not model.condition_on_radius:
        raise ValueError(
            f'{obs.name} supplies projected radii, but this model is '
            'not radius-conditioned (condition_on_radius=False) and '
            'needs sky positions. Build the Observation with pos_kpc, '
            "or use='position' if you have both")

    if (model.condition_on_radius and obs.use == 'position'
            and obs.R_proj_kpc is not None):
        # Reason: this model conditions on the projected radius, so
        # with use='position' it would recompute one as |pos| and the
        # R_proj_kpc you measured would never be read. Deriving a
        # quantity the catalog already reports is the silent kind of
        # wrong, so refuse rather than pick.
        raise ValueError(
            f'{obs.name} is radius-conditioned but was built with '
            "use='position' while R_proj_kpc is set. The model would "
            'recompute the radius as |pos| and ignore your measured '
            "R_proj_kpc. Pass use='radius' to use it, or drop "
            'R_proj_kpc to derive the radius from pos_kpc deliberately')

    theta = np.atleast_2d(np.asarray(theta, dtype=np.float32))
    n_cand, n_stars = theta.shape[0], obs.n_stars
    v_sys = np.broadcast_to(np.asarray(v_sys, dtype=np.float64), (n_cand,))
    if log_rstar is None:
        log_rstar = obs.log_rstar_kpc
    # Reason: broadcast_to returns a read-only view, which torch
    # refuses to wrap without a warning; copy once per call.
    log_rstar = np.broadcast_to(
        np.asarray(log_rstar, dtype=np.float32), (n_cand,)).copy()

    # One row per (candidate, star): stars tile, candidates repeat.
    vel = (obs.vlos_kms[None, :] - v_sys[:, None]).reshape(-1, 1)
    batch = StarBatch(
        pos=torch.from_numpy(obs.model_positions()).repeat(n_cand, 1),
        vel=torch.from_numpy(vel).float(),
        vel_error=torch.from_numpy(obs.vlos_err_kms).float().repeat(
            n_cand).view(-1, 1),
        theta=torch.from_numpy(theta).repeat_interleave(n_stars, dim=0),
        cond=torch.from_numpy(log_rstar).repeat_interleave(
            n_stars).view(-1, 1))

    log_p = model.log_prob(batch).view(n_cand, n_stars)
    total = log_p.sum(dim=1)
    total[~torch.isfinite(log_p).all(dim=1)] = -np.inf
    return total.cpu().numpy()


def cond_box(model):
    """The log10(r_star / kpc) range this checkpoint was trained on.

    Read off the model's own normalization rather than any registry:
    training maps the conditioning box onto [-1, 1], so the box is
    `cond_loc +- cond_scale`. For the production NLE that is
    [-2.0, 0.5], r_star from 10 pc to 3.2 kpc.

    Args:
        model: A trained NLE whose `norm_dict` carries `cond_loc` and
            `cond_scale`.

    Returns:
        (low, high) in log10(r_star / kpc).

    Raises:
        ValueError: If the model has no conditioning axis at all.
    """
    if 'cond_loc' not in model.norm_dict:
        raise ValueError('this model has no conditioning axis')
    loc = float(model.norm_dict['cond_loc'][0])
    scale = float(model.norm_dict['cond_scale'][0])
    return loc - scale, loc + scale


def sample_posterior(model, obs, prior, labels, fit_v_sys=True,
                     sample_log_rstar=True, n_effective=2048, n_active=512,
                     n_total=2048, seed=0, progress=True, pool=None):
    """Sample p(theta, log r_star | one galaxy) with pocomc.

    The prior is yours to build and pass in. It is the one part of an
    inference you cannot delegate without losing track of what you
    assumed, so this function deliberately does not construct one: the
    simulation box, the systemic-velocity window and the photometric
    prior on `log r_star` are all choices about the galaxy in front of
    you, not defaults.

    The dimension order is a contract. Columns must be

        [labels..., v_sys, log_rstar]

    with `v_sys` present when `fit_v_sys` and `log_rstar` present when
    `sample_log_rstar`. `prior.dim` is checked against that, so a prior
    built to a different layout fails here rather than silently
    assigning a density to the wrong parameter.

    Args:
        model: Trained NLE.
        obs: The observed galaxy.
        prior: A `pocomc.Prior` over the columns above.
        labels: Inferred parameter names, in the model's order. No
            default: a label list that silently disagrees with the
            checkpoint relabels every posterior column.
        fit_v_sys: Whether the prior carries a systemic-velocity
            column, fitted as a nuisance parameter.
        sample_log_rstar: Whether the prior carries a `log r_star`
            column. False pins it at `obs.log_rstar_kpc`, which asserts
            the photometry is exact.
        n_effective: Particle population pocomc maintains.
        n_active: Particles moved per step.
        n_total: Posterior samples drawn, and pocomc's `n_evidence`.
            Under pocoMC this is the work, not just the output size.
        seed: pocomc's random state.
        progress: Show the pocomc progress bar.
        pool: Optional worker pool. Leave None in a notebook: the
            likelihood already runs `n_active` particles per call.

    Returns:
        Dict with `posterior` (N, n_labels + 1) in model units with the
        conditioning column last, `labels` naming those columns,
        `nuisance` (N, 1) holding v_sys in km/s, `log_evidence` and
        `log_evidence_err`, and `sampler_stats`.

    Raises:
        ValueError: If `prior.dim` does not match the column contract.
    """
    labels = list(labels)
    n_labels = len(labels)
    expected = n_labels + int(fit_v_sys) + int(sample_log_rstar)
    if prior.dim != expected:
        raise ValueError(
            f'prior has {prior.dim} dimensions, but the column contract '
            f'[labels..., v_sys, log_rstar] with {n_labels} labels, '
            f'fit_v_sys={fit_v_sys} and '
            f'sample_log_rstar={sample_log_rstar} needs {expected}')

    def log_likelihood(theta):
        """Vectorized log-likelihood; pocomc applies the prior itself."""
        theta = np.atleast_2d(theta)
        log_like = galaxy_log_prob(
            model, obs, theta[:, :n_labels],
            v_sys=theta[:, n_labels] if fit_v_sys else 0.0,
            log_rstar=theta[:, -1] if sample_log_rstar else None)
        # Reason: pocomc's importance weights go NaN if a whole batch is
        # -inf, so an impossible particle gets a finite floor instead.
        # Nothing finite comes near it, so it is a rejection in practice.
        return np.where(np.isfinite(log_like), log_like, LOG_LIKE_FLOOR)

    sampler = pc.Sampler(
        prior=prior,
        likelihood=log_likelihood,
        vectorize=True,
        n_effective=n_effective,
        n_active=n_active,
        random_state=seed,
        pool=pool,
    )
    sampler.run(n_total=n_total, n_evidence=n_total, progress=progress)

    chain, _, _ = sampler.posterior(resample=True)
    log_evidence, log_evidence_err = sampler.evidence()

    nuisance = (chain[:, n_labels:n_labels + 1] if fit_v_sys
                else np.zeros((len(chain), 1)))
    cond = (chain[:, -1] if sample_log_rstar
            else np.full(len(chain), obs.log_rstar_kpc))
    posterior = np.column_stack([chain[:, :n_labels], cond])

    return {
        'posterior': posterior,
        'labels': labels + ['log_rstar'],
        'nuisance': nuisance,
        'nuisance_labels': ['v_sys_kms'],
        'log_evidence': float(log_evidence),
        'log_evidence_err': float(log_evidence_err),
        'sampler_stats': {
            'sampler': 'pocomc',
            'n_effective': n_effective,
            'n_active': n_active,
            'n_total': n_total,
        },
    }


def save_posterior(out_dir, obs, result):
    """Write the shared posterior file `<out_dir>/<obs.name>.npz`.

    The key names follow `sbi_infer.nle_backend`'s return shape:
    `posterior_model_units` is (N, n_labels + 1) with the conditioning
    column last, and the systemic velocity rides in `nuisance` rather
    than in the posterior. `log_rho_150` is computed per row from that
    row's own r_star, so a sampled conditioning is propagated into the
    density rather than collapsed onto the measured value.

    `posterior_kpc` is NOT written: converting radii to kpc depends on
    the model's `radius_units` and family, which only
    `sbi_infer.sampling.to_physical_kpc` knows.

    Args:
        out_dir: Directory to write into; created if missing.
        obs: The observed galaxy the posterior belongs to.
        result: Return value of `sample_posterior`.

    Returns:
        Path to the written file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{obs.name}.npz'

    posterior = result['posterior']
    # log_rho_150 broadcasts the shift, so a per-row r_star goes
    # through in one call.
    density = log_rho_150(posterior[:, :5], posterior[:, -1])

    np.savez(
        out_path,
        posterior_model_units=posterior,
        labels=np.array(result['labels']),
        nuisance=result['nuisance'],
        nuisance_labels=np.array(result['nuisance_labels']),
        log_rho_150=density,
        log_rstar_measured=obs.log_rstar_kpc,
        n_stars=obs.n_stars,
        log_evidence=result['log_evidence'],
        log_evidence_err=result['log_evidence_err'],
        sampler_stats=json.dumps(result['sampler_stats']),
        theta_true=(np.array([]) if obs.theta_true is None
                    else obs.theta_true),
    )
    return out_path
