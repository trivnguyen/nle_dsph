"""Neural Likelihood Estimation (NLE) module for per-star dwarf-galaxy data.

The model learns the single-star likelihood p(observables | theta, cond)
with a conditional normalizing flow, and can sample observables from it.
It deliberately does ONLY what it is trained on (per-star density + sampling)
-- assumptions like i.i.d. stars, priors, fixing/sampling `cond`, and MCMC
live outside the model.

Velocity is always modeled; sky position is modeled too by default, or fed
as context when `condition_on_position=True`. `theta` (inferred) and `cond`
(measured, e.g. the half-light radius) are always context, kept as separate
fields so callers can sample `theta` while fixing or MC-sampling `cond`.
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import zuko

from transforms import StarBatch
from utils import MLP, configure_optimizers, get_activation


def build_flow(
    features: int,
    context: int,
    flow_type: str = 'spline',
    num_transforms: int = 3,
    hidden_features=(64, 64),
    num_bins: int = 8,
    activation: str = None,
):
    """Build a conditional zuko flow for p(target | context)."""
    kwargs = dict(transforms=num_transforms, hidden_features=hidden_features)
    if activation is not None:
        kwargs['activation'] = get_activation(
            activation, return_instance=False)

    if flow_type == 'spline':
        return zuko.flows.NSF(features, context, bins=num_bins, **kwargs)
    if flow_type == 'maf':
        return zuko.flows.MAF(features, context, **kwargs)
    raise ValueError(f"Unknown flow_type: {flow_type!r} (use 'spline'/'maf')")


class NLE(pl.LightningModule):
    """Conditional-flow likelihood p(target | context) over single stars.

    Parameters
    ----------
    norm_dict : dict
        Per-field location/scale from `transforms.compute_field_norm`.
    condition_on_position : bool, default=False
        If True, model p(v | x, y, theta, cond, err); else p(x, y, v | ...).
    condition_on_radius : bool, default=False
        With `condition_on_position`, replace the (x, y) context by the
        single feature log10 R, the projected radius. The physical
        likelihood p(v_los | R, theta) is axisymmetric, so this removes
        a symmetry the flow would otherwise have to learn and makes the
        context independent of the sky orientation.
    condition_on_error : bool, default=True
        Include the per-star measurement sigma `vel_error` in the context.
        False for a model trained on noiseless velocities, which then
        learns p(v_true | ...) and takes batches with `vel_error=None`.
    flows_args : Mapping, optional
        Passed to `build_flow`.
    context_embedding_args : Mapping, optional
        If given, a single MLP embeds the full context before the flow.
        Keys: hidden_sizes, output_size, act_name, act_args, batch_norm,
        dropout.
    optimizer_args, scheduler_args : Mapping, optional
        Passed to `utils.configure_optimizers`.
    pre_transforms : callable, optional
        Observation pipeline (projection + uncertainty) applied to raw
        batches during training only.
    """

    def __init__(
        self,
        norm_dict: dict,
        condition_on_position: bool = False,
        condition_on_radius: bool = False,
        condition_on_error: bool = True,
        flows_args: dict = None,
        context_embedding_args: dict = None,
        optimizer_args: dict = None,
        scheduler_args: dict = None,
        pre_transforms=None,
    ):
        super().__init__()
        self.norm_dict = norm_dict
        self.condition_on_position = condition_on_position
        self.condition_on_radius = condition_on_radius
        self.condition_on_error = condition_on_error
        if condition_on_radius and not condition_on_position:
            raise ValueError('condition_on_radius requires '
                             'condition_on_position=True')
        self.flows_args = flows_args or {}
        self.context_embedding_args = context_embedding_args
        self.optimizer_args = optimizer_args or {}
        self.scheduler_args = scheduler_args or {}
        self.pre_transforms = pre_transforms

        self.save_hyperparameters(ignore=['pre_transforms'])

        self._register_norm(norm_dict)
        self._setup_model()

    # -- setup --------------------------------------------------------------

    def _register_norm(self, norm_dict: dict):
        self.has_cond = 'cond_loc' in norm_dict
        fields = ['pos', 'vel', 'vel_error', 'theta']
        if self.has_cond:
            fields.append('cond')
        for field in fields:
            self.register_buffer(
                f'{field}_loc',
                torch.tensor(norm_dict[f'{field}_loc'], dtype=torch.float32))
            self.register_buffer(
                f'{field}_scale',
                torch.tensor(norm_dict[f'{field}_scale'], dtype=torch.float32))

    def _setup_model(self):
        """Set up the field layout, optional context MLP, and flow."""
        dims = {f: len(self.norm_dict[f'{f}_loc'])
                for f in ('pos', 'vel', 'vel_error', 'theta')}
        if self.has_cond:
            dims['cond'] = len(self.norm_dict['cond_loc'])

        # Velocity is always the target; position is target unless conditioned
        # on. theta and cond are always context, followed by vel_error
        # (unless the model is noiseless).
        context = ['theta'] + (['cond'] if self.has_cond else [])
        if self.condition_on_error:
            context.append('vel_error')
        if self.condition_on_position:
            self.target_fields = ['vel']
            self.context_fields = ['pos'] + context
            if self.condition_on_radius:
                dims['pos'] = 1
        else:
            self.target_fields = ['pos', 'vel']
            self.context_fields = context

        target_size = sum(dims[f] for f in self.target_fields)
        context_size = sum(dims[f] for f in self.context_fields)

        if self.context_embedding_args:
            args = dict(self.context_embedding_args)
            act = get_activation(
                args.pop('act_name', 'relu'), args.pop('act_args', None))
            self.context_embedding = MLP(
                input_size=context_size, act=act, **args)
            flow_context_size = self.context_embedding.output_size
            n_ctx = sum(p.numel()
                        for p in self.context_embedding.parameters())
            print(f"[NLE] Context MLP {context_size} -> {flow_context_size} "
                  f"with {n_ctx:,} parameters")
        else:
            self.context_embedding = nn.Identity()
            flow_context_size = context_size

        self.flow = build_flow(
            target_size, flow_context_size, **self.flows_args)
        print(f"[NLE] Modeling p({','.join(self.target_fields)} | "
              f"{','.join(self.context_fields)})")
        n_flow = sum(p.numel() for p in self.flow.parameters())
        print(f"[NLE] Flow built with {n_flow:,} parameters")

    # -- normalization / assembly ------------------------------------------

    def _normalize(self, batch: StarBatch) -> StarBatch:
        """Normalize every present field to the training scale."""
        def norm(field, value):
            if value is None:
                return None
            if field == 'pos' and self.condition_on_radius:
                # Reason: log10 R in kpc spans ~[-3, 1.5]; a fixed
                # scaling keeps norm_dict unchanged for this option.
                log_r = torch.log10(value.norm(dim=1, keepdim=True) + 1e-6)
                return (log_r + 1.0) / 1.0
            return (value - getattr(self, f'{field}_loc')) / getattr(
                self, f'{field}_scale')

        return StarBatch(
            pos=norm('pos', batch.pos),
            vel=norm('vel', batch.vel),
            vel_error=norm('vel_error', batch.vel_error),
            theta=norm('theta', batch.theta),
            cond=norm('cond', batch.cond) if self.has_cond else None,
        )

    def _assemble(self, batch: StarBatch, fields) -> torch.Tensor:
        cols = []
        for field in fields:
            value = getattr(batch, field)
            if value is None:
                raise ValueError(f'batch.{field} is required but was None')
            cols.append(value)
        return torch.cat(cols, dim=1)

    def _flow_log_prob(self, norm_batch: StarBatch) -> torch.Tensor:
        """Per-star log p(target | context) for a normalized StarBatch."""
        target = self._assemble(norm_batch, self.target_fields)
        context = self.context_embedding(
            self._assemble(norm_batch, self.context_fields))
        return self.flow(context).log_prob(target)

    # -- training -----------------------------------------------------------

    def _prepare_batch(self, batch: StarBatch) -> StarBatch:
        """Augment (project + noise) then normalize a raw training batch.

        Batches flagged `transformed` (the fixed validation set) have been
        through the pipeline already and are only normalized.
        """
        if self.pre_transforms is not None and not batch.transformed:
            batch = self.pre_transforms(batch)
        return self._normalize(batch)

    def training_step(self, batch, batch_idx):
        loss = -self._flow_log_prob(self._prepare_batch(batch)).mean()
        self.log(
            'train/loss', loss, on_step=True, on_epoch=True, prog_bar=True,
            logger=True, batch_size=batch.pos.shape[0], sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Validation always runs in full fp32, whatever precision the
        # training step uses: the val loss selects checkpoints, and the
        # production model is evaluated in fp32 at inference, so the
        # selection metric must be the fp32 density.
        with torch.autocast(device_type=self.device.type, enabled=False):
            loss = -self._flow_log_prob(self._prepare_batch(batch)).mean()
        self.log(
            'val/loss', loss, on_step=False, on_epoch=True, prog_bar=True,
            logger=True, batch_size=batch.pos.shape[0], sync_dist=True)
        return loss

    def configure_optimizers(self):
        """Initialize optimizer and LR scheduler."""
        return configure_optimizers(
            self.parameters(), self.optimizer_args, self.scheduler_args)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return batch.to(device)

    # -- inference (observed data in physical units) -----------------------

    @torch.no_grad()
    def log_prob(self, batch: StarBatch) -> torch.Tensor:
        """Per-star log p(observables | theta, cond) for a physical batch.

        `batch` holds the observed per-star fields (`pos`, `vel`,
        `vel_error`) plus the candidate `theta` and `cond`, all broadcast to
        one row per star by the caller. No augmentation is applied. Returns
        a (n_stars,) tensor; summing over stars, priors, and MCMC are the
        caller's job.
        """
        self.eval()
        return self._flow_log_prob(self._normalize(batch.to(self.device)))

    @torch.no_grad()
    def sample(self, batch: StarBatch, num_samples: int = 1) -> torch.Tensor:
        """Sample observables from p(target | context) for each star.

        Uses the context assembled from `batch` (theta, cond, and the
        conditioning observables). Returns physical-unit samples of shape
        (n_stars, num_samples, target_dim).
        """
        self.eval()
        norm_batch = self._normalize(batch.to(self.device))
        context = self.context_embedding(
            self._assemble(norm_batch, self.context_fields))
        samples = self.flow(context).sample((num_samples,))

        loc = torch.cat(
            [getattr(self, f'{f}_loc') for f in self.target_fields])
        scale = torch.cat(
            [getattr(self, f'{f}_scale') for f in self.target_fields])
        samples = samples * scale + loc
        return samples.transpose(0, 1)
