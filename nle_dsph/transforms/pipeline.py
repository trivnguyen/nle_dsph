"""Compose the per-star NLE observation pipeline and measure field stats."""

import torch

from .basic import StarBatch
from .projection import RandomProjection
from .uncertainty import UncertaintySampler


class Compose:
    """Chain transforms, threading a StarBatch through each in order."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, batch: StarBatch) -> StarBatch:
        for transform in self.transforms:
            batch = transform(batch)
        return batch


def build_transformation(
    apply_projection: bool = True,
    projection_args: dict = None,
    apply_uncertainty: bool = False,
    uncertainty_args=None,
) -> Compose:
    """
    Build the observation pipeline (projection + measurement uncertainty).

    This is the stochastic, per-batch augmentation the NLE model applies in
    `_prepare_batch` during training. It produces physical fields only;
    normalization and the modeled-vs-conditioned split happen in the model.

    `uncertainty_args` accepts either a single dict or a list of dicts to
    chain multiple `UncertaintySampler` transforms, each perturbing its own
    velocity component (`vel_idx`) with its own `distribution_type` and
    parameters (e.g. one for v_los and one per proper-motion component).
    """
    transforms = []

    if apply_projection:
        transforms.append(RandomProjection(**(projection_args or {})))

    if apply_uncertainty:
        if uncertainty_args is None:
            raise ValueError(
                '`uncertainty_args` must be provided when '
                '`apply_uncertainty` is True.')
        # a single mapping (dict/ConfigDict) or a list of them, one per
        # perturbed velocity component
        if not isinstance(uncertainty_args, (list, tuple)):
            uncertainty_args = [uncertainty_args]
        for args in uncertainty_args:
            transforms.append(UncertaintySampler(**args))

    return Compose(transforms)


def compute_field_norm(
    batch: StarBatch, batch_size: int = 4096, **pre_transform_kwargs
) -> dict:
    """Measure per-field normalization stats for the NLE model.

    Streams `batch` through the observation pipeline built from
    `pre_transform_kwargs` (projection + uncertainty), chunk by chunk, then
    computes per-component location/scale for `pos`, `vel`, and `vel_error`.
    `theta` stats come from its midrange/half-range (mapped to ~[-1, 1]).
    Scales are floored at a small epsilon so unperturbed components (zero
    variance) don't divide by zero.

    Args:
        batch: Raw StarBatch with `pos`/`vel` (and `theta`) set.
        batch_size: Number of stars per streaming chunk.
        **pre_transform_kwargs: Same kwargs as `build_transformation`.

    Returns:
        Dict with `<field>_loc`/`<field>_scale` lists for pos, vel,
        vel_error, theta, and cond (cond only if `batch.cond` is set).
    """
    pipeline = build_transformation(**pre_transform_kwargs)

    # Running per-component min/max over the transformed chunks. Exactly the
    # min/max of the concatenation, without holding every chunk in memory
    # (for 530M stars the concatenation alone was 17 GB of scratch).
    n = batch.pos.shape[0]
    lo, hi = {}, {}

    def _update(name, value):
        v_min, v_max = value.min(dim=0)[0], value.max(dim=0)[0]
        lo[name] = v_min if name not in lo else torch.minimum(lo[name], v_min)
        hi[name] = v_max if name not in hi else torch.maximum(hi[name], v_max)

    for start in range(0, n, batch_size):
        stop = start + batch_size
        chunk = StarBatch(pos=batch.pos[start:stop], vel=batch.vel[start:stop])
        out = pipeline(chunk)
        _update('pos', out.pos)
        _update('vel', out.vel)
        _update('vel_error',
                torch.zeros_like(out.vel) if out.vel_error is None
                else out.vel_error)

    _floor = lambda scale: scale.clamp_min(1e-6)  # avoid divide-by-zero

    # normalized to ~[-1, 1] via midrange/half-range. pos/vel/vel_error
    # are measured AFTER the pipeline (projected, noisy), since that is
    # what the model sees; theta/cond are per-galaxy and untouched by it
    # (their min/max are the same whether given per star or per galaxy).
    for name in ('theta', 'cond'):
        field = getattr(batch, name, None)
        if field is not None:
            _update(name, field)

    norm = {}
    for name in ('pos', 'vel', 'vel_error', 'theta', 'cond'):
        if name not in lo:
            continue
        f_min, f_max = lo[name], hi[name]
        norm[f'{name}_loc'] = ((f_max + f_min) / 2).tolist()
        norm[f'{name}_scale'] = _floor((f_max - f_min) / 2).tolist()

    return norm
