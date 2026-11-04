"""Core per-star container for NLE.

The container holds physical phase-space fields only. Feature extraction,
the modeled-vs-conditioned split, and normalization are the NLE model's
job, not the data pipeline's.
"""

from dataclasses import dataclass

from torch import Tensor


@dataclass
class StarBatch:
    """A batch of stars in phase space for per-star NLE.

    Each row is one star. Fields are physical (unnormalized). After
    `RandomProjection`, `pos` is the 2-D sky position and `vel` holds the
    line-of-sight velocity (optionally with the two proper-motion
    components). `vel_error` is the per-component measurement sigma set by
    `UncertaintySampler` (training) or supplied from real data (inference),
    and is always used as conditioning context.

    Attributes:
        pos: Positions, (B, 3) raw Cartesian; (B, 2) after projection.
        vel: Velocities, (B, 3) raw; (B, k) after projection.
        vel_error: Per-velocity measurement sigma, (B, k). None until set.
        theta: Per-star inferred parameters, (B, n_params). None on real
            observations.
        cond: Per-star measured conditioning variables, (B, n_cond), e.g.
            the stellar half-light radius. Distinct from `theta`: measured
            (fixed or MC-sampled at inference), not inferred. None when unused.
        transformed: True once the observation pipeline (projection +
            noise) has been applied, so the model must not apply it again.
            Set by the deterministic validation set; raw loader batches
            leave it False.
    """

    pos: Tensor
    vel: Tensor
    vel_error: Tensor | None = None
    theta: Tensor | None = None
    cond: Tensor | None = None
    transformed: bool = False

    def clone(self) -> "StarBatch":
        """Return a deep copy so transforms never mutate their input."""
        return StarBatch(
            pos=self.pos.clone(),
            vel=self.vel.clone(),
            vel_error=(None if self.vel_error is None
                       else self.vel_error.clone()),
            theta=None if self.theta is None else self.theta.clone(),
            cond=None if self.cond is None else self.cond.clone(),
            transformed=self.transformed,
        )

    def to(self, device) -> "StarBatch":
        """Move all present tensor fields to `device`."""
        return StarBatch(
            pos=self.pos.to(device),
            vel=self.vel.to(device),
            vel_error=(None if self.vel_error is None
                       else self.vel_error.to(device)),
            theta=None if self.theta is None else self.theta.to(device),
            cond=None if self.cond is None else self.cond.to(device),
            transformed=self.transformed,
        )
