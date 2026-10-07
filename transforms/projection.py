"""Random line-of-sight projection of Cartesian phase space."""

import torch

from .basic import StarBatch


def random_rotation_matrix() -> torch.Tensor:
    """Return a uniformly random 3-D rotation matrix via a quaternion."""
    q = torch.randn(4)
    q /= torch.norm(q)

    q0, q1, q2, q3 = q.unbind()
    R = torch.tensor([
        [1 - 2*q2**2 - 2*q3**2, 2*q1*q2 - 2*q3*q0, 2*q1*q3 + 2*q2*q0],
        [2*q1*q2 + 2*q3*q0, 1 - 2*q1**2 - 2*q3**2, 2*q2*q3 - 2*q1*q0],
        [2*q1*q3 - 2*q2*q0, 2*q2*q3 + 2*q1*q0, 1 - 2*q1**2 - 2*q2**2]
    ])
    return R


class RandomProjection:
    """Apply a random line-of-sight projection to a batch of stars.

    A single random rotation is drawn per batch and applied to every star.
    Because stars from different galaxies may share a batch, this is a fast
    approximation to projecting each galaxy independently.

    By default only the line-of-sight velocity (along the removed axis) is
    kept, mimicking a radial-velocity-only observation. With
    ``use_proper_motions=True`` the two sky-plane velocity components (the
    proper motion in km/s) are kept as well.

    Columns are ordered cyclically from the line-of-sight axis: given LOS
    ``axis``, the sky-plane axes are ``(axis + 1) % 3, (axis + 2) % 3`` for
    both ``pos`` (sky-plane position) and ``vel`` (line-of-sight velocity
    first, then the two proper-motion components when enabled).

    ``components`` selects which of the three velocity columns
    (0 = line of sight, 1 and 2 = proper motion) are kept, in that order,
    e.g. ``(1, 2)`` for a proper-motion-only model. It implies
    ``use_proper_motions`` and is how models for different measured
    subsets of the velocity are trained for mixed-completeness data.
    """

    def __init__(self, axis=None, use_proper_motions: bool = False,
                 components=None):
        self.axis = axis
        self.components = None if components is None else tuple(components)
        self.use_proper_motions = use_proper_motions or (
            self.components is not None)

    def __call__(self, batch: StarBatch) -> StarBatch:
        batch = batch.clone()

        if self.axis is None:
            # one random rotation for the whole batch
            R = random_rotation_matrix().to(batch.pos.device)
            pos = torch.matmul(batch.pos, R)
            vel = torch.matmul(batch.vel, R)
            # by convention the observer looks along the last rotated axis
            axis = 2
        else:
            pos = batch.pos
            vel = batch.vel
            axis = self.axis

        # cyclic ordering of the two axes orthogonal to the LOS axis
        other1 = (axis + 1) % 3
        other2 = (axis + 2) % 3

        pos_proj = torch.stack([pos[:, other1], pos[:, other2]], dim=1)

        if self.use_proper_motions:
            # LOS velocity first, then the two proper-motion components
            vel_proj = torch.stack(
                [vel[:, axis], vel[:, other1], vel[:, other2]], dim=1
            )
            if self.components is not None:
                vel_proj = vel_proj[:, list(self.components)]
        else:
            vel_proj = vel[:, axis].unsqueeze(1)

        batch.pos = pos_proj
        batch.vel = vel_proj
        return batch
