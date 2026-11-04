"""Unit tests for the per-star dataloader assembly and the split."""

import numpy as np
import torch

from nle_dsph.datasets import (
    _broadcast_graph_features, _build_star_tensors, _split_by_galaxy,
    collate_star_batch, prepare_dataloaders,
)

LABELS = ('dm_alpha', 'dm_gamma')
COND_LABELS = ('stellar_log_rstar_kpc',)


def make_fake_dataset(n_galaxies=40, seed=0):
    """An in-memory stand-in for one shard, in the HDF5 reader's layout."""
    rng = np.random.default_rng(seed)
    num_stars = rng.integers(5, 20, size=n_galaxies)
    n_rows = int(num_stars.sum())

    node_feats = {
        'pos': rng.normal(size=(n_rows, 3)),
        'vel': rng.normal(scale=10.0, size=(n_rows, 3)),
    }
    graph_feats = {'num_stars': num_stars}
    for key in LABELS + COND_LABELS:
        graph_feats[key] = rng.normal(size=n_galaxies)
    return node_feats, graph_feats


def test_split_never_puts_one_galaxy_in_both_sets():
    """The whole point of splitting by galaxy rather than by star."""
    _, graph_feats = make_fake_dataset()
    num_stars = graph_feats['num_stars']

    train_mask, val_mask = _split_by_galaxy(num_stars, 0.8, seed=0)

    galaxy_id = np.repeat(np.arange(len(num_stars)), num_stars)
    shared = set(galaxy_id[train_mask]) & set(galaxy_id[val_mask])
    assert shared == set()


def test_split_is_a_partition_of_every_star():
    """No star is dropped and none is counted twice."""
    _, graph_feats = make_fake_dataset()
    num_stars = graph_feats['num_stars']

    train_mask, val_mask = _split_by_galaxy(num_stars, 0.8, seed=0)

    assert np.all(train_mask ^ val_mask)
    assert train_mask.sum() + val_mask.sum() == num_stars.sum()


def test_split_is_reproducible_for_a_given_seed():
    """A resumed run must reuse the split it trained on."""
    _, graph_feats = make_fake_dataset()
    num_stars = graph_feats['num_stars']

    first, _ = _split_by_galaxy(num_stars, 0.8, seed=7)
    again, _ = _split_by_galaxy(num_stars, 0.8, seed=7)
    other, _ = _split_by_galaxy(num_stars, 0.8, seed=8)

    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_graph_features_broadcast_to_each_galaxy_own_stars():
    """A galaxy's theta must land on its stars, not on its neighbour's."""
    num_stars = np.array([3, 1, 2])
    graph_feats = {'a': np.array([10.0, 20.0, 30.0])}

    out = _broadcast_graph_features(graph_feats, ['a'], num_stars)

    assert torch.allclose(
        out[:, 0], torch.tensor([10., 10., 10., 20., 30., 30.]))


def test_star_tensors_are_consistent_in_length():
    """pos, vel, theta and cond are all one row per star."""
    node_feats, graph_feats = make_fake_dataset()

    pos, vel, theta, cond, num_stars = _build_star_tensors(
        node_feats, graph_feats, LABELS, COND_LABELS)

    n_rows = int(num_stars.sum())
    assert pos.shape == (n_rows, 3)
    assert vel.shape == (n_rows, 3)
    assert theta.shape == (n_rows, len(LABELS))
    assert cond.shape == (n_rows, len(COND_LABELS))


def test_cond_is_none_when_no_cond_labels_are_asked_for():
    """A model with no conditioning must not be handed an empty one."""
    node_feats, graph_feats = make_fake_dataset()

    *_, cond, _ = _build_star_tensors(node_feats, graph_feats, LABELS, None)

    assert cond is None


def test_collate_returns_a_raw_untransformed_batch():
    """The loader only packages; the model applies the pipeline."""
    samples = [(torch.zeros(3), torch.zeros(3), torch.zeros(2),
                torch.zeros(1)) for _ in range(4)]

    batch = collate_star_batch(samples)

    assert batch.pos.shape == (4, 3)
    assert batch.cond.shape == (4, 1)
    assert batch.vel_error is None
    assert batch.transformed is False


def test_prepare_dataloaders_yields_star_batches_and_a_norm_dict():
    """End to end on a fake shard: the training script's contract."""
    node_feats, graph_feats = make_fake_dataset()

    train_loader, val_loader, norm_dict = prepare_dataloaders(
        node_feats, graph_feats, LABELS, cond_labels=COND_LABELS,
        train_batch_size=16, eval_batch_size=16, num_workers=0, seed=0,
        pre_transform_kwargs={
            'apply_projection': True,
            'apply_uncertainty': True,
            'uncertainty_args': {
                'distribution_type': 'jeffreys_varied', 'vel_idx': 0,
                'low_range': (0.01, 0.1), 'width_range': (5.0, 30.0)},
        })

    batch = next(iter(train_loader))
    assert batch.pos.shape[1] == 3, 'loaders yield raw 3-D phase space'
    assert batch.theta.shape[1] == len(LABELS)
    assert 'cond_loc' in norm_dict
    assert len(val_loader) > 0
