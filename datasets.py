"""HDF5 I/O and per-star dataloaders for NLE.

Each training example is a single star: the model learns the likelihood
p(x | theta), where theta is the star's host-galaxy parameters. Stars are
kept flat (no per-galaxy grouping) except when splitting train/val, where
all stars of a galaxy are forced into the same set so the two never share a
galaxy.
"""

import os
import warnings
import h5py

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from transforms import StarBatch, build_transformation, compute_field_norm


def read_graph_dataset(path, features_list=None, concat=False, to_array=True):
    """Read a graph dataset from an HDF5 file.

    Parameters
    ----------
    path          : str   path to the HDF5 file
    features_list : list  features to read; reads all if empty / None
    concat        : bool  if True, concatenate all node features into one array
    to_array      : bool  if True (and not concat), wrap node-feature lists in
                          a numpy object array

    Returns
    -------
    node_features  : dict
    graph_features : dict
    headers        : dict
    """
    if features_list is None:
        features_list = []

    with h5py.File(path, 'r') as f:
        headers = dict(f.attrs)

        if len(features_list) == 0:
            features_list = headers['all_features']

        node_features = {}
        for key in headers['node_features']:
            if key in features_list:
                if f.get(key) is None:
                    warnings.warn(f'Feature {key} not found in {path}')
                    continue
                if concat:
                    node_features[key] = f[key][:]
                else:
                    node_features[key] = np.split(f[key][:], f['ptr'][:-1])

        graph_features = {}
        for key in headers['graph_features']:
            if key in features_list:
                if f.get(key) is None:
                    warnings.warn(f'Feature {key} not found in {path}')
                    continue
                graph_features[key] = f[key][:]

    if not concat and to_array:
        node_features = {
            p: np.array(v, dtype='object') for p, v in node_features.items()}

    return node_features, graph_features, headers


def read_datasets(
    root, name, num_datasets=100, init=0, is_directory=True, concat=True,
    ext='.h5',
):
    """Read and concatenate multiple HDF5 dataset files."""
    if ext[0] != '.':
        ext = '.' + ext

    if is_directory:
        node_feats, graph_feats = {}, {}

        for i in tqdm(range(init, init + num_datasets)):
            data_path = os.path.join(root, name, f'data.{i}{ext}')
            if not os.path.exists(data_path):
                print(f'Warning: {data_path} does not exist. Skipping...')
                continue
            nodes, graphs, _ = read_graph_dataset(data_path, concat=concat)

            for k in nodes:
                node_feats.setdefault(k, []).append(nodes[k])
            for k in graphs:
                graph_feats.setdefault(k, []).append(graphs[k])

        if not node_feats or not graph_feats:
            raise ValueError(
                f'No valid datasets found in {root}/{name} with '
                f'init={init} and num_datasets={num_datasets}.')

        node_feats = {k: np.concatenate(v) for k, v in node_feats.items()}
        graph_feats = {k: np.concatenate(v) for k, v in graph_feats.items()}
    else:
        data_path = os.path.join(root, name + ext)
        node_feats, graph_feats, _ = read_graph_dataset(
            data_path, concat=concat)

    return node_feats, graph_feats


# ---------------------------------------------------------------------------
# Per-star tensor assembly
# ---------------------------------------------------------------------------

class StarDataset(Dataset):
    """Flat per-star dataset yielding (pos, vel, theta[, cond]) per star."""

    def __init__(self, pos: torch.Tensor, vel: torch.Tensor,
                 theta: torch.Tensor, cond: torch.Tensor = None):
        self.pos = pos
        self.vel = vel
        self.theta = theta
        self.cond = cond

    def __len__(self) -> int:
        return self.pos.shape[0]

    def __getitem__(self, i):
        if self.cond is None:
            return self.pos[i], self.vel[i], self.theta[i]
        return self.pos[i], self.vel[i], self.theta[i], self.cond[i]


def collate_star_batch(samples) -> StarBatch:
    """Stack per-star samples into a raw StarBatch (pos/vel/theta[/cond]).

    The plain-tensor analog of PyG's automatic `Batch.from_data_list`: it
    only packages the batch. The observation pipeline (projection,
    uncertainty) and normalization are applied later by the model, so a
    single per-batch random projection still covers the whole mixed-galaxy
    batch (the accepted fast approximation).
    """
    pos = torch.stack([s[0] for s in samples])
    vel = torch.stack([s[1] for s in samples])
    theta = torch.stack([s[2] for s in samples])
    cond = (torch.stack([s[3] for s in samples])
            if len(samples[0]) == 4 else None)
    return StarBatch(pos=pos, vel=vel, theta=theta, cond=cond)


def _broadcast_graph_features(graph_feats, keys, num_stars):
    """Broadcast per-galaxy features `keys` to per-star rows."""
    per_galaxy = np.column_stack(
        [graph_feats[k][:len(num_stars)] for k in keys])
    return torch.tensor(
        np.repeat(per_galaxy, num_stars, axis=0), dtype=torch.float32)


def _build_star_tensors(node_feats, graph_feats, labels, cond_labels=None,
                        max_graphs=None):
    """Assemble flat per-star pos/vel/theta[/cond] tensors.

    Each galaxy's `labels` (theta, inferred) and `cond_labels` (measured
    conditioning, e.g. the half-light radius) are broadcast to its stars.
    Returns the tensors plus the per-galaxy `num_stars` counts (needed for
    the galaxy-level split). `cond` is None when `cond_labels` is empty.
    """
    num_stars = np.asarray(graph_feats['num_stars'])
    if max_graphs is not None:
        num_stars = num_stars[:max_graphs]
    n_total = int(num_stars.sum())

    pos = torch.tensor(node_feats['pos'][:n_total], dtype=torch.float32)
    vel = torch.tensor(node_feats['vel'][:n_total], dtype=torch.float32)
    theta = _broadcast_graph_features(graph_feats, labels, num_stars)
    cond = (_broadcast_graph_features(graph_feats, cond_labels, num_stars)
            if cond_labels else None)

    return pos, vel, theta, cond, num_stars


def _split_by_galaxy(num_stars, train_frac, seed):
    """Split stars into train/val so no galaxy spans both sets.

    Galaxies (not stars) are permuted and partitioned; the split is then
    expanded to a per-star boolean mask via each star's galaxy id.
    """
    rng = np.random.default_rng(seed)
    num_galaxies = len(num_stars)
    perm = rng.permutation(num_galaxies)
    n_train = int(train_frac * num_galaxies)

    is_train_galaxy = np.zeros(num_galaxies, dtype=bool)
    is_train_galaxy[perm[:n_train]] = True

    galaxy_id = np.repeat(np.arange(num_galaxies), num_stars)
    train_mask = is_train_galaxy[galaxy_id]
    return train_mask, ~train_mask


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _compute_norm(pos, vel, theta, cond, pre_transform_kwargs,
                  norm_max_stars=None, seed=0):
    """Compute per-field normalization stats from training stars.

    Delegates to `transforms.compute_field_norm`, which streams the stars
    through the observation pipeline (projection + uncertainty) and measures
    per-component location/scale for pos, vel, vel_error, theta, and cond.
    The NLE model consumes this dict and applies the normalization itself.
    """
    return compute_field_norm(
        StarBatch(pos=pos, vel=vel, theta=theta, cond=cond),
        max_stars=norm_max_stars, seed=seed, **pre_transform_kwargs)


# ---------------------------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------------------------

def _mask(t, m):
    """Row-select a tensor by a boolean mask, passing None through."""
    return None if t is None else t[m]


def prepare_dataloaders(
    node_feats, graph_feats, labels, cond_labels=None, train_frac=0.8,
    train_batch_size=1024, eval_batch_size=1024, num_workers=1,
    norm_dict=None, seed=0, pre_transform_kwargs=None,
    norm_max_stars=None
):
    """Prepare per-star train/val dataloaders from Cartesian phase space.

    Stars are split by galaxy (`_split_by_galaxy`) so train and val never
    share a galaxy. Loaders yield fully raw physical StarBatches (raw 3-D
    pos/vel, physical theta/cond, no vel_error yet); the NLE model applies
    the observation pipeline (projection + uncertainty) and all
    normalization. `cond_labels` are the measured conditioning variables
    (e.g. the half-light radius), kept separate from the inferred `labels`.
    `pre_transform_kwargs` are the same kwargs passed to
    `transforms.build_transformation`; they're only needed when `norm_dict`
    isn't supplied, since the field stats are measured by streaming the
    training stars through that pipeline.
    """
    pre_transform_kwargs = pre_transform_kwargs or {}

    pos, vel, theta, cond, num_stars = _build_star_tensors(
        node_feats, graph_feats, labels, cond_labels)
    tr, va = _split_by_galaxy(num_stars, train_frac, seed)

    if norm_dict is None:
        print('Computing norm_dict from training stars...')
        norm_dict = _compute_norm(
            pos[tr], vel[tr], theta[tr], _mask(cond, tr),
            pre_transform_kwargs, norm_max_stars=norm_max_stars, seed=seed)

    train_loader = DataLoader(
        StarDataset(pos[tr], vel[tr], theta[tr], _mask(cond, tr)),
        batch_size=train_batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=collate_star_batch, pin_memory=False)
    val_loader = DataLoader(
        StarDataset(pos[va], vel[va], theta[va], _mask(cond, va)),
        batch_size=eval_batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=collate_star_batch, pin_memory=False)

    return train_loader, val_loader, norm_dict


def prepare_test_dataloader(
    node_feats, graph_feats, labels, cond_labels=None, batch_size=1024,
    num_workers=1, norm_dict=None, max_graphs=None, pre_transform_kwargs=None
):
    """Prepare a per-star test dataloader from Cartesian phase space."""
    pre_transform_kwargs = pre_transform_kwargs or {}

    pos, vel, theta, cond, _ = _build_star_tensors(
        node_feats, graph_feats, labels, cond_labels, max_graphs=max_graphs)

    if norm_dict is None:
        norm_dict = _compute_norm(pos, vel, theta, cond, pre_transform_kwargs)

    loader = DataLoader(
        StarDataset(pos, vel, theta, cond),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=collate_star_batch, pin_memory=False)

    return loader, norm_dict


# ---------------------------------------------------------------------------
# Device-resident data path (tara, single GPU with unified memory)
# ---------------------------------------------------------------------------
#
# Same statistics as the DataLoader path above -- per-galaxy train/val split,
# one random permutation of the training stars per epoch, consecutive
# batches, theta/cond broadcast per star -- but the broadcast happens by
# indexing a per-galaxy table through a galaxy id, the tensors live on the
# training device, and a batch is one gather instead of `batch_size` Python
# calls. Measured 2026-10-07: 0.4 ms vs 17 ms per 4096-star batch.


def read_datasets_f32(root, name, num_datasets, labels, cond_labels=None,
                      init=0, ext='.h5'):
    """Shard-wise float32 reader for the device path.

    Reads each shard once, casts to float32 immediately (the HDF5 files are
    float64) and writes into preallocated tensors, so the peak memory is the
    final footprint rather than two float64 copies of everything. Returns
    per-star `pos`/`vel` and per-galaxy `theta`/`cond`/`num_stars`.
    """
    if ext[0] != '.':
        ext = '.' + ext
    paths = []
    for i in range(init, init + num_datasets):
        path = os.path.join(root, name, f'data.{i}{ext}')
        if os.path.exists(path):
            paths.append(path)
        else:
            print(f'Warning: {path} does not exist. Skipping...')
    if not paths:
        raise ValueError(f'No valid datasets found in {root}/{name}.')

    # pass 1: sizes only
    n_stars, n_gal = [], []
    for path in paths:
        with h5py.File(path, 'r') as f:
            ns = f['num_stars'][:]
            n_stars.append(int(ns.sum()))
            n_gal.append(len(ns))
    S, G = sum(n_stars), sum(n_gal)
    print(f'[Data] {len(paths)} shards, {G:,} galaxies, {S:,} stars '
          f'(float32 pos+vel {S * 24 / 1e9:.1f} GB)')

    pos = torch.empty((S, 3), dtype=torch.float32)
    vel = torch.empty((S, 3), dtype=torch.float32)
    theta = torch.empty((G, len(labels)), dtype=torch.float32)
    cond = (torch.empty((G, len(cond_labels)), dtype=torch.float32)
            if cond_labels else None)
    num_stars = np.empty(G, dtype=np.int64)

    s = g = 0
    for path, n, ng in zip(tqdm(paths), n_stars, n_gal):
        with h5py.File(path, 'r') as f:
            # Reason: the original path takes the first num_stars.sum() rows
            # of pos/vel; identical here, shard by shard.
            pos[s:s + n] = torch.from_numpy(f['pos'][:n].astype(np.float32))
            vel[s:s + n] = torch.from_numpy(f['vel'][:n].astype(np.float32))
            theta[g:g + ng] = torch.from_numpy(np.column_stack(
                [f[k][:] for k in labels]).astype(np.float32))
            if cond is not None:
                cond[g:g + ng] = torch.from_numpy(np.column_stack(
                    [f[k][:] for k in cond_labels]).astype(np.float32))
            num_stars[g:g + ng] = f['num_stars'][:]
        s += n
        g += ng
    return dict(pos=pos, vel=vel, theta=theta, cond=cond, num_stars=num_stars)


class DeviceStarBatches:
    """Iterable of raw StarBatches gathered from device-resident tensors.

    One `torch.randperm` of `star_idx` per epoch when `shuffle`, consecutive
    slices of `batch_size`, theta/cond looked up through the galaxy id. The
    model applies the observation pipeline and normalization as before.
    """

    def __init__(self, pos, vel, theta_gal, cond_gal, gid, star_idx,
                 batch_size, shuffle):
        self.pos, self.vel = pos, vel
        self.theta_gal, self.cond_gal, self.gid = theta_gal, cond_gal, gid
        self.star_idx = star_idx
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return (len(self.star_idx) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        idx = self.star_idx
        if self.shuffle:
            idx = idx[torch.randperm(len(idx), device=idx.device)]
        for a in range(0, len(idx), self.batch_size):
            j = idx[a:a + self.batch_size]
            g = self.gid[j].long()
            yield StarBatch(
                pos=self.pos[j], vel=self.vel[j], theta=self.theta_gal[g],
                cond=None if self.cond_gal is None else self.cond_gal[g])


class FixedStarBatches:
    """Iterable over a validation set already passed through the pipeline.

    Yields consecutive slices flagged `transformed=True`, so the model only
    normalizes them: the validation loss is then a deterministic function
    of the weights, instead of changing with each evaluation's random
    projection and noise draw.
    """

    def __init__(self, batch: StarBatch, batch_size):
        self.batch = batch
        self.batch_size = batch_size

    def __len__(self):
        n = self.batch.pos.shape[0]
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        b = self.batch
        for a in range(0, b.pos.shape[0], self.batch_size):
            sl = slice(a, a + self.batch_size)
            yield StarBatch(
                pos=b.pos[sl], vel=b.vel[sl],
                vel_error=None if b.vel_error is None else b.vel_error[sl],
                theta=b.theta[sl],
                cond=None if b.cond is None else b.cond[sl],
                transformed=True)


def _apply_pipeline_fixed(pipeline, pos, vel, theta, cond, seed,
                          chunk=1_048_576):
    """Run the observation pipeline once over a star set with a fixed seed."""
    parts = []
    with torch.random.fork_rng(devices=[pos.device] if pos.is_cuda else []):
        torch.manual_seed(seed)
        for a in range(0, pos.shape[0], chunk):
            sl = slice(a, a + chunk)
            parts.append(pipeline(StarBatch(pos=pos[sl], vel=vel[sl])))
    out = StarBatch(
        pos=torch.cat([p.pos for p in parts]),
        vel=torch.cat([p.vel for p in parts]),
        vel_error=(None if parts[0].vel_error is None
                   else torch.cat([p.vel_error for p in parts])),
        theta=theta, cond=cond, transformed=True)
    return out


def prepare_device_dataloaders(
    data, train_frac=0.8, train_batch_size=1024, eval_batch_size=1024,
    norm_dict=None, seed=0, pre_transform_kwargs=None, max_val_galaxies=None,
    deterministic_val=True, device='cuda', norm_chunk=65536,
    norm_max_stars=None,
):
    """Device-path counterpart of `prepare_dataloaders`.

    `data` is the dict from `read_datasets_f32`. The galaxy-level split is
    `_split_by_galaxy` with the same seed, so the same galaxies land in the
    same set as on the DataLoader path. The validation galaxies can be
    capped (`max_val_galaxies`, a random subset since the split is a
    permutation) and the validation stars pre-transformed once with a fixed
    seed (`deterministic_val`). The norm_dict, when not supplied, is
    measured on the training stars exactly as before, but under a fixed
    seed so that two runs on the same data share it bit for bit.
    """
    pre_transform_kwargs = pre_transform_kwargs or {}
    num_stars = data['num_stars']
    G = len(num_stars)
    ns_t = torch.as_tensor(num_stars)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(G)
    n_train = int(train_frac * G)
    train_gal = perm[:n_train]
    val_gal = perm[n_train:]
    if max_val_galaxies is not None:
        val_gal = val_gal[:max_val_galaxies]

    # per-star galaxy id and per-galaxy star offsets
    gid = torch.repeat_interleave(torch.arange(G), ns_t)
    offsets = torch.zeros(G + 1, dtype=torch.int64)
    offsets[1:] = torch.cumsum(ns_t, 0)

    def stars_of(gals):
        gals = torch.as_tensor(np.sort(gals))
        starts, counts = offsets[gals], ns_t[gals]
        first = torch.repeat_interleave(torch.cumsum(counts, 0) - counts,
                                        counts)
        within = torch.arange(int(counts.sum())) - first
        return torch.repeat_interleave(starts, counts) + within

    train_idx = stars_of(train_gal)
    val_idx = stars_of(val_gal)
    print(f'[Data] device path: {len(train_gal):,} train galaxies '
          f'({len(train_idx):,} stars), {len(val_gal):,} val galaxies '
          f'({len(val_idx):,} stars)')

    pos = data['pos'].to(device)
    vel = data['vel'].to(device)
    theta_gal = data['theta'].to(device)
    cond_gal = None if data['cond'] is None else data['cond'].to(device)
    gid = gid.to(torch.int32).to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)

    if norm_dict is None:
        print('Computing norm_dict from training stars (fixed seed)...')
        # theta/cond are passed per training galaxy: their min/max (all the
        # norm uses) equal those of the per-star broadcast, at 1/100 the
        # memory.
        tg = torch.as_tensor(np.sort(train_gal), device=device)
        rng_devices = [device] if 'cuda' in str(device) else []
        with torch.random.fork_rng(devices=rng_devices):
            torch.manual_seed(seed)
            norm_dict = compute_field_norm(
                StarBatch(pos=pos[train_idx], vel=vel[train_idx],
                          theta=theta_gal[tg],
                          cond=None if cond_gal is None else cond_gal[tg]),
                batch_size=norm_chunk, max_stars=norm_max_stars,
                seed=seed, **pre_transform_kwargs)
        norm_dict = {k: [float(x) for x in v] for k, v in norm_dict.items()}
        if 'cuda' in str(device):
            torch.cuda.empty_cache()

    train_loader = DeviceStarBatches(
        pos, vel, theta_gal, cond_gal, gid, train_idx, train_batch_size,
        shuffle=True)

    gv = gid[val_idx].long()
    if deterministic_val:
        pipeline = build_transformation(**pre_transform_kwargs)
        val_batch = _apply_pipeline_fixed(
            pipeline, pos[val_idx], vel[val_idx], theta_gal[gv],
            None if cond_gal is None else cond_gal[gv], seed=seed + 1)
        val_loader = FixedStarBatches(val_batch, eval_batch_size)
    else:
        val_loader = DeviceStarBatches(
            pos, vel, theta_gal, cond_gal, gid, val_idx, eval_batch_size,
            shuffle=False)

    return train_loader, val_loader, norm_dict
