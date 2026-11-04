"""Training script for Neural Likelihood Estimation (NLE)."""

import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault('WANDB_DATA_DIR',
                      os.path.expanduser('~/scratch/wandb_data'))

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import wandb
import ml_collections
import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import summarize
import torch
from absl import flags
from ml_collections import config_flags

import datasets
import training
from nle import NLE
from transforms import build_transformation


def save_config_snapshot(
    config: ml_collections.ConfigDict, snapshot_dir: Path,
    config_path: str = None,
) -> None:
    """Write a JSON (and .py) config snapshot next to the checkpoints.

    Lets downstream consumers reconstruct the model / pre_transforms from
    the checkpoint directory alone, without wandb.
    """
    snapshot_path = snapshot_dir / 'config_snapshot.json'
    with open(snapshot_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2, default=str)
    print(f"[Setup] Wrote config snapshot -> {snapshot_path}")

    if config_path and os.path.exists(config_path):
        shutil.copy2(config_path, snapshot_dir / 'config_snapshot.py')


def prepare_data(config: ml_collections.ConfigDict, norm_dict=None):
    """Load data and build per-star train/val dataloaders.

    Returns (train_loader, val_loader, norm_dict). When `norm_dict` is given
    (e.g. reused from a resumed checkpoint) it is passed through unchanged
    instead of recomputing field stats from this call's training data.

    With `config.data_on_device` the whole dataset is loaded shard-wise in
    float32, moved to the accelerator and batched by indexing
    (`datasets.prepare_device_dataloaders`); otherwise the original
    DataLoader path is used.
    """
    if config.get('data_on_device', False):
        data = datasets.read_datasets_f32(
            config.data_root, config.data_name, config.num_datasets,
            labels=config.labels, cond_labels=config.get('cond_labels', None),
            init=config.get('init', 0))
        # Lightning's accelerator name is not a torch device name:
        # 'gpu' means CUDA here, 'mps' is Apple-silicon unified memory.
        device = {'gpu': 'cuda', 'mps': 'mps'}.get(
            config.accelerator, 'cpu')
        return datasets.prepare_device_dataloaders(
            data,
            train_frac=config.train_frac,
            train_batch_size=config.train_batch_size,
            eval_batch_size=config.eval_batch_size,
            norm_dict=norm_dict,
            seed=config.seed_data,
            pre_transform_kwargs=config.pre_transforms.to_dict(),
            max_val_galaxies=config.get('max_val_galaxies', None),
            norm_max_stars=config.get('norm_max_stars', None),
            deterministic_val=config.get('deterministic_val', True),
            device=device,
        )

    node_feats, graph_feats = datasets.read_datasets(
        config.data_root,
        config.data_name,
        config.num_datasets,
        init=config.get('init', 0),
        is_directory=True,
        concat=True,
    )

    train_loader, val_loader, norm_dict = datasets.prepare_dataloaders(
        node_feats,
        graph_feats,
        config.labels,
        cond_labels=config.get('cond_labels', None),
        train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size,
        train_frac=config.train_frac,
        num_workers=config.num_workers,
        seed=config.seed_data,
        norm_dict=norm_dict,
        pre_transform_kwargs=config.pre_transforms.to_dict(),
        norm_max_stars=config.get('norm_max_stars', None),
    )

    return train_loader, val_loader, norm_dict


def create_model(
    config: ml_collections.ConfigDict, pre_transforms, norm_dict,
) -> NLE:
    """Create the NLE model from the config."""
    print("[Model] Creating NLE model...")
    context_embedding = config.model.get('context_embedding', None)
    if context_embedding is not None:
        context_embedding = context_embedding.to_dict()

    return NLE(
        norm_dict=norm_dict,
        condition_on_position=config.model.get('condition_on_position', False),
        condition_on_radius=config.model.get('condition_on_radius', False),
        # Reason: a model trained without measurement noise has no sigma
        # to condition on.
        condition_on_error=config.pre_transforms.get(
            'apply_uncertainty', True),
        flows_args=config.model.flows.to_dict(),
        context_embedding_args=context_embedding,
        optimizer_args=config.optimizer.to_dict(),
        scheduler_args=config.scheduler.to_dict(),
        pre_transforms=pre_transforms,
    )


def create_callbacks(config: ml_collections.ConfigDict) -> list:
    """Create the training callbacks."""
    return training.create_base_callbacks(config)


def main(config: ml_collections.ConfigDict, config_path: str = None):
    """Train the NLE model with wandb logging."""
    resume_training = bool(config.get('checkpoint'))
    wandb_logger, project_dir = training.create_wandb_logger(config, tag='nle')

    print(f"[Setup] Resume training: {resume_training}")
    print(f"[Setup] Project directory: {project_dir}")

    print(f"[Setup] Saving config snapshot to: {project_dir}")
    save_config_snapshot(config, project_dir, config_path)

    checkpoint_path = None
    norm_dict = None
    if resume_training:
        checkpoint_path = training.get_checkpoint_path(config, project_dir)
        print(f"[Checkpoint] Resuming from: {checkpoint_path}")
        print("[Checkpoint] Reset optimizer: "
              f"{config.get('reset_optimizer', False)}")

        # Reuse the norm_dict from the resumed checkpoint's hyper_parameters.
        resume_checkpoint = torch.load(
            checkpoint_path, map_location='cpu', weights_only=False)
        norm_dict = resume_checkpoint['hyper_parameters']['norm_dict']
        print("[Checkpoint] Reusing norm_dict from resumed checkpoint")

    print("[Data] Loading datasets...")
    train_loader, val_loader, norm_dict = prepare_data(
        config, norm_dict=norm_dict)
    print(f"[Data] Train batches: {len(train_loader)}, "
          f"Val batches: {len(val_loader)}")
    if torch.cuda.is_available():
        # Return the data-preparation scratch (norm statistics, index
        # temporaries) to the system so concurrent runs can use it.
        torch.cuda.empty_cache()

    print("[Transforms] Building pre-transforms...")
    pre_transforms = build_transformation(**config.pre_transforms.to_dict())

    print("[Model] Creating NLE model...")
    model = create_model(config, pre_transforms, norm_dict)
    summarize(model, max_depth=3)
    training.report_param_counts(model)

    wandb_logger.watch(model, log="all", log_freq=1000, log_graph=False)

    callbacks = create_callbacks(config)
    print(f"[Callbacks] Created {len(callbacks)} callbacks")

    trainer_kwargs = dict(num_sanity_val_steps=0)
    vci = config.get('val_check_interval', None)
    if vci is not None:
        # Lightning requires the interval to fit inside one epoch.
        vci = min(int(vci), len(train_loader))
        trainer_kwargs['val_check_interval'] = vci
        print(f"[Trainer] Validation every {vci} steps")
    trainer = training.build_trainer(
        config, project_dir, callbacks, wandb_logger, **trainer_kwargs)

    pl.seed_everything(config.seed_training, workers=True)
    print(f"[Seed] Training seed set to: {config.seed_training}")

    training.fit(
        trainer, model, train_loader, val_loader,
        checkpoint_path=checkpoint_path,
        reset_optimizer=config.get('reset_optimizer', False),
    )

    wandb.finish()
    print("[WandB] Finished")


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file(
        "config",
        None,
        "File path to the training hyperparameter configuration.",
        lock_config=True,
    )
    FLAGS(sys.argv)
    main(
        config=FLAGS.config,
        config_path=config_flags.get_config_filename(FLAGS['config']),
    )
