"""Shared helpers for the NLE training script (setup, callbacks, fit loop).

Adapted from the jgnn package to keep the wandb/checkpoint/trainer
boilerplate identical to train_npe.
"""

import json
import os
import shutil
import socket
from pathlib import Path
from typing import Optional

import torch
import ml_collections
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)

from nle_dsph import datasets
from nle_dsph.nle import NLE


def _wandb_server_reachable(timeout: float = 3.0) -> bool:
    """Check whether the WandB API is reachable, with a short timeout.

    Lets us pick 'online' vs 'offline' up front, so training doesn't stall
    behind wandb's own connection retries on machines without internet
    (e.g. HPC compute nodes).
    """
    try:
        socket.create_connection(
            ("api.wandb.ai", 443), timeout=timeout).close()
        return True
    except OSError:
        return False


def create_wandb_logger(config: ml_collections.ConfigDict, tag: str):
    """Create a WandbLogger for the run, tagged with `tag`.

    Returns the logger and the project directory (where checkpoints and the
    config snapshot are saved). If `config.wandb_mode` is unset, the mode is
    'online' when the WandB API is reachable, else 'offline'.
    """
    workdir = Path(config.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    if config.get('debug', False):
        wandb_mode = 'disabled'
    else:
        wandb_mode = config.get('wandb_mode', None)
        if wandb_mode is None:
            wandb_mode = 'online' if _wandb_server_reachable() else 'offline'
    print(f"[WandB] Mode: {wandb_mode}")

    tags = set(config.get('tags', [])) | {tag}
    logger = WandbLogger(
        project=config.get("wandb_project", "nle_dsph"),
        name=config.get("name"),
        entity=config.get("entity", None),
        id=config.get("id") or None,
        save_dir=str(workdir),
        log_model="all",
        config=config.to_dict(),
        mode=wandb_mode,
        resume="allow",
        tags=list(tags),
    )
    project_dir = workdir / logger.experiment.project / logger.experiment.id
    project_dir.mkdir(parents=True, exist_ok=True)

    print(logger, project_dir)
    return logger, project_dir


def get_checkpoint_path(
    config: ml_collections.ConfigDict, project_dir: Optional[Path] = None,
) -> Optional[str]:
    """Resolve the checkpoint path to resume from, if any."""
    if not config.get('checkpoint'):
        return None

    ckpt = config.checkpoint
    if os.path.isabs(ckpt):
        return ckpt

    if project_dir is None:
        raise ValueError(
            "If `config.checkpoint` is a relative path, `project_dir` must be"
            " provided to resolve the full path.")
    return str(project_dir / 'checkpoints' / ckpt)


def create_base_callbacks(config: ml_collections.ConfigDict) -> list:
    """Standard early-stopping / checkpointing / LR-monitor callbacks."""
    return [
        EarlyStopping(
            monitor='val/loss',
            mode='min',
            patience=config.patience,
            verbose=True,
            check_on_train_epoch_end=False,
        ),
        ModelCheckpoint(
            filename="epoch={epoch}-step={step}-loss={val/loss:.4f}",
            monitor='val/loss',
            mode='min',
            save_top_k=config.get('save_top_k', 3),
            save_weights_only=False,
            auto_insert_metric_name=False,
        ),
        ModelCheckpoint(
            filename="last",
            save_weights_only=False,
            save_last=True,
            auto_insert_metric_name=False,
            enable_version_counter=False,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]


def report_param_counts(model) -> None:
    """Print total, trainable, and frozen parameter counts for a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total parameters: {total:,}")
    print(f"[Model] Trainable parameters: {trainable:,}")
    print(f"[Model] Frozen parameters: {total - trainable:,}")


def build_trainer(
    config: ml_collections.ConfigDict,
    project_dir: Path,
    callbacks: list,
    wandb_logger: WandbLogger,
    **trainer_kwargs,
) -> pl.Trainer:
    """Construct the PyTorch Lightning Trainer."""
    print(f"[Trainer] Max epochs: {config.num_epochs}, "
          f"Max steps: {config.num_steps}")
    print(f"[Trainer] Accelerator: {config.accelerator}")

    # Numerical precision of the training step. '32-true' (default) is
    # plain fp32. '16-mixed' runs matmuls in fp16 under autocast with a
    # GradScaler; weights, optimizer state and the validation step stay
    # fp32 (see NLE.validation_step). Measured on tara 2026-10-07: 1.35x at
    # batch 16384, 1.6x at 32768, with a per-star log p error of 0.002
    # nats (median) against fp32. 'bf16-mixed' is deliberately not
    # offered: its 8-bit mantissa gives 0.03-0.5 nats per star on narrow
    # velocity peaks (notes/nle_production_design.md, section 8).
    precision = config.get('precision', '32-true')
    if precision not in ('32-true', '16-mixed'):
        raise ValueError(
            f"config.precision={precision!r}; use '32-true' or '16-mixed'")
    print(f"[Trainer] Precision: {precision}")

    # TF32 matmuls ('high') bring no speed-up here (the step is launch-bound)
    # and lose knot precision, so default to full fp32 matmuls.
    torch.set_float32_matmul_precision(
        config.get('matmul_precision', 'highest'))

    return pl.Trainer(
        default_root_dir=str(project_dir),
        max_epochs=config.num_epochs,
        max_steps=config.num_steps,
        accelerator=config.accelerator,
        precision=precision,
        callbacks=callbacks,
        logger=wandb_logger,
        enable_progress_bar=config.get("enable_progress_bar", True),
        gradient_clip_val=config.get('gradient_clip_val', None),
        **trainer_kwargs,
    )


def fit(
    trainer: pl.Trainer,
    model,
    train_loader,
    val_loader,
    checkpoint_path: Optional[str],
    reset_optimizer: bool,
) -> None:
    """Run `trainer.fit`, handling checkpoint resume and weights-only reset."""
    if checkpoint_path and reset_optimizer:
        print("[Training] Loading model weights with fresh optimizer state")
        checkpoint = torch.load(
            checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        trainer.fit(model, train_loader, val_loader)
    elif checkpoint_path:
        print("[Training] Resuming training from full checkpoint")
        trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint_path)
    else:
        print("[Training] Starting fresh training")
        trainer.fit(model, train_loader, val_loader)

    print("[Training] Training complete!")


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
