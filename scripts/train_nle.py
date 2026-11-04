"""Training script for Neural Likelihood Estimation (NLE)."""

import os
import sys
from pathlib import Path

os.environ.setdefault('WANDB_DATA_DIR',
                      os.path.expanduser('~/scratch/wandb_data'))

# Reason: this script lives in scripts/, so the repo root -- the parent
# of the `nle_dsph` package -- has to be importable before the package
# imports below. Drop this once the project is pip-installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import wandb
import ml_collections
import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import summarize
import torch
from absl import flags
from ml_collections import config_flags

from nle_dsph import training
from nle_dsph.training import (
    create_model, prepare_data, save_config_snapshot,
)
from nle_dsph.transforms import build_transformation


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
    # Reason: the flow is initialized here, so the seed has to be set
    # before it, not merely before `fit`. Seeding only before training
    # left initialization drawing from torch's entropy-seeded default
    # generator, so two runs of one config produced different models.
    pl.seed_everything(config.seed_training, workers=True)
    model = create_model(config, pre_transforms, norm_dict)
    summarize(model, max_depth=3)
    training.report_param_counts(model)

    wandb_logger.watch(model, log="all", log_freq=1000, log_graph=False)

    callbacks = training.create_base_callbacks(config)
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
