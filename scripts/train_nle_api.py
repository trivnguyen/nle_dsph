"""Training entry point that goes through `nle_dsph.api.train`.

The same job as `train_nle.py`, with the procedure living in the
package instead of in this file. Everything here is argument handling;
`api.train` owns the run.

    python scripts/train_nle_api.py --config configs/nle_8p_v4.py
    python scripts/train_nle_api.py --config configs/nle_8p_v4.py \
        --config.num_steps=50000

Both scripts read the same configs, call the same helpers in
`nle_dsph.training`, and write the same run directory, so a model
trained by either is loaded the same way.

WHAT STILL DIFFERS FROM train_nle.py
------------------------------------
Resume (`config.checkpoint`, `config.reset_optimizer`) now behaves the
same in both. The remainder, all of it in `train_nle.py` and not here:

  * `summarize(model, max_depth=3)` and `training.report_param_counts`,
    which print the layer table and the parameter counts.
  * `wandb_logger.watch(model, log='all', log_freq=1000)`, which logs
    gradient and weight histograms. A real difference in what wandb
    records, not just in what is printed.
  * `torch.cuda.empty_cache()` after the data pass, which returns the
    normalization scratch on a shared CUDA box.
  * `wandb.finish()` at the end.

And in the other direction, two things `api.train` does that
`train_nle.py` does not:

  * it resolves the run directory from
    `trainer.checkpoint_callback.dirpath` rather than trusting
    `create_wandb_logger`'s `project_dir`. With `debug=True` the
    disabled-wandb stub reports project 'dummy' and the two disagree,
    which puts the config snapshot and the checkpoints in different
    directories.
  * it raises if training wrote no checkpoint at all, which happens
    when `num_steps` is shorter than one epoch and
    `val_check_interval` is unset: the checkpoint callbacks monitor
    `val/loss`, and without a validation pass there is nothing to
    monitor.

Those four-plus-two are the whole difference. Folding them into
`api.train` and reducing `train_nle.py` to this file would leave one
implementation and nothing left to drift; until then, prefer this
script for short runs and `train_nle.py` for long logged ones.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault('WANDB_DATA_DIR',
                      os.path.expanduser('~/scratch/wandb_data'))

# Reason: this script lives in scripts/, so the repo root -- the parent
# of the `nle_dsph` package -- has to be importable before the package
# imports below. Drop this once the project is pip-installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings  # noqa: E402
warnings.filterwarnings('ignore', category=UserWarning)

import ml_collections  # noqa: E402
from absl import flags  # noqa: E402
from ml_collections import config_flags  # noqa: E402

from nle_dsph import api  # noqa: E402


def main(config: ml_collections.ConfigDict, config_path: str = None) -> Path:
    """Train from `config` and report where the run landed.

    Args:
        config: The training config.
        config_path: Path to the config file, copied into the run
            directory beside the JSON snapshot.

    Returns:
        The run directory.
    """
    run_dir = api.train(config, config_path=config_path)
    print(f'[Done] Run directory: {run_dir}')
    return run_dir


if __name__ == '__main__':
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file(
        'config',
        None,
        'File path to the training hyperparameter configuration.',
        lock_config=True,
    )
    FLAGS(sys.argv)
    main(
        config=FLAGS.config,
        config_path=config_flags.get_config_filename(FLAGS['config']),
    )
