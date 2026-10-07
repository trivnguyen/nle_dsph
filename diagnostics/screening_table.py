"""Tabulate the screening arms: best fixed-val NLL, parameters, time.

    python diagnostics/screening_table.py
Reads the run dirs named in the local log manifests of
~/scratch/local_logs/nle/*screen_<arm>_* and prints one row per arm:
arm, wandb run id, parameter count, best val/loss (nats per star, the
deterministic 50k-galaxy validation set), the step of that best, the
val/loss at 30k steps for an equal-budget comparison, and wall time.
"""

import glob
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LOGS = Path(os.path.expanduser('~/scratch/local_logs/nle'))
MODELS = Path(os.path.expanduser(
    '~/scratch/projects/sbi_dsph/trained_models/nle/nle_8p_ZhaoPlumCOM_v4'))
ARMS = {
    'A': 'control 4x[128,128] 8 bins ctx64',
    'B': '5x[256,256] 16 bins ctx128',
    'C': '8x[256,256] 16 bins ctx128',
    'D': 'B with silu',
    'E': 'B without context MLP',
}


def run_of(arm):
    """Latest local log dir for the arm and the wandb run id it used."""
    # Reason: the detached launcher leaves a second, empty dir behind.
    dirs = sorted((d for d in glob.glob(str(LOGS / f'*screen_{arm}_*'))
                   if os.path.exists(f'{d}/stdout.log')),
                  key=os.path.getmtime)
    if not dirs:
        return None, None
    d = dirs[-1]
    text = open(f'{d}/stdout.log').read()
    m = re.search(r'nle_8p_ZhaoPlumCOM_v4/([a-z0-9]{8})', text)
    return d, (m.group(1) if m else None)


def main():
    rows = []
    for arm, desc in ARMS.items():
        d, rid = run_of(arm)
        if rid is None:
            rows.append((arm, desc, '-', '-', '-', '-', 'not run'))
            continue
        cks = glob.glob(str(MODELS / rid / 'checkpoints' / '*loss=*.ckpt'))
        losses = {int(re.search(r'step=(\d+)', c).group(1)):
                  float(re.search(r'loss=(-?\d+\.\d+)', c).group(1))
                  for c in cks}
        if not losses:
            rows.append((arm, desc, rid, '-', '-', '-', 'no ckpt'))
            continue
        best_step = min(losses, key=losses.get)
        ck = torch.load(cks[0], map_location='cpu', weights_only=False)
        npar = sum(v.numel() for v in ck['state_dict'].values())
        man = open(f'{d}/manifest.txt').read()
        t = re.findall(r'\[(\S+ \S+)\] =+ run (start|end)', man)
        wall = '-'
        if len(t) == 2:
            f = '%Y-%m-%d %H:%M:%S'
            wall = str(datetime.strptime(t[1][0], f)
                       - datetime.strptime(t[0][0], f))
        rows.append((arm, desc, rid, f'{npar:,}', f'{losses[best_step]:.4f}',
                     best_step, wall))
    hdr = ('arm', 'architecture', 'run', 'params', 'best val/loss',
           'at step', 'wall')
    w = [max(len(str(r[i])) for r in rows + [hdr]) for i in range(len(hdr))]
    for r in [hdr] + rows:
        print('  '.join(str(x).ljust(w[i]) for i, x in enumerate(r)))


if __name__ == '__main__':
    main()
