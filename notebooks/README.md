# Tutorials

Hand-maintained notebooks. Edit them in Jupyter directly.

| notebook | question |
|---|---|
| `00_nle_frontend.ipynb` | How do I train a model and sample a posterior from it? |
| `01_nle_internals.ipynb` | What do those four API calls actually do? |

Start with `00`. Read `01` once when you want to change something the API
does not expose; its last cell maps each step back onto the `api` call that
replaces it.

Run from this directory, with the `torch` venv.

`example_config.py` is the documented config: every field carries a comment.
`CONFIG.md` has the same notes grouped into tables. Copy `example_config.py`
and strip the comments when you make a config of your own.

The batch twins live in `scripts/`: `train_nle.py` for training (same config
file, same run directory) and `diagnostics/` for the per-test scripts.
