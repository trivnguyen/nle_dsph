# Tutorials

Hand-maintained notebooks that walk through one test each, with the
derivation, the operations written out cell by cell, and the batch
results. They are shared material: edit them in Jupyter directly. The
batch versions of the same steps are `diagnostics/*.py`.

| notebook | question | batch script |
|---|---|---|
| `03_error_convolution.ipynb` | Does a noise-trained NLE equal the noiseless NLE convolved with the error kernel? | `diagnostics/check_error_convolution.py` |
| `04_mock_sf.ipynb` | Does modelling the sky positions, p(x, y, v), change the posterior relative to p(v \| R)? | `diagnostics/mock_sf_compare.py`, `plot_mock_sf.py` |

Run from this directory with the `torch` venv and `~/my_modules` on
`PYTHONPATH` (the first cell sets both up).
