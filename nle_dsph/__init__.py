"""Per-star neural likelihood estimation for dwarf spheroidals.

Submodules are imported explicitly (`from nle_dsph.nle import NLE`)
rather than re-exported here, so that importing one does not drag in
torch and pytorch_lightning when only the numpy-level helpers in
`profiles` or `priors` are wanted.
"""
