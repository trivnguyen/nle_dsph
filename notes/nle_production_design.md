# Production per-star NLE on 8p_ZhaoPlumCOM_v4 — design (2026-10-07, tara)

Status: design only; nothing has been trained. Draft configs:
`configs/nle_8p_v4_prod.py` (production) and `configs/nle_8p_v4_screen.py`
(screening arms A–E). Code changes required before the production run are in
section 4; the current code runs the configs unchanged but ~10x slower.

## 0. Starting point

The model under test is the T2 run `nle/nle_8p_ZhaoPlumCOM_v4/37vbi58p`
(Trillium, 2026-10-06): 5 shards (500k galaxies, 49M stars), NSF 4 x [128,128],
8 bins, context MLP [128,128] -> 64, batch 4096, 100k steps (0.41B star-draws,
8.7 epochs), AdamW 1e-3, cosine to 1e-6. Best val loss -2.5629 at step 77k;
val still improving ~0.004/epoch at the end while the LR annealed. On Draco it
reaches the Jeans width (0.058 vs 0.059 dex) and recovers the 12 mocks with
RMS z 1.08, mean z +0.5 (a 1.8 sigma hint of upward bias), and it pushes
dm_alpha to the upper prior edge on real Draco and on CuspIso.

Why a production run: a per-star likelihood error of e nats that is coherent
across the stars of one galaxy becomes 500e nats at the galaxy level for a
510-star catalogue. Errors invisible in the per-star NLL (0.01 nats) are
therefore large in the posterior. More distinct theta (5.5M vs 0.5M), more
capacity and more optimisation are the levers; multiple seeds measure what is
left.

## 1. Measurements that fix the design (all on tara, 2026-10-07)

Hardware: one NVIDIA GB10, 121 GB unified memory, 20 cores, 3.3 TB free NVMe,
no scheduler. Launcher: `slurm/train_nle.sh` (detached, manifest, log dir
`~/scratch/local_logs/nle/`). wandb online works (today's smoke tests synced).

Dataset on tara: `~/scratch/projects/sbi_dsph/datasets/8p_ZhaoPlumCOM_v4`,
57 shards, 5,480,787 galaxies, 540,046,491 stars (60–142 per galaxy, median 98).
float32 footprint: pos+vel 13.0 GB; theta broadcast per star 15.1 GB (avoid:
store per galaxy, gather by a 2.2 GB int32 galaxy id); float64 h5 read of
pos+vel 25.9 GB (convert shard by shard, never concatenate in float64).

Velocity dynamic range: vel_scale in norm_dict is 203 km/s (midrange/half-range
over all stars after noise) while the per-galaxy sigma_los percentiles are
[1, 16, 50, 84, 99]% = [0.02, 0.27, 2.5, 12.6, 27.3] km/s. A Draco-like galaxy
is 0.045 in normalised units. The question was whether 8-bin splines resolve
this. Answer below: they do.

Calibration of 37vbi58p on held-out shard 30 (40k galaxies, 3.9M stars, fixed
projection, per-galaxy errors cycled through 0.3 / 1 / 3 km/s). PIT = Phi(z)
of the flow's base coordinate at the true theta; the Gaussian baseline uses
the galaxy's TRUE intrinsic sigma convolved with the error (a floor no Gaussian
LOSVD model could beat):

| sigma_true (km/s) | err | NLL model | NLL Gauss(true sigma) | PIT mean | PIT std | frac abs(z)>3 |
|---|---|---|---|---|---|---|
| [0, 0.3)  | 0.3 | -4.993 | -4.994 | 0.478 | 0.290 | 0.0025 |
| [0.3, 1)  | 0.3 | -4.363 | -4.315 | 0.486 | 0.292 | 0.0027 |
| [1, 3)    | 0.3 | -3.411 | -3.311 | 0.495 | 0.289 | 0.0026 |
| [3, 10)   | 0.3 | -2.285 | -2.183 | 0.501 | 0.290 | 0.0026 |
| [10, 40)  | 0.3 | -1.170 | -1.077 | 0.502 | 0.291 | 0.0029 |
| [1, 3)    | 1.0 | -3.221 | -3.172 | 0.498 | 0.290 | 0.0028 |
| [3, 10)   | 1.0 | -2.269 | -2.174 | 0.502 | 0.290 | 0.0029 |
| [3, 10)   | 3.0 | -2.090 | -2.043 | 0.503 | 0.290 | 0.0026 |
| [10, 40)  | 3.0 | -1.150 | -1.071 | 0.501 | 0.292 | 0.0032 |

Expected for a calibrated model: PIT mean 0.500, std 0.2887, tail 0.0027.
Overall NLL -2.687 vs -2.635 for the Gaussian floor. Reading: the per-star
density is calibrated in every regime, including sigma < 1 km/s, and beats the
true-sigma Gaussian by 0.05–0.10 nats wherever the intrinsic dispersion
exceeds the error (the non-Gaussian, radius-dependent LOSVD is being learned).
The one visible defect is PIT mean 0.478–0.486 at sigma < 1 km/s with 0.3 km/s
errors: a shift of ~0.03 sigma, irrelevant for any real catalogue (errors >= 1
km/s). Conclusion: no theta-dependent velocity rescaling is needed; the
production run is a scale-up, not a re-design. Whatever the galaxy-level
posterior error is, it is below the resolution of a 4M-star per-star test —
which is exactly why the posterior-level checks in section 6 matter.

Throughput, on-device random-index batches, synthetic data, 60 timed steps:

| model | params | batch | ms/step | M stars/s | 1M steps |
|---|---|---|---|---|---|
| current 4x[128,128], 8 bins, ctx 64 | 137k | 4096  | 11.0 | 0.37 | 3.1 h |
| current | 137k | 8192  | 11.3 | 0.73 | 3.1 h |
| current | 137k | 16384 | 13.2 | 1.24 | 3.7 h |
| candidate 6x[256,256], 16 bins, ctx 128 | 767k | 8192  | 16.9 | 0.48 | 4.7 h |
| candidate | 767k | 16384 | 23.5 | 0.70 | 6.5 h |
| candidate | 767k | 32768 | 58.8 | 0.56 | 16 h |

The step is kernel-launch bound up to batch ~16k (zuko's autoregressive
transforms are many small kernels), so batch 16384 is nearly free and 32768
is where compute starts to bind. One epoch of 530M training stars at 0.7 M/s
is 13 min. Peak GPU memory for the largest case: 0.9 GB plus the dataset.

The current `StarDataset` + `DataLoader(num_workers=0)` path costs 17 ms per
4096-star batch on tara's CPU (4096 Python `__getitem__` calls + `torch.stack`)
against 0.4 ms for a whole-batch index gather; the T2 run's 58 ms/step on
Trillium is consistent with that path dominating. Replacing it (section 4)
is the single change that makes the budget below affordable.

## 2. Production recipe (`configs/nle_8p_v4_prod.py`)

| knob | T2 run | production | why |
|---|---|---|---|
| shards | 5 (0.5M gal) | 57 (5.48M gal, 540M stars) | 11x more distinct theta in the 8-d context; same cost per step; 15 GB resident |
| train / val | 0.95 / 0.05 | 0.98 / 0.02, val capped at 50k galaxies (4.9M stars) | val NLL standard error ~0.0007 nats, enough to resolve 0.003 differences |
| val cadence | per epoch (stochastic projection + noise) | every 10k steps on a fixed, pre-projected, pre-noised set | epoch-to-epoch jitter in T2 was 0.003–0.004, the size of the effects being selected on |
| batch | 4096 | 16384 | step time flat to 16k; 4x star-draws per step for free |
| steps | 100k (0.41B draws, 8.7 ep) | 500k (8.2B draws, ~15 ep) | T2 was still improving when the LR ran out; 20x the draws costs ~3 h |
| LR schedule | warmup 2k, cosine 100k, 1e-3 -> 1e-6 | warmup 5k, cosine 500k, 1e-3 -> 1e-6 | same peak LR (spline flows + clip 0.5 are stable there); longer warmup for the larger batch |
| optimizer | AdamW 1e-3, wd 1e-4, clip 0.5 | unchanged | no evidence against it |
| flow | NSF 4 x [128,128], 8 bins, tanh | NSF 5 x [256,256], 16 bins, tanh (= arm B) | with a 1-d target each transform is one conditioner MLP -> 3*bins-1 spline knots; width and bins add shape resolution, depth adds little |
| context MLP | [128,128] -> 64 | [256,256] -> 128 | the 10-d context (log R, 7 theta, log r_half, sigma_v) is where all the structure is |
| precision | fp32 | fp32 (`matmul_precision='highest'`) | run is launch-bound, TF32 buys nothing and costs knot precision |
| early stopping | patience 20 epochs | patience 20 val checks (200k steps) | effectively off; cosine runs keep improving slowly |
| checkpoints | top-3 + last | top-3 + last, plus the fixed-val PIT table at each top-k save | selection by a deterministic metric |
| seeds | 1 | 2 (A: 792139710, B: 46598) | the two posteriors' disagreement is the NLE's epistemic error bar |

Wall-clock: ~3.3 h per seed for arm B at 23.5 ms/step; ~1.8 h if arm A wins.
Two runs fit concurrently on the GB10 (launch-bound, 2 x 15 GB data + 2 x 1 GB
model), so both seeds can run overnight side by side.

Observation pipeline unchanged: random projection per batch, log-uniform
per-star sigma_v within a per-batch window drawn from low in (0.01, 0.1) and
width in (5, 30) km/s, i.e. sigma_v in 0.01–30 km/s, bracketing Walker 2023
(~1–3), Pace, DESI and the UFD catalogues. Prior box unchanged (= inference
box in `infer_nle.py`, with r_half from 10 pc to 3.2 kpc).

## 3. Screening before committing (optional, ~2.5 h total)

Five arms at 57 shards, batch 16384, 60k steps (1B draws, 2.4x T2's total),
seed 111, same fixed val set; `configs/nle_8p_v4_screen.py:<arm>`. Run two
at a time.

| arm | change vs B | what it tests |
|---|---|---|
| A | T2 architecture | is capacity the limit at all? if A is within 0.002 of B, ship A (half the time) |
| B | 5 x [256,256], 16 bins, ctx 128 | the default |
| C | 8 transforms | depth vs width for a 1-d target (expected: no gain) |
| D | SiLU conditioner | tanh saturation in the conditioner MLPs |
| E | no context MLP | whether the 64/128-d embedding helps or only adds a layer |

Decision rule: best fixed-val NLL; ties (< 0.002 nats) broken toward the
smaller model. Also record the per-regime PIT table of section 1 for each
arm; any arm with PIT std off by > 0.005 in a sigma >= 1 km/s bin is out
regardless of NLL. If screening is skipped, run B directly.

## 4. Code changes required (train_nle.py / datasets.py / training.py)

1. Shard-wise float32 loading. `read_datasets` concatenates float64 arrays
   across shards; for 57 shards that peaks near 55 GB before conversion.
   Read each shard, cast to float32, write into preallocated tensors.
2. Per-galaxy theta/cond with a per-star int32 galaxy id instead of the
   `np.repeat` broadcast (saves 15 GB; the gather is one indexing op).
3. On-device batch sampler replacing `StarDataset`/`DataLoader`: dataset
   tensors live on the GPU; each step draws `torch.randint` indices and builds
   the `StarBatch` by indexing (what the benchmark above measured). Lightning
   accepts any iterable with `__len__`; `transfer_batch_to_device` already
   exists. Expose via `config.data_on_device`.
4. Deterministic validation: apply `pre_transforms` once at setup with a fixed
   generator to the capped val split, store the result (with `vel_error`), and
   make `validation_step` skip the pipeline for pre-transformed batches.
   `config.max_val_galaxies`, `config.deterministic_val`.
5. `val_check_interval` passed to `pl.Trainer` (`config.val_check_interval`,
   steps); `torch.set_float32_matmul_precision(config.matmul_precision)`.
6. Per-regime PIT/NLL table (the section-1 diagnostic, ~10 s on GPU) logged to
   wandb at every validation, on the fixed val set binned by true sigma_los
   and sigma_v. This is the production acceptance metric, not the NLL alone.
7. `train_nle.py` hard-codes `WANDB_DATA_DIR=/scratch/tvnguyen/wandb_data`
   (Trillium path, fails on tara unless the launcher exports it; it does).
   Make the default `~/scratch/wandb_data`.

None of these change the model class or the checkpoint format, so
`infer_nle.py` and the T2 comparison scripts load the production model as is.

## 5. Inference side for production (43 Paper-4 targets, 37–3313 stars)

- `infer_nle.py` evaluates one walker at a time on CPU in a process pool. For
  43 targets x 2 seeds, evaluate all 64 walkers in one batch on the GPU
  (emcee `vectorize=True`): 64 x 3313 stars is one 200k-star forward pass per
  step, seconds per target instead of minutes.
- Keep the 8p_v4 prior box, v_sys as the eighth parameter, r_half fixed at the
  catalogue value (as in T2) — the comparison with the T2 numbers must be
  like for like before anything else changes.
- Output layout unchanged: `~/scratch/projects/sbi_dsph/npe_gap/nle/<tag>/<target>.npz`
  with `tag = nle_8p_v4_prod_<seed>`.

## 6. Acceptance tests (what makes it "production")

1. Per-regime PIT table on two held-out shards (not shard 30, which this note
   used; e.g. 31 and 32): PIT mean within 0.005 of 0.5 and std within 0.003 of
   0.2887 in every sigma >= 1 km/s bin; NLL <= T2's in every bin.
2. T2 replay: the 12 Draco mocks + 3 real Draco catalogues with the same emcee
   settings. Report the same table as the T2 page (median, half-width, z,
   width ratio to the GNN). Pass if RMS z <= 1.1 and the half-widths are within
   10% of T2's; a *narrower* production posterior with the same coverage is
   the hoped-for outcome, a wider one needs explaining.
3. The open T2 question (mean z +0.5): coverage on ~200 held-out simulated
   galaxies at ~100 stars (SBC / PIT of log rho(150 pc)) plus, if the mock
   generator is at hand, 100 Draco-like 510-star mocks with Walker-23 errors.
   Pass if the PIT of log rho(150 pc) is flat (KS p > 0.05) and the mean z is
   within 2 sigma of 0.
4. Seed agreement: on real Draco and the 12 mocks, the two seeds' log rho(150)
   medians differ by less than 0.3 of the half-width. Larger differences mean
   the NLE itself contributes to the width and should be reported as such.
5. dm_alpha at the prior edge on Draco (seen in T2): check whether it persists
   with the production model and with each seed; if it does, it is a
   data/prior statement, if not, it was NLE error.

## 7. Open choices left to the user

- Screening (2.5 h) vs. straight to arm B (3.3 h/seed).
- Whether to include proper motions (`use_proper_motions`) now; this design
  keeps v_los only, matching T2 and the NPE.
- Whether the production model should also condition on sky position
  (`condition_on_radius=False`) for non-spherical targets; the mocks and the
  simulator are spherical, so no.

## 8. Precision (measured 2026-10-07, after the user asked about float32/16)

Training is already float32 end to end (h5 float64 is cast on load; model
weights are float32). Lower precision, candidate model, on-device batches:

| precision | bs 16384 ms/step | bs 32768 ms/step | M stars/s |
|---|---|---|---|
| fp32 (current) | 20.6 | 50.3 | 0.65–0.80 |
| TF32 matmul (`'high'`) | 19.9 | 48.9 | 0.67–0.82 |
| bf16 autocast | 15.4 | 30.2 | 1.06–1.09 |
| fp16 autocast + GradScaler | 15.0 | 32.0 | 1.02–1.09 |

TF32 does nothing (the run is launch-bound, not matmul-bound). bf16 and fp16
both give ~1.35x at bs 16384 and ~1.6x at bs 32768, i.e. ~1 h off a 3.3 h run.

Accuracy of the trained 5-shard model's log p under autocast vs fp32, on 2.0M
held-out stars of shard 31 (errors 1 and 3 km/s), inference-time check:

| precision | per-star abs dlogp median / 99% / max | per-galaxy abs dlogL (sum of ~98 stars) median / 99% / max |
|---|---|---|
| bf16 | 0.026 / 0.47 / 1.70 | 0.34 / 6.4 / 11.1 |
| fp16 | 0.0017 / 0.034 / 0.14 | 0.027 / 0.33 / 0.86 |

bf16's 8-bit mantissa (3 significant digits) is too coarse for spline knots
resolving peaks of width 0.01 in normalised velocity: the error is largest at
sigma < 3 km/s (99% per-star 0.35–0.60 nats, with a systematic -0.005 to -0.015
nats/star), which scaled to a 510-star catalogue is several nats of
systematic log-likelihood error. fp16 (11-bit mantissa) is 15x tighter, 0.03
nats per 98-star galaxy at the 99th percentile.

Implemented 2026-10-07 (logic unchanged): `config.precision` in
`training.build_trainer` ('32-true' default, '16-mixed' = autocast + GradScaler
via Lightning; bf16 rejected with an error), `config.matmul_precision`, and
`NLE.validation_step` wrapped in `autocast(enabled=False)` so the checkpoint
selection metric is always the fp32 density. Smoke-tested in debug mode (1
shard, 40 steps each mode; 1 epoch under 16-mixed to exercise validation):
weights stay float32 in the checkpoint, no non-finite losses. Screening
config uses '16-mixed'; production config defaults to '32-true'.

Why the choice is set by the model and not by the catalogues. The real data
(`~/scratch/vr_catalogs/vr_catalogs`, 26 velocity columns across walker23,
desi, pace and ufdcomp, members and non-members) carry heliocentric v_los up
to 450–675 km/s with per-star errors down to 0.05–0.4 km/s. fp16 spacing at
300–500 km/s is 0.25–0.5 km/s, i.e. up to 5.4x the smallest error (pace scl)
and ~0.2x the median error: raw velocities cannot be stored in fp16. After
subtracting v_sys (done in fp32 inside `infer_nle.py`) the range is < 60 km/s
and the spacing 0.03–0.06 km/s, still 10–50% of the smallest errors. fp32
spacing is 3e-5 km/s everywhere: the data need fp32 and nothing more (RA/Dec
at 260 deg have fp32 resolution 0.05 arcsec = 0.02 pc at Draco, irrelevant).
Mixed precision never touches the data: inputs, normalisation, the spline
evaluation's non-matmul ops, weights and optimizer state stay fp32; only the
conditioner matmuls run in fp16. The accuracy table above is therefore about
the model's internal arithmetic, which is why bf16 fails on narrow peaks that
the data themselves represent with ease.

Decision: keep fp32 as the default for production (a 3.3 h run does not need
the hour), use '16-mixed' for screening arms and extra seeds (1.35x per run,
or ~1 h per arm over the five-arm screening), and keep bf16 out entirely. If the speed is wanted — e.g. for
screening or for more seeds — use fp16 autocast with GradScaler, and always
run inference (`infer_nle.py`) in fp32: the autocast error above was measured
at inference on fp32-trained weights; training under fp16 lets the weights
adapt, and the forward/backward through the spline is the part that needs the
range check (no non-finite losses in 140 fp16 steps on synthetic data, but a
real screening arm should log the GradScaler scale).

## 9. Error-convolution test (2026-10-07)

Question (user): a model trained without measurement noise, p_clean(v | log R,
theta, log r_half), convolved with the error kernel, should equal the
noise-trained model p_noisy(v_obs | log R, theta, log r_half, sigma). Does it?

Setup. Two shards (190k training galaxies), T2 architecture, batch 16384,
30k steps, same data seed and galaxy split: `errtest_clean` (run voj63drr,
no noise, no sigma in the context; `condition_on_error=False`) and
`errtest_noisy` (run 6xaw28uw, the usual pipeline). Also compared: the
5-shard T2 model 37vbi58p (`noisy2`). Test set: 3000 galaxies of held-out
shard 40, v_obs = v_true + N(0, sigma^2) at fixed sigma = 0.3, 1, 3, 10 km/s,
both log-densities converted to per km/s, the convolution by trapezoid
quadrature on 1025 points over the overlap of |v| < 12 sigma_true and
v_obs +- 8 sigma. Script `diagnostics/check_error_convolution.py`; table
`.../voj63drr/error_convolution_shard40_v2.csv`, figure `..._v2.png`.

d = log p_noisy - log Int p_clean N, in nats, at the true theta.
Columns: sigma = assumed error (km/s); sigma_true bin = the galaxy's
intrinsic line-of-sight dispersion (km/s); star_mean / star_rms = mean and
rms of d per star; gal_mean / gal_rms = mean and rms of d summed over a
galaxy's ~98 stars. Excerpt for the 2-shard noisy model (full table in
the csv, the T2 model is the same or tighter):

| sigma | sigma_true | star_mean | star_rms | gal_mean | gal_rms |
|---|---|---|---|---|---|
| 0.3 | [0, 1)   | -0.006 | 0.139 | -0.56 | 1.95 |
| 0.3 | [3, 10)  | +0.002 | 0.106 | +0.22 | 1.07 |
| 1   | [1, 3)   | -0.001 | 0.098 | -0.07 | 1.00 |
| 1   | [3, 10)  | +0.002 | 0.097 | +0.19 | 0.99 |
| 3   | [3, 10)  | -0.001 | 0.090 | -0.05 | 0.85 |
| 3   | [10, 40) | +0.002 | 0.113 | +0.21 | 1.21 |
| 10  | [3, 10)  | -0.001 | 0.060 | -0.13 | 0.59 |
| 10  | [10, 40) | -0.002 | 0.107 | -0.16 | 1.10 |

Result. The two agree: the mean of d is within +-0.006 nats per star in
every regime and within +-0.6 nats per galaxy, with no trend in sigma or
sigma_true. The per-star scatter (rms 0.04-0.14 nats) adds up to about 1
nat per galaxy; it is the sum of both models' own approximation errors plus
quadrature error, and the galaxy-level rms is ~sqrt(98) times the per-star
rms, i.e. the per-star errors are not coherent across a galaxy's stars.
The 1-D log-likelihood scans in dm_log_rho0 (figure, six galaxies, sigma =
1 and 3) coincide to the eye over 30 nats of dynamic range.

Two things the test turned up on the way. (1) `NLE.log_prob` returns the
density per normalized velocity unit and the two models have different
`vel_scale` (193.8 vs 205.2 km/s); forgetting the Jacobian gives a
constant 0.057 nats/star offset that looks like a model disagreement. Any
code that compares densities across models (or feeds them to a sampler
with a theta-dependent scale) must subtract log vel_scale. (2) The
quadrature grid must resolve the narrower of the intrinsic peak and the
error kernel; a grid spanning the union of the two supports gave 0.44
nats/star rms at sigma_true < 1 km/s under 10 km/s errors.

Consequence. The noise augmentation is learned consistently with the
physics (the model is not using sigma as a shortcut). The user's reasons
for keeping the noiseless model in play stand regardless: (1) the paper
should test this property, and (2) the noise in real data need not be
Gaussian, and only the noiseless model lets an arbitrary per-star error
kernel be applied at inference. The cost of doing so is measured in the
next subsection.

### 9b. Cost of the noiseless model with error integration (2026-10-07)

Measured on 300 held-out galaxies (29.7k stars) while a screening run
shared the GPU, so absolute times are inflated but ratios hold. Noisy
model (137k params): 1.9 M stars/s per log p evaluation. Convolved
noiseless model, trapezoid on the intersection grid of
`check_error_convolution.py`, time relative to the noisy model and rms
difference per star against a 1025-point reference, in nats, by the
galaxy's intrinsic sigma_true bin (km/s); last column the rms of the
difference summed over a galaxy's ~98 stars:

| nodes | time | [0,1) | [1,3) | [3,10) | [10,40) | per galaxy |
|---|---|---|---|---|---|---|
| sigma = 1 km/s |||||||
| 17  | 19x  | 0.16  | 0.033 | 0.003 | 0.0005 | 5.7 |
| 33  | 37x  | 0.046 | 0.007 | 0.0006 | 0.0001 | 0.98 |
| 65  | 70x  | 0.010 | 0.0014 | 0.0002 | 0.0000 | 0.15 |
| 129 | 147x | 0.0015 | 0.0003 | 0.0000 | 0.0000 | 0.019 |
| sigma = 10 km/s |||||||
| 33  | 35x  | 0.047 | 0.091 | 0.040 | 0.005 | 3.4 |
| 65  | 67x  | 0.010 | 0.028 | 0.006 | 0.001 | 0.83 |
| 129 | 139x | 0.0016 | 0.005 | 0.002 | 0.0003 | 0.15 |
| 257 | 286x | 0.0003 | 0.0007 | 0.0003 | 0.0001 | 0.016 |

The time ratio is about 1.1x the node count (the gather overhead). The
accuracy floor to aim for is the model-to-model scatter of section 9,
about 1 nat per galaxy: 65 nodes reach it, 129 nodes are 5-10x below it.
So the noiseless route costs 70-150x the noisy model's evaluation time
per likelihood call, independent of the kernel shape (a non-Gaussian
kernel is evaluated on the same grid; only its support bounds change,
and heavy tails widen the grid and raise the node count).

Per target with the current emcee settings (64 walkers x 5000 steps):
Draco (510 stars) is 163M per-star evaluations, i.e. ~1.5 min for the
noisy model on the GPU at the measured contended rate and ~3.5 h at 129
nodes (1.7 h at 65). The 43 Paper-4 targets (37-3313 stars, 44k stars in
total) would take ~2.3 h noisy and ~13 days at 129 nodes on one GB10 at
this rate, so for the population the noiseless route needs either fewer
MCMC steps (a 64 x 5000 chain is far longer than the autocorrelation
time demands; 64 x 1000 would cut it 5x), importance reweighting of the
noisy-model chain by the convolved likelihood (one convolved evaluation
per posterior sample instead of per proposal, ~10^4 evaluations per
target, minutes), or a larger per-call batch (the 2 M stars/s above is
at 30k-star batches with a shared GPU; uncontended and at 10^6-star
batches it should be several times higher).

Recommended use: train both models in production (the noiseless one is
the same cost to train); sample with the noisy model; validate on the
paper's test cases by reweighting or re-sampling with the convolved
noiseless likelihood, which is also the path for a non-Gaussian error
kernel.
