# Development log

All decisions taken without user interaction are recorded here with their reason.

## 1. Environment probe (2026-11-10)

- Python 3.13.18, Linux x86_64 (Ubuntu 24.04), 2 CPU cores (Intel Xeon 2.1 GHz), 7.8 GiB RAM, 22 GiB free disk.
- TeX Live 2023 (pdfTeX 1.40.25); IEEEtran was missing and was installed with
  `apt-get update && apt-get install -y --no-install-recommends texlive-publishers texlive-latex-extra texlive-science texlive-pictures latexmk poppler-utils`.
- Host reachability: `data.open-power-system-data.org` unreachable (egress proxy refused the connection),
  `raw.githubusercontent.com` reachable, `pypi.org` reachable, `doi.org` unreachable, `api.crossref.org` unreachable,
  `download.pytorch.org` unreachable. No blocked host was retried through another route.
- Software: torch 2.15.0 (PyPI wheel, the CPU index was unreachable), highspy 1.15.1, scipy 1.18.1, numpy 2.4.6,
  scikit-learn 1.9.1, tsam 4.1.1, kmedoids 0.5.5, pandas 3.0.5.

## 2. Data tier

- Tier A (OPSD) unreachable, no user attachments: **Tier B** is used (ENTSO-E Transparency Platform data for Spain,
  2015-2018, as distributed with the Darts library).
- `energy_dataset.csv`: 6,062,625 bytes, SHA-256 `6ccdba32...886c5` (identical to the protocol), 35,064 rows.
- `costs_2030.csv` (PyPSA technology-data v0.13.0): 265,861 bytes, SHA-256 `0407b2c6...c5573` (identical).
- Gap filling: PV and onshore wind had 16 missing hours in 2015 and 1 in 2016 and 2018; prices had none.
- Capacity factor proxy CF_t = gen_t / Q_y(0.999), clipped to [0, 1].

## 3. Modelling decisions

- The bound `0 <= (1-lambda)^h S_d + intra_{sigma_d,h} <= E` for all d and h is implemented exactly with per-cluster
  auxiliaries Smax_k >= S_d and Smin_k <= S_d (d in cluster k). Because (1-lambda)^h > 0, this is an exact projection
  of the per-(d, h) constraints and reduces the row count from 2 x 8760 to 2 x 365 + 2 x 24K per storage.
  Verified by `tests/test_storage_linking.py` (identity clustering equals the chronological model to 1e-7).
- Tier B HC+X extremes: the LT heat demand is schedule-based in Tier B (identical on every weekday), so the
  "maximum LT heat day" is degenerate. It is replaced by the day with the lowest mean PV capacity factor (the darkest
  day, a heating-season proxy); the other two extremes are the minimum PV-plus-wind day and the maximum price day.
  The new-cluster semantics of tsam (days closer to an extreme day than to their medoid join it) are reimplemented
  because the extreme criteria use a derived quantity (PV plus wind).
- tsam normalizes every column to [0, 1] internally. Column weights equal to each feature's range in standardized
  units undo that normalization, so all methods cluster on the same z-scored features (training-year statistics).
  `preserve_column_means=False` keeps pure centroid and medoid representations.
- KMed uses `kmedoids.fasterpam` (Euclidean distances on z-scored daily vectors); tsam 4 k-medoids needs a MILP solver.
- Reconstruction term of the VITA loss: squared distances are averaged over the 24F entries of a day
  (per-entry mean squared error) so that its scale does not depend on F.
- Rademacher directions of the Taylor targets are drawn once per anchor (fixed for the whole training).
- CO2 metric: the oracle emits no CO2 at 150 and 250 EUR/t in most years, so a relative error with respect to the
  oracle is undefined. The CO2 error is reported as |E(x-hat) - E(x*)| divided by the emissions of a fossil reference
  in which all high-temperature heat comes from the gas boiler (both evaluated at full resolution), in percent.
- Full-resolution design LP: HiGHS dual simplex (390-430 s per year); interior point with and without crossover and
  with split design columns was slower on this machine. Full-resolution operational LP: dual simplex (about 3.5 s).

## 4. Compute budget (Section 9.6)

Pilot timings on 2 cores: full-resolution design LP 400 s; full-resolution operational LP 3.5 s; aggregated
operational LP at K = 8 about 100 ms cold and 10-25 ms warm-started; one VITA step at K = 8 (60 LP solves) about 1.5 s,
at K = 16 about 4 s. The projected wall time of the full protocol (5 K values, 3 prices, 5 seeds, J0 = 24, S = 50)
exceeded 20 hours on two cores. Reductions applied in the prescribed order:

1. spatial transfer: not applicable in Tier B (one region);
2. K grid reduced to {4, 8, 16};
3. carbon prices: **kept at {80, 150, 250}** (deviation from the prescribed order). Reason: the carbon price grid
   requires no VITA retraining (scenario transfer), it costs only oracle and baseline evaluations, and 80 EUR/t is the
   only price at which the oracle builds fossil capacity, so dropping it would remove the only informative scenario;
4. VITA seeds reduced to 3;
5. J0 reduced to 16.

The projection still exceeded the budget, so two further changes were made and are reported in the paper:

- S (Adam steps per outer iteration) reduced from 50 to 25 (R = 6 kept);
- persistent HiGHS models are warm-started across steps; to keep warm starts effective, each minibatch of 6 anchors
  per training year is reused for 10 consecutive steps and the day sequence sigma (no gradient) is refreshed at the
  start of each block of 10 steps.

## 5. Development loop (development fold: train 2015-2016, validate 2017, 150 EUR/t, K = 8)

Validation regret of the baselines (2017, 150 EUR/t):

| K | KM | KMed | HC | HC+X | DP | AE | CO | IX |
|---|---|---|---|---|---|---|---|---|
| 8 | 0.278 | 0.255 | 0.734 | 0.334 | 0.511 | 0.781 | 0.310 | 0.398 |
| 16 | 0.352 | 0.092 | 0.263 | 0.608 | 0.313 | 0.237 | 0.198 | 0.736 |

Acceptance target: VITA mean validation regret (3 seeds) at K = 8 at most 0.85 x 0.255 = 0.217 %, and at K = 16
at most 0.092 %. VITA's validation regret is the regret of the early-stopped parameters (best of the R + 1
validation evaluations, including the initialization).

- **Cycle 0 (protocol defaults).** Seeds 0 and 1: best validation regret 0.920 % (iteration 2) and 0.333 %
  (iteration 3); seed 2 was killed by the out-of-memory killer after iteration 4 (logged regrets 1.257, 3.182, 0.850,
  3.333 %) because a second development run had been started in parallel. Diagnostics: the operational value
  V_y(x) is negative at near-optimal designs (export revenue of 150 MW PV and 61 MW wind exceeds import cost) and
  changes sign across anchors, so the normalization by V_y(x^j)^2 is ill-conditioned (relative errors up to 1.8);
  the prototypes were initialized at k-means++ seeds only, which left unbalanced clusters (weights 1, 180, 1, 98, ...).
- **Cycle 1 (item 1, correctness).** (a) Loss denominators TAC_y(x^j)^2 = (kappa^T x^j + V_y(x^j))^2 instead of
  V_y(x^j)^2 (units and sign: TAC is the quantity whose relative error defines regret and it is always positive);
  (b) prototypes initialized by k-means with k-means++ seeding (10 restarts) on the embeddings. Result: 0.2035,
  0.9495, 0.8965 % (mean 0.683 %, 2.68 times the best baseline). Kept (correctness).
- **Cycle 2 (item 2, loss balance, lambda_rec = 0.2).** Seed 0 best 0.647 % (iterations: 0.639, 3.110, 25.05, 2.084,
  2.238, 0.647), seed 1 at least 2.236 % after three iterations. Worse than cycle 1 on both seeds: rejected and
  stopped early to save compute.
- **Cycle 3 (item 3, straight-through phase over all 150 steps, tau annealed to 0.05).** Seed 0: 2.586, 2.789, 1.300,
  372.0 % after four iterations (a design with large unserved energy). Rejected and stopped.
- **Cycle 4 (deviation, not in the prescribed list): value matching up to a year-specific constant offset.** Reason
  tested: the aggregated value has a systematic optimistic bias of about 1 MEUR at near-optimal designs (smoothing
  reduces curtailment), and level matching can remove it only by reassigning days between clusters. Seed 0: 0.581,
  2.046, 2.981, 3.017 %. Rejected and stopped; the option is not used.
- **Cycle 5 (item 7, learning rates x 0.3).** Seeds 0 and 1 after one iteration: 2.106 and 1.793 %. Rejected and
  stopped.

Diagnosis after cycle 5: the validation regret jumps between 0.2 % and more than 100 % from one outer iteration to the
next, and the minibatch value error does not decrease. With layer-normalized 16-dimensional embeddings the squared
distances between a day and its second-nearest prototype exceed the nearest one by several units, so the soft
assignment saturates once tau falls below about 0.5 (after the first outer iteration); the assignment gradients then
vanish and Adam rescales the remaining gradient noise to full-size steps.
- **Cycle 6 (item 3, annealing speed and straight-through phase: tau annealed only to 0.5 so that assignment
  gradients do not vanish, straight-through hard forward pass for all 150 steps).** Seed 0: 0.267, 1.161, 3.643 %;
  seed 1: 33.77, 1.220, 203.99 % after three iterations. Not better than cycle 1 on either seed: rejected and stopped.

**Freeze.** Seven cycles were used (the limit is eight). No change after the correctness fixes of cycle 1 improved
the validation regret, and the acceptance target was not met (cycle 1: 0.683 % against 0.217 %). The configuration of
cycle 1 is frozen: TAC-normalized value loss, k-means prototype initialization, alpha = 0.5, lambda_rec = 0.05,
lambda_ent = 0.01, tau from 1.0 to 0.05, 20 straight-through steps, R = 6, S = 25, rho = 3, Adam learning rates
3e-3 (encoder) and 1e-2 (prototypes), weight decay 1e-5, m = 16, convolutional encoder, mean representation,
6 anchors per training year and per minibatch, M = 4 Taylor directions of 5 %. The cycle-1 models of the development
fold (seeds 0-2, K = 8) are reused as the fold-2018 models of the final runs (identical configuration and seeds).
The K = 16 part of the acceptance target is checked on the final fold-2018 models (same training and validation
years as the development fold). Test data were not used before this point.

## 6. Final runs

- Frozen configuration of cycle 1; K in {4, 8, 16}; seeds 0, 1, 2; four leave-one-year-out folds; carbon prices
  80, 150 and 250 EUR/t (VITA trained at 150 EUR/t only).
- Measured training time with the frozen configuration: about 6 min at K = 8 and about 14 min at K = 16 per model
  on one core.
- Ablations A1 to A4 are retrained with one seed (seed 0) per fold instead of three, because the projected wall
  time with three seeds exceeded the remaining budget by about one hour; A5 (no inter-period storage linking) and
  A6 (medoid snapping) change only the deployment of the trained full models (seed 0). The best baseline used in
  A5 is KMed, the best baseline on the validation year at K = 8 (chosen on validation data, not on test data).

## 7. Results of the final runs (test years evaluated once, after the freeze)

- Development-fold check at K = 16 with the frozen configuration: mean validation regret 0.393 % against 0.092 %
  for KMed, so the K = 16 part of the acceptance target was not met either.
- Temporal test set (12 instances per K): VITA 2.081 / 0.800 / 0.494 % at K = 4 / 8 / 16; best baselines KM 0.779 %
  (K = 4), KM 0.295 % (K = 8), KMed 0.117 % (K = 16). Holm-corrected Wilcoxon tests: no baseline significantly
  worse than VITA; VITA significantly worse than KMed at K = 16 (adjusted p = 0.008). The paper reports these results
  as they are and analyzes the causes (gradient dominated by the weekday channel, saturation of the soft assignment).
- Early stopping selected the initialization (iteration 0) in 18 of 36 final runs.
- The full-resolution design LP of 2018 at 150 EUR/t was paused (SIGSTOP) for about 40 minutes while development
  runs used both cores, so its recorded wall time (2670 s) is not a solver time; the paper reports the median
  solve time of the 12 oracle solves (454 s), which is not affected.
- A transient extra training process was started on the idle core while the oracle process finished and stopped
  after four outer iterations because three concurrent K = 16 trainings would have exceeded the memory; the run was
  repeated in full by the driver (cached results are keyed by inputs, so no partial result was used).
