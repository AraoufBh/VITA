# VITA: value-informed neural temporal aggregation for sector-coupled industrial energy hubs

This repository contains the code, configuration, tests, cached results and scripts that regenerate every number,
table and figure of the paper "Value-Informed Neural Temporal Aggregation for the Design of Sector-Coupled
Industrial Energy Hubs" (A. Bahi).

VITA is a neural time-series aggregation map. A temporal convolutional encoder embeds each day of a year, learnable
prototypes assign days softly to K clusters, and the representative days and weights are soft cluster means and
masses of real days. The map is trained so that the value function of the aggregated operational LP of a
sector-coupled hub matches the full-resolution value function, and its first-order behavior, on a set of anchor
designs. Gradients through the aggregated LP are exact LP sensitivities computed from HiGHS primal and dual solutions
(a custom `torch.autograd.Function`), verified by finite differences. At deployment, VITA aggregates a new year in one
forward pass without any full-resolution solve.

## Contents

```
configs/     default.yaml (every assumption of the case study), tier_b.yaml (data tier B and splits)
src/vita/    data.py, costs.py, lp_models.py, sensitivities.py, baselines.py, vita.py, evaluate.py,
             pipeline.py, stats.py, export.py, utils.py
scripts/     probe_env.py, run_pipeline.py, run_oracles.py, run_final.sh, make_assets.py, dev_summary.py,
             lint_paper.py, check_numbers.py, check_refs.py, add_doi_notes.py
tests/       LP checks, finite-difference sensitivities, gradient check of the full map, storage linking,
             aggregation and determinism tests
data/raw/    CHECKSUMS.txt (raw files are downloaded by `make data`)
data/processed/  daily feature tensors (npz) and data statistics
results/     cached LP results, trained VITA weights, metrics CSV files and logs
paper/       LaTeX sources, figure data (paper/data) and generated tables and macros (paper/generated)
docs/        development_log.md (all decisions), data_provenance.md
```

## Installation

Python 3.11 or newer.

```
python -m pip install -r requirements.txt
```

If the PyTorch CPU index is reachable, `pip install torch --index-url https://download.pytorch.org/whl/cpu` gives a
smaller installation. LaTeX (pdfLaTeX, BibTeX, latexmk, IEEEtran, pgfplots) is needed only for `make paper`.

## Quick start

```
make test          # unit tests (finite differences, gradient check, storage linking, determinism)
make assets        # regenerate all CSV files, tables and macros of the paper from results/ (minutes)
make paper         # compile paper/main.tex
```

## Full reproduction

```
make all           # data, oracles, anchors, baselines, VITA training, evaluation, ablations, assets, paper
```

`make all` reruns every computation from the raw data. Results are cached under `results/` keyed by a hash of the
inputs, so interrupted runs resume. The published run used two CPU cores (Intel Xeon, 2.1 GHz) and 7.8 GiB of RAM;
measured runtimes on that machine: full-resolution design LP 454 s (median over 12 years and carbon prices),
full-resolution operational LP about 3.6 s, VITA training 3.4, 5.6 and 12.3 minutes per model at K = 4, 8 and 16.
`make all` takes about 4 hours on two cores (the development loop of `docs/development_log.md` is not repeated).
The experiment protocol, all deviations from it and the development cycles are documented in
`docs/development_log.md`; `scripts/run_final.sh` is the driver used for the final runs and
`scripts/quality_gate.sh` runs the automatic checks of the deliverables.

## Results summary

Mean test regret [%] over 12 instances per K (four held-out years, leave-one-year-out, under carbon prices of 80, 150
and 250 EUR/t; VITA trained at 150 EUR/t and averaged over three seeds). Every design is evaluated with the
full-resolution operational LP of the test year.

| Method | K = 4 | K = 8 | K = 16 |
|---|---|---|---|
| VITA | 2.081 | 0.800 | 0.494 |
| KM | 0.779 | 0.295 | 0.259 |
| KMed | 0.836 | 0.437 | 0.117 |
| HC | 2.505 | 0.450 | 0.400 |
| HC+X | 3.792 | 0.754 | 0.382 |
| DP | 1.053 | 0.581 | 0.453 |
| AE | 1.737 | 0.536 | 0.377 |
| CO | 3.843 | 0.658 | 0.312 |
| IX | 6.289 | 0.917 | 0.339 |

VITA did not outperform the strongest input-based baselines (KM and KMed) on this case study; the Holm-corrected
Wilcoxon tests found VITA significantly worse than KMed at K = 16 and no significant difference elsewhere at K = 8 and
16. The paper analyzes the causes (the value-loss gradient acts mainly on the weekday share of representative days,
and the soft assignment saturates during training).

## Data tier and licenses

The published run uses data tier B: ENTSO-E Transparency Platform generation and day-ahead price data for Spain,
2015 to 2018, as compiled in the dataset file distributed with the Darts time-series library (see
`docs/data_provenance.md` for the URL, checksum and processing). Techno-economic parameters come from PyPSA
technology-data v0.13.0 (CC-BY-4.0, DOI 10.5281/zenodo.15640601). Data tier A (Open Power System Data and
When2Heat, Germany) is implemented in `src/vita/data.py` but could not be downloaded in the computing environment.

The code is released under the MIT license.
