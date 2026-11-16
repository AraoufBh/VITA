#!/usr/bin/env bash
# Final experiments with the frozen configuration (development cycle 1).
set -e
cd "$(dirname "$0")/.."
PY=python3
$PY scripts/run_pipeline.py anchors --jobs 1
$PY scripts/run_pipeline.py train --jobs 1 --folds F2018 --K 16,4
# wait for the background oracle process (one core) before using two workers
while pgrep -f run_oracles.py > /dev/null; do sleep 30; done
$PY scripts/run_pipeline.py oracle --jobs 2
$PY scripts/run_pipeline.py train --jobs 2
$PY scripts/run_pipeline.py baselines --jobs 2
$PY scripts/run_pipeline.py evaluate --jobs 2
$PY scripts/run_pipeline.py ablations --jobs 2
echo FINAL_DONE
