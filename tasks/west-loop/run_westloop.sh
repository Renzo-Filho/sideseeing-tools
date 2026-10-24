#!/usr/bin/env bash
# Unattended West Loop run (sideseeing-tools tasks/west-loop-run.md). Rerun to resume: finished stages are
# skipped via logs/westloop/status, and finished work inside a stage is skipped by the library.
# Logs: ~/sideseeing/logs/westloop/{run.log,status,gpu.csv,summary.json}
set -u
source ~/miniforge3/etc/profile.d/conda.sh && conda activate sideseeing
cd ~/sideseeing
logs=~/sideseeing/logs/westloop; mkdir -p "$logs"; status=$logs/status; touch "$status"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 30 >> "$logs/gpu.csv" &
gpu_logger=$!
trap 'kill $gpu_logger 2>/dev/null' EXIT

note() { echo "$(date '+%F %T') $*" | tee -a "$status"; }

run() {  # run STAGE COMMAND...: skip if done; a failed stage is recorded and the run goes on
  local stage=$1; shift
  if grep -q "^.* done $stage$" "$status"; then echo "skip $stage (done)"; return; fi
  note "start $stage"
  if "$@"; then note "done $stage"; else note "FAILED $stage (exit $?)"; fi
}

if [ -f links.txt ]; then  # a failed download stops everything: later stages must not read truncated files
  run fetch ./fetch_all.sh ~/sideseeing/raw/chicago/west-loop links.txt
  grep -q " done fetch$" "$status" || { note "stopped: fetch failed"; exit 1; }
fi
for stage in link whisper frames segment redact events report summary; do
  run "$stage" python -u westloop_step.py "$stage"
done
note "finished"
