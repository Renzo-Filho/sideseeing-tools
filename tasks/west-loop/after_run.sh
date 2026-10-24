#!/usr/bin/env bash
# Runs after run_westloop.sh finishes: the large-v3 Whisper variant (failed in the main run; fixed loader),
# then the summary again. Appends to the same status file.
source ~/miniforge3/etc/profile.d/conda.sh && conda activate sideseeing
cd ~/sideseeing
status=~/sideseeing/logs/westloop/status
until grep -qE " (finished|stopped)" "$status"; do sleep 300; done
note() { echo "$(date "+%F %T") $*" >> "$status"; }
note "start whisper (large-v3 retry)"
if python -u westloop_step.py whisper >> logs/westloop/whisper-retry.log 2>&1; then note "done whisper (large-v3 retry)"; else note "FAILED whisper (large-v3 retry)"; fi
python -u westloop_step.py summary >> logs/westloop/whisper-retry.log 2>&1
note "after_run finished"
