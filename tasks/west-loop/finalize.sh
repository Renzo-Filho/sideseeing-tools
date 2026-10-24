#!/usr/bin/env bash
# After run_westloop.sh and after_run.sh: switch to the new code (smoothed GPS, grouped dataviz, redacted
# frames in the viewer), regenerate events and the report, build Whisper review packet v2 and restart the viewer.
source ~/miniforge3/etc/profile.d/conda.sh && conda activate sideseeing
status=~/sideseeing/logs/westloop/status
note() { echo "$(date "+%F %T") $*" >> "$status"; }
until grep -q " after_run finished" "$status"; do sleep 300; done
cd ~/GitHub/sideseeing-tools && git reset -q --hard incoming && note "finalize: code at $(git log --oneline -1)"
cd ~/sideseeing
for stage in events report summary; do
  if python -u westloop_step.py $stage >> logs/westloop/finalize.log 2>&1; then note "finalize: done $stage"; else note "finalize: FAILED $stage"; fi
done
if python whisper_review.py workspaces/west-loop review/whisper-v2 --windows-per-session 8 >> logs/westloop/finalize.log 2>&1; then note "finalize: done whisper review v2"; else note "finalize: FAILED whisper review v2"; fi
pkill -f "[s]ideseeing_tools.dataviz -i /home/renzo/sideseeing/workspaces/west-loop"; sleep 2
cd ~/GitHub/sideseeing-tools && setsid nohup python -m sideseeing_tools.dataviz -i ~/sideseeing/workspaces/west-loop > ~/sideseeing/logs/viewer.log 2>&1 < /dev/null &
note "finalize finished"
