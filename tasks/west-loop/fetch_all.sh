#!/usr/bin/env bash
# Download captures from presigned links ("capture size name url md5" per line) into RAW/<capture>/.
# Resumes partial files, checks size and MD5 (when RustFS has one), deletes the link file when all pass.
set -u
raw=$1; links=$2; failed=0
while read -r capture size name url md5; do
  mkdir -p "$raw/$capture"
  file="$raw/$capture/$name"
  if [ "$(stat -c %s "$file" 2>/dev/null || echo 0)" != "$size" ]; then
    curl -fsS --retry 5 -C - -o "$file" "$url" || true
  fi
  actual=$(stat -c %s "$file" 2>/dev/null || echo 0)
  if [ "$actual" != "$size" ]; then echo "FAILED size $capture/$name expected $size got $actual"; failed=1; continue; fi
  if [ "$md5" != "-" ] && [ "$(md5sum "$file" | cut -d' ' -f1)" != "$md5" ]; then echo "FAILED md5 $capture/$name"; failed=1; continue; fi
  if [ "$md5" = "-" ] && [[ "$name" == *.mp4 ]]; then  # no MD5 in RustFS: the container must at least open fully
    duration=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$file") || { echo "FAILED probe $capture/$name"; failed=1; continue; }
    echo "probe $capture/$name duration ${duration}s"
  fi
  echo "ok $capture/$name $size ${md5:0:8}"
done < "$links"
[ $failed = 0 ] && rm -f "$links"
echo "fetch done failed=$failed"
exit $failed
