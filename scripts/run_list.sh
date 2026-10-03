#!/usr/bin/env bash
# Run a list of experiment commands (one per line in a file); record exit codes.
# Usage: scripts/run_list.sh commands.txt logname
set -u
while IFS= read -r cmd; do
  [ -z "$cmd" ] && continue
  echo "### $cmd" >> "results/logs/$2.log"
  bash -c "$cmd" >> "results/logs/$2.log" 2>&1
  echo "### EXIT $? $(date -u +%FT%TZ)" >> "results/logs/$2.log"
done < "$1"
