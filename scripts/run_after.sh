#!/usr/bin/env bash
# Wait until queue <prev_file> (log results/logs/<prev_log>.log) has recorded an exit
# code for every command, then run <next_file> with scripts/run_list.sh.
# Usage: scripts/run_after.sh <prev_file> <prev_log> <next_file> <next_log>
set -u
n=$(grep -c . "$1")
while [ "$(grep -c '^### EXIT' "results/logs/$2.log" 2>/dev/null || echo 0)" -lt "$n" ]; do sleep 60; done
exec scripts/run_list.sh "$3" "$4"
