#!/usr/bin/env bash
# Runs unattended. Lets CKD scoring continue for a bounded time, then stops it
# and completes every remaining task so the project is left in a coherent,
# verified state regardless of how far scoring got.
set -u
cd "$(dirname "$0")"
D=$(pwd)
LOG=results/finish_up.log
: > $LOG
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }

BUDGET=${1:-2400}          # seconds of further scoring
say "allowing CKD scoring up to ${BUDGET}s more"
END=$(( $(date +%s) + BUDGET ))
while [ "$(date +%s)" -lt "$END" ]; do
  n=$(wc -l < results/llm_ckd_llama32_3b_list_f10.jsonl 2>/dev/null || echo 0)
  [ "$n" -ge 400 ] && { say "scoring reached 400"; break; }
  pgrep -f "score_llm.py --task ckd" >/dev/null || { say "scorer exited at $n"; break; }
  sleep 30
done

say "stopping scorer"
for p in $(ps -eo pid=,cmd= | awk '/score_llm.py --task ckd/ && !/awk/ {print $1}'); do
  kill -9 "$p" 2>/dev/null
done
sleep 2
ollama stop llama3.2:3b >/dev/null 2>&1
say "scored $(wc -l < results/llm_ckd_llama32_3b_list_f10.jsonl) of 400 records"

say "analysing (patient-level, complete patients only)"
( cd /tmp && python3 -P -u "$D/src/analyze_ckd.py" ) >> "$D/$LOG" 2>&1

say "integrating into the manuscript"
( cd /tmp && python3 -P -u "$D/src/integrate_ckd.py" ) >> "$D/$LOG" 2>&1

say "verifying every claim against the artifacts"
( cd /tmp && python3 -P -u "$D/src/check_claims.py" ) >> "$D/$LOG" 2>&1
say "claim check exit=$?"

say "regenerating figures"
( cd /tmp && python3 -P -u "$D/src/make_figures.py" ) >> "$D/$LOG" 2>&1

say "DONE - see results/finish_up.log and FINAL_REPORT.md"
