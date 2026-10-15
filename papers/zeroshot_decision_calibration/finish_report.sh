#!/usr/bin/env bash
cd "$(dirname "$0")"
until grep -q "^\[.*DONE" results/finish_up.log 2>/dev/null; do sleep 30; done
( cd /tmp && python3 -P -u "$(cd "$(dirname "$0")" && pwd)/src/make_final_report.py" ) >> results/finish_up.log 2>&1
echo "[$(date +%H:%M:%S)] FINAL_REPORT.md generated" >> results/finish_up.log
