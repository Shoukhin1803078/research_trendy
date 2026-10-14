#!/usr/bin/env bash
# Queued LLM arms. Launch when the dev stack is down.
# Preflight: >5GB RAM available, load < 4, governor on performance.
set -u
cd "$(dirname "$0")/src"
R=../results
mkdir -p $R

avail=$(free -m | awk '/^Mem:/{print $7}')
load=$(awk '{print int($1)}' /proc/loadavg)
gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)
echo "available RAM ${avail}MB | load ${load} | governor ${gov}"
[ "$avail" -lt 5000 ] && echo "WARNING: <5GB free, the model will thrash. Ctrl-C to abort." && sleep 10
[ "$gov" != "performance" ] && echo "HINT: sudo cpupower frequency-set -g performance  (2.7x faster)"

# Start small: --limit counts ITEMS (not subjects), enough to measure throughput;
# raise it once the per-item cost is known (900 items = full BenSParX).
for model in qwen2.5:3b-instruct-q8_0 llama3.2:3b; do
  for prompt in list lines sentence; do
    tag=$(echo "${model}_${prompt}" | tr ':/.' '___')
    python3 -P -u score_llm.py --task bensparx --model "$model" --prompt "$prompt" \
        --limit 20 --out "$R/llm_bensparx_${tag}.jsonl"  2>&1 | tee -a "$R/llm_bensparx.log"
    python3 -P -u score_llm.py --task ckd      --model "$model" --prompt "$prompt" \
        --limit 20 --out "$R/llm_ckd_${tag}.jsonl"       2>&1 | tee -a "$R/llm_ckd.log"
  done
done
echo "All cells written to $R. Analyse with: python3 -P analyze_llm.py"
