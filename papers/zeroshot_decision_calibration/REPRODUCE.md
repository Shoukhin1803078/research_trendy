# Reproducing the results

All commands are run from this directory. Python 3.12; `numpy`, `scikit-learn`,
`scipy`, `matplotlib`. Use `python3 -P` (it keeps the current directory off the
import path) and run from a directory other than a downloaded-data directory.

Nothing here needs a GPU. Only step 5 needs an LLM.

## 0. Data

| dataset | how to obtain | local path |
|---|---|---|
| BenSParX feature table | `curl -L -o BenSParX.csv https://raw.githubusercontent.com/Riad071/BenSParX/HEAD/BenSParX.csv` (MIT) | `/mnt/extra-storage/data/bensparx/` |
| MedIMeta BiomedCLIP features | pre-extracted and cached | `/mnt/extra-storage/features/*.npz` |
| Diabetic CKD cohort | Mendeley doi:10.17632/hjkzgbxgv5.2 | `/mnt/extra-storage/data/diabetic_ckd/` |

The published-result cells are transcribed by hand in `src/harvest.py`, with the
paper and table recorded for each row. No network access is needed to re-run the
audit.

## 1. Audit scripts (read-only, no artifacts modified)

```
python3 -P audit/a1_circularity.py      # shared-term analysis of the validation
python3 -P audit/a2_formula.py          # formula verification and error direction
python3 -P audit/a3_harvest.py          # clustering, robustness, data integrity
python3 -P audit/a4_leakage_stats.py    # leakage scan, claim spot-checks
```

## 2. Image endpoints

```
python3 -P src/run_arm_a.py             # original 9 endpoints  -> results/arm_a_medimeta.json
python3 -P src/run_budget.py            # label budget          -> results/budget.json
python3 -P src/run_endpoints_v2.py      # expanded set          -> results/endpoints_v2.json
python3 -P src/validate_diagnostic_v2.py   # RQ1 with clustering
python3 -P src/analyze_budget.py
```

Runtime: a few minutes each on CPU. `run_budget.py` is the longest (about 10
minutes) because of the leave-one-out tuning.

## 3. Speech, no LLM

```
python3 -P src/bensparx_reference.py    # supervised reference (logistic regression)
python3 -P src/bensparx_aggregation.py  # aggregation rules, feature subsets, leakage
python3 -P src/stats_v2.py              # cluster-aware intervals -> results/stats_v2.json
```

## 4. The audit of published results

```
python3 -P src/harvest.py               # -> results/harvest.json
python3 -P src/harvest_sensitivity.py   # -> results/harvest_sensitivity.json
```

## 5. LLM scoring (the only step needing a model)

Requires ollama with `llama3.2:3b`. Check first that at least 5 GB of RAM is free
and the load average is low; the run is resumable, so an interruption costs
nothing.

```
python3 -P src/score_llm.py --task bensparx --model llama3.2:3b \
    --prompt list --groups 40 --nfeat 10 \
    --out results/llm_bensparx_llama32_3b_list_f10.jsonl
python3 -P src/analyze_bensparx.py
```

Measured throughput: about 15 s per item when the machine is free, so 304 items
take roughly 75 minutes.

**Not run:** the CKD arm, `src/score_llm.py --task ckd --groups 40 --nfeat 10`.
See `draft_paper.md` §13.

## 6. Figures and verification

```
python3 -P src/make_figures.py          # -> figures/*.pdf and *.png
python3 -P src/check_claims.py          # 44 claims re-verified; exits non-zero on any failure
```

`check_claims.py` is the gate: it must pass before the manuscript is submitted.

## Seeds

Every sampling loop uses an explicit seed derived from the draw index and the
label budget (`np.random.default_rng(10_000 * seed + k)` and similar), so runs
reproduce exactly. The LLM is queried at temperature 0 with a single forced
token.
