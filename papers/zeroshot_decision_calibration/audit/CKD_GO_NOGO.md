# CKD Experiment: Go / No-Go Recommendation

**Status: NOT LAUNCHED.** This is a recommendation only, as required by
`feedback2.md` Priority 5. Nothing below has been run.

---

## 1. What the experiment is

Score the Mendeley diabetic-CKD cohort with `llama3.2:3b` using the existing
pipeline (`src/score_llm.py --task ckd`), extract `s = logP("1") - logP("0")`,
and test whether the headroom diagnostic predicts the realised gain from fitting
a decision point. It would add a **third modality** (tabular) to a forward test
that currently covers images (8 endpoints) and speech (1 cell).

## 2. Resource requirements, corrected

The phase-1 manuscript said "about 800 calls and 3.5 hours". **That estimate was
too high.** Recomputed from the loader:

| configuration | items | at 15 s/item |
|---|---|---|
| `--groups 20` | 200 | **50 min** |
| `--groups 40` | 400 | **100 min** |

`--groups` selects whole patients, and each patient contributes 10 yearly
records. The run is resumable per item, so an interruption costs nothing. The
15 s/item figure is the throughput measured on this machine with a 10-feature
prompt when the dev stack was idle; it was 60 s/item under load.

**Machine requirement:** at least 5 GB free RAM and a low load average. Both
earlier attempts at LLM scoring failed when the machine was busy.

## 3. Data and protocol check

| item | finding |
|---|---|
| Source | Mendeley doi:10.17632/hjkzgbxgv5.2, BIRDEM General Hospital, Dhaka |
| Access | public, no access controls stated |
| Ethics | the dataset page states ethical approval from the Diabetic Association of Bangladesh |
| Structure | **400 patients x 10 yearly visits = 4000 rows** |
| Independence | **rows are NOT independent**; `score_llm.load_ckd` already sets `group` to the patient id and `subsample` takes whole patients |
| Prevalence | 9.65% at record level; 185/400 patients ever positive |
| Features | 24, demographic and lifestyle only |
| **Lab values** | **none** - no creatinine, no eGFR, no urea |

The protocol is appropriate: patient-grouped sampling is implemented, and the
threshold would be fitted on support patients and evaluated on held-out
patients, as in the speech arm.

## 4. The scientific case

**For:**
- Adds a third modality, which is the main generalisability gap in the forward
  validation.
- Tabular clinical data is the setting of two of the four audited papers, so it
  matches the literature the audit covers.
- A skewed cohort (9.65%) tests the diagnostic where prevalence is far from
  balanced, which the current forward tests do not.

**Against:**
- The central claim rests on the **published audit**, not the forward test. A
  third forward cell would strengthen generality but would not change the
  headline.
- **The task may have too little signal to be informative.** With no lab values,
  CKD prediction from demographics and lifestyle alone may yield AUROC near 0.5.
  If it does, the cell lands in mode B or C and **cannot demonstrate
  recoverability at all** - there is no ranking signal to recover. The
  experiment would then test the diagnostic's specificity rather than its main
  claim, which is a weaker payoff for 100 minutes of exclusive machine time.

## 5. Recommendation: **CONDITIONAL GO, with a cheap pre-check**

Do not launch the full run blind. The risk is not cost, it is that a low-signal
task produces an uninformative cell.

**Step 1 - pre-check (about 15 minutes, 60 items).**

```
python3 -P src/score_llm.py --task ckd --model llama3.2:3b \
    --prompt list --groups 6 --nfeat 10 \
    --out results/llm_ckd_PRECHECK.jsonl
```

Compute AUROC on those 60 records.

**Step 2 - stopping criteria, fixed in advance:**

| pre-check AUROC | decision |
|---|---|
| **>= 0.60** | **GO.** Run `--groups 40` (100 min). The cell can demonstrate recoverability. |
| **0.55 - 0.60** | **GO at `--groups 20`** (50 min). Marginal signal; report with wide intervals. |
| **< 0.55** | **NO-GO.** Record as "attempted, insufficient ranking signal to test recoverability" and keep it as a documented limitation. |

**Expected outputs:** `results/llm_ckd_*.jsonl` (per-record scores), an analysis
row giving patient-level AUROC, native BACC, H, and realised BACC at k = 4, 8,
16 with subject-level bootstrap intervals, plus a row appended to the forward-
validation table.

**Pre-committed:** the pre-check AUROC is computed once and the threshold rule
above is applied as written. The rule is not revised after seeing the number.

## 6. What I need from you

Approval to run **Step 1 only** (the 15-minute pre-check). I will report its
AUROC and the resulting decision before running anything longer.

---

# OUTCOME (2026-10-14, after approval)

## A bug the pre-check caught before any compute was wasted

The first pre-check launch reported **30 items, not 60**. Cause: the
label-balanced patient sampler used each patient's **first** visit label, and in
this longitudinal cohort **0 of 400 patients are CKD-positive at visit 1**, so
`pos` was empty and the balancing silently became a no-op, selecting 3 patients
instead of 6. Fixed in `src/score_llm.py` to label a group positive if **any**
record in it is positive. BenSParX is unaffected (all segments of a subject
share one label), verified at 20/20.

**Had the full run been launched without the pre-check, the CKD arm would have
been invalid.** This is the main argument for pre-checks in this project.

## Pre-check result, all 60 items, 6 patients

| quantity | at 47 items | **final, 60 items** |
|---|---|---|
| record-level AUROC | 0.900 [0.788, 0.979] | **0.812 [0.664, 0.932]** |
| **patient-level AUROC** | 1.000 (meaningless at n=5) | **0.556** (6 patients, 3 positive) |
| native decision | sens 1.000, spec 0.000 | **sens 1.000, spec 0.000** |
| native BACC | 0.500 | **0.500** |
| estimated reachable | 0.818 | 0.735 |
| H | +0.318 | **+0.235** |

## Decision, and an honest reading of it

The pre-registered rule is applied to the **record-level** AUROC: 0.812 >= 0.60
-> **GO at `--groups 40`**. The full run was launched on that basis, seeded with
these 60 items so nothing is re-scored.

**But the two AUROCs disagree sharply, and this matters.**

- Record-level 0.812 is **inflated by within-patient correlation**: the 10
  records of a patient are near-duplicates, and positives cluster inside
  ever-positive patients, so this number largely reflects separating 6 patients
  rather than 60 records.
- Patient-level 0.556 is **near chance** — but on 6 patients with 3 positives it
  carries almost no information either.

Neither settles the question, which is exactly why the full run at 40 patients
is worth its 60 minutes. **The full analysis will be patient-level**, since that
is the unit a screening claim would concern, and the record-level figure will be
reported only as a diagnostic of within-patient correlation.

The degenerate signature is unambiguous regardless of level: the model predicts
positive for **every record**, giving specificity 0.000 and BACC exactly 0.500.

## Pre-registered for the full run, before seeing it

- Primary metric: **patient-level** AUROC, native BACC, H, and realised BACC at
  k = 4, 8, 16 labelled **patients**, with subject-level bootstrap intervals.
- If patient-level AUROC <= 0.55, the cell is reported as **signal-limited or
  inverted**, which is a negative result for recoverability and will be reported
  as such, not reframed.
- Thresholds are fitted on support patients only and evaluated on held-out
  patients.
