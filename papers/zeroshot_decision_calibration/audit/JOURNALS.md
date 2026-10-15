# Journal Shortlist

**Date checked:** 2026-10-14.

## IMPORTANT LIMITATION ON THIS DOCUMENT

**I could not reach SCImago (scimagojr.com) directly — every fetch returned
HTTP 403.** The quartile figures below come from search-result snippets of
SCImago pages and from third-party aggregators. Under the rule in `feedback2.md`
("Do not claim Q1 status without a current authoritative source"), these are
recorded as **unverified** and must be confirmed on the SCImago or JCR site, or
via an institutional subscription, before being used in a submission decision.

One aggregator (paperguide.ai) returned visibly corrupted values and was
discarded.

---

## Candidates, ranked by scope fit, not prestige

### 1. Journal of Biomedical Informatics (Elsevier)

- **Scope fit: strong.** Publishes methodological critiques of clinical
  informatics evaluation, which is exactly what this paper is.
- **Quartile:** snippet of the SCImago profile gives **SJR 2025 = 1.320, Q1
  band**, Health Informatics, H-index 145, with Q1 reported for Health
  Informatics for each year 2005-2025. **Unverified — SCImago not reachable.**
- **Article type:** would fit as a methodological or research paper.
- **Why first:** an audit of how a field reports evaluation metrics is squarely
  in its remit, and the paper's contribution is methodological rather than
  model-building.

### 2. Artificial Intelligence in Medicine (Elsevier)

- **Scope fit: strong.** Evaluation methodology for medical AI.
- **Quartile:** aggregators give **SJR 2025 = 1.659**, 2024 = 1.396, best
  quartile **Q1** in Artificial Intelligence. JCR categories listed as Computer
  Science AI, Engineering Biomedical, Medical Informatics. **Unverified — I
  could not confirm the Medicine-category SJR quartile on SCImago itself.**
- **Caveat:** the "best quartile" figure refers to the Artificial Intelligence
  category, which is not necessarily the category a reviewer would cite.

### 3. Computers in Biology and Medicine (Elsevier)

- **Scope fit: moderate to strong.** Broad; publishes audits and benchmarks.
- **Quartile:** SCImago category listing snippet gives **1.375, Q1** in Health
  Informatics, H-index 164. **Unverified.**
- **Caveat:** broader and higher-volume; a methodological critique may attract
  less expert review than at (1) or (2).

### 4. IEEE Journal of Biomedical and Health Informatics

- **Scope fit: moderate.** Strong on methods; the audit framing is less typical.
- **Quartile:** **conflicting figures** — the journal profile snippet gives SJR
  1.624 / H-index 179, while the category ranking page gives 1.395 / H-index 53,
  with metrics stated as of March 2026. The two disagree and **neither is
  verified**. This discrepancy alone means it should not be cited without
  checking.

### 5. npj Digital Medicine (Nature Portfolio)

- **Scope fit: moderate.** High-profile; would likely require the audit expanded
  well beyond 4 papers.
- **Quartile:** not checked.
- **Assessment:** premature. The audit is 37 cells from 4 papers with a
  paper-level interval of [17.6%, 46.2%]. A journal at this level would
  reasonably ask for a systematic review-scale audit.

---

## Honest fit assessment

The paper's realistic positioning is **(1) or (2)**. Reasons:

- The contribution is **methodological and negative-leaning**: it audits how a
  field reports, finds a decision-point artifact in a third of auditable cells,
  and explicitly claims no new method.
- The audit is **small (4 papers)** and **heterogeneous (0% to 50% per paper)**.
- A substantial part of the paper is corrections and retractions of its own
  earlier results, which is good practice but reads as a methods-and-evidence
  paper rather than a flagship result.

A plausible alternative framing, if a reviewer finds the audit too small, is to
submit it as a **short report or methods note** emphasising the diagnostic and
the reporting-practice finding (3 of 5 recent papers report no
threshold-dependent metric), with the audit as supporting evidence.

---

## Required before submission

1. Verify each quartile on SCImago or JCR directly, recording the ranking year
   and subject category.
2. Check each journal's article-type limits and whether an audit of published
   results requires any particular declaration.
3. Check open-access options and APCs — **not verified here**.
4. Find 2-3 recent papers in the chosen journal with comparable methodology, to
   confirm fit and to cite.

---

## Decision, 2026-10-15: Option B (short report)

The user chose the fallback framing described above. The submission version is
`draft_paper_short.md` (~3,300 words), which **leads with the reporting
finding** — three of five screened papers report no threshold-dependent metric
and therefore cannot be audited by anyone — and carries the 32.4% rate as
supporting evidence rather than as the headline.

**Why:** the rate rests on 37 cells from 4 papers with a per-paper range of 0%
to 50% and a paper-level interval of [17.6%, 46.2%]. The reporting finding does
not depend on the estimator being correct, so it survives that sample size.

`draft_paper.md` is retained in full as supplementary material or a preprint.

**Target, in order:** (1) *Journal of Biomedical Informatics*, (2) *Artificial
Intelligence in Medicine*. Both take short reports / methods notes.

### Still required before submission — NOT DONE

1. **Verify the quartile for the chosen journal on SCImago or JCR directly**,
   recording ranking year and subject category. Blocked here by HTTP 403; this
   must be done by hand. Nothing in the manuscript should state a quartile until
   it is.
2. Confirm the journal's short-report / methods-note word and figure limits, and
   reformat to them. The current draft is ~3,300 words with one figure.
3. Check article-type rules on auditing published results — some journals
   require the audited authors be notified or given right of reply.
4. Check open-access options and APCs.
5. Find 2-3 recent papers in the chosen journal with comparable methodology, to
   confirm fit and to cite.
