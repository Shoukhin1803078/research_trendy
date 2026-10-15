# Search Log

Every search run during the phase-2 audit, with its yield. Failed and
low-yield searches are recorded, as required by `feedback2.md`.

## Target

Papers evaluating LLMs on clinical classification without task-specific
training, reporting **both** a ranking metric and a threshold-dependent metric
(see `audit/EXTRACTION_PROTOCOL.md`).

## Searches

| # | query | yield |
|---|---|---|
| S1 | "zero-shot LLM clinical prediction paper reports balanced accuracy sensitivity specificity AUROC table 2026 screening diagnosis" | 1 eligible (2607.29082); surfaced 1 new candidate (pediatric wrist fracture, Nature Sci Rep) |
| S2 | "pediatric wrist fracture zero-shot multimodal LLM GPT-4o Claude Gemini CNN baseline accuracy sensitivity specificity PMC" | **0** — no open version of the candidate found; related papers were elbow or general fracture studies, not the target |
| S3 | "LLM4CKD chronic kidney disease zero-shot large language models github code repository Mirzapur" | **0** — no code repository located for either CKD paper |

## Direct fetches

| paper | result |
|---|---|
| arXiv:2607.20453 (delirium) | fetched; **AUROC only**, excluded E2 |
| arXiv:2607.29082 (malnutrition) | fetched; **eligible**, 15 rows extracted |
| arXiv:2605.10286 (AgentRx) | fetched; **AUROC, AUPRC, ECE only**, excluded E2 |
| Nature s41746-026-02539-z (ClinicRealm) | **blocked** by paywall redirect |
| arXiv:2407.18525v3 (ClinicRealm) | PDF fetched, text extracted locally with `pdftotext`; **AUROC + AUPRC only**, excluded E2 |
| Nature s41598-026-58763-w (wrist fracture) | **blocked** by paywall redirect; no open version found |
| scimagojr.com (x2) | **blocked**, HTTP 403, for journal quartile verification |

## Yield summary

- Papers examined in phase 2: **5**
- Newly eligible: **1** (2607.29082)
- Excluded for reporting no threshold-dependent metric: **3**
- Inaccessible: **1**

## Honest assessment of search coverage

This is **not a systematic review**. It is a targeted search seeded by papers
identified in phase 1, extended by three queries. A systematic search would
require a registered protocol, multiple databases (PubMed, Scopus, IEEE Xplore,
ACL Anthology), and a documented screening process with two reviewers. The
manuscript describes the audit as a convenience sample and does not claim
representativeness.

**A selection effect is now documented:** our audit can only include papers that
report both metric families, and 3 of the 5 examined in this phase do not. The
audited sample is therefore drawn from a minority of the literature, and its
rate should not be read as the field's rate.
