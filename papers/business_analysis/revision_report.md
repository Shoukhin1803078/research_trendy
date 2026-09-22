# Revision Report — "Business Analytics for Audit and Fraud Detection"

Companion to `audit_fraud_detection.tex` / `.pdf`. This file records what was changed, what still
needs author input, residual reviewer risks, and a reference audit. No experiments, datasets,
statistics, citations, authors or findings were invented at any point.

---

## 1. CHANGELOG (feedback item → change)

**Item 1 — Sharpen contribution.** Title changed to signal the review type and the framework
("An Integrative Review ... and a Conceptual AI-Assisted Prioritisation Framework"). The Introduction
was restructured into problem → traditional-audit limits → rise of analytics → gap → stakeholder
implications → four explicit Research Questions (RQ1–RQ4) → four explicit contributions (C1–C4:
methodological, conceptual, comparative, framework) → paper roadmap. Generic "cohesive framework"
language replaced with concrete claims. No "first/novel/unique" claims were added.

**Item 2 — Reproducible methodology.** Methodology reorganised into: Theoretical Foundations; Review
Design and Analytical Strategy; Search Strategy and Source Selection; Coding, Synthesis and Validation.
Inclusion/exclusion criteria are now stated explicitly. Missing details are not invented — they are
marked `[AUTHOR INPUT REQUIRED: ...]` (search strings; reviewer count/independence/IRR; screening
software; coders/reliability). "Integrative review" is used consistently; the paper is not labelled a
systematic review.

**Item 3 — Consistent selection logic.** The funnel is now stated and shown in Table I with explicit
arithmetic (272 academic + 25 professional = 297; −25 duplicates = 272; −118 = 154; −70 = 84). Academic
vs industry/professional sources are distinguished, and sources are tagged by type at coding so grey
literature is not presented as equivalent to peer-reviewed evidence.

**Item 4 — Illustrative numbers.** The two quantitative-looking charts (maturity line chart; red-flag
pie chart) were **removed from the paper** and replaced with conceptual diagrams (TikZ). No fabricated
data replaced them. The original PNGs remain in `figs/` unused, so they can be reinstated if a citable
source for the values is supplied.

**Item 5 — Red-flag taxonomy.** Converted from a bullet list into Table IV with, per cluster:
definition, representative indicators, reported analytical techniques, audit relevance and limitations.
Categories are defined to be conceptually distinct; co-occurrence is acknowledged.

**Item 6 — AI prioritisation framework.** Now defines every variable, normalisation to [0,1], the weight
constraints (w_i ≥ 0, Σw_i = 1), the range of s(r), threshold semantics, above/below-threshold behaviour
and workflow integration. An explicit statement — repeated in the abstract — states the framework is
conceptual and **not empirically validated**, with no accuracy/precision/recall claimed.

**Item 7 — Maturity model.** Added Table III (descriptive → diagnostic → predictive → prescriptive)
linking each level to its contribution and to data/skills/governance/explainability conditions; maturity
framed as socio-technical (tied to the TOE lens).

**Item 8 — Tool comparison.** Table III(a) rebuilt: five platforms across seven shared dimensions; prose
and table now both treat Python/R as one open-source category (resolving the earlier prose-vs-table
inconsistency). Only dimensions supported by the sources are used; no new commercial-tool claims added.

**Item 9 — Discussion.** Rewritten to distinguish reported evidence from the authors' interpretation, and
to discuss implications for auditors, organisations and AI adoption. A dedicated "Synthesis of
Implementation Challenges" subsection with Table VI was added.

**Item 10 — Overclaiming.** Removed/softened: "guarantee", "significantly improves", "demonstrates
effectiveness", "sharpens", "upgrades corporate resilience", "expands the precision", "most effective",
"substantially higher", "successful". Replaced with "reported", "indicates", "suggests", "may", "can".
Retained strong language only where the review evidence supports it.

**Item 11 — Limitations.** Expanded (Section IV-C) to cover: no primary data; heterogeneous literature;
publication/selection bias; English-language and database coverage limits; grey-literature differences;
conceptual (unvalidated) framework; and inability to establish causal effects.

**Item 12 — Conclusion.** Rewritten to state what the review established, what the framework contributes,
what practitioners can take, what remains unvalidated, and what future research should test — rather
than restating the abstract.

**Item 13 — References.** Audited; incomplete entries flagged with a dagger and the
`[REFERENCE VERIFICATION REQUIRED]` marker (see §4). Nothing invented. No duplicates remain (verified).

**Item 14 — Internal consistency.** Section/table/figure/equation/algorithm numbering verified; "integrative
review" used consistently; five tools described consistently; source counts consistent; cross-references
resolved (no undefined references).

**Item 15 — Tables.** Now six information-dense tables: I review protocol; II conventional vs
analytics-enabled audit; III analytics maturity; IV audit-analytics platforms; V red-flag taxonomy;
VI implementation challenges and mitigations. Tables that merely restated prose were removed.

**Item 16 — Figures.** Three conceptual figures: (1) analytics-driven audit workflow; (2) red-flag
taxonomy (new TikZ); (3) AI-assisted prioritisation workflow (new TikZ). No decorative or fake-quantitative
figures.

**Item 17 — Academic writing.** Full copyedit: shorter sentences, removed repetition, consistent tense
and terminology, British spelling normalised, marketing language removed.

**Item 18 — Abstract.** Rewritten to the requested structure (background → objective → methodology →
key synthesis → contribution → practical implications → limitations), and explicitly states that the
framework is conceptual.

**Item 19 — Introduction.** Reorganised to the requested arc, with RQs and contributions; unsupported
generic statements removed or cited.

**Item 20 — Reviewer simulation.** See §3.

---

## 2. AUTHOR INPUT REQUIRED (cannot be fixed without information/evidence)

These items are **no longer printed in the PDF**. To keep the compiled output clean they were converted
to `%` LaTeX comments in the `.tex` source and are tracked here. Reword the surrounding sentences when
you supply the values, or delete them.

1. `[AUTHOR INPUT REQUIRED: exact database search strings + date of final search]` (§II-C).
2. `[AUTHOR INPUT REQUIRED: single vs multiple independent reviewers; inter-rater agreement if any]` (§II-C).
3. `[AUTHOR INPUT REQUIRED: screening/reference software; final split of the 84 sources into peer-reviewed
   vs industry/professional]` (§II-C).
4. `[AUTHOR INPUT REQUIRED: number of coders and any coding-reliability measure]` (§II-D).
5. A citable source for the "~50% of effort in data preparation" statement — currently marked
   `[NEEDS EVIDENCE]` in §II-G. (If unavailable, the sentence should be deleted.)
6. Complete bibliographic data for the 15 flagged references in §4, plus confirmation of the two
   stub entries ([15] Wolniak; [28] Abdelwahed).
7. Author names, affiliations and emails (title-page block is a placeholder).
8. Target venue (to finalise citation style and page limit; current build is 11 pages).

---

## 3. REMAINING REVIEWER RISKS (top unresolved)

- **R1 (major) — Novelty framing.** Even sharpened, this remains a synthesis. The strongest defence is
  C1–C4 (protocol + technique-linked taxonomy + maturity synthesis + a transparently conceptual
  framework). A reviewer may still ask "why this review over existing audit-analytics reviews?" — the
  Introduction now answers this, but it is inherently arguable.
- **R2 (major) — Reproducibility.** Until the four methodology placeholders are filled with real
  search strings and reviewer/coding details, "PRISMA-informed" cannot be fully reproduced.
- **R3 (major) — Conceptual framework.** Eq. (1)/Algorithm 1 are unvalidated. This is now stated
  explicitly, but a reviewer may still view the framework as under-determined (weights/threshold not
  derived). Options: keep as conceptual (current), or remove the framework and present only the taxonomy.
- **R4 (major) — Grey literature.** Part of the evidence rests on industry/professional reports; tagging
  helps, but reviewers may still discount claims whose sole support is grey literature.
- **R5 (moderate) — Reference quality.** 15 entries lack complete metadata; two are stubs. This is a
  desk-reject risk if unresolved.
- **R6 (moderate) — Evidence for quantitative claims.** Exact percentages (30–50%, 73/68/87/23%) were
  removed. The remaining claims are qualitative and hedged; if reviewers want numbers, the underlying
  sources must be produced.
- **R7 (minor) — Citation style.** Bibliography is APA-like; the target venue may require IEEE numeric
  style with vol./no./pp. — conversion needs otherwise-missing metadata.

---

## 4. REFERENCE AUDIT

Verified programmatically: 67 `\bibitem` entries, 67 `\cite` keys, **0 undefined citations, 0 uncited
entries, 0 duplicate entries**.

### 4.1 Metadata completed via public-source verification

These entries, previously incomplete, were completed by locating the source online and are no longer
flagged:

| # | Resolved citation |
|---|------------------|
| 4  | Johnson-Rokosu & Enobi (2025), *Journal of Accounting and Financial Management*, 11(5), 93-117 |
| 8  | Darwish (Ed.) (2024), *Emerging Trends in Cloud Computing Analytics...*, IGI Global (ISBN 979-8-3693-0901-8) |
| 21 | Syam et al. (2025), *Economics and Business Quarterly Reviews*, 8(2), 49-66, doi:10.31014/aior.1992.08.02.662 |
| 28 | Abdelwahed & Abu-Musa (2024), *South African Journal of Accounting Research*, 38(2), 113-145, doi:10.1080/10291954.2023.2279751 |
| 31 | Lokanan (2024), *Research Square* preprint, doi:10.21203/rs.3.rs-5635767/v1 |
| 38 | Javaid (2024), *Integrated Journal of Science and Technology*, 1(8) — corrected from 1(3) |
| 49 | Akter (2025), Bachelor's thesis, Theseus repository (URN:NBN:fi:amk-202504227203) |
| 52 | Alex-Omiogbemi et al. (2024), *World Journal of Advanced Research and Reviews*, 24(3), 1155-1162, doi:10.30574/wjarr.2024.24.3.3752 |
| 57 | Oko-Odion & Udoh (2024), *International Journal of Science and Research Archive*, 13(2), 3077-3100, doi:10.30574/ijsra.2024.13.2.2549 |
| 63 | Sarkola (2023), Master's thesis, LUT University |
| 64 | Mai (2022), Master's thesis, Ca' Foscari University of Venice |
| 65 | George (2022), *World Journal of Advanced Engineering Technology and Sciences*, 7(1), 174-185 — malformed "10-30574" corrected |

### 4.2 Still flagged `[REFERENCE VERIFICATION REQUIRED]` (3 of 15 remain)

| # | Entry | Status |
|---|-------|--------|
| 15 | Wolniak (2024) | Source manuscript gave no title/venue. A likely candidate was found (Scientific Papers of Silesian Univ. of Technology, Organization & Management Series, No. 198, pp. 619-635), but its cybersecurity-analytics topic does not match the audit claim it supports. **Author must confirm or replace.** |
| 25 | Prosper (2025) | No journal venue exists; the work appears only as a self-archived paper (ResearchGate, March 2025). **Author to supply a venue or confirm as a preprint.** |
| 62 | Carreño (2024) | Only an institute/professional source (ICLBT) and a ResearchGate record were located; peer-review status unclear. **Author to confirm.** |

### 4.3 Other notes
- Author-name strings garbled in the source (Adepoju et al.; Balogun et al.) were reduced to surname +
  first initial; confirm against the originals.
- Duplicates present in the original manuscript (Akula & Garibay 2021; Pamisetty 2021; Pinto 2024;
  Sewpersadh 2025; Thakkar et al. 2025) were already consolidated to single entries.
- Bibliography is APA-like; convert to the venue's numeric IEEE style at typesetting. Some sources are
  theses, preprints or professional reports by nature, which is disclosed in the limitations.
