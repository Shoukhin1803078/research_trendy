I want you to substantially improve the quality of this research paper for conference submission.

Paper file:
audit_fraud_detection.pdf

First, carefully inspect the entire paper and understand its current structure, methodology, claims, figures, tables, equations, and references.

IMPORTANT:
- Do NOT blindly rewrite the paper.
- Do NOT invent experimental results, datasets, citations, statistics, authors, publication venues, or findings.
- Do NOT fabricate evidence to make the paper appear stronger.
- Preserve all claims that are actually supported by the existing manuscript.
- If a claim is too strong for the available evidence, weaken/rephrase it appropriately.
- If something requires new evidence that is not available, explicitly mark it as [NEEDS EVIDENCE] rather than fabricating it.
- Maintain the paper's core topic and contribution.
- The final paper should remain a REVIEW / INTEGRATIVE REVIEW paper unless genuine empirical experiments are available in the source material.
- Do not pretend that the proposed framework has been experimentally validated.

TARGET:
Improve this paper from a roughly borderline conference-quality review paper into a substantially stronger, rigorous, publication-ready conference manuscript.

The paper is:
"Leveraging Business Analytics for Audit and Fraud Detection: A Review of Digital Tools, Red-Flag Indicators and Practical Implications"

CORE PROBLEMS TO FIX
====================

1. SHARPEN THE RESEARCH CONTRIBUTION
------------------------------------
The current paper has a potentially useful combination of:
- business analytics for audit/fraud detection
- digital audit tools
- red-flag taxonomy
- AI-enabled red-flag prioritization
- analytics maturity
- organizational readiness
- practical implementation

However, the novelty is currently not sufficiently sharp.

Rewrite the Introduction and Contribution section so that it clearly answers:

1. What exact research gap exists?
2. Why are existing reviews/frameworks insufficient?
3. What does this paper add?
4. What is the specific conceptual contribution?
5. What is the practical contribution?

Make the contributions explicit and non-generic.

The strongest potential contribution appears to be the integration of:
- a structured red-flag taxonomy,
- analytics capabilities,
- AI-assisted prioritization,
- audit workflow,
- organizational/implementation factors.

Make this contribution coherent without overstating novelty.

Do NOT claim "first", "novel", "unique", etc. unless the manuscript actually provides evidence for such claims.


2. STRENGTHEN THE REVIEW METHODOLOGY
-------------------------------------
This is one of the most important weaknesses.

The manuscript currently describes:
- 84 sources
- Scopus
- Web of Science
- IEEE Xplore
- SSRN
- Google Scholar
- 2015–2025
- PRISMA-based screening
- initial records and exclusion process

But the methodology is not sufficiently reproducible.

Improve the methodology section by making the following information explicit wherever the existing manuscript supports it:

- research questions/objectives
- databases searched
- publication period
- search strategy
- inclusion criteria
- exclusion criteria
- duplicate-removal procedure
- title/abstract screening
- full-text screening
- final included sources
- thematic/coding process
- synthesis procedure

VERY IMPORTANT:
Do NOT invent exact search strings, reviewer counts, inter-rater reliability, quality scores, or screening software if they are not present.

If critical methodological information is missing, insert a clearly marked placeholder such as:

[AUTHOR INPUT REQUIRED: provide exact database search strings]

or

[AUTHOR INPUT REQUIRED: specify whether screening was performed by one or multiple reviewers]

Do not fabricate these details.


3. MAKE THE 84-SOURCE SELECTION LOGIC CONSISTENT
------------------------------------------------
Check the numerical flow carefully.

The manuscript currently discusses:
- 297 initial records
- 272 after duplicate removal
- 118 excluded during title/abstract screening
- 70 excluded during full-text screening
- 84 final sources

Verify all arithmetic and wording.

Make the PRISMA explanation internally consistent.

Clearly distinguish:
- academic literature
- industry reports
- professional publications
- other sources

If the methodology mixes peer-reviewed papers with industry/professional sources, explain the distinction clearly instead of presenting all sources as equivalent evidence.


4. DO NOT PRESENT ILLUSTRATIVE NUMBERS AS EMPIRICAL RESULTS
-------------------------------------------------------------
This is a major conference-reviewer risk.

Figures 2 and 3 currently contain numerical/visual summaries that are explicitly described as illustrative/indicative rather than meta-analysis results.

Reviewers may interpret these as fabricated quantitative evidence.

Do one of the following:

Preferred:
- Replace the illustrative quantitative-looking figures with conceptual diagrams, evidence matrices, or qualitative synthesis figures.

OR, if retaining them:
- Make the captions and surrounding text extremely explicit that the values are conceptual/illustrative and are NOT empirical measurements, pooled estimates, or quantitative findings.

Do not allow the paper to visually imply statistical evidence that was not actually calculated.

If you think a figure should be removed completely, recommend removal rather than inventing replacement data.


5. IMPROVE THE RED-FLAG TAXONOMY
--------------------------------
The five red-flag categories are potentially one of the strongest parts of the manuscript:

- transaction irregularities
- behavioral anomalies
- structural anomalies
- documentation discrepancies
- relational fraud patterns

Make this taxonomy academically stronger.

For each category, clearly explain:
- definition
- what it detects
- typical examples
- relevant analytics techniques
- audit relevance
- limitations

Create a stronger table if appropriate.

Do not invent specific empirical findings.

Make sure the categories are conceptually distinct and do not substantially overlap.


6. STRENGTHEN THE AI-ENABLED RED-FLAG PRIORITIZATION FRAMEWORK
---------------------------------------------------------------
The manuscript proposes:

s(r) = w1 a(xr) + w2 b(xr) + w3 c(xr) + w4 f(xr)

with weights summing to 1 and a threshold τ.

Improve the mathematical and conceptual presentation.

Clearly define:
- every variable
- every feature
- normalization assumptions
- weight constraints
- threshold interpretation
- what happens above/below threshold
- how the framework would theoretically integrate with an audit workflow

IMPORTANT:
This is a PROPOSED CONCEPTUAL FRAMEWORK, not a validated ML model.

Do not imply that it has demonstrated improved fraud detection accuracy.

Explicitly distinguish:
"conceptual framework"
from
"empirically validated model."


7. IMPROVE THE ANALYTICS MATURITY MODEL
----------------------------------------
Strengthen the analytics maturity discussion.

Clearly distinguish levels such as:
- descriptive analytics
- diagnostic analytics
- predictive analytics
- prescriptive / AI-assisted analytics

Explain what each level contributes to fraud/audit workflows.

Connect maturity to:
- data availability
- organizational capability
- technology
- governance
- workforce skills
- explainability
- audit integration

Avoid generic business-analytics language.


8. IMPROVE TOOL COMPARISON
--------------------------
The paper discusses:
- ACL
- IDEA
- Power BI
- Tableau
- Python
- and mentions R in some places

Check for inconsistency between the prose and tables.

If the table compares five tools, make sure the prose also consistently describes five tools.

Create a more academically useful comparison using dimensions such as:
- primary use
- analytical capability
- automation
- scalability
- anomaly detection
- visualization
- programmability
- audit integration
- explainability
- limitations

Do not make unsupported claims about commercial tools.


9. STRENGTHEN THE DISCUSSION
----------------------------
The Discussion currently repeats some findings.

Make it more analytical rather than descriptive.

For each major finding, discuss:
- what the synthesis suggests
- why it matters
- implications for auditors
- implications for organizations
- implications for AI/analytics adoption
- limitations of the evidence

Clearly distinguish:
- evidence from the reviewed literature
- authors' conceptual interpretation
- proposed framework

Do not present author interpretation as empirical evidence.


10. REDUCE OVERCLAIMING
-----------------------
Search the entire manuscript for language such as:

"proves"
"demonstrates"
"significantly improves"
"substantial improvement"
"highly accurate"
"effective"
"successful"
"superior"
"better"
"ensures"
"guarantees"
"revolutionizes"

Where the review evidence does not support strong causal/evaluative language, replace it with appropriately cautious academic language such as:

"suggests"
"indicates"
"the reviewed literature reports"
"is associated with"
"may support"
"has the potential to"
"can facilitate"

Do not weaken claims unnecessarily when they are genuinely supported.


11. STRENGTHEN THE LIMITATIONS
--------------------------------
The current limitations are useful but should be more precise.

Clearly acknowledge:

- no primary empirical validation
- heterogeneous literature
- publication/selection bias
- English-language bias
- limitations of database coverage
- possible grey-literature differences
- conceptual rather than empirical validation of the proposed framework
- inability to establish causal effects from an integrative review

Make the limitations consistent with the actual methodology.


12. IMPROVE THE CONCLUSION
--------------------------
Rewrite the conclusion so it does NOT simply repeat the abstract.

It should contain:

1. What the review established
2. What the proposed framework contributes
3. What practitioners can take from the synthesis
4. What remains unvalidated
5. What future empirical research should test

Do not claim empirical validation.


13. REFERENCES — VERY IMPORTANT
--------------------------------
Audit the entire reference list.

Identify:
- incomplete references
- inconsistent formatting
- missing venue information
- missing volume/issue/pages
- questionable bibliographic entries
- inconsistent author formatting
- inconsistent DOI formatting
- references cited in text but missing from bibliography
- bibliography entries never cited in the manuscript

Known suspicious/incomplete entries include:
- [15]
- [28]
- [31]
- [49]
- [52]
- [57]
- [62]
- [64]
- [65]

Inspect these carefully.

Do NOT invent missing bibliographic information.

For anything that cannot be verified from the manuscript, use:

[REFERENCE VERIFICATION REQUIRED]

Do not hallucinate DOI, journal, volume, pages, or authors.

Standardize the bibliography to the target conference citation style.


14. CHECK INTERNAL CONSISTENCY
------------------------------
Perform a full consistency audit.

Check:
- section numbering
- table numbering
- figure numbering
- equation numbering
- terminology
- number of tools
- number of sources
- dates
- framework names
- abbreviations
- references
- citations
- claims
- captions
- cross-references

Especially check whether the manuscript consistently uses:
- "review"
- "integrative review"
- "systematic review"
- "PRISMA-based review"

Do not call it a systematic review unless the methodology actually supports that designation.


15. IMPROVE TABLES
-----------------
Review all tables.

Make tables more information-dense and academically useful.

Avoid tables that simply repeat prose.

Prioritize tables such as:

Table 1:
Review methodology / source selection

Table 2:
Analytics tools and capabilities

Table 3:
Red-flag taxonomy

Table 4:
Evidence synthesis by fraud/audit problem

Table 5:
Implementation challenges and mitigation strategies

Only create a table when the underlying information actually exists in the manuscript.


16. IMPROVE FIGURES
------------------
Make figures academically meaningful.

Preferred conceptual figures:

Figure 1:
Overall analytics-driven audit/fraud detection framework

Figure 2:
Red-flag taxonomy

Figure 3:
AI-assisted red-flag prioritization workflow

Avoid decorative diagrams.

Do not create fake quantitative charts.

If existing figures are useful, improve their labels, captions, and explanation.


17. IMPROVE ACADEMIC WRITING
---------------------------
Edit the entire manuscript for:

- concise academic English
- logical transitions
- reduced repetition
- precise terminology
- stronger topic sentences
- less marketing-style language
- consistent tense
- consistent terminology
- shorter sentences where needed

Do NOT unnecessarily increase word count.

Prefer clarity and precision over complexity.


18. IMPROVE ABSTRACT
--------------------
Rewrite the abstract using this structure:

Background/problem
→ objective
→ methodology
→ key synthesis/findings
→ proposed contribution
→ practical implications
→ limitations/future direction

The abstract must clearly indicate that this is a review/integrative synthesis and that the proposed AI framework is conceptual unless empirical validation exists.


19. IMPROVE INTRODUCTION
------------------------
The Introduction should follow:

Problem
→ why traditional audit/fraud detection faces challenges
→ rise of business analytics/AI
→ existing research limitation/gap
→ why an integrated synthesis is needed
→ research objective/questions
→ contributions
→ paper organization

Avoid generic statements that are not supported by citations.


20. CONFERENCE REVIEWER SIMULATION
----------------------------------
After editing, perform a strict conference-review simulation.

Evaluate:

A. Novelty
B. Technical/methodological rigor
C. Literature coverage
D. Reproducibility
E. Contribution
F. Clarity
G. Reference quality
H. Figure/table quality
I. Validity of claims
J. Conference suitability

Do NOT give a political-style or arbitrary score.

Instead, identify:
- Major weaknesses
- Minor weaknesses
- Reviewer objections
- Required revisions
- Optional improvements

Pay special attention to the question:

"Why should a conference accept this review paper instead of the many existing reviews on fraud detection, audit analytics, and AI?"

The manuscript must answer that convincingly.


FINAL DELIVERABLE
=================

Work directly on the source/LaTeX project if available.

Produce:

1. A revised, polished version of the paper.
2. A detailed CHANGELOG explaining every major modification.
3. A list called "AUTHOR INPUT REQUIRED" containing only things that cannot be safely fixed without additional information/evidence.
4. A section called "REMAINING REVIEWER RISKS" with the top unresolved risks.
5. A final reference audit report.

IMPORTANT:
- Never fabricate evidence.
- Never fabricate citations.
- Never fabricate experimental results.
- Never convert conceptual proposals into claimed empirical findings.
- Never hide methodological weaknesses.
- Preserve the core contribution.
- Prioritize scientific validity over making the paper sound impressive.

Before making major structural changes, inspect the complete manuscript and establish its current structure and evidence base.