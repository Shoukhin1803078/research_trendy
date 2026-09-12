# Harness Engineering — Paper Collection

Downloaded 2026-09-10. "Harness engineering" is the emerging discipline of designing the
model-external runtime layer of LLM agents: how context is constructed, tools are invoked,
results are verified, state is persisted, and failures are recovered. This folder collects
primary sources (papers + key engineering write-ups) found via web search.

## Downloaded PDFs

| # | Title | Source | Pages | File |
|---|-------|--------|-------|------|
| 1 | Agentic Harness Engineering: Observability-Driven Automatic Evolution of Coding-Agent Harnesses | arXiv:2604.25850 | 35 | `arxiv-2604.25850-agentic-harness-engineering.pdf` |
| 2 | Self-Harness: Harnesses That Improve Themselves | arXiv:2606.09498 | — | `arxiv-2606.09498-self-harness.pdf` |
| 3 | TTHE: Test-Time Harness Evolution | arXiv:2607.08124 | 15 | `arxiv-2607.08124-tthe-test-time-harness-evolution.pdf` |
| 4 | Harness Engineering: Anatomy, Architecture, and Evolution of Coding Agents — A Source-Code Study of Eleven Systems | arXiv:2609.00006 | 83 | `arxiv-2609.00006-anatomy-architecture-evolution.pdf` |
| 5 | HarnessDev: Can LLMs Create and Evolve Their Own Agent Harness? | arXiv:2609.01437 | 41 | `arxiv-2609.01437-harnessdev.pdf` |
| 6 | Harness-Native Software Engineering: The Control Plane of Coding Agents | research.chaitanya.science | 29 | `harness-native-software-engineering.pdf` |
| 7 | Agent Harness Engineering: A Survey | picrew.github.io/LLM-Harness | — | `agent-harness-engineering-survey.pdf` |

## Notes on each

1. **Agentic Harness Engineering (AHE)** — automates harness-level evolution using
   "matched observability pillars" across component editing, trajectory inspection, and
   decision making. Treats harness engineering as a manual craft to be automated.
2. **Self-Harness** — a paradigm where an LLM-based agent improves its own operating
   harness without human engineers or stronger external agents.
3. **TTHE: Test-Time Harness Evolution** — argues agent behavior depends on the harness
   (context construction, tool calls, verification, recovery) and evolves it at test time
   rather than freezing a workflow after deployment.
4. **Anatomy, Architecture, and Evolution** — source-code anatomy of eleven production
   coding harnesses (incl. Claude Code); the largest empirical study in the set.
5. **HarnessDev** — asks whether LLMs can create and evolve their own agent harness while
   model weights stay fixed.
6. **Harness-Native Software Engineering** — formalizes the runtime layer as the "Agent
   Harness Control Plane" with eight functions (context ingress, action mediation,
   execution substrate, state persistence, verification/review, recovery/debugging,
   delegation, ...).
7. **Agent Harness Engineering: A Survey** — ETCLOVG seven-layer taxonomy plus a catalog
   of 170+ open-source agent-harness projects.

## Not downloaded (blocked)

- **Harness Engineering: A Governance Framework for AI-Driven Software Engineering**
  (Kim, Jiun — Hanyang University). Zenodo record `19166436`, DOI `10.2139/ssrn.6372119`.
  Direct PDF download returned HTTP 403 (bot/IP restriction).
  Abstract: defines a harness as a governance system for structural consistency of code
  artifacts, organized along three dimensions — Context, Constraint, Convergence — with
  the goal of "structural idempotence".

## Related non-PDF resources

- OpenAI — *Harness engineering: leveraging Codex in an agent-first world*:
  https://openai.com/index/harness-engineering/
- Anthropic — *Harness design for long-running application development*:
  https://www.anthropic.com/engineering/harness-design-long-running-apps
- Curated lists: https://github.com/walkinglabs/awesome-harness-engineering ,
  https://github.com/ai-boost/awesome-harness-engineering
