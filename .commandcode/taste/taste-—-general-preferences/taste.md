# Taste — General Preferences
- Prefers written deliverables as LaTeX `.tex` files formatted for IEEE publication (IEEEtran, conference style). Confidence: 0.85
- Cares strongly about research/output quality and rigor — expects careful, high-quality work rather than quick drafts. Confidence: 0.6
- When writing a paper modeled on a reference paper, wants a genuinely novel architecture/contribution ("similar but different"), not a close reproduction of the source. Confidence: 0.6
- Wants papers to include results/analysis figures (plots, charts) pulled from the actual experimental data to make the work more compelling than text and tables alone. Confidence: 0.6
- Prefers figures to be generated programmatically from a runnable Python script (e.g., a `.py` that emits images) rather than hand-made, and expects the script to be run to produce them. Confidence: 0.7
- Wants image files embedded by reference inside the document at the corresponding location (Markdown `![](path)` / `\includegraphics`), not just produced as loose files. Confidence: 0.55
- Prefers generated/derived output written to a NEW file (e.g., a separate `*_illustrated.md`) rather than overwriting the original source document. Confidence: 0.5
- Wants figures delivered as raster image files (e.g., high-resolution PNG) rather than PDF/vector or live TikZ, and expects `\includegraphics` to point at those image files. Confidence: 0.6
- Prefers concrete quantitative content over empty placeholders in drafts — when a table has TBD/blank fields, wants it filled with real, "reliable" values sourced from published research (with citations), not left empty; this extends to cells describing their own not-yet-run system, where they ask for values "based on other research" to stand in until the experiment is done. Confidence: 0.7
