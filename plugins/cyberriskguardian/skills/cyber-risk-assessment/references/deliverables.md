# Deliverable conventions

## Formats
- **Management report** — Word (.docx) when the user asks for Word; otherwise the environment's document format. Build Word files with the `docx` npm package (docx-js): US Letter or A4 as requested, Arial, navy headings, tables with explicit DXA widths, `ShadingType.CLEAR`, numbered/bulleted lists via numbering config, a table of contents with `features: { updateFields: true }`. Render to PDF (LibreOffice headless) and inspect pages before delivering.
- **Workbook** — Excel (.xlsx) built with openpyxl using **live formulas** (never hard-coded results), Arial, blue = inputs, black = formulas, green = cross-sheet links, yellow fill = key assumptions. Recalculate with LibreOffice headless and confirm **zero formula errors**; then cross-check key cells against `scripts/crg_calc.py`.
- Use `scripts/build_workbook.py <scenarios.json> <out.xlsx>` for the standard CRG workbook (Parameters, Analyse, Inputs, Portfolio, Sensitivity, Case_View, Scenarios, Candidates, KRIs, Evidence).
- Keep one data model (JSON, see `assets/scenario-template.json`) as the single source of truth for both the workbook and the report so numbers cannot diverge.

## Report outline
Executive summary · scope and methodology (incl. evidence register and formulas) · organizational and technology context (crown jewels) · risk appetite (with appetite sensitivity table) · scenario development and selection (candidates, screening, consolidation, watch list) · scenario portfolio table · detailed scenarios (statement, stakeholders, background, threat, vulnerabilities, assets, existing controls, narrative, event sequence, consequences, parameter table with rationale/evidence/confidence, CVSS vector and metric rationale, calculation table, treatment and result callout) · cross-scenario analysis · sensitivity · prioritization and treatment portfolio · cybersecurity budget · implementation roadmap · KRIs · cross-check with any organization-native scoring method · assumptions, limitations, methodological notes · management decisions required.

## Scenario-selection workbook (Sections A–E)
README · A_Context · A_AssetMap · A_Dependencies · B_Candidates · C_Weights · C_Screening (12 criteria 1–5, totals, weighted scores, escalation flags, evidence basis, confidence, disposition, family) · C_Sensitivity (family score = MAXIFS of member candidates under each weighting scheme; ranks; "selected by judgement" and "explicit review" flags) · D_Portfolio · D_WhyTest · D_QuantPrep · D_Validation · E_Coverage · E_Quality · Changes. A reference implementation is in `examples/medibec/source/build_sel.py` of the repository.

## Delivery
- State results in one or two sentences; do not recap every step.
- Put the key numbers (counts above/at/below tolerance, portfolio cost, budget %) and the decisions required in the reply.
- Flag every judgement call where the selection or recommendation deviates from the scores.
- Keep deliverable file names stable and descriptive (`<Org>_Cybersecurity_Risk_Assessment[_variant].docx`, `<Org>_CyberRiskGuardian_Workbook[_variant].xlsx`, `<Org>_Scenario_Selection.xlsx`).
