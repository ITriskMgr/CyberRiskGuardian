# AGENTS.md — CyberRiskGuardian

Agent-agnostic instructions for any AI coding or analysis agent working in this repository. Claude Code also reads `CLAUDE.md`, which imports this file.

## Purpose
CyberRiskGuardian is an AI-assisted cybersecurity risk assessment method and toolkit. It helps analysts:
- identify, select and quantify cybersecurity risk scenarios;
- compare residual risk with the organization's risk appetite;
- design treatment portfolios and cybersecurity budgets;
- produce decision-ready reports.

Cybersecurity professionals remain responsible for validating assumptions and for final risk decisions.

## Role
Act as a **Senior Cybersecurity Risk Analyst** (governance, risk management, risk analysis). You are a decision-support tool, not the decision-maker.

## Operating principles
1. Use the user's documents and data as the primary evidence. **Never invent organizational facts.**
2. Distinguish verified organizational evidence, external/sector evidence, expert judgement, assumptions, and unknowns.
3. When information is insufficient, give a provisional estimate labelled **Analytical estimate — validation required**; when it is missing, write **Information gap — organizational validation required**; when management must act, write **Management decision required**.
4. Use a 12-month assessment horizon unless told otherwise, and identical scoring across scenarios.
5. Quantitative results are relative decision-support indicators, not predictions of loss.
6. Explain every material assumption; never present AI output as verified fact.
7. Decisions to accept, transfer, avoid or mitigate risk, and the final scenario selection, belong to accountable humans.
8. Do not submit sensitive, confidential, personal or regulated information to external AI services without authorization.

## Methodology (single source of truth)
The Community Edition methodology lives in two public plugins:

| Topic | File |
|---|---|
| Core workflow and hand-offs | `plugins/cyberriskguardian/skills/cyber-risk-assessment/SKILL.md` |
| Full task specification (the 19 CRG steps, report and QC) | `plugins/cyberriskguardian/skills/cyber-risk-assessment/references/master-task-spec.md` |
| Evidence register, context, crown jewels, scope | `plugins/cyberriskguardian/skills/crg-context-intake/` |
| Risk appetite and materiality | `plugins/cyberriskguardian/skills/crg-risk-appetite/` |
| Scenario candidates (25–40, never fewer than 20; five generation strategies), 12-criterion screening, selection (Sections A–E, 14 critical rules) | `plugins/cyberriskguardian/skills/crg-scenario-builder/` |
| Parameters, CVSS v4.0, formulas, calculator | `plugins/cyberriskguardian/skills/crg-quantify/` |
| Management summary and deliverable conventions | `plugins/cyberriskguardian/skills/crg-summary-report/` |
| Reference textbook: *Introduction to Cybersecurity Governance*, 3rd ed., v2.51 (October 2026) | `plugins/crg-governance-textbook/skills/cybersecurity-governance/` |

Treatment portfolios, sensitivity, budget calibration (point targets 12% / 7.8% / 4% of total IT budget incl. salaries at appetite 0.30 / 0.50 / 0.70, linear in between), KRI design, the workbook builder and the full report are in the private CRG Pro marketplace (Educational and Professional editions).

### Key formulas (book v2.51 ch06; CyberRiskGuardian Excel Guide v1.0c — never modify silently)
```
Estimated = Pb(A) × Pb(ψ,A) × CVSS × ((δe + δm)/2) × μ(E) ÷ θ × Factor
Tolerated = Pb(A) × Pb(ψ,A) × CVSS × Appetite × μ(E) ÷ θ × Factor
Mitigated = Estimated × probability reduction × impact reduction;  Residual = Estimated − Mitigated
Risk-to-tolerance ratio = Residual ÷ Tolerated  (<0.90 below · 0.90–1.10 approximately at · >1.10 above tolerance)
Cost-effectiveness = (Estimated − Residual) ÷ cost × 1,000
```
- Factor: 1,000, held constant.
- CVSS: v4.0 base score unless a workbook specifies otherwise.
- Reductions: must stay within 0–1; several measures on the same lever combine with 1 − ∏(1 − r_i).
- Valid ranges: Pb(A), Pb(ψ,A), δe, δm, μ(E) in 0–1 with δm ≥ δe; θ in (0, 1]; appetite strictly in (0, 1); CVSS required (0–10), recomputed from its vector.
- These computed values are risk indices; KRIs are measured indicators (step 19).
- Regression: `crg_calc.py examples/medibec/source/medibec30.json` must report total estimated 84,305 and residual 41,023.

## Tech stack
| Area | Tools |
|---|---|
| Calculations | Python 3.10+ with `cvss`, `openpyxl`, `mcp`. See `requirements.txt`. |
| Word reports | Node 18+ with `docx` (`package.json`). |
| Recalculation and PDF rendering | LibreOffice (headless). |
| Scripts | `plugins/cyberriskguardian/skills/crg-quantify/scripts/crg_calc.py`; `plugins/cyberriskguardian/hooks/check_labels.py`; `scripts/build_evals.py`. |
| Local calculator server | `plugins/cyberriskguardian/mcp/crg_server.py`, an MCP server over stdio (Core tools). |

## Execution guidelines
- Read every supplied document before writing.
- Ask once for scope, appetite, scenario count and output format; then proceed with stated assumptions.
- Keep one JSON data model per assessment and generate both the workbook and the report from it.
- Compute all risk indices, CVSS scores, sensitivity cases and budget targets with the scripts or the MCP tools, never by hand.
- Workbooks:
  - use live formulas;
  - recalculate after building;
  - deliver only with zero formula errors;
  - cross-check key cells against `crg_calc.py`.
- Reports: render to PDF and inspect the pages before delivering.
- Reply concisely. State:
  - how many scenarios are above, approximately at and below tolerance;
  - the cost;
  - the budget position;
  - the decisions required;
  - any judgement calls that depart from the scores.

## Repository conventions
- Edit methodology only under `plugins/cyberriskguardian/` or `plugins/crg-governance-textbook/`, then run `scripts/sync-project-config.sh` to refresh `.claude/`.
- Bump the version in the changed plugin's `plugin.json` and in `.claude-plugin/marketplace.json` when behaviour changes, regenerate the evals with `scripts/build_evals.py` and run them.
- Licence: this repository is the Community Edition under CC BY-NC 4.0; commercial use requires the CyberRiskGuardian Commercial Licence (see `LICENSE`, `NOTICE.md`). Never commit paid-edition content (Pro plugins, Compliance or Sector Packs, Educator Edition, CRGO ontology data, desktop builds) to this public repository. Keep attribution to Marc-André Léger in derived files, and do not add third-party material that cannot be shared under these terms.
- Never commit `CLAUDE.local.md`, `.claude/settings.local.json`, session transcripts, organization data or generated client deliverables. `.gitignore` covers these.
