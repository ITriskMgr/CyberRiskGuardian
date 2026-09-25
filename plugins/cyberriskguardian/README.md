# cyberriskguardian (plugin)

Scenario-driven cybersecurity risk assessment with the CyberRiskGuardian methodology.

## Components
| Type | Name | Purpose |
|---|---|---|
| Skill | `cyber-risk-assessment` | Full 19-step assessment workflow. Includes references (master spec, formulas, deliverables), scripts (`crg_calc.py`, `build_workbook.py`, `budget_calibrator.py`) and a data-model template. |
| Skill | `scenario-selection` | 20-step selection procedure with Sections A–E output and the 14 critical rules |
| Skill | `cybersecurity-governance` | The author's textbook *Introduction to Cybersecurity Governance* (3rd ed., v2.1g), one file per chapter with an index |
| Skill | `cyber-budget-calibration` | 4% / 7.8% / 12% of IT budget guideline, calibrated to appetite, with reconciliation and project budgets |
| Commands | `risk-assessment`, `select-scenarios`, `calibrate-budget`, `crg-calc`, `qc-review` | Entry points |
| Agents | `risk-scenario-analyst`, `risk-quantifier`, `risk-qc-reviewer` | Delegated generation and screening, quantification, and independent review |
| Output styles | `CyberRiskGuardian Analyst`, `CyberRiskGuardian Executive Brief` | Analyst voice with evidence markers; one-page board brief |
| Hook | SessionStart | Injects the methodology guardrails |
| MCP | `crg-calculator` (stdio, local Python) | `crg_risk`, `cvss4_score`, `budget_target`, `budget_check`, `sensitivity` |

## Setup
- Requirements: Python 3.10+ and `pip install cvss openpyxl "mcp>=1.2"`.
- For Word output, Node with `docx`; for recalculation and rendering, LibreOffice.
- Optional environment variables (defaults in brackets): `CRG_DEFAULT_APPETITE` [0.30], `CRG_FACTOR` [1000].
- No credentials or external services are required.

## Licence
© 2026 Marc-André Léger. Licensed under CC BY-NC 4.0; see `LICENSE`.

## Notes
- The formulas implement the CyberRiskGuardian Excel Guide v1.0c, s.11, unchanged.
- Estimates the agent produces are labelled for validation, and decisions stay with management.
