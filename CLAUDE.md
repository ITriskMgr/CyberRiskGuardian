# CLAUDE.md — CyberRiskGuardian

@AGENTS.md

## Claude Code specifics

### Components in this project
| Component | Location | Purpose |
|---|---|---|
| Skills | `.claude/skills/` (source: `plugins/cyberriskguardian/skills/`) | `cyber-risk-assessment`, `scenario-selection`, `cyber-budget-calibration`, `cybersecurity-governance` (the author's textbook, one reference file per chapter). They auto-trigger on risk-assessment, scenario-selection, budget and governance-concept requests. |
| Commands | `.claude/commands/` | `/risk-assessment`, `/select-scenarios`, `/calibrate-budget`, `/crg-calc`, `/qc-review` |
| Subagents | `.claude/agents/` | `risk-scenario-analyst` (candidates and screening), `risk-quantifier` (parameters, CVSS, KRIs, workbook), `risk-qc-reviewer` (independent read-only review) |
| Output styles | `.claude/output-styles/` | `CyberRiskGuardian Analyst` (default via settings), `CyberRiskGuardian Executive Brief` |
| Rules | `.claude/rules/` | Evidence discipline and data protection (always); workbook, report and methodology-maintenance rules (path-gated) |
| MCP | `.mcp.json` | `crg-calculator` — tools `crg_risk`, `cvss4_score`, `budget_target`, `budget_check`, `sensitivity` |
| Settings | `.claude/settings.json` | Model, output style, environment defaults, permissions, SessionStart guardrail hook |

### How to work
- For a full assessment, follow the `cyber-risk-assessment` skill. It hands off to the other skills:
  - scenario selection → `scenario-selection`;
  - budget → `cyber-budget-calibration`.
- Use the MCP tools or `scripts/crg_calc.py` for every number, and never compute KRIs or CVSS scores by hand.
- Delegation:
  - **`risk-scenario-analyst`** — the document-heavy scenario generation and screening step.
  - **`risk-quantifier`** — quantification and building the workbook.
  - **`risk-qc-reviewer`** — before any board-level delivery.
- Keep outputs private until the user decides to share them.
- For documents the user will edit, ask which format they want (Word/Excel files or the environment's document format).

### Environment defaults (from settings.json `env`)
| Variable | Default |
|---|---|
| `CRG_DEFAULT_APPETITE` | 0.30 |
| `CRG_FACTOR` | 1000 |
| `CRG_HORIZON_MONTHS` | 12 |
| `CRG_CVSS_VERSION` | 4.0 |

Override them in `.claude/settings.local.json`, which is never committed.
