# CyberRiskGuardian Core

Free, scenario-driven cybersecurity risk assessment with the CyberRiskGuardian methodology. Core takes you from organizational context to a management summary: evidence and context, risk appetite, 20 candidate scenarios narrowed to the 10 most material, quantification with CVSS v4.0, and the decisions management must take.

## Install

```
/plugin install cyberriskguardian --marketplace ITriskMgr/CyberRiskGuardian
```

On older Claude Code versions:

```
/plugin marketplace add ITriskMgr/CyberRiskGuardian
/plugin install cyberriskguardian@cyberriskguardian-marketplace
```

In Claude Cowork, upload `cyberriskguardian.plugin` (a zip of this folder) through the plugin install flow.

## Components

| Type | Name | Purpose |
|---|---|---|
| Skill | `cyber-risk-assessment` | Core workflow; hands each step to the skills below |
| Skill | `crg-context-intake` | Evidence register, context, crown jewels, scope and dependencies; interview mode when there are no documents |
| Skill | `crg-risk-appetite` | Appetite on a 0–1 scale with rationale, and materiality criteria |
| Skill | `crg-scenario-builder` | Causal-chain candidates, 12-criterion screening, consolidation, selection, scenario development (Sections A–E) |
| Skill | `crg-quantify` | Pb(A), Pb(ψ,A), δe, δm, θ, μ(E) with confidence; CVSS v4.0; estimated and tolerated risk |
| Skill | `crg-summary-report` | Management summary with portfolio table, priorities and decisions |
| Commands | `crg-assess`, `crg-quick-scan`, `select-scenarios`, `crg-calc`, `qc-review` | Entry points |
| Agents | `risk-scenario-analyst`, `risk-qc-reviewer` | Delegated scenario screening and independent read-only review |
| Output styles | `CyberRiskGuardian Analyst`, `CyberRiskGuardian Executive Brief` | Analyst voice with evidence markers; one-page board brief |
| Hooks | SessionStart, PostToolUse, Stop | Methodology guardrails; checks that outputs carry the estimate, evidence-gap and decision markers |
| MCP | `crg-calculator` (local Python) | `crg_risk`, `cvss4_score` |
| Evals | `evals/` | One suite per skill, run with `claude plugin eval` |

The governance textbook is a separate free plugin, `crg-governance-textbook`, in the same marketplace.

## CRG Pro

Costed treatment portfolios, sensitivity analysis, the appetite-aligned cybersecurity budget, KRI design, the live-formula Excel workbook and the Word board report are in **CRG Pro Quantitative**, available under the CyberRiskGuardian Commercial or Educational Licence. See www.leger.ca.

## Setup

- Python 3.10+ and `pip install cvss "mcp>=1.2"`.
- Optional environment variables: `CRG_DEFAULT_APPETITE` (0.30), `CRG_FACTOR` (1000).
- No credentials or external services are needed.
- The skills, commands and agents work in Claude Code, Cowork and claude.ai. The `crg-calculator` MCP server and the hooks run locally, so they work in Claude Code and Cowork; on claude.ai the skills use the bundled script or a transparent manual calculation.

## What the plugin runs and sends

- **SessionStart hook:** prints `hooks/session-context.md` into the session.
- **PostToolUse and Stop hooks:** run `hooks/check_labels.py`, which reads only the file just written (Markdown, text, HTML, CSV, JSON or Word) or the last reply, and reports missing methodology markers. On Stop it asks Claude once to add a missing estimate marker.
- **MCP server:** `mcp/crg_server.py` performs deterministic calculations (CRG formulas, CVSS v4.0) locally.
- **Calculator script:** `skills/crg-quantify/scripts/crg_calc.py` reads the JSON data model you supply.
- No component makes network calls, stores data or collects telemetry.

## Licence

© 2026 Marc-André Léger. Community Edition under CC BY-NC 4.0; commercial use requires the CyberRiskGuardian Commercial Licence. See `LICENSE` and `NOTICE.md`.
