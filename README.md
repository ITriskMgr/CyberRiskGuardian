# CyberRiskGuardian Agent

AI-assisted, scenario-driven cybersecurity risk assessment for Claude Code, Claude Cowork and claude.ai. The free **CyberRiskGuardian Core** plugin acts as a Senior Cybersecurity Risk Analyst:

- building the evidence register, context and crown-jewel map;
- setting the risk appetite;
- generating, screening and selecting causal-chain scenarios;
- quantifying risk with the CyberRiskGuardian formulas and CVSS v4.0;
- comparing estimated with tolerated risk;
- writing the management summary.

Treatment portfolios, sensitivity analysis, budget calibration, KRI design, the live-formula workbook and the full board report are in **CRG Pro**, available under a commercial or educational licence.

People stay accountable for validation and risk decisions.

The latest version is CyberRiskGuardian with Threat Live-feed, available here: https://github.com/ITriskMgr/CyberRiskGuardian_ThreatIntel

Methodology and textbook © Marc-André Léger. This repository is the free **Community Edition**, licensed under **CC BY-NC 4.0**. Paid editions are available under a commercial or educational licence (see [Licence](#licence)). Packaged with Claude, version 2.0.0 (October 2026).

## What's inside

Version 2.0 turns this repository into a marketplace of separate free plugins.

```
CyberRiskGuardian/
├── .claude-plugin/marketplace.json     # cyberriskguardian-marketplace: lists the plugins below
├── plugins/
│   ├── cyberriskguardian/              # CyberRiskGuardian Core (free)
│   │   ├── skills/  cyber-risk-assessment, crg-context-intake, crg-risk-appetite,
│   │   │            crg-scenario-builder, crg-quantify, crg-summary-report
│   │   ├── commands/  crg-assess, crg-quick-scan, select-scenarios, crg-calc, qc-review
│   │   ├── agents/    risk-scenario-analyst, risk-qc-reviewer
│   │   ├── hooks/     SessionStart guardrails + check_labels.py (PostToolUse, Stop)
│   │   ├── mcp/       crg_server.py: crg_risk, cvss4_score
│   │   └── evals/     one suite per skill (claude plugin eval)
│   └── crg-governance-textbook/        # the governance textbook as a skill (free)
├── docs/                               # quick-start guides (EN, FR) and demo script
├── education/                          # teaching package for instructors and students (EN, FR)
├── examples/medibec/                   # worked example (fictional teaching case)
├── scripts/  sync-project-config.sh, build_evals.py, package-plugins.sh
├── CLAUDE.md  AGENTS.md  .claude/      # project mode for Claude Code
└── LICENSE  NOTICE.md
```

Paid plugins (CRG Pro Quantitative, and the Compliance Packs to come) are distributed from a separate private marketplace to licensed customers.

## Install

**Quick start:** see `docs/quickstart-en.md` or, en français, `docs/demarrage-rapide-fr.md`.

**Claude Code:**
```
/plugin install cyberriskguardian --marketplace ITriskMgr/CyberRiskGuardian
/plugin install crg-governance-textbook --marketplace ITriskMgr/CyberRiskGuardian
```
On older versions:
```
/plugin marketplace add ITriskMgr/CyberRiskGuardian
/plugin install cyberriskguardian@cyberriskguardian-marketplace
/plugin install crg-governance-textbook@cyberriskguardian-marketplace
```

**Claude Cowork:** upload `cyberriskguardian.plugin` and `crg-governance-textbook.plugin` (built with `bash scripts/package-plugins.sh`) through the plugin install flow.

**Project mode (Claude Code):** clone the repository and open it with `claude`; `CLAUDE.md`, `.claude/` and `.mcp.json` load automatically. Approve the `crg-calculator` MCP server when prompted.

**Prerequisites:** Python 3.10+ (`pip install -r requirements.txt`); Node 18+ with `npm install` for Word output; LibreOffice for rendering.

**Version 1.1.1** (single plugin with workbook, sensitivity and budget tools) remains available on the `release/1.x` branch.

## Teaching with CyberRiskGuardian

`education/` is a free teaching package, in English and French, for courses on cybersecurity risk management, governance, IT audit and business technology management:

- **Instructor Guide** — the method in one page, learning outcomes, tool tracks, four integration models from a 3-hour workshop to a full term, lab facilitation notes, generative AI policy.
- **Student Lab Workbook** — twelve labs on the MediBec teaching case, with self-check answers.
- **Assessment Kit** — briefs, 0–10 rubrics and self-checks for the risk assessment, IAM policy and audit plan.
- **Case Library** — five fictional business cases (student versions, v2.0c).
- **Introductory slide deck** and the **blank CyberRiskGuardian Excel model** (`education/templates/`).

Start with `education/README.md` (or `education/LISEZMOI.md`). Answer keys, the graded quiz bank, the instructor annexes of the cases and the final-exam case form the Educator Edition, supplied with the Educational Licence.

## Use

| Ask or run | What happens |
|---|---|
| "Run a cyber risk assessment for <org>" / `/crg-assess` | Core workflow: context, appetite, scenarios, quantification, management summary |
| "Quick cyber risk scan" / `/crg-quick-scan` | 20 candidates, 10 scenarios, pre-treatment verdicts in one table |
| "Select the most material scenarios" / `/select-scenarios` | Sections A–E scenario selection |
| `/crg-calc 0.75 0.5 0.7 0.95 0.35 0.95 0.75 0.75 CVSS:4.0/…` | KRIs for one scenario |
| "Explain the three lines model" | Answer from the textbook with chapter citation |
| `/qc-review <files>` | Independent read-only quality review |

Calculator: `python3 plugins/cyberriskguardian/skills/crg-quantify/scripts/crg_calc.py data.json [--appetite 0.30] [--factor 1000] [--json]`. The data model is `plugins/cyberriskguardian/skills/crg-quantify/assets/scenario-template.json`.

Label check: `python3 plugins/cyberriskguardian/hooks/check_labels.py report.md` lists missing methodology markers.

## Maintain and release
1. Edit only under `plugins/`, then run `bash scripts/sync-project-config.sh`.
2. Validate each plugin and the marketplace with `claude plugin validate`.
3. Run `python3 scripts/build_evals.py`, then the eval suites:
   - `claude plugin eval plugins/cyberriskguardian --allow-tools Bash Write`
   - `claude plugin eval plugins/crg-governance-textbook`
4. Regression: `crg_calc.py examples/medibec/source/medibec30.json` must report total estimated 84,491 and residual 41,104.
5. Bump the versions (semver) in each changed `plugin.json` and in `marketplace.json`; current version 2.0.0.

## Deliberately excluded
- `CLAUDE.local.md` and `.claude/settings.local.json`, which hold private preferences.
- Session transcripts (`~/.claude/projects/`).
- Personal memory, `.env` files and credentials.
- Organization data other than the fictional MediBec teaching case.
- Third-party course materials held in the author's project knowledge base. The author's own methodology specifications and governance textbook are bundled.

## Licence
© 2026 Marc-André Léger. CyberRiskGuardian uses an open-core model.

| Edition | Licence | Who | Price |
|---|---|---|---|
| **Community Edition** (this repository) | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (SPDX `CC-BY-NC-4.0`) | Students, non-profit organisations and anyone using it for non-commercial purposes | Free |
| **Paid editions** for colleges and universities | CyberRiskGuardian Educational Licence | Colleges, CEGEPs and universities, as institutions | Reduced educational price, installation and customization included |
| **Paid editions** for other organizations, and commercial use of the Community Edition | CyberRiskGuardian Commercial Licence | Companies, consultants, MSSPs, government bodies | Professional, Team, Enterprise or MSSP tiers |

The Community Edition in this repository covers:
- the CyberRiskGuardian Core and CRG Governance Textbook plugins (methodology, skills, agents, commands, hooks, calculator, evals);
- the governance textbook;
- the MediBec example.

You may share and adapt it for non-commercial purposes with attribution. Using it to deliver paid work, or using it inside a company or government body, is commercial use and requires the Commercial Licence.

The paid editions are not in this repository. They include the Pro Quantitative, Policy Suite, SecOps Bridge, Third-Party Risk and Audit & Assurance plugins, Compliance and Sector Packs, the Educator Edition, the hosted CRGO ontology service and CyberRiskGuardian Desktop versions released after 1.7.0 Beta.

See `LICENSE` for the CC BY-NC 4.0 legal code, `NOTICE.md` for when a paid licence is needed and for attribution wording, and www.leger.ca for tiers and the educational-licence contact.
