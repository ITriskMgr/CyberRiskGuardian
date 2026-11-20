# CyberRiskGuardian Agent

> **This edition is retired (19 November 2026).** CyberRiskGuardian Core (last release: Core plugin 2.2.1, with the
> governance textbook plugin 2.1.1) is no longer maintained. Use **CyberRiskGuardian with Threat Live-feed**
> ([`ITriskMgr/CyberRiskGuardian_ThreatIntel`](https://github.com/ITriskMgr/CyberRiskGuardian_ThreatIntel)): the same
> formulas and the same MediBec totals, plus dated threat-context calibration and the textbook skill. Existing installs keep
> working; disable this plugin (`claude plugin disable cyberriskguardian`) before enabling Threat Live-feed, so two methods
> never answer the same request. This repository stays readable under CC BY-NC 4.0.

AI-assisted, scenario-driven cybersecurity risk assessment for Claude Code, Claude Cowork and claude.ai. The free **CyberRiskGuardian Core** plugin acts as a Senior Cybersecurity Risk Analyst:

- building the evidence register, context and crown-jewel map;
- setting the risk appetite;
- generating, screening and selecting causal-chain scenarios;
- quantifying risk with the CyberRiskGuardian formulas and CVSS v4.0;
- comparing estimated with tolerated risk;
- writing the management summary.

Treatment portfolios, sensitivity analysis, budget calibration, KRI design, the live-formula workbook and the full board report are in **CRG Pro Quantitative**, part of the **Educational edition** (Educational Licence, for institutions) and the **Professional editions** (Commercial Licence).

People stay accountable for validation and risk decisions.

The latest version is CyberRiskGuardian with Threat Live-feed, available here: https://github.com/ITriskMgr/CyberRiskGuardian_ThreatIntel

Methodology and textbook © Marc-André Léger. This repository is the free **Community Edition**, licensed under **CC BY-NC 4.0**. The Educational and Professional editions are available under the Educational or Commercial Licence (see [Licence](#licence)). Packaged with Claude: Core 2.2.1 and textbook plugin 2.1.1 (November 2026), aligned with the textbook *Introduction to Cybersecurity Governance*, 3rd ed., v2.51b (the corrected v2.51). Release notes: `CHANGELOG.md`.

## What's inside

Version 2.0 turns this repository into a marketplace of separate free plugins.

```
CyberRiskGuardian/
├── .claude-plugin/marketplace.json     # cyberriskguardian-marketplace: lists the plugins below
├── plugins/
│   ├── cyberriskguardian/              # CyberRiskGuardian Core (free)
│   │   ├── skills/  cyber-risk-assessment, crg-context-intake, crg-risk-appetite,
│   │   │            crg-scenario-builder, crg-quantify, crg-summary-report
│   │   ├── commands/  crg-assess, crg-quick-scan, select-scenarios, crg-calc, qc-review, label-check
│   │   ├── agents/    risk-scenario-analyst, risk-qc-reviewer
│   │   ├── hooks/     SessionStart guardrails + check_labels.py (PostToolUse, Stop)
│   │   ├── mcp/       crg_server.py: crg_risk, cvss4_score
│   │   └── evals/     one suite per skill (claude plugin eval)
│   └── crg-governance-textbook/        # the governance textbook as a skill (free)
├── docs/                               # quick-start guides (EN, FR) and demo script
├── examples/medibec/                   # worked example (fictional teaching case)
├── scripts/  sync-project-config.sh, build_evals.py, add_cis_notice.py, package-plugins.sh
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

The teaching package (instructor guide, student lab workbook, assessment kit, case library, introductory slides and the
blank CyberRiskGuardian Excel model, in English and French) was published here up to **Education Package 2.1.2**
(8 November 2026). Since 20 November 2026 it is maintained outside this repository and supplied with the **Educational
edition**, together with the Educator Edition (lab answer keys, graded quiz bank, rubric calibration examples, case teaching
notes and the textbook's Instructor Resource Guide). Copies obtained from this repository's history or its earlier release
assets keep the CC BY-NC 4.0 licence they were published under. Enquiries: sales@cyberriskguardian.ca.

## Use

| Ask or run | What happens |
|---|---|
| "Run a cyber risk assessment for <org>" / `/crg-assess` | Core workflow: context, appetite, scenarios, quantification, management summary |
| "Quick cyber risk scan" / `/crg-quick-scan` | At least twenty candidates (typically 25–40), 10 scenarios, pre-treatment verdicts in one table |
| "Select the most material scenarios" / `/select-scenarios` | Sections A–E scenario selection |
| `/crg-calc 0.75 0.5 0.7 0.95 0.35 0.95 0.75 0.75 CVSS:4.0/… 0.30` | Risk indices for one scenario |
| "Explain the three lines model" | Answer from the textbook with chapter citation |
| `/qc-review <files>` | Independent read-only quality review |
| `/label-check <files>` | Checks the three standard markers (EN/FR) and reports retired wording |

Calculator: `python3 plugins/cyberriskguardian/skills/crg-quantify/scripts/crg_calc.py data.json [--appetite 0.30] [--factor 1000] [--json] [--it-budget N --cyber-spend N]`, or `--budget-target 0.35` for the appetite-consistent budget target. Every input is validated; CVSS scores are recomputed from their vectors. The data model is `plugins/cyberriskguardian/skills/crg-quantify/assets/scenario-template.json`.

Label check: `python3 plugins/cyberriskguardian/hooks/check_labels.py report.md` lists missing methodology markers and outdated marker wording.

## Maintain and release
1. Edit only under `plugins/`, then run `bash scripts/sync-project-config.sh`.
2. Validate each plugin and the marketplace with `claude plugin validate`.
3. Run `python3 scripts/build_evals.py`, then the eval suites:
   - `claude plugin eval plugins/cyberriskguardian --allow-tools Bash Write`
   - `claude plugin eval plugins/crg-governance-textbook`
4. Regression: `crg_calc.py examples/medibec/source/medibec30.json` must report total estimated 84,305 and residual 41,023.
5. Bump the versions (semver) in each changed `plugin.json` and in `marketplace.json`; current versions: Core 2.2.1, textbook 2.1.1. Record the change in `CHANGELOG.md`.

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
| **Community Edition** (this repository) | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (SPDX `CC-BY-NC-4.0`) | Students (own coursework, including graded work), non-profit organisations (own risk), instructors (non-commercial adaptation with attribution) and anyone using it for non-commercial purposes | Free |
| **Educational edition** | CyberRiskGuardian Educational Licence | Colleges, CEGEPs and universities, as institutions | Reduced educational price; installation, customization, the current desktop application, CRG Pro Quantitative and the Educator Edition included |
| **Professional editions** | CyberRiskGuardian Commercial Licence | Companies, consultants, MSSPs, government bodies, and commercial use of the Community Edition | Professional, Team, Enterprise or MSSP tiers |

The Community Edition in this repository covers:
- the core method specifications and the CyberRiskGuardian Core and CRG Governance Textbook plugins (methodology, skills, agents, commands, hooks, calculator, evals);
- the governance textbook (3rd ed., v2.51b) packaged as a reference skill (English; the French reference set awaits the French v2.51b source);
- the MediBec example.

You may share and adapt it for non-commercial purposes with attribution. Using it to deliver paid work, or using it inside a company or government body, is commercial use and requires the Commercial Licence.

Educational and Professional edition content is not in this repository. It includes the Pro Quantitative, Policy Suite, SecOps Bridge, Third-Party Risk and Audit & Assurance plugins, Compliance and Sector Packs, the Educator Edition, the hosted CRGO ontology service and CyberRiskGuardian Desktop from release 1.8.0 onward (1.7.0 Beta and earlier remain under CC BY-NC 4.0).

See `LICENSE` for the CC BY-NC 4.0 legal code, `NOTICE.md` for when a paid licence is needed and for attribution wording, and https://cyberriskguardian.ca (sales@cyberriskguardian.ca) for tiers and licence enquiries.
