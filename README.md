# CyberRiskGuardian Agent

AI-assisted, scenario-driven cybersecurity risk assessment for Claude Code and Claude Cowork. The agent acts as a Senior Cybersecurity Risk Analyst and covers the whole assessment:

- selecting scenarios;
- quantifying risk with the CyberRiskGuardian formulas and CVSS v4.0;
- comparing residual risk with appetite;
- designing the treatment portfolio;
- calibrating the budget to appetite;
- producing management reports.

People stay accountable for validation and risk decisions.

The latest version is CyberRiskGuardian with Threat Live-feed, available here: https://github.com/ITriskMgr/CyberRiskGuardian_ThreatIntel

Methodology and textbook © Marc-André Léger. This repository is the free **Community Edition**, licensed under **CC BY-NC 4.0**. Paid editions are available under a commercial or educational licence (see [Licence](#licence)). Packaged with Claude, version 1.1.2 (October 2026).

## What's inside

```
cyberriskguardian-agent/
├── CLAUDE.md                     # Claude Code memory: imports AGENTS.md + Claude-specific components
├── AGENTS.md                     # Agent-agnostic instructions (role, principles, formulas, stack, conventions)
├── LICENSE                       # CC BY-NC 4.0 legal code
├── NOTICE.md                     # Licensing at a glance (free vs paid), attribution, third-party material
├── .mcp.json                     # Project MCP config: local crg-calculator server
├── .claude/
│   ├── settings.json             # Model, output style, env defaults, permissions, SessionStart hook
│   ├── skills/                   # (synced) cyber-risk-assessment, scenario-selection, cyber-budget-calibration, cybersecurity-governance
│   ├── commands/                 # (synced) /risk-assessment /select-scenarios /calibrate-budget /crg-calc /qc-review
│   ├── agents/                   # (synced) risk-scenario-analyst, risk-quantifier, risk-qc-reviewer
│   ├── output-styles/            # (synced) CyberRiskGuardian Analyst, Executive Brief
│   ├── rules/                    # evidence-discipline, data-protection (always); workbook, report, maintenance (path-gated)
│   └── hooks/session-context.md  # guardrails injected at session start
├── .claude-plugin/marketplace.json   # Marketplace listing this repository's plugin
├── plugins/cyberriskguardian/    # THE PLUGIN - single source of truth
│   ├── .claude-plugin/plugin.json
│   ├── skills/…/SKILL.md, references/, scripts/, assets/
│   │   └── cybersecurity-governance/references/   # the governance textbook, one file per chapter
│   ├── commands/  agents/  output-styles/
│   ├── hooks/hooks.json          # SessionStart guardrails
│   ├── mcp/crg_server.py         # MCP server: crg_risk, cvss4_score, budget_target, budget_check, sensitivity
│   └── .mcp.json
├── examples/medibec/             # Worked example (fictional teaching case): input, source builders, deliverables
├── scripts/sync-project-config.sh
├── requirements.txt  package.json  .gitignore
```

## Install

**Prerequisites:**
- Python 3.10+ — `pip install -r requirements.txt` installs cvss, openpyxl and mcp.
- Node 18+ — `npm install` for Word reports.
- LibreOffice, for recalculating workbooks and rendering PDFs.

**A. Use as a project (Claude Code):** clone or unzip the repository and open it with `claude`. Claude Code then loads everything automatically:
- `CLAUDE.md`, `.claude/` and `.mcp.json`;
- approve the `crg-calculator` MCP server when prompted.

**Public install from GitHub (Claude Code):** from https://github.com/ITriskMgr/CyberRiskGuardian:
```
claude plugin marketplace add ITriskMgr/CyberRiskGuardian
claude plugin install cyberriskguardian@cyberriskguardian-marketplace
```
**B. Install as a plugin from the marketplace (Claude Code):**
```
/plugin marketplace add <git-url-or-local-path-of-this-repository>
/plugin install cyberriskguardian@cyberriskguardian-marketplace
```
The plugin brings the following into any project:
- skills;
- commands (namespaced, e.g. `/cyberriskguardian:risk-assessment`);
- subagents and output styles;
- the SessionStart hook;
- the MCP calculator.

Rules and project settings stay project-level. Copy `.claude/rules/` and `.claude/settings.json` if you want them elsewhere.

**C. Claude Cowork:** upload `cyberriskguardian.plugin` (a zip of `plugins/cyberriskguardian/`) through the plugin install flow.

## Use

| Ask or run | What happens |
|---|---|
| "Create 10 risk scenarios for <org> and complete a risk assessment" / `/risk-assessment` | Full workflow: context, selection, quantification, treatment, budget, report and workbook |
| "Select the most material scenarios" / `/select-scenarios` | Sections A–E scenario-selection workbook |
| "Explain the three lines model" / any governance concept | Answers from the author's textbook (`cybersecurity-governance` skill), with chapter citation |
| "Determine the appropriate cybersecurity budget" / `/calibrate-budget` | Appetite-consistent budget vs the 4% / 7.8% / 12% guideline, risk-driven budget, project budgets |
| `/crg-calc 0.75 0.5 0.7 0.95 0.35 0.95 0.75 0.75 CVSS:4.0/…` | KRIs for one scenario |
| `/qc-review <files>` | Independent read-only quality review |

**Scripts** (under `plugins/cyberriskguardian/skills/cyber-risk-assessment/scripts/`):
```
python3 crg_calc.py data.json [--appetite 0.30] [--factor 1000] [--json]
python3 build_workbook.py data.json out.xlsx --title "Org - CyberRiskGuardian Workbook"
python3 budget_calibrator.py --it-budget 100000000 --appetite 0.30 --spend 5.05e6 5.55e6 4.87e6
```
The data model is described in `assets/scenario-template.json`.

## Maintain and release
1. Edit only under `plugins/cyberriskguardian/`, then run `bash scripts/sync-project-config.sh`.
2. Validate:
   - `claude plugin validate plugins/cyberriskguardian/.claude-plugin/plugin.json`
   - `claude plugin validate .claude-plugin/marketplace.json`
3. Regression-check the MediBec example. `crg_calc.py examples/medibec/source/medibec30.json` must report total estimated 84,491 and residual 41,104.
4. Bump the version (semver) in `plugin.json` and `marketplace.json`; current version 1.1.2.

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
- methodology;
- skills, agents and commands;
- scripts;
- the governance textbook;
- the MediBec example.

You may share and adapt it for non-commercial purposes with attribution. Using it to deliver paid work, or using it inside a company or government body, is commercial use and requires the Commercial Licence.

The paid editions are not in this repository. They include the Pro Quantitative, Policy Suite, SecOps Bridge, Third-Party Risk and Audit & Assurance plugins, Compliance and Sector Packs, the Educator Edition, the hosted CRGO ontology service and CyberRiskGuardian Desktop versions released after 1.7.0 Beta.

See `LICENSE` for the CC BY-NC 4.0 legal code, `NOTICE.md` for when a paid licence is needed and for attribution wording, and www.leger.ca for tiers and the educational-licence contact.
