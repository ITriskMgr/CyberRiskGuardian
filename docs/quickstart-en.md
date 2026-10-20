# CyberRiskGuardian quick start

*Version 2.0 · Version française : `demarrage-rapide-fr.md`*

CyberRiskGuardian helps you run a structured cybersecurity risk assessment with Claude: from your organization's context to a ranked set of risk scenarios and the decisions management must take. It is a decision-support tool. You validate its estimates and management makes the decisions.

## 1. Install (5 minutes)

**Claude Code**

```
/plugin install cyberriskguardian --marketplace ITriskMgr/CyberRiskGuardian
/plugin install crg-governance-textbook --marketplace ITriskMgr/CyberRiskGuardian
```

Then install the calculator dependencies once: `pip install cvss "mcp>=1.2"`. Restart Claude Code and approve the `crg-calculator` server when prompted.

**Claude Cowork**: in the plugin install flow, upload `cyberriskguardian.plugin` and `crg-governance-textbook.plugin` from the GitHub release.

**claude.ai**: the skills work there too; calculations use the bundled script instead of the local calculator server.

## 2. Your first assessment (20 minutes)

1. Gather what you have about the organization: a profile, an IT overview, a supplier list, audit or incident notes. No documents? Say so, and Claude will interview you, five questions at a time.
2. Type `/crg-quick-scan` followed by a short description, or attach the documents. Example:
   `/crg-quick-scan Regional clinic network, 10 sites, electronic health records, remote access for clinicians.`
3. Answer any questions. Claude estimates the risk appetite, builds 20 candidate scenarios, selects the 10 most material and computes estimated and tolerated risk with the calculator.
4. Read the result table: each scenario is **above**, **approximately at** or **below** tolerance before treatment.

For the full Core workflow with a written management summary, use `/crg-assess`.

## 3. Read the markers

Every output carries three markers. They show you what still needs work:

| Marker | Meaning | What to do |
|---|---|---|
| **Analyst estimate — validation required** | A value Claude estimated | Confirm it with the people who know the system or process |
| **Evidence gap — organizational input required** | A fact nobody supplied | Find the document or ask the owner |
| **Management decision required** | A choice only management can make | Put it on the agenda |

A built-in check warns you when a deliverable is missing a marker it needs.

## 4. Useful commands

| Command | Use |
|---|---|
| `/crg-quick-scan` | 20 candidates, 10 scenarios, one table |
| `/crg-assess` | Full Core workflow and management summary |
| `/select-scenarios` | Scenario screening and selection only (Sections A–E) |
| `/crg-calc` | KRIs for one scenario from its parameters |
| `/qc-review` | Independent review of a report before you send it |

Ask governance questions in plain language, for example "What is the difference between risk appetite and risk tolerance?" The textbook plugin answers with a chapter citation.

## 5. Going further: CRG Pro

CRG Pro adds the costed treatment portfolio, sensitivity analysis, the appetite-aligned cybersecurity budget, KRI design, the Excel workbook with live formulas and the full Word board report. It is available under the CyberRiskGuardian Commercial Licence or, for colleges and universities, the Educational Licence. Details: www.leger.ca.

## Licence

The Community Edition is free for students, non-profit organisations and non-commercial use (CC BY-NC 4.0). Using it to deliver paid work, or inside a company or government body, requires a commercial licence.
