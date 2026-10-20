---
name: crg-summary-report
description: >
  This skill should be used when the user asks to "write the risk assessment summary", "summarize the assessment for management",
  "produce the management summary", "what should management decide", "write the executive summary of the cyber risk assessment",
  or when the cyber-risk-assessment workflow has finished quantification.
metadata:
  version: "2.0.0"
  author: "Marc-André Léger (methodology) - packaged with Claude"
  edition: "Core (free, CC BY-NC 4.0)"
---

# Summary report (CyberRiskGuardian Core)

Write a management summary that a director can read in five minutes and act on. Read `references/deliverables.md` for naming and format conventions.

## Structure

1. **Conclusion first.** Two to four sentences: how many scenarios are above, approximately at and below tolerance before treatment, which three need attention first, and what management must decide.
2. **Scope and method.** Organization, scope, 12-month horizon, risk appetite (approved or estimated), evidence used, and the CyberRiskGuardian formulas in one line.
3. **Portfolio table.** ID, scenario, estimated risk, tolerated risk, ratio, classification, main evidence gap.
4. **Priorities.** A tiered attention list built on several dimensions, never one score: estimated risk, maximum damage, legal and privacy obligations, safety, risk velocity, control maturity, dependencies. Flag every place where judgement departs from the scores.
5. **Indicative treatment directions.** For each priority scenario, the type of measure that would reduce it (for example phishing-resistant MFA, immutable backups, supplier contract clauses). Mark these indicative and uncosted.
6. **Assumptions, evidence gaps and limitations.** List what must be validated and by whom.
7. **Management decisions required.** Each decision on its own line, marked **Management decision required**.
8. **Next steps.** Costed treatment portfolio, sensitivity analysis, appetite-aligned budget, KRIs with thresholds and the live-formula workbook are produced with CRG Pro Quantitative.

## Rules

- Use the numbers from the calculator output exactly; round risk units to whole numbers and ratios to two decimals.
- Keep AI estimates labelled **Analyst estimate — validation required** and missing facts labelled **Evidence gap — organizational input required**.
- Connect each technical event to its business consequence.
- Write in the language the user writes in (French or English).

## Format

Default to a document in the environment's document format. Use Word only when the user asks for Word. Name files `<Org>_CyberRiskGuardian_Summary[_variant]`.

## Language

Reply in the user's language. In French, write the markers as **Estimation de l'analyste — validation requise**, **Lacune de preuve — information organisationnelle requise** and **Décision de la direction requise**.
