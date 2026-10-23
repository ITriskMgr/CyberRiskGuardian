---
name: crg-summary-report
description: >
  This skill should be used when the user asks to "write the risk assessment summary", "summarize the assessment for management",
  "produce the management summary", "what should management decide", "write the executive summary of the cyber risk assessment",
  or when the cyber-risk-assessment workflow has finished quantification.
metadata:
  version: "2.1.1"
  author: "Marc-André Léger (methodology) - packaged with Claude"
  edition: "Community Edition (Core plugin, free, CC BY-NC 4.0)"
---

# Summary report (CyberRiskGuardian Community Edition)

Write a management summary that a director can read in five minutes and act on. Read `references/deliverables.md` for naming and format conventions.

## Structure

1. **Conclusion first.** Two to four sentences: how many scenarios are above, approximately at and below tolerance before treatment, which three need attention first, and — always as the last sentence of this paragraph — what management must decide.
2. **Scope and method.** Organization, scope, 12-month horizon, risk appetite (approved or estimated), evidence used, and the CyberRiskGuardian formulas in one line.
3. **Portfolio table.** ID, scenario, estimated risk, tolerated risk, risk-to-tolerance ratio, classification, main information gap.
4. **Priorities.** A tiered attention list built on several dimensions, never one score: estimated risk, maximum damage, legal and privacy obligations, safety, risk velocity, control maturity, dependencies. Flag every place where judgement departs from the scores.
5. **Indicative treatment directions (uncosted).** Use this heading. For each priority scenario, the type of measure that would reduce it (for example phishing-resistant MFA, immutable backups, supplier contract clauses). Mark these indicative and uncosted.
6. **Assumptions, evidence gaps and limitations.** List what must be validated and by whom.
7. **Management decisions required.** Each decision on its own line, marked **Management decision required**.
8. **Next steps.** Costed treatment portfolio, sensitivity analysis, appetite-aligned budget, KRIs with thresholds and the live-formula workbook are produced with CRG Pro Quantitative (Educational and Professional editions), or by hand with the Excel template in a course.

## Rules

- Use the numbers from the calculator output exactly; round risk units to whole numbers and ratios to two decimals.
- Keep AI estimates labelled **Analytical estimate — validation required** and missing facts labelled **Information gap — organizational validation required**.
- Connect each technical event to its business consequence.
- Write in the language the user writes in (French or English).

## Format

Default to a document in the environment's document format. Use Word only when the user asks for Word. Name files `<Org>_CyberRiskGuardian_Summary[_variant]`.

## Language

Reply in the user's language. In French, write the markers as **Estimation analytique — validation requise**, **Lacune d'information — validation organisationnelle requise** and **Décision de la direction requise**.
