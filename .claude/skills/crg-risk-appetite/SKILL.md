---
name: crg-risk-appetite
description: >
  This skill should be used when the user asks to "estimate our cybersecurity risk appetite", "set the risk appetite for the
  assessment", "what risk appetite should we use", "translate risk appetite into materiality criteria", or "calibrate appetite on a
  0 to 1 scale", or when the cyber-risk-assessment workflow reaches step 4.
metadata:
  version: "2.1.1"
  author: "Marc-André Léger (methodology) - packaged with Claude"
  edition: "Community Edition (Core plugin, free, CC BY-NC 4.0)"
---

# Risk appetite and materiality (CyberRiskGuardian step 4)

The risk appetite sets the tolerated risk for every scenario, so it drives every tolerance verdict in the assessment. Use the value management has approved. Estimate one only when none is supplied, and say that it is an estimate. The appetite must lie strictly between 0 and 1 (0.1–0.9 in practice); never leave a template placeholder such as 0.5 in place of a reasoned value.

## Scale and anchors (0 to 1)

| Value | Meaning |
|---|---|
| 0.1–0.2 | very high risk aversion, very low appetite |
| 0.3 | low appetite, risk averse |
| 0.5 | risk neutral, moderate appetite |
| 0.7 | relatively high appetite |
| 0.8–0.9 | very high appetite |

## Factors to weigh

Industry and mission; sensitivity of information; safety and impact on individuals; legal and regulatory obligations; dependence on technology; financial capacity; tolerance for operational disruption; customer and public expectations; reputational exposure; any statements management has made about risk.

Avoid the extremes without strong evidence. A health, public-safety or critical-infrastructure organization rarely justifies more than 0.3; a small unregulated firm rarely justifies less than 0.4.

## Output

Always give all three parts below, even in a short answer.

1. Start with this line, word for word and with this capitalization, the value on two decimals: `Estimated Risk Appetite: 0.30` (or `Approved Risk Appetite: 0.30` when management approved it). In French: `Appétit pour le risque estimé : 0,30`. Then give the rationale, citing the factors from the organization's own documents that pushed the value up or down (for example data sensitivity, safety or continuity of service, sector obligations), and the confidence level. Under that line, write both markers literally, each on its own line: **Analytical estimate — validation required** and **Management decision required** (management must approve the appetite).
2. A short appetite sensitivity note: with the CRG formulas, the pre-treatment ratio equals average damage divided by appetite. At 0.30, any scenario whose average damage exceeds 0.33 starts above tolerance. Say what changes at the next anchor up and down. For reference, the appetite-consistent budget target is 12% of total IT budget incl. salaries at 0.30, 7.8% at 0.50 and 4% at 0.70, linear in between (`crg_budget_target`).
3. Materiality criteria for scenario selection derived from the appetite: which impact levels, regulatory consequences, safety effects or service outages count as material for this organization.

Hand off to `crg-scenario-builder` next.

## Language

Reply in the user's language. In French, write the markers as **Estimation analytique — validation requise**, **Lacune d'information — validation organisationnelle requise** and **Décision de la direction requise**.
