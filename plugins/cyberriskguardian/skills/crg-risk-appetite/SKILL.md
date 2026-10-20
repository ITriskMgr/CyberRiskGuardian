---
name: crg-risk-appetite
description: >
  This skill should be used when the user asks to "estimate our cybersecurity risk appetite", "set the risk appetite for the
  assessment", "what risk appetite should we use", "translate risk appetite into materiality criteria", or "calibrate appetite on a
  0 to 1 scale", or when the cyber-risk-assessment workflow reaches step 4.
metadata:
  version: "2.0.0"
  author: "Marc-André Léger (methodology) - packaged with Claude"
  edition: "Core (free, CC BY-NC 4.0)"
---

# Risk appetite and materiality (CyberRiskGuardian step 4)

The risk appetite sets the tolerated risk for every scenario, so it drives every tolerance verdict in the assessment. Use the value management has approved. Estimate one only when none is supplied, and say that it is an estimate.

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

1. One line: `Estimated Risk Appetite: X.XX` (or `Approved Risk Appetite: X.XX`), followed by the rationale, the factors that pushed the value up or down, and the confidence level. Mark an estimate **Analytical estimate — validation required** and **Management decision required**.
2. A short appetite sensitivity note: with the CRG formulas, the pre-treatment ratio equals average damage divided by appetite. At 0.30, any scenario whose average damage exceeds 0.33 starts above tolerance. Say what changes at the next anchor up and down.
3. Materiality criteria for scenario selection derived from the appetite: which impact levels, regulatory consequences, safety effects or service outages count as material for this organization.

Hand off to `crg-scenario-builder` next.

## Language

Reply in the user's language. In French, write the markers as **Estimation analytique — validation requise**, **Lacune d'information — validation organisationnelle requise** and **Décision de la direction requise**.
