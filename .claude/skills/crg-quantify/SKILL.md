---
name: crg-quantify
description: >
  This skill should be used when the user asks to "quantify these scenarios", "estimate Pb(A), Pb(ψ,A), damage, resilience and
  criticality", "calculate estimated and tolerated risk", "compute CyberRiskGuardian KRIs", "score this CVSS v4.0 vector", "is this
  scenario above tolerance", or when the cyber-risk-assessment workflow reaches steps 9 to 11.
metadata:
  version: "2.0.0"
  author: "Marc-André Léger (methodology) - packaged with Claude"
  edition: "Core (free, CC BY-NC 4.0)"
---

# Quantification (CyberRiskGuardian steps 9–11)

Turn each developed scenario into the CyberRiskGuardian parameters, score its CVSS v4.0 vector where it applies, and compute estimated and tolerated risk with the calculator. Read `references/crg-formulas.md` before calculating.

## 1. Parameters (0 to 1)

| Symbol | Key in the data model | Meaning |
|---|---|---|
| Pb(A) | `PbA` | probability that the threat is present in the 12-month period |
| Pb(ψ,A) | `Pbx` | probability that the threat exploits the weakness |
| δe(ψ,A) | `De` | normalized expected damage |
| δm(ψ,A) | `Dm` | normalized plausible maximum damage |
| θ(ψ,A) | `Th` | organizational resilience: prevent, detect, contain, respond, recover (higher is stronger) |
| μ(E) | `Mu` | importance of the affected service or asset to the mission |

For every value give: the estimate, a qualitative reading, the rationale, the evidence reference (E-number), the confidence (High, Medium, Low) and the evidence that would raise confidence. Use the anchor table in `references/crg-formulas.md`. Round to two decimals and avoid false precision. Mark each value **Analyst estimate — validation required** unless the organization supplied it.

## 2. CVSS v4.0

When the scenario rests on an identifiable technical weakness, write the CVSS v4.0 vector, justify each base metric in one line, and get the score from the `cvss4_score` MCP tool or `scripts/crg_calc.py`. Never estimate a score by eye. For non-technical scenarios (insider misuse, fraud, supplier concentration, human error), score a representative weakness and label it *limited applicability*. CVSS never ranks or selects scenarios.

## 3. Calculate

Use the `crg_risk` MCP tool for one scenario, or write the data model (`assets/scenario-template.json` schema) and run:

```
python3 scripts/crg_calc.py scenarios.json [--appetite A] [--factor 1000] [--json]
```

Show the substitution for at least one scenario so the reader can follow it. Keep the factor at 1,000 for every scenario. Before treatment, set `red_p` and `red_i` to 0.

## 4. Interpret

Classify each scenario on the residual-to-tolerated ratio: below 0.90 below tolerance, 0.90 to 1.10 approximately at tolerance, above 1.10 above tolerance. Explain the result for management in one sentence per scenario.

Remind the reader that the ratio depends only on damage, treatment and appetite. Probabilities, CVSS, resilience and criticality change the magnitude of estimated risk (the ranking), not the verdict. Always read the ratio together with estimated risk.

## Output

A parameter table per scenario, the CVSS vector and metric rationale, and a results table: ID, scenario, CVSS, estimated risk, tolerated risk, ratio, classification. Costed treatment, sensitivity and budget are CRG Pro steps; hand them to the Pro skills when installed, otherwise go to `crg-summary-report`.

## Language

Reply in the user's language. In French, write the markers as **Estimation de l'analyste — validation requise**, **Lacune de preuve — information organisationnelle requise** and **Décision de la direction requise**.
