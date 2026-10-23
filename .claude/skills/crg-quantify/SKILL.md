---
name: crg-quantify
description: >
  This skill should be used when the user asks to "quantify these scenarios", "estimate Pb(A), Pb(ψ,A), damage, resilience and
  criticality", "calculate estimated and tolerated risk", "compute CyberRiskGuardian risk indices", "score this CVSS v4.0 vector", "is this
  scenario above tolerance", or when the cyber-risk-assessment workflow reaches steps 9 to 11.
metadata:
  version: "2.1.1"
  author: "Marc-André Léger (methodology) - packaged with Claude"
  edition: "Community Edition (Core plugin, free, CC BY-NC 4.0)"
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
| μ(E) | `Mu` | organizational criticality (expected utility): importance of the affected service or asset to the mission, not a second damage estimate |

For every value give: the estimate, a qualitative reading, the rationale, the evidence reference (E-number), the confidence (High, Medium, Low) and the evidence that would raise confidence. Use the anchor table in `references/crg-formulas.md`. Round to two decimals and avoid false precision. Valid ranges: every value in 0–1, **δm ≥ δe**, θ strictly above 0 (up to 1); avoid 0 and 1 for the probabilities without strong evidence. The calculator rejects any value outside these ranges. Mark each value **Analytical estimate — validation required** unless the organization supplied it.

## 2. CVSS v4.0

CVSS is a **mandatory multiplier** (CRG step 10): every scenario needs a score; a blank is an error, never 0. When the scenario rests on an identifiable technical weakness, write the CVSS v4.0 vector, justify each base metric in one line, and get the score from the `cvss4_score` MCP tool or `scripts/crg_calc.py`. Never estimate a score by eye, and always store the **vector next to the score**: the calculator recomputes the score from the vector and flags a stored score that disagrees. A malicious document delivered by e-mail is scored **AV:L** (it is opened locally), as the FIRST user guide requires.

For non-technical weaknesses (process, people, governance, physical, third-party, configuration), score the weakness **by equivalency** (book ch06 §Non-technical vulnerabilities by equivalency): map it to the CVSS metric it most resembles, justify each metric in one line, label the score *limited applicability* and mark it **Analytical estimate — validation required**. CVSS never ranks or selects scenarios.

## 3. Calculate

Use the `crg_risk` MCP tool for one scenario, or write the data model (`assets/scenario-template.json` schema) and run:

```
python3 scripts/crg_calc.py scenarios.json [--appetite A] [--factor 1000] [--json]
```

Run the calculator once, in a single command: write the data model and run it together (for one scenario, a heredoc piped to `crg_calc.py -` is enough). Write the data model in the current working directory, not in a temporary folder. If no folder is writable, pipe it: `python3 scripts/crg_calc.py - <<'EOF' … EOF`. Never fall back to calculating by hand.

**When the calculator rejects the input** (exit code 2, "invalid input"), do not compute anything from that input and do not correct the values yourself. Report each error in plain words — for example, that the maximum damage δm cannot be lower than the expected damage δe — repeat any CVSS warning (a stored score that disagrees with its vector), and ask for corrected values. Give no risk indices until the input is valid.

Show the substitution for at least one scenario so the reader can follow it. Keep the factor at 1,000 for every scenario. Before treatment, set `red_p` and `red_i` to 0. Record the approved or estimated appetite as `APPETITE` in the data model: if it is missing, the calculator uses the default (0.30) and reports it as an **Analytical estimate — validation required**. When a scenario carries a `cost`, the calculator adds cost-effectiveness (risk units removed per $1,000).

## 4. Interpret

Classify each scenario on the **risk-to-tolerance ratio** (residual ÷ tolerated): below 0.90 below tolerance, 0.90 to 1.10 approximately at tolerance, above 1.10 above tolerance. These computed values are **risk indices**, not KRIs: KRIs are measured monitoring indicators defined at step 19. Explain the result for management in one sentence per scenario.

Remind the reader that the ratio depends only on damage, treatment and appetite. Probabilities, CVSS, resilience and criticality change the magnitude of estimated risk (the ranking), not the verdict. Always read the ratio together with estimated risk.

## Output

A parameter table per scenario, the CVSS vector and metric rationale, and a results table: ID, scenario, CVSS, estimated risk, tolerated risk, risk-to-tolerance ratio, classification.

Always state these figures in the reply itself, copied from the calculator's output (never only "see the table above" or a file):
- **one scenario:** the CVSS score, estimated, tolerated, mitigated and residual risk, the risk-to-tolerance ratio and its classification, and the cost-effectiveness whenever a cost was given;
- **a data model with several scenarios:** the calculator's two totals, **total estimated risk** and **total residual risk**, as printed on its `Total` line, and how many scenarios are above, approximately at and below tolerance. Costed treatment, sensitivity and budget calibration are CRG Pro Quantitative steps (Educational and Professional editions); hand them to the Pro skills when installed, otherwise go to `crg-summary-report`. The `crg_budget_target` MCP tool (or `crg_calc.py --budget-target A`) gives the appetite-consistent budget target for reference.

## Language

Reply in the user's language. In French, write the markers as **Estimation analytique — validation requise**, **Lacune d'information — validation organisationnelle requise** and **Décision de la direction requise**.
