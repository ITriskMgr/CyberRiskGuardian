# CyberRiskGuardian formulas, workbook conventions and interpretation

## Formulas (authoritative, never modify silently)

Source: book *Introduction to Cybersecurity Governance* (3rd ed., v2.51) ch06 §The risk index and underlying formulas; CyberRiskGuardian Excel Guide v1.0c, s.11. These computed values are **risk indices** (Re, Rt, residual, ratio); they are not KRIs.

```
Estimated risk (Re) = Pb(A) × Pb(ψ,A) × CVSS × ((δe + δm) / 2) × μ(E) ÷ θ(ψ,A) × Factor
Tolerated risk (Rt) = Pb(A) × Pb(ψ,A) × CVSS × Appetite × μ(E) ÷ θ(ψ,A) × Factor
Mitigated           = Re × Probability reduction (rp) × Impact reduction (ri)
Residual            = Re − Mitigated        (template v2-6: MAX(0, Re − Mitigated) with a Checks column; identical while reductions stay in 0–1)
Risk-to-tolerance   = Residual ÷ Rt         (pre-treatment ratio = Re ÷ Rt)
Cost-effectiveness  = (Re − Residual) ÷ cost × 1,000   (risk units removed per $1,000)
```

Worked example (guide s.16 parameters, MediBec ransomware, vector `CVSS:4.0/AV:L/AC:L/AT:N/PR:N/UI:P/VC:H/VI:H/VA:H/SC:H/SI:H/SA:H` = CVSS-B 9.3; book ch06 §Worked example): Pb(A) 0.70, Pb(ψ,A) 0.60, CVSS-B 9.3, δe 0.60, δm 0.80, θ 0.50, μ(E) 0.65, appetite 0.35, factor 1,000, probability reduction 0.65, impact reduction 0.70, cost $62,000 → Re 3,554.46; Rt 1,777.23; pre-treatment ratio 2.00; mitigated 1,617.28; residual 1,937.18; ratio 1.09 (approximately at tolerance); cost-effectiveness 26.09. The document is delivered by e-mail but opened locally, so the Attack Vector is Local (FIRST CVSS v4.0 user guide). These values are illustrative and differ from MediBec's published 30-scenario register.

## Valid ranges (enforced by `crg_calc.py`, the MCP tools and the v2-6 template)

| Input | Range | Note |
|---|---|---|
| Pb(A), Pb(ψ,A), δe, δm, μ(E) | 0–1 | avoid 0 and 1 without strong evidence; **δm ≥ δe** (error otherwise) |
| θ(ψ,A) | above 0, up to 1 | divides the index |
| CVSS | 0.0–10.0, **required** | a blank is an error, never 0; keep the vector: the score is recomputed from it and a disagreeing stored score is flagged |
| Appetite | strictly between 0 and 1 | warning outside 0.1–0.9; if missing from the data model the default (0.30) is used and reported as an **Analytical estimate — validation required** |
| rp, ri | 0–1 each | a value above 1 is rejected |
| Factor | constant, 1,000 | warning when another value is used |

## Conventions
- **Factor**: 1,000, constant across all scenarios of an assessment and across assessments that will be compared. It scales indices, never ratios.
- **CVSS**: v4.0 base score (CVSS-B) unless the workbook specifies otherwise. Compute scores with `scripts/crg_calc.py` or the `crg-calculator` MCP tool `cvss4_score`; never estimate a score by eye. Non-technical weaknesses are scored by equivalency, labelled *limited applicability* (book ch06).
- **Classification band** (analyst convention reflecting estimate precision): ratio < 0.90 below tolerance; 0.90–1.10 approximately at tolerance; > 1.10 above tolerance.
- **Combining several measures (product rule).** The package reductions combine per lever: probability reductions together, impact reductions together, with
  `r_package = 1 − ∏(1 − r_i)`. Each r_i must lie in 0–1, so the package value can never exceed 1. This is the rule used by the desktop application, the v2-6 Excel template (M sheets, including the "Other" row as one more factor) and the CRG Pro planner. It models overlap and diminishing returns; adjust the package estimate further when measures clearly duplicate each other, and record the rationale. Per-control coefficients are never simply added.
- **FTE valuation** in the template: 1,950 h × $80 = $156,000 per year.
- **Cost allocation**: when one initiative serves several scenarios, split its cost across those scenarios (equal split is an acceptable default) so no scenario carries the full enterprise cost; totals come from the portfolio, not the sum of scenario allocations.
- **Transfer and avoidance**: transfer (insurance, contract terms) is a costed portfolio line plus a management decision; record the share of loss covered (0–1, informational) and the approver, but claim no reduction in the index. Avoidance is a management decision with an owner.
- **Budget calibration** (step 18, best practice): target = 12% of total IT budget including salaries at appetite 0.30 or below, 7.8% at 0.50, 4% at 0.70 or above, linear interpolation in between (clamped outside). Compare current spending plus the portfolio with the target; report the gap and the implied appetite; closing the gap or revising the appetite is a **Management decision required**. Use `crg_calc.py --budget-target A` or the `crg_budget_target` MCP tool.
- **Workbook structure** (template v2-6, `education/templates/`): Analyse (row per scenario: E–L parameters with G CVSS score and H CVSS vector, M estimated, N tolerated, P cost, Q Pb reduction, R impact reduction, S mitigated, T residual, Y pre-treatment ratio, Z risk-to-tolerance ratio, AA verdict, AB cost-effectiveness, W recommend 0/1, AC checks, AD transfer share of loss covered and AE approver (informational); appetite E14, factor E15; budget calibration block Y14:AC25 with current cybersecurity spending, gap to target and implied appetite), S1–S10 scenario inputs, M1–M10 mitigation sheets (rows 2–6 "Other" package lines; reductions combined with the product rule; catalogues ISO/IEC 27002:2022 93 controls, NIST CSF 2.0 106 subcategories, CIS Controls v8.1 18 controls, with indicative default costs by effort profile for a mid-sized organization (about 500 staff; cost basis table under each catalogue, Analytical estimate — validation required) and French ISO/IEC 27002:2022 titles as an unofficial working translation). The template holds 10 scenarios; the CRG Pro workbook builder extends the same formulas to any number of rows.

## Interpretation notes (methodological observations)
1. **The tolerance test depends only on damage, treatment and appetite.** Residual ÷ Tolerated simplifies to ((δe + δm)/2) × (1 − rp × ri) ÷ Appetite: probabilities, CVSS, resilience and criticality cancel. They drive magnitude (ranking) but not the tolerance verdict, so a rare scenario and a frequent one with equal damage and treatment get the same verdict. Always read the ratio together with estimated risk.
2. **Pre-treatment ratio** = average damage ÷ appetite, so with appetite 0.30 nearly every scenario with average damage above 0.33 starts above tolerance.
3. **Mitigation is multiplicative**: 80% probability and 80% impact reduction remove 64% of risk. Design rules: residual **no greater than tolerated (ratio ≤ 1.0)** requires rp × ri ≥ 1 − Appetite ÷ average damage; **below tolerance (ratio < 0.90)** requires rp × ri > 1 − 0.90 × Appetite ÷ average damage. A package that reduces only probability, or only impact, mitigates nothing.
4. **CVSS is a mandatory multiplier.** For non-technical scenarios (insider misuse, detection gaps, supplier concentration, fraud, human error) score a representative weakness by equivalency and label it *limited applicability*; low CVSS values depress estimated risk for fraud and error scenarios whose impact is carried by the damage estimates — say so in the report rather than adjusting the score.
5. **Two lenses may disagree.** A simpler normalized score (T × E × I × (1 − control maturity)) with escalation thresholds (e.g. 0.25 executive review, 0.40 Board) often places post-treatment risks inside authority limits that the CRG ratio flags above tolerance. Tie governance thresholds to one method and state which.

## Sensitivity convention (applied by CRG Pro `crg-sensitivity`)
Lower / central / higher cases: shift Pb(A), Pb(ψ,A), δe, δm by ∓/± 0.10, θ by ±/∓ 0.10 and both reductions by ±/∓ 0.10, clamped to [0.05, 0.99]. Flag scenarios whose classification changes between the lower and higher case; those whose classification is "above" even in the lower case need transfer or formal acceptance rather than more spending. For the worked example above the lower and higher cases give 0.69 and 1.53.

## Parameter guidance (anchors used so far)
| Parameter | Low (≈0.2–0.35) | Moderate (≈0.4–0.6) | High (≈0.65–0.85+) |
|---|---|---|---|
| Pb(A) | rare actor/event, needs special position | plausible for sector | observed or constant (phishing, scanning) |
| Pb(ψ,A) | strong, evidenced controls | partial or unknown controls | weakness evidenced (e.g. push MFA approved, alert closed) |
| δe / δm | limited records or one site | material, several sites | network-wide, safety, catastrophic data loss |
| θ | resilience weak: no tested recovery, small team | partial | tested recovery, 24×7 detection |
| μ(E) organizational criticality (expected utility) | supporting process | important process | mission-critical (care delivery, identity) |
Each value needs rationale, evidence reference and confidence; mark **Analytical estimate — validation required**.
