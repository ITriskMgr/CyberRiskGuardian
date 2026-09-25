# CyberRiskGuardian formulas, workbook conventions and interpretation

## Formulas (Excel Guide v1.0c, s.11 — authoritative, never modify silently)

```
Estimated risk (Re) = Pb(A) × Pb(ψ,A) × CVSS × ((δe + δm) / 2) × μ(E) ÷ θ(ψ,A) × Factor
Tolerated risk (Rt) = Pb(A) × Pb(ψ,A) × CVSS × Appetite × μ(E) ÷ θ(ψ,A) × Factor
Mitigated           = Re × Probability reduction × Impact reduction
Residual            = Re − Mitigated        (template: ABS(L − R); identical while reductions stay in 0–1)
Risk-to-tolerance   = Residual ÷ Rt
Cost-effectiveness  = (Re − Residual) ÷ cost × 1,000   (risk units removed per $1,000)
```

Worked example (guide s.16, MediBec): Pb(A) 0.70, Pb(ψ,A) 0.60, CVSS-B 9.4, δe 0.60, δm 0.80, θ 0.50, μ 0.65, appetite 0.35, factor 1,000, impact reduction 0.70, probability reduction 0.65, cost $62,000 → Re 3,592.68; Rt 1,796.34; mitigated 1,634.67; residual 1,958.01; ratio 1.09; cost-effectiveness 26.37.

## Conventions
- **Factor**: 1,000 (constant across all scenarios of an assessment). The blank template uses 10,000; the factor scales scores, not ratios.
- **CVSS**: v4.0 base score (CVSS-B) unless the workbook specifies otherwise. Compute scores with `scripts/crg_calc.py` or the `crg-calculator` MCP tool `cvss4_score`; never estimate a score by eye.
- **Classification band** (analyst convention reflecting estimate precision): ratio < 0.90 below tolerance; 0.90–1.10 approximately at tolerance; > 1.10 above tolerance.
- **Reductions**: probability and impact reductions of a package must each stay within 0–1. The template's M sheets *sum* control coefficients (default 0.06 each), so ~20 selected controls exceed 1 and residual risk rises through ABS(). Check before interpreting a large portfolio.
- **FTE valuation** in the template: 1,950 h × $80 = $156,000 per year.
- **Cost allocation**: when one initiative serves several scenarios, split its cost across those scenarios (equal split is an acceptable default) so no scenario carries the full enterprise cost; totals come from the portfolio, not the sum of scenario allocations.
- **Workbook structure**: Analyse (row per scenario: L estimated, M tolerated, O cost, P Pb reduction, Q impact reduction, R = P×Q×L, S residual, T tolerated, V recommend 0/1), S1–S10 scenario inputs, M1–M10 mitigation catalogues (ISO/IEC 27002:2013, NIST 800-53/CSF, CIS v7.1 — 382 controls, many with generic default costs to recalibrate). The template holds 10 scenarios; `scripts/build_workbook.py` extends the same formulas to any number of rows.

## Interpretation notes (methodological observations)
1. **The tolerance test depends only on damage, treatment and appetite.** Residual ÷ Tolerated simplifies to ((δe + δm)/2) × (1 − P × Q) ÷ Appetite: probabilities, CVSS, resilience and criticality cancel. They drive magnitude (ranking) but not the tolerance verdict, so a rare scenario and a frequent one with equal damage and treatment get the same verdict. Always read the ratio together with estimated risk.
2. **Pre-treatment ratio** = average damage ÷ appetite, so with appetite 0.30 nearly every scenario with average damage above 0.33 starts above tolerance.
3. **Mitigation is multiplicative**: 80% probability and 80% impact reduction remove 64% of risk. "Below tolerance" requires P × Q ≥ 1 − appetite ÷ average damage.
4. **CVSS is a mandatory multiplier.** For non-technical scenarios (insider misuse, detection gaps, supplier concentration, fraud, human error) score a representative weakness and label it *limited applicability*; low CVSS values depress estimated risk for fraud and error scenarios whose impact is carried by the damage estimates.
5. **Two lenses may disagree.** A simpler normalized score (T × E × I × (1 − control maturity)) with escalation thresholds (e.g. 0.25 executive review, 0.40 Board) often places post-treatment risks inside authority limits that the CRG ratio flags above tolerance. Tie governance thresholds to one method and state which.

## Sensitivity convention
Lower / central / higher cases: shift Pb(A), Pb(ψ,A), δe, δm by ∓/± 0.10, θ by ±/∓ 0.10 and both reductions by ±/∓ 0.10, clamped to [0.05, 0.99]. Flag scenarios whose classification changes between the lower and higher case; those whose classification is "above" even in the lower case need transfer or formal acceptance rather than more spending.

## Parameter guidance (anchors used so far)
| Parameter | Low (≈0.2–0.35) | Moderate (≈0.4–0.6) | High (≈0.65–0.85+) |
|---|---|---|---|
| Pb(A) | rare actor/event, needs special position | plausible for sector | observed or constant (phishing, scanning) |
| Pb(ψ,A) | strong, evidenced controls | partial or unknown controls | weakness evidenced (e.g. push MFA approved, alert closed) |
| δe / δm | limited records or one site | material, several sites | network-wide, safety, catastrophic data loss |
| θ | resilience weak: no tested recovery, small team | partial | tested recovery, 24×7 detection |
| μ(E) | supporting process | important process | mission-critical (care delivery, identity) |
Each value needs rationale, evidence reference and confidence; mark **Analyst estimate — validation required**.
