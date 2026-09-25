---
name: cyber-budget-calibration
description: >
  This skill should be used when the user asks to "determine the cybersecurity budget", "check whether the security budget is
  appropriate", "align the cyber budget with risk appetite", "calculate the budget for IT and cybersecurity projects", or needs
  the 4%–12% of IT budget guideline applied to a risk assessment's treatment portfolio.
metadata:
  version: "1.0.0"
  author: "Marc-André Léger (methodology) - packaged with Claude"
---

# Cybersecurity Budget Calibration

Set the cybersecurity budget with the same logic as risk decisions: the risk appetite fixes where the budget sits in the guideline band, and the risk assessment shows what the money must buy. Read `references/budget-guideline.md` before starting.

## Rule

Cybersecurity budget = **4% to 12% of the total IT budget, including salaries**. 4% = risk seeking (appetite 0.70) · 7.8% = median, risk neutral (0.50) · 12% = risk averse (0.30). Map appetite to target by linear interpolation between anchors, clamped outside. Compute targets and implied appetites with `../cyber-risk-assessment/scripts/budget_calibrator.py` or the `crg-calculator` MCP tools `budget_target` and `budget_check`.

## Procedure

1. Confirm the IT budget (including salaries) and the approved or estimated appetite; flag implausible figures (e.g. budget inconsistent with headcount) as **Evidence gap — organizational input required**.
2. Derive the appetite-consistent target (% and amount); show the appetite sensitivity table (0.2–0.8).
3. Estimate current security spend (security staff, tooling, security embedded in IT contracts); flag it if below the 4% floor.
4. Build the risk-driven budget bottom-up from the treatment portfolio: phase by fiscal year (one-time costs spread between start and end month, recurring from completion); split each shared initiative between cyber and IT budgets and state why; count shared controls once; add new security roles (state FTE valuation), risk-transfer premiums (placeholder until quoted) and ~10% contingency.
5. Compare options against the band (status quo; portfolio; + staffing; + second-wave treatments) and compute each option's implied appetite.
6. Show what each funding level buys: counts above / approximately at / below tolerance and total residual risk (re-run CRG calculations with improved reductions only where a funded treatment justifies them, labelled as estimates).
7. Reconcile budget and appetite — fund to appetite (ramp limited by delivery capacity; any balancing reserve released only against scenario-linked business cases), formally revise the appetite, or accept residual risk formally. Mark **Management decision required**.
8. Report the cybersecurity project budget (committed and gated reserve) separately from security-relevant IT projects; note IT-funded prerequisites.
9. Test sensitivity to the IT budget figure; above the 12% ceiling, phase or trim starting with the lowest cost-effectiveness items.
10. Add KRIs: cyber spend % of IT budget (target = appetite-consistent %, warning < 7.8%, critical < 4%); share of spend tied to scenarios; reserve releases tied to scenario IDs.

## Output

A budget section in the management report and budget sheets in the workbook (Budget_Params with the live mapping formula, Budget_Lines with gross/cyber/IT splits by year, Budget_Summary with options, band check, implied appetite, project budgets and IT-budget sensitivity, Budget_Risk with tolerance status by funding level). Recalculate and verify zero errors. In the reply, state the appetite-consistent target, the risk-driven budget and its implied appetite, the recommendation, and the decisions required.
