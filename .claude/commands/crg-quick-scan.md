---
description: Quick CyberRiskGuardian scan - at least 20 candidates (typically 25-40), 10 selected, pre-treatment verdicts in one table (no treatment)
argument-hint: "[organization description or documents] [--candidates N]"
---

Run a quick CyberRiskGuardian scan for: $ARGUMENTS

This is steps 1–11 of the 19-step method (book v2.51b, Chapters 5 and 6) in compressed form. It stops before treatment: no measures, costs, residual risk or budget.

1. **Evidence and context (steps 1–3).** With `crg-context-intake`, read everything supplied and build a short evidence register (E1, E2…), the crown jewels and the 12-month horizon. Do not interview at length: list each missing fact as **Information gap — organizational validation required**.
2. **Appetite (step 4).** With `crg-risk-appetite`, use the approved appetite or estimate one on the CRG anchors (0.1–0.2 · 0.3 · 0.5 · 0.7 · 0.8–0.9), strictly between 0 and 1, with its rationale and **Analytical estimate — validation required**. Approving it is a **Management decision required**.
3. **Candidates (step 5).** With `crg-scenario-builder`, generate **at least twenty** causal-chain candidates, **typically 25–40** (`--candidates N` sets the number; never fewer than 20): threat source → vulnerability or condition → asset/process → event → consequence, using at least three of the five strategies (top-down, bottom-up, threat-driven, dependency-driven, localized generic). Include the cloud patterns where the organization uses cloud or SaaS.
4. **Screen and select (steps 6–7).** Score every candidate 1–5 on the 12 screening criteria (the total out of 60 is a screening aid, not a rule), apply the escalation flags, consolidate variants and select the **10** most material for materiality, coverage and decision usefulness, with a one-line rationale each. Say which selections rest on judgement rather than score. Never select or rank on CVSS.
5. **Parameters and CVSS (steps 8–10).** With `crg-quantify`, state each selected chain in one sentence, estimate Pb(A), Pb(ψ,A), δe, δm (≥ δe), θ and μ(E) to two decimals, and score the representative weakness with a CVSS v4.0 Base vector; for a process, people, governance, physical, third-party or configuration weakness use the equivalency method and label the score *limited applicability*.
6. **Risk indices (step 11).** Compute estimated and tolerated risk and the pre-treatment ratio with the `crg-calculator` MCP tool `crg_risk` (red_p = red_i = 0) or `skills/crg-quantify/scripts/crg_calc.py` — never by hand. Classify: below (< 0.90), approximately at (0.90–1.10), above tolerance (> 1.10).

**Return one table:**

| ID | Scenario (causal chain, short) | CVSS | Re | Rt | Pre-treatment ratio | Verdict | Main information gap |
|---|---|---|---|---|---|---|---|

Then, briefly: the candidate count and dispositions in one line, the strongest unselected candidate, the top three priorities and the decisions required (each marked **Management decision required**: appetite approval, which scenarios to take to full treatment). Recall that the pre-treatment ratio equals average damage ÷ appetite, so at a low appetite most material scenarios start above tolerance, and that the values are risk indices, not KRIs. Label every estimate with the markers in the language of the reply (FR: **Estimation analytique — validation requise**, **Lacune d’information — validation organisationnelle requise**, **Décision de la direction requise**). Finish with `/label-check` on any file written. For treatment, residual risk, portfolio and budget, continue with `/crg-assess` and CRG Pro (Educational and Professional editions).
