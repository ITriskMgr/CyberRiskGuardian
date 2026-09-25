---
name: scenario-selection
description: >
  This skill should be used when the user asks to "select risk scenarios", "screen candidate scenarios", "choose the 10 most
  material cyber scenarios", "build a scenario screening matrix", "review a scenario portfolio", or wants Sections A–E
  (context, candidates, screening, final portfolio, quality review) for a cybersecurity risk assessment.
metadata:
  version: "1.0.0"
  author: "Marc-André Léger (methodology) - packaged with Claude"
---

# Cybersecurity Risk Scenario Selection

Select a defensible, evidence-based, decision-useful portfolio of cybersecurity risk scenarios representing the organization's most material exposures and uncertainties. The goal is not to list every conceivable threat. Treat selection as strategic filtering and prioritization aligned with ISO 31000, ISO/IEC 27001/27005, NIST guidance and FAIR concepts. Read `references/scenario-selection-instructions.md` for the complete 20-step procedure and critical rules before starting.

## Procedure

1. **Context first (Section A).** Scope, 12-month horizon, objectives, critical processes, assets, obligations, maturity, known weaknesses, third parties, continuity needs. Build the Objective → Process → Primary asset → Supporting assets → Dependencies → Consequences map. List single points of failure, concentration risk and peripheral assets. Translate appetite into materiality criteria. List assumptions and gaps; never invent facts.
2. **Candidate population (Section B).** Generate about 25–40 candidates for 10 final scenarios (scale proportionally) using top-down, bottom-up, threat-driven, dependency-driven and localized generic strategies. Write each as threat + vulnerability/condition + asset/process + adverse event + material consequence; use multi-stage chains where justified. Tag CIA, strategy and source.
3. **Screening (Section C).** Score every candidate 1–5 on: business criticality, asset criticality, threat relevance, exposure/vulnerability, potential impact, regulatory/legal, operational resilience, dependency/concentration, evidence strength, risk velocity, control uncertainty, decision usefulness. Record rationale. Add mandatory escalation flags (safety, essential service, catastrophic data loss, systemic failure, severe privacy, major regulatory, existential financial, privileged infrastructure, backup/recovery destruction, primary+recovery failure, supply-chain concentration, multi-unit spread). Rate selection confidence H/M/L and evidence basis. Apply FAIR reasoning (threat event frequency, capability, resistance strength, loss magnitude) where information allows.
4. **Consolidate.** Group variants and stages of the same exposure into families; keep scenarios separate only when assets, actors, consequences, controls, owners or treatment decisions differ materially. Prevent any one technique's variants from dominating.
5. **Sensitivity.** Score families under at least four weighting schemes (equal, impact-led, evidence-led, decision-led) using the highest member score per family. Flag families selected by judgement (not top-10 in all schemes) and unselected candidates that rank in the top 10 (explicit review).
6. **Select (Section D).** Choose 10 unless evidence justifies otherwise, balancing materiality, tolerance exceedance, evidence, uncertainty, regulatory/safety weight, dependencies, coverage and decision usefulness — not the ten highest totals. For each: statement, objective, assets, threat, vulnerabilities, consequences, dependencies, known controls, appetite relevance, selection rationale, evidence basis, confidence, validation needs; answer the six "why this scenario?" questions; prepare quantification inputs (Pb(A), Pb(ψ,A), δe, δm, θ, μ, CVSS applicability, control effectiveness, mitigations, reductions, cost) as estimate + rationale + confidence + evidence required; map cross-functional validators (R/C) and anticipated disagreements.
7. **Quality review (Section E).** Coverage matrix (C/I/A, identity, ransomware, cloud/SaaS, third party, insider, application/API, infrastructure, endpoint, OT/IoT/medical, privacy, fraud/BEC, resilience, AI, safety — only where credible); duplication; dependencies; low-frequency/high-impact; third-party and systemic risk; regulatory and safety; decision usefulness; blind spots and watch list with promotion triggers; evidence-collection plan; critical-rules check.

## Output

Default to an Excel workbook with sheets README, A_Context, A_AssetMap, A_Dependencies, B_Candidates, C_Weights, C_Screening, C_Sensitivity, D_Portfolio, D_WhyTest, D_QuantPrep, D_Validation, E_Coverage, E_Quality, Changes (when a previous portfolio exists). Use live formulas for totals, weighted scores, ranks (`_xlfn.MAXIFS` for family scores), coverage counts and disposition counts; blue text for editable scores, weights and dispositions; recalculate and verify zero errors. A reference implementation is `examples/medibec/source/build_sel.py` in the CyberRiskGuardian repository.

In the reply, report: candidates screened, dispositions, the final 10, which selections rest on judgement rather than score (and why), the strongest unselected candidate, coverage gaps, and what must be validated next.

## Critical rules

A threat is not a scenario · never auto-select severe vulnerabilities or rank on CVSS · never confuse probability with impact · generic threats are not organizational evidence · never auto-exclude low-probability/high-consequence scenarios · never double-count overlapping scenarios · no artificial precision · never present AI assumptions as facts · no portfolio dominated by one technique · always link technical events to business consequences · always document why each scenario was selected · always identify assumptions and gaps · the human analyst remains accountable for the final selection.
