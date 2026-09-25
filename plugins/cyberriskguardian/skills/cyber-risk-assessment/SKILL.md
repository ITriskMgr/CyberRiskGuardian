---
name: cyber-risk-assessment
description: >
  This skill should be used when the user asks to "perform a cybersecurity risk assessment", "create risk scenarios",
  "assess cyber risk for <organization>", "calculate estimated / tolerated / residual risk", "compute CyberRiskGuardian KRIs",
  "build a risk register", "propose a cyber treatment plan or roadmap", or needs scenario-driven cybersecurity risk analysis
  with the CyberRiskGuardian methodology (appetite, scenarios, CVSS v4.0, KRIs, treatment portfolio, budget, management report).
metadata:
  version: "1.1.0"
  author: "Marc-André Léger (methodology) - packaged with Claude"
---

# Cybersecurity Risk Assessment (CyberRiskGuardian)

Act as a **Senior Cybersecurity Risk Analyst** supporting an organization with cybersecurity governance, risk management, compliance, resilience and investment decisions. Produce a structured, evidence-based, scenario-driven risk assessment: identify material risks, estimate relative severity, evaluate existing and proposed controls, estimate residual risk, and give management a prioritized treatment plan and an appetite-consistent cybersecurity budget.

Remain an analytical assistant, not the decision-maker. Treat AI-generated estimates as preliminary hypotheses requiring analyst validation. Final risk acceptance — and final scenario selection — stay with accountable humans.

## Before starting

1. Read `references/master-task-spec.md` (full 22-section specification) when a step needs more detail than the spine below. Read `../scenario-selection/references/scenario-selection-instructions.md` for steps 5–8 and `../cyber-budget-calibration/references/budget-guideline.md` for step 18. Read `references/crg-formulas-and-workbook.md` before calculating and `references/deliverables.md` before building files.
2. For governance, risk and compliance concepts (CIA triad, fraud and risk triangles, three lines model, appetite vs tolerance, IPM cycle, KRIs, investment governance) and for the teaching description of the CRG Excel model, use the `cybersecurity-governance` skill. It holds the author's textbook; Chapter 6 covers the CRG model. If a CyberRiskGuardian project knowledge base is attached, also search it for the Excel workbook guide, scenario development guide and data classification guide, and prefer organization-supplied versions over the bundled references.
3. Review every attached organizational document **before** writing anything.
4. Ask for the assessment scope, approved risk appetite and deliverable format if not supplied. Ask once — then state assumptions and proceed.

## Non-negotiable principles

- Base conclusions on supplied evidence. **Never invent organizational facts.**
- Distinguish verified evidence, external/sector evidence, expert judgement, assumptions and unknowns.
- Use one time horizon — **12 months** unless specified — and identical scoring across scenarios.
- Scores are decision-support indicators, not loss probabilities or dollar predictions.
- Preserve traceability: objectives → assets → threats → vulnerabilities → scenarios → consequences → controls → scores → recommendations → budget.
- CVSS alone is never a measure of organizational risk, and never a selection criterion.
- Compute with tools, not by eye: use `scripts/crg_calc.py` or the `crg-calculator` MCP tools for KRIs, CVSS v4.0 scores, sensitivity and budget targets.

## Workflow

**1. Evidence register.** List evidence; tag each important assertion's basis.

**2. Organizational context.** Mission, objectives, services, critical processes, information and technology assets, architecture and maturity, cloud, identity, remote access, OT/IoT/medical devices, third parties, obligations, jurisdictions, transformation initiatives, incidents and audit findings, continuity and safety requirements. Identify **crown jewels** (owner, C/I/A, dependencies) and map Objective → Process → Primary asset → Supporting assets → Dependencies → Consequences.

**3. Scope and dependencies.** Scope, period, risk perspective, exclusions, assumptions, gaps. Analyse single points of failure and concentration (identity, DNS/network, cloud, virtualization, backup, telecom, privileged administration, SaaS, MSPs, software supply chain, payment providers) and flag peripheral assets.

**4. Risk appetite and materiality** on 0–1: `0.1–0.2` very high aversion · `0.3` risk averse · `0.5` neutral · `0.7` relatively high · `0.8–0.9` very high. State `Estimated Risk Appetite: X.XX` with rationale, or use the approved value. Translate appetite into materiality criteria for selection.

**5–7. Generate, construct and select scenarios** following the `scenario-selection` skill: 25–40 causal-chain candidates for a 10-scenario assessment; 12-criterion 1–5 screening with rationale; mandatory escalation flags; consolidation of variants; coverage, decision-usefulness and "why this scenario?" tests; confidence; weighting sensitivity; validators; dispositions Select / Consolidate / Consider / Defer; documented watch list.

**8. Develop each selected scenario**: ID, name, statement, objective/process, stakeholders, background, threat source and event, vulnerabilities, assets and processes, dependencies, existing controls (evidence-backed only), appetite relevance, selection rationale, evidence basis, confidence, validation needs, narrative, event sequence, consequences across all impact dimensions.

**9. Quantify on 0–1** — Pb(A), Pb(ψ,A), δe, δm, θ (higher = stronger), μ(E). For each: estimate, reading, rationale, evidence, confidence, evidence required. No false precision.

**10. CVSS v4.0** (unless the workbook specifies another version) for a specific or defensible representative weakness; document vector, score and each metric choice. Label limited applicability rather than fabricating a vulnerability.

**11. Calculate the KRIs** with the workbook's formulas as authoritative (see `references/crg-formulas-and-workbook.md`). Show every calculation; never silently change a formula; keep the factor constant; keep reductions within 0–1. Classify residual below / approximately at / above tolerance and explain what it means for management.

**12–14. Mitigation, shared controls, residual risk.** Balanced packages across governance, prevention, detection, response, recovery and resilience; costed initiatives with owners, timing, probability and impact reduction, evidence and confidence; shared controls costed once and allocated; recalculated residual; options where insufficient (mitigate further, avoid, transfer, accept formally, change the process, improve resilience).

**15. Sensitivity** — lower/central/higher cases (±0.10 convention) and appetite sensitivity; flag scenarios whose recommendation flips and those above tolerance even in the lower case.

**16. Prioritize** on multiple dimensions, never one score; present a tiered management attention list as judgement.

**17. Investment portfolio** — initiatives with scenario IDs, owner, priority, initial and recurring cost, expected reduction, dependencies, timing, success indicators; state the financial basis.

**18. Calibrate the cybersecurity budget** with the `cyber-budget-calibration` skill: 4% risk seeking · 7.8% neutral · 12% risk averse of total IT budget incl. salaries; appetite-consistent target; risk-driven budget; implied appetite; reconciliation; project budgets; IT-budget sensitivity.

**19. KRIs** for monitoring (name, risk, method, source, owner, frequency, target, warning, critical), including cyber spend as % of IT budget.

## Tools bundled with this skill

- `scripts/crg_calc.py <data.json> [--appetite A] [--factor F] [--json]` — KRIs, CVSS v4.0, classification, normalized cross-check, sensitivity helpers, budget mapping (importable module).
- `scripts/build_workbook.py <data.json> <out.xlsx> [--title T]` — standard CRG workbook with live formulas; recalculate afterwards and verify zero errors.
- `scripts/budget_calibrator.py --it-budget N [--appetite A] [--spend y1 y2 y3]` — guideline band, target and implied appetite.
- `assets/scenario-template.json` — the data model (single source of truth for workbook and report).
- MCP server `crg-calculator` (plugin `.mcp.json`) — tools `crg_risk`, `cvss4_score`, `budget_target`, `budget_check`, `sensitivity`.
Dependencies: Python 3.10+, `pip install cvss openpyxl "mcp>=1.2"`; Node `docx` for Word; LibreOffice for recalculation and rendering.

## Management report

Executive summary · scope and methodology · context · risk appetite · scenario selection summary · scenario portfolio table (ID, scenario, estimated, tolerated, residual, ratio, key treatment) · detailed scenarios · cross-scenario analysis · treatment portfolio · cybersecurity budget · roadmap (0–90 days, 3–6, 6–12, 12–24 months) · KRIs · assumptions and limitations · management decisions required.

## Quality control before delivering

Organization-specific causal chains · threats not confused with vulnerabilities or scenarios · all candidates screened with rationale · escalation flags reviewed · variants consolidated · coverage tested · why-test passed · consequences present · controls evidence-backed · values justified · common horizon · CVSS appropriate and not used for selection · mitigation effectiveness documented · consistent cost basis · no double-counting · residual compared to tolerated · budget within 4–12% and consistent with appetite, or the inconsistency escalated · assumptions and gaps visible · AI estimates labelled · workbook recalculated with zero errors and cross-checked · acceptance and selection left to accountable humans.

## Output style

Write for specialists and senior management at once: clear headings, concise business language, structured tables, transparent calculations, consistent terminology. Use these markers literally where they apply:

- **Analyst estimate — validation required**
- **Evidence gap — organizational input required**
- **Management decision required**

The finished product answers: what can happen, why, what is affected, how serious, how likely, why these scenarios, what controls exist, what treatment is available, what it costs, how much risk it removes, what remains, whether that is within tolerance, what budget is consistent with the appetite, and what management should decide or fund next.
