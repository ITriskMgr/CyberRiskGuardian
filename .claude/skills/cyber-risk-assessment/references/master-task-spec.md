# Master task specification — Perform a Cybersecurity Risk Assessment

Source: specification provided by the CyberRiskGuardian author (Marc-André Léger), September 2026. Designed so organization documents, the CyberRiskGuardian workbook and evidence can be attached to the task without rewriting the methodology each time. The skill `cyber-risk-assessment` summarizes this specification; consult this file when a step needs more detail. Aligned with the book *Introduction to Cybersecurity Governance* (3rd ed., v2.51), ch05–ch06: the sections below are numbered as the **19 CRG steps** (ch06 §The steps of a CRG assessment), followed by the management report and the quality control review.

## Role
Act as a **Senior Cybersecurity Risk Analyst** supporting an organization with cybersecurity governance, risk management, compliance, resilience and investment decision-making. Conduct a structured, evidence-based **scenario-driven cybersecurity risk assessment**: identify material risks, estimate relative severity, evaluate existing and proposed controls, estimate residual risk, and give management a prioritized treatment plan. The analyst is an analytical assistant, not the final decision-maker. Treat AI-generated estimates as preliminary hypotheses; identify assumptions, uncertainties, evidence gaps and areas needing SME validation.

## Assessment principles
1. Base conclusions on the organizational information and evidence provided.
2. Distinguish facts supported by evidence, reasonable analytical assumptions, estimates, and unknown or missing information.
3. Do not invent organizational facts.
4. When evidence is incomplete, make a reasonable provisional estimate and label it as an assumption.
5. Keep one assessment horizon, preferably **12 months**.
6. Use the same scoring methodology across all scenarios.
7. Treat scores as **decision-support indicators**, not exact probabilities or dollar-loss predictions.
8. Preserve traceability: objectives → assets and processes → threats → vulnerabilities → scenarios → consequences → controls → scores → recommendations.
9. Do not treat CVSS alone as a measure of organizational risk.
10. Final risk acceptance stays with organizational management.

## Inputs
Use everything supplied: organizational description, annual reports, architecture diagrams, asset inventories, network and cloud architecture, application inventories, data classifications, business processes, critical services, risk registers, previous assessments, vulnerability and penetration-test reports, audit reports, incident records, threat intelligence, BCP/DR information, vendor information, policies and standards, legal/regulatory/contractual/privacy obligations, security-control documentation, cybersecurity budgets, CyberRiskGuardian templates or workbooks. Review files before beginning.

## Step 1. Evidence register
Create an **Evidence Register** (E1, E2, …; type documented, exercise or simulation, SME input, external; strength); for each important assertion indicate whether it rests on documented evidence, external evidence, SME input, analyst inference, or an AI-generated estimate requiring validation. Exercise material is indicative evidence to validate, never a confirmed incident.

## Step 2. Organizational context and crown jewels
Organization, sector, mission, products and services, customers or beneficiaries, critical processes, critical information and technology assets, sensitive information, infrastructure, cloud services, major applications, identity systems, remote access, OT/IoT, third parties, geography, regulatory obligations, cybersecurity dependencies, continuity requirements, public-safety implications. Identify **crown jewels** with business purpose, owner, C/I/A requirements, technical and third-party dependencies.

## Step 3. Scope
Assessment scope; period (default 12 months); risk perspective (confidentiality, integrity, availability, privacy, operations, finance, compliance, reputation, safety, strategy); exclusions; assumptions; material information gaps.

## Step 4. Risk appetite and materiality
Estimate on 0–1 unless supplied: 0.1–0.2 very high aversion; 0.3 risk averse; 0.5 neutral; 0.7 relatively high; 0.8–0.9 very high. Consider industry, mission, sensitivity, safety, obligations, technology dependence, financial capacity, customer expectations, availability requirements, reputation, management statements. Provide **Estimated Risk Appetite: X.XX** with justification; avoid extremes without strong evidence; use an approved value if supplied. The appetite must lie strictly between 0 and 1 (0.1–0.9 in practice); never leave a template placeholder. Translate the appetite into materiality criteria for selection. Approving the appetite is a **Management decision required**.

## Step 5. Generate candidate scenarios
About 25–40 candidates for a 10-scenario assessment (scaled proportionally), **never fewer than 20** (20 is the lab quick version), using at least three of the five generation strategies: **top-down, bottom-up, threat-driven, dependency-driven and localized generic**. Generative AI is a drafting aid, not a sixth strategy.

### Scenario construction standard
Never "Ransomware", "Phishing", "Data breach" or "Cloud compromise" alone. Causal chain: **threat source/event → vulnerability or predisposing condition → affected asset/process → cybersecurity event → business consequence**. Example: *An external threat actor compromises a privileged cloud identity through credential phishing and insufficient phishing-resistant authentication, allowing unauthorized access to sensitive information and disruption of critical services.* Each candidate identifies ID, name, threat source, threat event, vulnerability/condition, primary and supporting assets, process affected, undesired outcome, principal consequences.

## Step 6. Screen
Score every candidate 1–5 on the **12 screening criteria** (book ch05): business criticality, asset criticality, threat relevance, exposure/vulnerability, potential impact, regulatory/legal, operational resilience, dependency/concentration, evidence strength, risk velocity, control uncertainty, decision usefulness. The total (out of 60) is a screening aid, not a mechanical rule. Add the mandatory escalation flags, selection confidence and disposition; consolidate variants into families; test the selection under at least four weighting schemes.

## Step 7. Select
Select the most material scenarios (10 unless specified) on materiality, potential to exceed tolerance, evidence, uncertainty, regulatory and safety significance, dependencies, coverage and decision usefulness — not the ten highest totals. Explain each selection (why-test), flag selections made by judgement, keep a watch list with promotion triggers. Never rank on CVSS. The detailed procedure is in the scenario-selection instructions. The final selection is the accountable analyst's.

## Step 8. Develop each selected scenario as a causal chain
ID and name; concise statement; stakeholders; business background; threat source; threat event; vulnerabilities and predisposing conditions; assets and processes at risk; existing controls (never assumed without evidence); detailed narrative; event sequence adapted to the scenario (initial access, exploitation, privilege escalation, persistence, discovery, lateral movement, collection or manipulation, exfiltration or disruption, detection, business impact); consequences for confidentiality, integrity, availability, privacy, operations, customers, finances, compliance, contracts, reputation, strategy and safety.

## Step 9. Quantitative parameters (0–1)
| Parameter | Symbol | Meaning |
|---|---|---|
| Probability threat present | Pb(A) | Threat source or hazardous condition present during the period |
| Probability of exploitation | Pb(ψ,A) | Threat succeeds against relevant weaknesses |
| Estimated expected damage | δe(ψ,A) | Normalized expected damage |
| Maximum damage | δm(ψ,A) | Plausible maximum damage |
| Organizational resilience | θ(ψ,A) | Ability to prevent, detect, respond, contain, recover — higher is stronger |
| Organizational criticality (expected utility) | μ(E) | Importance of the affected activity or asset to the mission |

For each: estimate, qualitative interpretation, rationale, evidence, confidence High/Medium/Low. No unsupported precision (two decimals). Valid ranges: all 0–1, δm ≥ δe, θ strictly above 0.

## Step 10. CVSS
CVSS is a **mandatory multiplier**. Where a specific or defensible representative technical weakness exists, use the workbook's CVSS version, otherwise **v4.0**. Document version, vector, base score and every metric (AV, AC, AT, PR, UI, impacts); keep the vector next to the score and compute the score with the calculator. For a process, people, governance, physical, third-party or configuration weakness, score a representative weakness **by equivalency**: justify every metric, label the score *limited applicability* and mark it **Analytical estimate — validation required**. Only where no defensible representative weakness exists at all, state the limited applicability explicitly rather than fabricating a vulnerability.

## Step 11. Risk indices and tolerance test
Use the supplied workbook's formulas as authoritative. Calculate estimated and tolerated risk (and, at step 14, mitigated and residual risk and the **risk-to-tolerance ratio**). These are computed **risk indices**, not KRIs. Show calculations; never silently modify formulas; flag the gap if no formula is supplied. Classify residual vs tolerated: below / approximately at / above tolerance, and explain the management significance.

## Step 12. Mitigation measures
Balanced across governance, prevention, detection, response, recovery and resilience (IAM, MFA, PAM, vulnerability and patch management, EDR/XDR, email security, awareness, segmentation, Zero Trust, secure configuration, application security, encryption, DLP, logging, SIEM, SOC, immutable backup, DR, IR, supplier risk, cloud security, data governance, penetration testing, tabletop exercises). For each: name, description, scenarios, type, owner, dependencies, timeframe, cost, recurring cost, probability reduction, impact reduction, supporting evidence, confidence. Distinguish indicative costs from quotations and approved budgets. Several measures on the same lever combine with the product rule: package reduction = 1 − ∏(1 − r_i), probability and impact reductions separately; never add coefficients.

## Step 13. Shared controls
Identify controls reducing multiple scenarios; never charge one enterprise control at full cost to every scenario; record dependencies (e.g. PAM depends on identity governance and account inventory).

## Step 14. Mitigated and residual risk, risk-to-tolerance ratio
Apply treatment effectiveness; compute mitigated amount, residual risk and the revised risk-to-tolerance relationship. Where insufficient: mitigate further, avoid, transfer, accept with formal authorization, change the process, or improve resilience. Not all risks can or should be eliminated. Design rules: ratio ≤ 1.0 needs rp × ri ≥ 1 − appetite ÷ average damage; below tolerance (< 0.90) needs rp × ri > 1 − 0.90 × appetite ÷ average damage. Transfer is a costed line plus a decision (share of loss covered and approver recorded, no reduction claimed); avoidance is a decision.

## Step 15. Sensitivity analysis
Lower, central and higher cases varying threat probability, exploitation, damages, resilience and mitigation effectiveness. Highlight scenarios whose recommendation changes materially.

## Step 16. Prioritize
Estimated and residual risk, risk-to-tolerance, maximum consequence, business criticality, legal and privacy obligations, safety, velocity, threat activity, control maturity, feasibility, cost, shared benefits, dependencies, time to benefit. Produce a management attention list explained as judgement, not mathematical certainty.

## Step 17. Investment portfolio
Per initiative: name, scenario IDs, description, owner, priority (Immediate / Near-term / Planned), initial cost, recurring cost, expected risk reduction, dependencies, target start and completion, success indicators. Avoid double-counting; state the financial basis (first-year, annual recurring, implementation or multi-year TCO).

## Step 18. Budget calibration
Compare the portfolio, together with existing cybersecurity spending, with the appetite-consistent target. Best practice: **12%** of total IT budget including salaries at appetite 0.30 or below, **7.8%** at 0.50, **4%** at 0.70 or above, linear interpolation in between (a reasonableness check, not a formula that dictates the budget). Report the gap and the implied appetite; closing the gap or revising the appetite is a **Management decision required**.

## Step 19. Key risk indicators
Measured monitoring values (distinct from the computed risk indices), connected to risk drivers (phishing-resistant MFA coverage, critical vulnerabilities within SLA, MTTD, MTTC, unsupported critical systems, backup-restore success, supplier assessments, phishing failure rate, security exceptions, EDR coverage, log-source coverage, cyber spend as % of IT budget). For each: name, risk addressed, method, data source, owner, frequency, target, warning, critical threshold.

## Management report
1. Executive summary — exposure, most material scenarios, residual vs tolerance, systemic weaknesses, investment themes, approximate cost, decisions required.
2. Scope and methodology.
3. Organizational and technology context.
4. Risk appetite.
5. Risk scenario portfolio — ID, scenario, estimated, tolerated, residual, risk-to-tolerance, key treatment.
6. Detailed scenarios.
7. Cross-scenario analysis — root causes, common vulnerabilities and control weaknesses, concentration risk, systemic dependencies, shared mitigations.
8. Treatment portfolio — investments, costs, sequence, owners.
9. Cybersecurity budget — appetite-consistent target, risk-driven budget, options, recommended budget, project budgets.
10. Implementation roadmap — 0–90 days, 3–6 months, 6–12 months, 12–24 months.
11. KRIs.
12. Assumptions and limitations.
13. Management decisions required — approve investment, accept residual risk, request further assessment, assign risk owners, obtain evidence, change practices, transfer, avoid.

## Quality control
Scenarios organization-specific; causal chains; threats and vulnerabilities not confused; business consequences present; existing controls evidence-backed; values justified and within their valid ranges; common horizon; CVSS appropriate and not treated as business risk; mitigation effectiveness documented; consistent cost basis; shared controls not double-counted; residual compared with tolerated; assumptions visible; evidence gaps identified; AI assumptions labelled; recommendations feasible; risk acceptance left to management.

## Output style
For specialists and senior management at once: clear headings, concise business language, structured tables, transparent calculations, consistent terminology, traceable assumptions. Avoid unsupported certainty, jargon, generic recommendations, arbitrary precision, invented facts. Use literally:
- **Analytical estimate — validation required**
- **Information gap — organizational validation required**
- **Management decision required**

The final product lets management understand what can happen, why, what is affected, how serious and likely it is, what controls exist, what treatment is available and what it costs, how much it reduces risk, what remains, whether that is within tolerance, and what to decide or fund next.

## Resolved methodology choices
- **CVSS version** — when the workbook does not specify, use 4.0 (not the 3.1 of the older seven-prompt procedure).
- **Appetite scale** — use the Excel guide anchors (0.1–0.2 · 0.3 · 0.5 · 0.7 · 0.8–0.9).
- **Multiplication factor** — 1,000 (Excel guide worked example and the v2-6 template default); keep it constant within an assessment. It scales indices but not ratios.
- **Residual** — Re − Mitigated. The v2-6 template computes MAX(0, Re − Mitigated) with a Checks column; reductions stay within 0–1, so a negative residual is always an estimation error.
- **Combining measures** — product rule 1 − ∏(1 − r_i) per lever everywhere (engine documentation, desktop, template, Pro planner); the template no longer sums coefficients with a cap.
- **Input validation** — engine, MCP tools, desktop and template reject Pb(A), Pb(ψ,A), δe, δm, μ(E), rp, ri outside 0–1, θ outside (0, 1], appetite outside (0, 1), δm < δe and a missing CVSS; the CVSS score is recomputed from its vector and a disagreeing stored score is flagged.
- **MediBec reference** — S2 and S8 are scored AV:L (malicious e-mailed document opened locally); reference totals 84,305 estimated and 41,023 residual (30 scenarios).
- **Terminology** — computed Re, Rt, residual and ratio are risk indices; KRIs are measured indicators (step 19). μ(E) is organizational criticality (expected utility).
