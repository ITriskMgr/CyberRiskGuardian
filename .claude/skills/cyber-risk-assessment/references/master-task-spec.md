# Master task specification — Perform a Cybersecurity Risk Assessment

Source: specification provided by the CyberRiskGuardian author (Marc-André Léger), September 2026. Designed so organization documents, the CyberRiskGuardian workbook and evidence can be attached to the task without rewriting the methodology each time. The skill `cyber-risk-assessment` summarizes this specification; consult this file when a step needs more detail.

## Role
Act as a **Senior Cybersecurity Risk Analyst** supporting an organization with cybersecurity governance, risk management, compliance, resilience and investment decision-making. Conduct a structured, evidence-based **scenario-driven cybersecurity risk assessment**: identify material risks, estimate relative severity, evaluate existing and proposed controls, estimate residual risk, and give management a prioritized treatment plan. The analyst is an analytical assistant, not the final decision-maker. Treat AI-generated estimates as preliminary hypotheses; identify assumptions, uncertainties, evidence gaps and areas needing SME validation.

## 1. Assessment principles
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

## 2. Inputs
Use everything supplied: organizational description, annual reports, architecture diagrams, asset inventories, network and cloud architecture, application inventories, data classifications, business processes, critical services, risk registers, previous assessments, vulnerability and penetration-test reports, audit reports, incident records, threat intelligence, BCP/DR information, vendor information, policies and standards, legal/regulatory/contractual/privacy obligations, security-control documentation, cybersecurity budgets, CyberRiskGuardian templates or workbooks. Review files before beginning. Create an **Evidence Register**; for each important assertion indicate whether it rests on documented evidence, external evidence, SME input, analyst inference, or an AI-generated estimate requiring validation.

## 3. Organizational context
Organization, sector, mission, products and services, customers or beneficiaries, critical processes, critical information and technology assets, sensitive information, infrastructure, cloud services, major applications, identity systems, remote access, OT/IoT, third parties, geography, regulatory obligations, cybersecurity dependencies, continuity requirements, public-safety implications. Identify **crown jewels** with business purpose, owner, C/I/A requirements, technical and third-party dependencies.

## 4. Scope
Assessment scope; period (default 12 months); risk perspective (confidentiality, integrity, availability, privacy, operations, finance, compliance, reputation, safety, strategy); exclusions; assumptions; material information gaps.

## 5. Risk appetite
Estimate on 0–1 unless supplied: 0.1–0.2 very high aversion; 0.3 risk averse; 0.5 neutral; 0.7 relatively high; 0.8–0.9 very high. Consider industry, mission, sensitivity, safety, obligations, technology dependence, financial capacity, customer expectations, availability requirements, reputation, management statements. Provide **Estimated Risk Appetite: X.XX** with justification; avoid extremes without strong evidence; use an approved value if supplied.

## 6. Generate candidate scenarios
At least 20 candidates (25–40 for a 10-scenario assessment per the scenario-selection instructions), from top-down, bottom-up, threat-driven, asset-driven, third-party, human and resilience perspectives.

## 7. Scenario construction standard
Never "Ransomware", "Phishing", "Data breach" or "Cloud compromise" alone. Causal chain: **threat source/event → vulnerability or predisposing condition → affected asset/process → cybersecurity event → business consequence**. Example: *An external threat actor compromises a privileged cloud identity through credential phishing and insufficient phishing-resistant authentication, allowing unauthorized access to sensitive information and disruption of critical services.* Each candidate identifies ID, name, threat source, threat event, vulnerability/condition, primary and supporting assets, process affected, undesired outcome, principal consequences.

## 8. Screen and select
Plausibility, materiality, relevance, uniqueness, evidence, threat and vulnerability exposure, consequence severity, regulatory, operational and safety significance. Remove duplicates. Select the most material scenarios (10 unless specified). Explain each selection. Never rank on CVSS alone. The detailed procedure is in the scenario-selection instructions.

## 9. Develop each selected scenario
ID and name; concise statement; stakeholders; business background; threat source; threat event; vulnerabilities and predisposing conditions; assets and processes at risk; existing controls (never assumed without evidence); detailed narrative; event sequence adapted to the scenario (initial access, exploitation, privilege escalation, persistence, discovery, lateral movement, collection or manipulation, exfiltration or disruption, detection, business impact); consequences for confidentiality, integrity, availability, privacy, operations, customers, finances, compliance, contracts, reputation, strategy and safety.

## 10. Quantitative parameters (0–1)
| Parameter | Symbol | Meaning |
|---|---|---|
| Probability threat present | Pb(A) | Threat source or hazardous condition present during the period |
| Probability of exploitation | Pb(ψ,A) | Threat succeeds against relevant weaknesses |
| Estimated expected damage | δe(ψ,A) | Normalized expected damage |
| Maximum damage | δm(ψ,A) | Plausible maximum damage |
| Organizational resilience | θ(ψ,A) | Ability to prevent, detect, respond, contain, recover — higher is stronger |
| Expected utility / criticality | μ(E) | Importance of the affected activity or asset |

For each: estimate, qualitative interpretation, rationale, evidence, confidence High/Medium/Low. No unsupported precision.

## 11. CVSS
Where a specific or defensible representative technical weakness exists, use the workbook's CVSS version, otherwise **v4.0**. Document version, vector, base score and every metric (AV, AC, AT, PR, UI, impacts). Where CVSS does not fit (pure policy failure, supplier concentration, some insider scenarios), state its limited applicability rather than fabricating a vulnerability.

## 12. Risk indicators
Use the supplied workbook's formulas as authoritative. Calculate estimated, tolerated, mitigated, residual risk and residual-to-tolerance ratio. Show calculations; never silently modify formulas; flag the gap if no formula is supplied. Classify residual vs tolerated: below / approximately at / above tolerance, and explain the management significance.

## 13. Mitigation measures
Balanced across governance, prevention, detection, response, recovery and resilience (IAM, MFA, PAM, vulnerability and patch management, EDR/XDR, email security, awareness, segmentation, Zero Trust, secure configuration, application security, encryption, DLP, logging, SIEM, SOC, immutable backup, DR, IR, supplier risk, cloud security, data governance, penetration testing, tabletop exercises). For each: name, description, scenarios, type, owner, dependencies, timeframe, cost, recurring cost, probability reduction, impact reduction, supporting evidence, confidence. Distinguish indicative costs from quotations and approved budgets.

## 14. Shared controls
Identify controls reducing multiple scenarios; never charge one enterprise control at full cost to every scenario; record dependencies (e.g. PAM depends on identity governance and account inventory).

## 15. Recalculate residual risk
Apply treatment effectiveness; compute mitigated amount, residual risk and the revised risk-to-tolerance relationship. Where insufficient: mitigate further, avoid, transfer, accept with formal authorization, change the process, or improve resilience. Not all risks can or should be eliminated.

## 16. Sensitivity analysis
Lower, central and higher cases varying threat probability, exploitation, damages, resilience and mitigation effectiveness. Highlight scenarios whose recommendation changes materially.

## 17. Prioritize
Estimated and residual risk, risk-to-tolerance, maximum consequence, business criticality, legal and privacy obligations, safety, velocity, threat activity, control maturity, feasibility, cost, shared benefits, dependencies, time to benefit. Produce a management attention list explained as judgement, not mathematical certainty.

## 18. Investment portfolio
Per initiative: name, scenario IDs, description, owner, priority (Immediate / Near-term / Planned), initial cost, recurring cost, expected risk reduction, dependencies, target start and completion, success indicators. Avoid double-counting; state the financial basis (first-year, annual recurring, implementation or multi-year TCO). Calibrate the total against the cybersecurity budget guideline (4–12% of IT budget).

## 19. Key risk indicators
Measurable, connected to risk drivers (phishing-resistant MFA coverage, critical vulnerabilities within SLA, MTTD, MTTC, unsupported critical systems, backup-restore success, supplier assessments, phishing failure rate, security exceptions, EDR coverage, log-source coverage, cyber spend as % of IT budget). For each: name, risk addressed, method, data source, owner, frequency, target, warning, critical threshold.

## 20. Management report
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

## 21. Quality control
Scenarios organization-specific; causal chains; threats and vulnerabilities not confused; business consequences present; existing controls evidence-backed; values justified; common horizon; CVSS appropriate and not treated as business risk; mitigation effectiveness documented; consistent cost basis; shared controls not double-counted; residual compared with tolerated; assumptions visible; evidence gaps identified; AI assumptions labelled; recommendations feasible; risk acceptance left to management.

## 22. Output style
For specialists and senior management at once: clear headings, concise business language, structured tables, transparent calculations, consistent terminology, traceable assumptions. Avoid unsupported certainty, jargon, generic recommendations, arbitrary precision, invented facts. Use literally:
- **Analyst estimate — validation required**
- **Evidence gap — organizational input required**
- **Management decision required**

The final product lets management understand what can happen, why, what is affected, how serious and likely it is, what controls exist, what treatment is available and what it costs, how much it reduces risk, what remains, whether that is within tolerance, and what to decide or fund next.

## Resolved methodology choices
- **CVSS version** — when the workbook does not specify, use 4.0 (not the 3.1 of the older seven-prompt procedure).
- **Appetite scale** — use the Excel guide anchors (0.1–0.2 · 0.3 · 0.5 · 0.7 · 0.8–0.9).
- **Multiplication factor** — 1,000 (Excel guide worked example); keep it constant within an assessment. The blank workbook uses 10,000; the choice scales scores but not ratios.
- **Residual** — the template computes ABS(Estimated − Mitigated); keep reductions within 0–1 so the ABS is never needed, and treat a negative residual as an estimation error.
