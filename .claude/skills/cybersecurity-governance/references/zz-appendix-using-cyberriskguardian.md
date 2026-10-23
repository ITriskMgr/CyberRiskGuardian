<!-- Léger, M.-A. (2026). Introduction to Cybersecurity Governance for Business Technology Management (3rd ed., v2.51). Appendix: Using CyberRiskGuardian with this book. CC BY-NC 4.0 -->

# Appendix: Using CyberRiskGuardian with this book

This book teaches cybersecurity governance as a set of managerial decisions: what to protect, how much risk to accept, which treatments to fund, and how to show that the decisions were sound. Chapters 5 and 6 turn those decisions into a repeatable method, scenario-based risk assessment, and the remaining chapters return to it whenever a domain—identity, cloud, third parties, artificial intelligence, investment—needs to be connected to business risk. CyberRiskGuardian is the set of tools that supports that method in practice. This appendix explains what it is, which edition to use, how each chapter connects to it, and where to obtain it.

The appendix is written for students, instructors, and program coordinators. It does not repeat the method itself; the formulas, parameters, and interpretation rules are explained in Chapters 5 and 6, and the tools follow those chapters exactly. **All teaching activities with this book use the CyberRiskGuardian Community Edition or its Educational edition.**

## What CyberRiskGuardian is

**CyberRiskGuardian (CRG)** is the author's open, scenario-driven method for assessing cybersecurity risk, together with the tools that support it. It is a decision-support tool, not a decision-maker: it structures the analysis and speeds it up, while the analyst remains accountable for every estimate and management remains accountable for every risk decision.

The method follows the workflow taught in Chapters 4 to 6. The analyst documents the organizational context in an evidence register, identifies the crown jewels, sets the risk appetite on a 0–1 scale, drafts twenty candidate scenarios written as causal chains, and selects the ten most material. Each selected scenario receives six parameters—the probability that the threat is present, Pb(A); the probability of exploitation, Pb(ψ,A); expected damage, δe; maximum damage, δm; organizational resilience, θ; and organizational criticality or expected utility, μ(E)—plus the CVSS v4.0 score of the weakness it exploits, all on a common 12-month horizon. The calculation engine then produces the estimated risk (Re), the tolerated risk (Rt), the mitigated and residual risk of each treatment package, and the risk-to-tolerance ratio, read as below tolerance (under 0.90), approximately at tolerance (0.90 to 1.10), or above tolerance (over 1.10). The results feed a treatment portfolio, an appetite-consistent cybersecurity budget, key risk indicators (KRIs), and a management report.

Three markers appear on every CRG deliverable and throughout Chapters 5 and 6: *Analytical estimate — validation required* for a value the analyst or an AI tool estimated, *Information gap — organizational validation required* for a fact nobody has supplied, and *Management decision required* for a choice only management can make. Students who use these markers correctly have understood the difference between evidence, inference, and judgment, which is the central skill this book tries to build.

| CyberRiskGuardian calculates and structures; people validate and decide. A number the verified engine has not recomputed is not a result. |
|-------------------------------------------------------------------------------------------------------------------------------------------|

## Editions and who should use which

CyberRiskGuardian follows an open-core model. The editions differ in who may use them and for what purpose, not in the method, which is the same everywhere.

- **Community Edition (free).** The public repository is published under the Creative Commons Attribution-NonCommercial 4.0 International licence (CC BY-NC 4.0). Students may use it free of charge for their own coursework, including graded work, and non-profit organizations may use it free of charge to assess their own risk. Instructors may read, adapt, and reuse the free materials for non-commercial teaching, with attribution. The Community Edition includes the core method specifications, the AI-assistant plugin with its skills, commands, review agents and verified calculator, a version of this textbook packaged as a reference skill, the MediBec worked example, and the education package.

- **Educational edition (Educational Licence).** Colleges, CEGEPs, and universities, as institutions, use the CyberRiskGuardian Educational Licence to deliver a course built on CRG. It is offered at a reduced educational price and includes installation and customization for the institution, the current desktop application, and the Educator Edition of the teaching materials: lab answer keys, a graded quiz bank, rubric calibration examples, case teaching notes, and a reserved final-examination case.

- **Professional editions.** Separate paid editions exist for companies, consultants, managed security service providers, and government bodies that use CRG for their own risk management or to deliver paid work.

The practical rule for this book is simple. A student working alone uses the Community Edition. A course delivered by an institution uses the Educational edition, so that every section works with the same installation, the same answer keys, and the same examination safeguards. Where this summary and the licence text differ, the licence governs; this appendix is not legal advice.

## Three ways to work

The education package describes three tool tracks. The method and the formulas are identical in all of them; only the interface changes. Instructors should choose one track per course, or a declared hybrid, so that teams are graded on the same footing.

- **Track A, the AI assistant.** The free plugin runs inside an approved AI-assistant platform. It interviews the user about the organization, builds the evidence register and crown-jewel map, proposes the risk appetite, generates and screens scenarios, and computes the results with the bundled, verified calculator. A quick-scan command produces twenty candidates, ten selected scenarios, and their pre-treatment verdicts in one table; a full-assessment command adds the management summary. A read-only quality-review agent and a label check flag missing markers.

- **Track B, the desktop application.** The desktop application runs locally in a web browser, keeps data on the student's computer unless the student chooses to send it, and switches its whole interface between English and French. It offers classroom workspaces, a scenario editor, a CVSS v4.0 tool, a risk calculator that shows each step of the formula, treatment proposals and a control catalogue, forecasts, a KRI dashboard, security-operations simulation exercises, and policy and audit modules.

- **Track C, the Excel model.** The blank CRG Excel model shows every formula openly and works offline, which suits business students who want to see the arithmetic. How to use it is described below.

A hybrid works well: students draft context and scenarios with the AI assistant, then quantify and treat them in the desktop application or the workbook, where every number is recomputed. Whatever the track, the first step is to confirm that the engine agrees with the reference implementation: the MediBec example must produce a total estimated risk of 84,491 and a total residual risk of 41,104.

## Chapter-by-chapter guide

The table below maps each chapter to the CRG features that support it. Features confirmed in the CRG documentation are listed plainly; features distributed in separate or paid modules, or not yet documented for the teaching editions, are marked "where available".

| **Chapter**                  | **CyberRiskGuardian features**                                                                                                                                               | **Typical activity**                                                  |
|------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| 1 Basic concepts             | Textbook reference skill; three evidence markers; causal-chain template                                                                                                      | Explain the risk triangle with a cited chapter answer                 |
| 2 Governance                 | Risk appetite estimator; *Management decision required* marker; quality review; multi-user RACI (where available)                                                            | Identify which decisions belong to the board and which to management  |
| 3 Compliance                 | Audit management (audit universe, audit plan); compliance packs and compliance management (where available)                                                                  | Link one obligation to a scenario and its key control                 |
| 4 Risk management            | Context intake; evidence register; crown jewels; appetite on a 0–1 scale; budget guideline                                                                                   | Set and justify an appetite value for a case                          |
| 5 Risk scenarios             | Scenario builder: twenty candidates, screening, ten selected; scenario editor                                                                                                | Write and screen causal-chain scenarios                               |
| 6 Risk assessment            | Six parameters; CVSS v4.0 scoring; risk calculator; treatment proposals and control catalogue; tolerance verdicts; sensitivity and Monte Carlo forecasts; management summary | Calculate Re, Rt, residual risk and the ratio, then test sensitivity  |
| 7 Operations                 | Security-operations evidence recorded on scenarios; KRI dashboard                                                                                                            | Record operations evidence without changing parameters by hand        |
| 8 Detection and response     | SIEM and SOAR simulation exercises; CVSS v4.0 tool; threat intelligence feed (where available)                                                                               | Triage alerts and confirm what turns a case into an incident          |
| 9 Control frameworks         | Control catalogue; policy statements referenced to ISO/IEC 27002:2022                                                                                                        | Map a treatment package to current framework editions                 |
| 10 IAM                       | Security policy library (access control and authentication template); IAM policy assignment                                                                                  | Draft five testable IAM policy statements                             |
| 11 OT, IoT, CPS              | Scenario builder and calculator applied to cyber-physical scenarios                                                                                                          | Score a safety-relevant scenario with δm and μ(E)                     |
| 12 Incidents and resilience  | Resilience parameter θ; restore-test KRIs; SOAR exercises                                                                                                                    | Show how tested recovery changes θ                                    |
| 13 Architecture              | Causal-chain traceability from scenario to treatment; CRGO ontology (where available)                                                                                        | Trace a design decision to an enterprise scenario                     |
| 14 Baselines                 | KRI dashboard and KRI projections                                                                                                                                            | Define configuration KRIs with owners and thresholds                  |
| 15 Cloud                     | Scenario register; cloud scenario patterns in the scenario builder                                                                                                           | Assess a cloud misconfiguration scenario                              |
| 16 Privacy                   | Privacy scenarios in the same register; evidence register                                                                                                                    | Add a Law 25 privacy scenario beside security scenarios               |
| 17 Economics                 | Cost-effectiveness per \$1,000; budget guideline; drivers, sensitivity, and Monte Carlo forecasts                                                                            | Compare the risk-driven portfolio with the appetite-consistent budget |
| 18 AI                        | Governed AI with payload preview, anonymization, and local-model option; label check                                                                                         | Use AI output only after validation and recomputation                 |
| 19 Third parties             | Supplier scenarios; third-party risk module (where available)                                                                                                                | Assess a supplier-concentration scenario                              |
| 20 Cryptography              | Scenario builder; CVSS by equivalency for process weaknesses                                                                                                                 | Model a certificate-expiry scenario                                   |
| 21 People and accountability | Risk owners; *Management decision required*; quality review; multi-user RACI (where available)                                                                               | Assign owners and decision rights for residual risks                  |

Three cautions apply across the table. First, CVSS is a mandatory multiplier in the risk index but never a ranking or selection criterion; for process, people, governance, physical, and third-party weaknesses, students score a representative weakness by equivalency, as Chapter 6 explains, and label it as of limited applicability. Second, the risk index is a relative indicator for comparison, not a dollar amount and not a KRI; KRIs are the measured indicators, with owners, sources, and thresholds, that monitor whether a risk is moving. Third, a treatment package reduces the calculated risk only when it acts on both probability and impact, because the two reductions are multiplied; a single-lever measure such as insurance is recorded as risk transfer, not as mitigation.

## The Excel teaching template

The education package includes a free, blank CyberRiskGuardian Excel model, used for the team risk-assessment assignment in courses built on this book. It contains an Analyse sheet with one row per scenario, ten scenario input sheets (S1 to S10), and ten mitigation sheets (M1 to M10) holding a control catalogue. Its transparency is its main strength: every intermediate value can be inspected, which helps students check their hand calculations against the engine.

The corrected template that accompanies this edition of the book follows the CyberRiskGuardian engine exactly. Students should keep the following points in mind when they use it.

- **Recalculate deliberately.** The workbook opens in manual calculation mode, so values change only after Calculate Now (F9).

- **Enter every required input.** CVSS cells start blank and accept scores from 0 to 10; a missing score is flagged as "CVSS required" rather than silently producing a zero. Resilience must be greater than 0 and at most 1, and the appetite must lie strictly between 0 and 1. The multiplication factor is 1,000; it scales the scores but never changes a ratio or a verdict.

- **Keep reductions between 0 and 1.** A package's probability reduction and impact reduction are each capped at 1. When the summed coefficients of selected controls would exceed 1, the workbook caps the value and displays a warning so that the package can be reviewed; a single package-level estimate can also be entered in an "Other" row.

- **Read the verdict columns.** The workbook computes the pre-treatment ratio, the risk-to-tolerance ratio with its verdict (below, approximately at, or above tolerance), cost-effectiveness per \$1,000, and the best-practice budget benchmark for the organization's appetite.

- **Use current catalogues.** The mitigation sheets list ISO/IEC 27002:2022 controls, NIST CSF 2.0 outcomes, and the eighteen CIS Controls v8.1, as Chapter 9 describes. Default costs and coefficients are placeholders to be replaced with case-specific estimates.

- **Count shared controls once.** When one initiative serves several scenarios, split its cost across them for scenario analysis and count it once in the portfolio total.

## Using AI with CyberRiskGuardian responsibly

CyberRiskGuardian is itself AI-assisted, so the rules of Chapter 18 apply to it directly. AI may draft scenarios, propose parameter values with rationales, critique estimates, and suggest controls. It may not supply facts about a case that the case does not contain, produce a final number that the student has not recomputed with the verified engine, or invent standards, control identifiers, legal obligations, statistics, or sources. Every AI-proposed value carries the *Analytical estimate — validation required* marker until a person has checked it.

Data protection comes first. Students should use only an AI service their institution has approved, and they should never paste confidential information about a real organization into an external service without written authorization. Fictional cases, such as the five in the CRG Case Library, are safe for coursework. In the desktop application, AI requests show the exact content before anything is sent, can be anonymized, and can be routed to a local model or switched off entirely, which is the appropriate choice whenever the data are sensitive.

## Where to get it

The Community Edition, including the education package, the Excel template, the MediBec example, and the quick-start guides in English and French, is published in the public GitHub repository ITriskMgr/CyberRiskGuardian. Start with the education README, which lists the Instructor Guide, the Student Lab Workbook, the Assessment Kit, the Case Library, the introductory slide deck, and the blank Excel model.

Product information, the Educational Licence for colleges and universities, and the professional editions are presented on the product site, cyberriskguardian.ca, and licence enquiries can be sent to sales@cyberriskguardian.ca. Instructors planning a course should contact the institution's program coordinator early, so that installation, customization, and the Educator Edition are in place before the first lab.

| Whichever edition or track a course uses, the test is the same: every number traceable to evidence, every estimate marked, every risk decision left to the people accountable for it. |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
