---
title: "CyberRiskGuardian in the Classroom"
subtitle: "Instructor Guide — Education Package 1.0"
author: "Marc-André Léger, PhD"
date: "October 2026"
---

# About this guide

This guide helps instructors, teaching assistants and program coordinators bring CyberRiskGuardian (CRG) into a course on cybersecurity risk management, governance, IT audit or business technology management. It explains what the method teaches, which version of the tool to use, how to fit it into a single workshop or a full term, how to run the labs, and how to assess the work fairly when students are allowed to use generative AI.

The package was built from the material that ships with CyberRiskGuardian (the method specifications, the governance textbook, the MediBec worked example and the desktop application's teaching features) and from the author's own course, *BTM 440 Business Information Systems Risks, Security and Audit* (Concordia University, Fall 2026), where CRG has been used for the team risk-assessment assignment.

## What is in the package

| Component | Audience | Edition |
|---|---|---|
| **Instructor Guide** (this document) | Instructors, TAs, coordinators | Community (free) |
| **Student Lab Workbook** — twelve labs on the MediBec case, with self-check answers | Students | Community (free) |
| **Assessment Kit** — assignment briefs, 0–10 rubrics, self-checks, peer-review form, AI acknowledgement | Students and instructors | Community (free) |
| **Case Library** — five fictional business cases with a guide to choosing one | Students and instructors | Community (free) |
| **Introductory slide deck** — the CRG method in one class | Instructors | Community (free) |
| **Educator Edition** — lab answer keys, graded quiz bank, rubric calibration and feedback bank, case teaching notes, the final-exam case and its blueprint | Instructors only | Educational Licence |

Everything except the Educator Edition is published with the Community Edition under CC BY-NC 4.0, in English and French. The Educator Edition is kept out of the public repository so that answer keys and the exam case stay useful from one term to the next.

## Who may use it, and when a licence is needed

- **Students** may use CyberRiskGuardian and this package free of charge for their own coursework, including work handed in for a grade.
- **Instructors** may read, adapt and reuse the free materials for non-commercial teaching, with attribution.
- **Colleges, CEGEPs and universities, as institutions,** need the CyberRiskGuardian **Educational Licence** to deliver a course built on CRG. It is offered at a significantly reduced price and includes installation, customization and the Educator Edition.
- **Non-profit organizations** may use CRG free of charge to assess their own risk.

Licence enquiries: **sales@cyberriskguardian.ca** · product site: **cyberriskguardian.ca**. The CC BY-NC 4.0 legal code shipped in the repository governs where this summary and the licence differ. This is not legal advice.

# CyberRiskGuardian in one page

CyberRiskGuardian is a scenario-driven method for assessing cybersecurity risk, supported by an AI assistant (the Claude plugin and project), a desktop application and an Excel model. It is a **decision-support tool**: it structures the analysis and speeds it up, while the analyst stays accountable for every estimate and management stays accountable for every risk decision.

**The workflow.** Organizational context → crown jewels → risk appetite → 20 candidate scenarios → the 10 most material → six parameters and CVSS for each → estimated and tolerated risk → treatment portfolio → residual risk against tolerance → recommendations, budget and KRIs.

**A scenario is a causal chain, never a threat name.** *Threat source or event → vulnerability or weakness → affected asset or process → cybersecurity event → organizational consequence.* "Ransomware" is a threat; "a ransomware affiliate uses stolen VPN credentials and the absence of MFA on remote access to encrypt the clinical scheduling servers, stopping appointments at all sites for several days" is a scenario.

**Six parameters, each on a 0–1 scale.**

| Symbol | Parameter | Question it answers |
|---|---|---|
| Pb(A) | Probability that the threat is present | Will this threat source act against us in the next 12 months? |
| Pb(ψ,A) | Probability of exploitation | If it acts, will it get through our weakness? |
| δe(ψ,A) | Expected damage | How much harm, reasonably expected? |
| δm(ψ,A) | Maximum damage | How much harm, plausibly at worst? |
| θ(ψ,A) | Organizational resilience | How well can we prevent, detect, contain, respond, recover and keep operating? |
| μ(E) | Expected utility / criticality | How important is the affected service to the mission? |

**The formulas** (Excel Guide v1.0c, section 11 — authoritative, never changed silently):

```
Estimated risk  Re = Pb(A) × Pb(ψ,A) × CVSS × ((δe + δm) / 2) × μ(E) ÷ θ × Factor
Tolerated risk  Rt = Pb(A) × Pb(ψ,A) × CVSS × Appetite × μ(E) ÷ θ × Factor
Mitigated risk     = Re × Probability reduction × Impact reduction
Residual risk      = Re − Mitigated risk
Ratio              = Residual ÷ Tolerated
```

A ratio below 0.90 is read as **below tolerance**, 0.90 to 1.10 as **approximately at tolerance**, above 1.10 as **above tolerance**. The factor (usually 1,000) scales the scores; it never changes a ratio.

**Three markers on every deliverable** keep the evidence honest:

| Marker | Meaning |
|---|---|
| **Analytical estimate — validation required** | A value the analyst (or the AI) estimated |
| **Information gap — organizational validation required** | A fact nobody has supplied |
| **Management decision required** | A choice only management can make |

These markers are the single most useful habit the method teaches. A student who uses them correctly has understood the difference between evidence, inference and judgement.

# Learning outcomes

CRG supports learning outcomes that most risk, security and audit courses already have. The table maps them to the labs and assessments in this package. Adapt the wording to your program's competency framework.

| # | By the end of the course, students will be able to… | Labs | Assessment |
|---|---|---|---|
| LO1 | Describe an organization's context, critical processes and crown-jewel assets from case evidence, separating facts from assumptions | 1 | A1 (context), exam |
| LO2 | Estimate and justify a cybersecurity risk appetite and explain how it differs from tolerance and from a risk score | 2 | A1, quiz |
| LO3 | Write plausible cybersecurity risk scenarios as complete causal chains and select a material portfolio | 3 | A1 (S1), exam |
| LO4 | Estimate the six CRG parameters with rationale, evidence and confidence, and score a representative vulnerability with CVSS v4.0 | 4, 5 | A1 |
| LO5 | Calculate estimated, tolerated, mitigated and residual risk and interpret the result against tolerance | 6 | A1, quiz |
| LO6 | Design a cost-effective, non-duplicated treatment portfolio tied to the causal chain | 7 | A1 recommendation |
| LO7 | Test how conclusions change under different assumptions | 8 | A1, exam |
| LO8 | Define key risk indicators with owners, sources and thresholds | 9 | A1, A2 |
| LO9 | Communicate risk to senior management in a decision-oriented memorandum | 10 | A1 memo, exam |
| LO10 | Use generative AI responsibly: verify its output, label estimates, protect data and disclose its use | All | All (professionalism criterion) |
| LO11 | Translate risk results into a security policy and a risk-based audit plan | 12 | A2, A3 |

# Choosing a tool track

The method is the same in every track; only the interface changes. Pick one track per course and say so in the outline, so that teams are graded on the same footing.

| Track | What students use | Strengths | Watch for |
|---|---|---|---|
| **A. Claude (plugin or project)** | The free CyberRiskGuardian Core plugin in Claude Code or Claude Cowork, or a Claude project carrying the CRG instructions | Fast scenario drafting, interview-style context intake, built-in label checks, `/crg-quick-scan` and `/crg-assess` commands, verified calculator | Students must verify AI output; institutional AI policy and data-protection rules apply |
| **B. Desktop application** | CyberRiskGuardian Desktop (Beta 1.6.x/1.7.x), served locally in a browser | Everything on one screen; classroom workspaces; engine self-check; *Start over* for teaching cases; simulation data for SIEM/SOAR exercises; EN/FR interface; no data leaves the computer unless the student sends it | Requires Python 3 to serve the folder; keep the same port so workspaces are found |
| **C. Excel model** | The free CyberRiskGuardian Excel model (Analyse sheet, S1–S10, M1–M10), in `templates/` | Transparent formulas, familiar to business students, works offline | Manual calculation mode (press F9); no input validation; red highlighting on the Analyse sheet can mislead |

A **hybrid** works well in practice: students draft context and scenarios with Claude (Track A), then quantify and treat in the desktop application or the workbook (Track B or C), where every number is recomputed by the verified engine.

## Minimum set-up per track

**Track A — Claude.**

```
/plugin install cyberriskguardian --marketplace ITriskMgr/CyberRiskGuardian
/plugin install crg-governance-textbook --marketplace ITriskMgr/CyberRiskGuardian
pip install cvss "mcp>=1.2"
```

Restart Claude and approve the `crg-calculator` server when prompted. In claude.ai, the skills work too; calculations use the bundled `crg_calc.py` script.

**Track B — Desktop application.** Unzip the release, then double-click `start.command` (macOS, Linux) or `start.bat` (Windows). The application must be served over `http://localhost:8099`; double-clicking `index.html` does not work. Install it as an app from Chrome or Edge for offline use.

**Track C — Excel.** The blank model ships free with this package: `templates/CyberRiskGuardian_Excel_Model_Blank_v2-4e.xlsx`. Post it on your learning platform. Remind students that it opens in manual calculation mode, and that its defaults (appetite 0.5 in E14, factor 10,000 in E15, CVSS 0 in G3:G12) must be replaced with the case's values.

## The check every student does first

Whatever the track, the first thing a student does is confirm that the calculation engine agrees with the reference implementation:

- Desktop: the sidebar footer must read **engine verified · 84,491 / 41,104**.
- Claude or command line: `python3 crg_calc.py medibec30.json` must report **total estimated 84,491** and **residual 41,104**.

If it does not, nothing produced on that installation should be trusted. Make this check part of Lab 0 and ask for a screenshot.

# Integration models

Choose the model that fits the time you have. Each model lists the sessions, the labs and a suggested weighting. The labs are described in the Student Lab Workbook.

## Model 1 — One workshop (3 hours)

For an existing course that wants a single hands-on introduction.

| Time | Activity |
|---|---|
| 0:00–0:30 | Introductory deck (slides 1–9): why scenarios, the six parameters, the markers |
| 0:30–0:45 | Lab 0: install or open the tool, verify the engine |
| 0:45–1:30 | Lab 3 (shortened): teams write three causal-chain scenarios for MediBec and critique each other's |
| 1:30–1:45 | Break |
| 1:45–2:30 | Labs 4 and 6: quantify one scenario and calculate its risk; compare with the worked example |
| 2:30–3:00 | Debrief: why the tolerance verdict depends only on damage, treatment and appetite; what management would decide |

Assessment: a one-page reflection or the Lab 6 output (pass/fail).

## Model 2 — Three-week module

For a course that dedicates a unit to risk assessment.

| Week | Class | Labs | Deliverable |
|---|---|---|---|
| 1 | Context, appetite and scenarios | 0, 1, 2, 3 | Team scenario set (20 candidates → 10) |
| 2 | Quantification, CVSS and interpretation | 4, 5, 6, 8 | Quantified portfolio with rationale |
| 3 | Treatment, KRIs and the management memo | 7, 9, 10 | One-page memo and two-page recommendation |

Suggested weight: 20–30% of the course grade, using the A1 rubric of the Assessment Kit.

## Model 3 — Full term (12 weeks)

This is the model used in BTM 440. The risk assessment (A1) feeds the security policy (A2) and the audit plan (A3), so students follow one organization from risk to governance to assurance.

| Week | Course topic | CRG activity | Due |
|---|---|---|---|
| 1 | Governance, compliance, the security professional | Course overview; introductory deck; Lab 0 | — |
| 2 | Endpoint and network security | Lab 1: context and crown jewels (MediBec) | — |
| 3 | Risk management and privacy; performing a risk assessment | Labs 2, 3 and 4; teams receive their case | — |
| 4 | Threat landscape; security policies | Lab 5 (CVSS v4.0); teams draft their original scenario S1 | — |
| 5 | Malicious code; social engineering; protecting information assets | Lab 6; team check-in on S1–S10 | — |
| 6 | Security assessment and testing; cybersecurity audits | Labs 7 and 8 | — |
| 7 | Cryptography and PKI | Lab 10 (memo clinic) | **A1 risk assessment** |
| 8 | Identity and access management; Zero Trust | Lab 9 (KRIs); A1 feedback | — |
| 9 | Resilience; monitoring and incident response | Lab 11 (optional, desktop): MediBec simulation pack | — |
| 10 | Cloud, wireless and mobile security | Lab 12, part 1: from risk to audit plan | — |
| 11 | Digital forensics; standards and certification | Lab 12, part 2: from risk to IAM policy | **A3 audit plan** (individual) |
| 12 | Application security; review | Exam preparation; quiz bank (Educator Edition) | **A2 IAM policy** |

Suggested weights, as in BTM 440: A1 20% (team), A2 10% (team), A3 10% (individual), final exam 40% (including an essay on a business case), quizzes and labs 20%.

## Model 4 — Capstone or graduate seminar

Students work on a richer case (CNIC or ACRB) or, with the organization's written authorization and the data-protection precautions in section 9, on a real non-profit. Use the full scenario-selection procedure (Sections A–E: context, candidates, screening, final portfolio, quality review), the sensitivity analysis and the budget calibration. Require an oral defence of the portfolio before a panel acting as the board.

# Before the term: a checklist

1. **Pick the track** (section 4) and test it on the machines students will use, including the lab computers.
2. **Pick the cases** (Case Library). Use MediBec for the labs, because its full worked example is public and students can check themselves against it. Assign a *different* case to each team for the graded assignment, and rotate cases from term to term.
3. **Prepare a classroom workspace** (desktop track): Organization → Workspaces → New workspace → *a copy of the MediBec example*, type *Classroom business case*, tick *Educational case used for training purposes*. When it is ready to hand out, Organization → Reset data → *For the instructor* → **Set the current state as the starting point**, so that teams can start over.
4. **Decide which steps students redo** after *Start over* (by default, steps 2 and 3: context and crown jewels).
5. **Load the simulation pack** if you plan Lab 11 (Settings → Simulation data → MediBec pack). The import report must show 40 alerts, 13 rules, 15 indicators, 4 cases and one refused self-approval (APR-MB-04).
6. **Write your AI rules into the outline** (section 9) and attach the AI acknowledgement template from the Assessment Kit.
7. **Publish the Assessment Kit** with your due dates, page limits and weights.
8. **Check the institution's data-protection rules** for AI tools and decide which AI providers students may use. In the desktop application, AI features can be routed to a local model or switched off.

# Running the labs

Each lab in the workbook states its objectives, timing, inputs, steps, deliverable and three self-check questions. The notes below are for the person running the session.

| Lab | Time | What to watch for | A question that unlocks the room |
|---|---|---|---|
| 0. Getting started | 30 min | Students opening `index.html` directly (footer stuck on *engine loading…*); different ports losing workspaces | "Why should you not trust any number until the footer says *verified*?" |
| 1. Context and crown jewels | 60 min | Copying the case text instead of analysing it; inventing facts | "Which of these facts would change your risk ranking if it were wrong?" |
| 2. Risk appetite | 45 min | Confusing appetite (0–1 attitude) with a risk score; ignoring the value the case states | "MediBec states 0.40. Why might an analyst propose 0.30?" |
| 3. Scenarios | 90 min | Threat names ("phishing") instead of causal chains; ten variants of one technique | "Which step of your chain would a control interrupt?" |
| 4. Quantification | 75 min | Merging Pb(A) and Pb(ψ,A); δe greater than δm; θ set to zero; numbers without reasons | "Which two values are you least sure of, and what evidence would settle them?" |
| 5. CVSS v4.0 | 60 min | Scoring the whole scenario instead of one weakness; estimating a score by eye | "What vulnerability, exactly, did you score?" |
| 6. Calculation | 45 min | Reading the estimated risk as a dollar figure; ignoring tolerated risk | "Why does a rare scenario get the same verdict as a frequent one with equal damage?" |
| 7. Treatment | 90 min | "Control dumping"; counting a shared control twice; reductions summing above 1 | "What does this control add that the organization does not already have?" |
| 8. Sensitivity | 45 min | Treating the central case as the truth | "Which decision changes between the optimistic and the pessimistic case?" |
| 9. KRIs | 45 min | Indicators without owners, sources or thresholds | "Who will measure this on the first of the month, and from which system?" |
| 10. Memo | 60 min | Restating tables; burying the decision | "What do you want the CFO to approve on Monday?" |
| 11. Security operations (optional) | 120 min | Treating an alert as an incident; using a source confidence as a probability | "Who confirmed this incident, and when?" |
| 12. From risk to policy and audit | 90 min | Policies full of technical values; audit plans that only check a policy exists | "How would an auditor prove this control works over a period?" |

# Teaching the hard concepts

Experience with the method shows that a handful of ideas cause most of the confusion. Teach them explicitly.

**Appetite is not tolerance, and neither is a score.** The appetite (0–1) expresses how much risk management is willing to pursue; the tolerated risk is the appetite applied to a specific scenario through the formula. The textbook plugin answers "What is the difference between risk appetite and risk tolerance?" with a chapter citation — a good warm-up for Lab 2.

**Threat presence and exploitation are two different probabilities.** Pb(A) asks whether the threat source will act; Pb(ψ,A) asks whether it will succeed against this weakness given existing controls. Merging them double-counts.

**CVSS scores a vulnerability, not a scenario.** CVSS v4.0 Base measures the technical severity of one defined weakness. For phishing, insider or governance scenarios, students score a *representative* weakness and label it *limited applicability*, or explain why CVSS does not apply. A high CVSS never means the highest business risk.

**The tolerance verdict depends only on damage, treatment and appetite.** Residual ÷ Tolerated simplifies to ((δe + δm)/2) × (1 − P × Q) ÷ Appetite. Probabilities, CVSS, resilience and criticality cancel out: they drive the *size* of the risk (the ranking) but not the *verdict*. This is a striking result for students. Use it to show why the ratio must always be read together with estimated risk.

**Before treatment, almost everything is above tolerance.** The pre-treatment ratio is average damage ÷ appetite. With an appetite of 0.30, any scenario whose average damage exceeds 0.33 starts above tolerance. Students who expect a mixed picture before treatment are usually surprised.

**Mitigation is multiplicative.** An 80% probability reduction and an 80% impact reduction remove 64% of the risk, not 80%. "Below tolerance" requires P × Q ≥ 1 − appetite ÷ average damage. A package that reduces only probability, or only impact, mitigates nothing under the workbook convention.

**Shared controls are counted once.** When one initiative serves several scenarios, its cost is split across them (an equal split is an acceptable default) and the portfolio total comes from the initiatives, not from the sum of scenario allocations.

**Two lenses may disagree.** A simple normalized score (threat × exposure × impact × (1 − control maturity)) with escalation thresholds can place a treated risk within authority limits while the CRG ratio flags it above tolerance. Ask students to say which method a governance threshold is tied to.

# Generative AI in the course

CyberRiskGuardian is itself an AI-assisted tool, so the course must take a clear position on AI. The position below is the one used in BTM 440; adapt it to your institution's policy.

**Allowed on assignments,** with an explicit acknowledgement (template in the Assessment Kit). Students are responsible for everything they submit, including anything an AI produced.

**Not allowed during the exam.** Students may prepare material with AI before the exam (for example risk scenarios submitted with the exam) if they disclose it.

**What AI may do:** draft scenarios, propose parameter values with rationales, critique estimates, suggest controls, check grammar, summarize sources.

**What AI may not do:** state facts about the case that the case does not contain, produce a final number that the student has not recomputed with the verified engine, invent standards, control identifiers, legal obligations, statistics or sources.

**Verification is the skill being graded.** AI tools invent control identifiers, CVSS vectors and laws with great confidence. Grade the verification, not the prose: a fabricated or untraceable source is both an evidence problem and an academic-integrity concern.

**Data protection.** Students do not paste confidential material into an external AI service unless authorized. Fictional cases are safe; real organizations are not, unless the organization has agreed in writing and the institution's rules allow it. In the desktop application, AI requests show the exact payload before anything is sent, can be anonymized, and can be routed to a local model.

**The markers are the bridge.** Ask students to keep the three markers in every deliverable. A marker-free deliverable is a warning sign that AI output was taken as fact. The Claude plugin includes a label check (`check_labels.py`) that lists missing markers.

# Assessment strategy

The Assessment Kit contains:

- the **0–10 scale** and how criterion scores combine into an assignment score;
- **briefs and rubrics** for the risk assessment (A1), the IAM security policy (A2) and the risk-based audit plan (A3), plus four shorter formats (lab portfolio, individual scenario paper, in-class quick-scan challenge, exam essay);
- **self-check lists**, a **peer-review form** for team assignments and the **AI acknowledgement**.

The Educator Edition adds the lab answer keys, a graded quiz bank with answers, rubric calibration examples, a feedback-comment bank, teaching notes for each case and the final-exam case with its blueprint.

**Protecting the integrity of the assessment.**

- Use MediBec for practice only. Its full answer is public.
- Rotate cases between terms and between sections. The Case Library lists five cases; the Educator Edition holds a sixth, reserved for the exam.
- Require the original scenario S1 to be distinct from every scenario supplied in the case and in the guides.
- Require the workbook or workspace export with the memo, and check that the numbers agree. Disagreement between files is the most common sign of unverified AI output.
- Ask for a short oral walk-through of one scenario's parameters from a randomly chosen team member.

# Bilingual and inclusive delivery

The desktop application switches its whole interface between English and French (EN/FR in the top bar) while leaving case data in its original language. All the free materials in this package exist in both languages. A French case remains French in an English interface; teams may write in either language if your institution allows it.

For accessibility, the labs can be done entirely with the keyboard in the desktop application and in Excel; the slide deck uses high-contrast text; every figure in the workbook is also given as text. Allow extra time for Lab 3 and Lab 10, which are writing-intensive.

# Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Footer stuck on *engine loading…* | The application was opened as a file | Start it with `start.command` or `start.bat`; open `http://localhost:8099` |
| A student's workspaces have disappeared | Different address, port or browser | Use the same address and port; import the last workspace export |
| Excel shows formulas but old numbers | Manual calculation mode | Formulas → Calculate Now (F9) |
| A `#DIV/0!` error in the workbook | θ left blank or at zero | θ must always be above zero |
| A scenario's ratio did not change after adding measures | Measures not applied, or only one dimension reduced | Apply the package; add a measure for the other dimension |
| Residual risk *rises* in the Excel model | Control coefficients sum above 1 on an M sheet | Keep each reduction within 0–1; fewer, better-justified controls |
| AI proposes a law the case does not name | AI invention | Remove it; cite only obligations the case names or that a primary source confirms |
| The simulation pack will not import | The workspace is not a MediBec copy | Import into the MediBec example or a copy of it |
| `engine DIVERGES` banner | The engine disagrees with the reference | Stop and restore a clean copy of the application |

# Support, feedback and updates

- Repository: **github.com/ITriskMgr/CyberRiskGuardian** (Community Edition, CC BY-NC 4.0)
- Product site and Educational Licence: **cyberriskguardian.ca** · **sales@cyberriskguardian.ca**
- Desktop application: user guide under **Help → User guide**, in English and French

Report errors in this package through the repository's issue tracker, naming the document, the section and the language version. Suggested improvements from instructors who have taught with CRG are especially welcome.

*© 2026 Marc-André Léger. Community materials licensed under CC BY-NC 4.0. Cases are fictional; any resemblance to real organizations is coincidental. CyberRiskGuardian is decision support: estimates are initial analytical hypotheses, and risk decisions remain with management.*
