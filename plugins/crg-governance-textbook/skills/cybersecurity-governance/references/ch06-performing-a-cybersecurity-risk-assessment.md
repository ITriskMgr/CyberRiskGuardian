<!-- Léger, M.-A. (2026). Introduction to Cybersecurity Governance for Business Technology Management (3rd ed., v2.51). Chapter 6: Performing a Cybersecurity Risk Assessment. CC BY-NC 4.0 -->

# Chapter 6: Performing a Cybersecurity Risk Assessment

## Learning outcomes

After completing this chapter, you should be able to:

1.  Explain how a CyberRiskGuardian assessment implements ISO/IEC 27001:2022 Clauses 6.1.2 and 6.1.3, ISO/IEC 27005:2022, NIST SP 800-30 Rev. 1, the NIST RMF Prepare step, and NIST CSF 2.0 GV.RM and ID.RA.

2.  Describe the inputs and the nineteen-step CRG workflow, and apply its evidence discipline and three standard markers.

3.  Calculate estimated risk (Re), tolerated risk (Rt), mitigated and residual risk, the risk-to-tolerance ratio, and cost-effectiveness using the CRG formulas.

4.  Interpret the tolerance bands and the cancellation property, and use the design rule to determine how strong a treatment package must be.

5.  Score a representative weakness with CVSS v4.0, including its vector string, nomenclature, and severity band, and score non-technical weaknesses by equivalency.

6.  Evaluate treatment packages, shared controls, transfer, and avoidance under CRG's multiplicative convention, and prioritize scenarios on multiple criteria.

7.  Calculate an appetite-consistent cybersecurity budget target and the appetite implied by actual funding, and recommend how to report any gap.

8.  Design key risk indicators, a management report, and a quality control review, and justify the role of incident management, continuity, and recovery in handling residual risk.

Chapter 4 explained what cybersecurity risk is and how an organization expresses the amount of risk it is prepared to accept. Chapter 5 showed how to turn that understanding into risk scenarios: causal chains that link a threat source, a weakness, an affected asset or process, a cybersecurity event, and a business consequence. This chapter takes the next step. It shows how to move from a portfolio of scenarios to a concrete, repeatable cybersecurity risk assessment that produces numbers management can compare, treatment options management can fund, and residual risk management can formally accept or refuse.

The governance problem is not a shortage of methods. Organizations can choose among dozens of risk assessment approaches, from simple heat maps to probabilistic loss models. The problem is that many assessments are neither repeatable nor defensible. Two analysts given the same evidence produce different results; numbers appear without rationale; tolerance is asserted rather than tested; and the report ends with a list of controls rather than a set of decisions. When a regulator, an auditor, an insurer, or a board member asks why a scenario was rated as it was, or why one investment was preferred over another, the answer is often a shrug. A risk assessment that cannot be explained cannot support accountability.

This chapter teaches one method in depth, the **CyberRiskGuardian (CRG)** method developed for this book and implemented in an open, scenario-driven toolset: an AI-assisted analyst track, a desktop application, a calculation engine, and an Excel model. The method is deliberately transparent. Every formula is published, every parameter has a defined range and meaning, every estimate carries a rationale and an evidence reference, and every result is read against the organization's risk appetite. Where the book and the CRG implementation might differ, this chapter follows CRG exactly, so that what students learn here is what the tool computes.

The method is intentionally simplified for teaching and introductory practice, but it is not a toy. It implements the risk assessment and risk treatment logic of ISO/IEC 27001:2022 and ISO/IEC 27005:2022, it follows the risk factors of NIST SP 800-30 Rev. 1, and it produces the inputs that the NIST Cybersecurity Framework (CSF) 2.0 expects in its Govern and Identify functions. It also has limits, and this chapter states them openly, including a property of the tolerance test that surprises most people the first time they see it.

All teaching activities associated with this book should use CyberRiskGuardian's **Community Edition**, which is free for students and non-profit organizations under the Creative Commons CC BY-NC 4.0 licence, or its **Educational** edition, licensed to colleges and universities as institutions, which includes installation and customization. The worked example in this chapter uses MediBec, the fictional medical centre whose complete CRG assessment is published with the Community Edition, so that students can check every number themselves.

The chapter proceeds in the order an analyst works. It begins with purpose and standards alignment, then describes the inputs and the CRG workflow, the formulas and their variables, and the scoring of vulnerabilities with CVSS v4.0, including how to score weaknesses that are not technical at all. It then walks through the procedure in the spreadsheet and the application, works a complete example, and covers prioritization, budget calibration, key risk indicators, the management report, the quality control review, and the role of incident management and continuity planning in dealing with what remains. It closes with responsible use of generative AI, an illustrative scenario, and the questions managers and boards should ask.

## From concepts to practice: purpose and standards alignment

A risk assessment is the bridge between the conceptual vocabulary of Chapters 1 to 4 and the decisions of later chapters. It is where threats, vulnerabilities, and impacts stop being abstractions and become a ranked list of scenarios, each with an owner, an estimate, a tolerance test, and a recommended response.

### Why we assess risk

In previous chapters, risk was defined as the intersection of three components: a **threat** or threat agent, a **vulnerability** or predisposing condition that can be exploited, and a **potential exposure to loss or negative outcome**. When a threat exploits a vulnerability in a way that produces a negative outcome for the organization, such as financial loss, operational disruption, a regulatory breach, harm to people, or reputational damage, the potential for loss has materialized into actual damage. Before that point, the situation was a risk. Once it materializes, it has become an incident with actual consequences.

Risk assessment helps organizations move from reactive to proactive behaviour. Instead of waiting for incidents to occur, management asks four questions, and a good assessment answers each of them explicitly:

- **What could realistically go wrong in our environment?** The answer is a portfolio of organization-specific scenarios, not a list of threat names.

- **How severe would the consequences be** for finances, operations, compliance, safety, and strategy, both in the expected case and in a plausible worst case?

- **How likely is each scenario** over a defined horizon, given the threats we actually face and the controls we can actually evidence?

- **Which combination of mitigation, transfer, avoidance, or acceptance makes sense** given our risk appetite and limited resources, and who must decide?

### The IPM cycle and the GRC continuum

In the governance, risk, and compliance (GRC) continuum, risk assessment belongs to the **R**. Governance defines objectives, obligations, and risk appetite. Risk management uses those parameters to identify, analyze, and evaluate scenarios and to choose responses. Compliance and assurance functions provide independent confirmation that decisions and controls are appropriate and effective. A risk assessment that ignores the governance parameters, for example by using a default appetite instead of the approved one, produces results that are internally consistent but organizationally meaningless.

The practical assessment described here follows the **IPM** logic introduced earlier in the book. **Identification** of threats, vulnerabilities, and potential adverse impacts is achieved through scenarios, crown-jewel analysis, and business impact information. **Prioritization** of scenarios uses calculated risk indices, the organization's risk appetite, and a set of managerial criteria. **Mobilization** of resources follows through the chosen risk responses: mitigation projects, transfer mechanisms, formal risk acceptance, and preparedness through incident management, business continuity, and disaster recovery. CRG operationalizes this cycle in a structured, traceable way that students and, later, practitioners can apply.

### Purpose of the CyberRiskGuardian method

CRG has five design objectives. It aims to illustrate an end-to-end assessment workflow that a team can complete within a course module; to make explicit the link between qualitative judgment and quantitative indicators; to show how risk appetite drives treatment decisions; to remain aligned with recognized standards; and to provide free, transparent tools that prepare students for the commercial platforms and methodologies they will meet in practice. Above all, CRG is a **decision-support** method. It structures and accelerates the analysis, but the analyst remains accountable for every estimate and management remains accountable for every risk decision.

### Alignment with ISO/IEC 27001 and ISO/IEC 27005

ISO/IEC 27001:2022 (as amended in 2024) requires the organization to define and apply an information security risk assessment process in **clause 6.1.2**. That process must establish and maintain risk criteria, including risk acceptance criteria and criteria for performing assessments; ensure that repeated assessments produce consistent, valid, and comparable results; identify risks associated with the loss of confidentiality, integrity, and availability and assign risk owners; analyze consequences and likelihood; and evaluate risks against the criteria to prioritize them for treatment. **Clause 8.2** then requires the organization to perform those assessments at planned intervals or when significant changes are proposed or occur, and to retain documented information about the results. Clause 6.1.3 and clause 8.3 do the same for risk treatment. CRG implements each of these requirements: risk criteria are the appetite and the tolerance bands; consistency comes from one formula, one horizon, and one set of anchors; risk owners are named per scenario; consequences and likelihood are separate parameters; and evaluation is the tolerance test.

ISO/IEC 27005:2022 provides guidance for performing these activities. It describes context establishment, risk identification, analysis, and evaluation, risk treatment, acceptance, communication and consultation, recording and reporting, and monitoring and review. It also recognizes both asset-based and event-based approaches to identification and gives prominence to **risk scenarios**, which is exactly the construct CRG uses. Each scenario in a CRG assessment is one row of analysis; each row can be traced back through its causal chain to the assets, threats, and weaknesses that produced it.

### Alignment with NIST guidance

NIST SP 800-30 Rev. 1 structures a risk assessment around threat sources, threat events, vulnerabilities and predisposing conditions, likelihood, and impact, and it organizes the work into preparing for, conducting, communicating, and maintaining the assessment. CRG's parameters map onto these factors directly: the probability that the threat is present and the probability that it exploits the weakness express likelihood; expected and maximum damage express impact; resilience captures the organization's ability to limit harm. In the NIST Risk Management Framework described in NIST SP 800-37 Rev. 2, the assessment corresponds to the risk assessment tasks of the **Prepare** step, at the organization level and at the system level, not to the RMF **Assess** step, which verifies whether selected controls are implemented correctly and operating as intended. This distinction matters, because students often confuse "assessing risk" with "assessing controls".

NIST CSF 2.0 places the governance parameters of an assessment in its **Govern** function, notably the Risk Management Strategy category (GV.RM), which expects risk appetite and risk tolerance statements to be established, communicated, and maintained. The assessment itself sits in the Risk Assessment category of the **Identify** function (ID.RA), which expects threats, vulnerabilities, likelihoods, and impacts to be used to understand inherent risk and inform the prioritization of risk responses, and expects those responses to be chosen, prioritized, planned, tracked, and communicated. A CRG assessment therefore provides documented evidence for both categories at once.

| A risk assessment is defensible when another analyst, given the same evidence, the same appetite, and the same method, would reach substantially the same conclusions—and when every number can be traced to a rationale. |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## Inputs and the CyberRiskGuardian workflow

Before any calculation, an assessment needs inputs, and it needs a workflow that turns those inputs into decisions in a fixed order. CRG defines both. Following the order is not bureaucracy: each step depends on the outputs of the one before, and skipping a step, such as setting appetite after the numbers are known, is the most common way an assessment loses its credibility.

### Inputs to the assessment

Some inputs are produced in earlier chapters; others come from the organization or, in a course, from the business case. Typically, the following are required:

1.  **Organizational documents and evidence.** Descriptions of the mission, services, critical processes, systems and data, architecture, suppliers, obligations, existing controls, incident history, audit findings, continuity arrangements, and the approximate IT and cybersecurity budgets. Every document is reviewed before anything is written.

2.  **An approved risk appetite**, if one exists, on a 0–1 scale, together with any statements management has made about risk. If none is supplied, the analyst estimates one and labels it as an estimate.

3.  **Candidate risk scenarios**, created as described in Chapter 5. CRG expects at least twenty candidates for a ten-scenario assessment, and preferably 25 to 40.

4.  **Information to estimate the six scenario parameters**: threat intelligence for the sector, evidence of weaknesses, business impact information, and evidence of resilience, such as restore-test records or incident exercise results.

5.  **Information to score each scenario's representative weakness with CVSS v4.0**: vulnerability scanner output, advisories, penetration test findings, or, for non-technical weaknesses, the equivalency analysis described later in this chapter.

6.  **Information about treatment options and costs**: indicative costs, vendor quotations where available, and the cost conventions the organization uses, such as the value of one full-time equivalent.

### The steps of a CRG assessment

CRG organizes a complete assessment into nineteen numbered steps, followed by the management report and an independent quality control review. The steps fall into seven groups.

1.  **Steps 1 to 3: evidence, context, and scope.** Build the **evidence register** (each source numbered E1, E2, and so on, and typed as documented, exercise, SME input, or external evidence); describe the organizational context and identify the **crown jewels** with their owners, confidentiality, integrity, and availability requirements, and technical and third-party dependencies; and define the scope, risk perspectives, exclusions, and a single **12-month horizon** unless another is specified.

2.  **Step 4: risk appetite and materiality.** Record the approved appetite or estimate one on the CRG anchors, and translate it into materiality criteria for scenario selection.

3.  **Steps 5 to 8: scenarios.** Generate candidates from several perspectives, screen them on a defined set of criteria, select the most material (ten by default), and develop each selected scenario fully as a **causal chain**.

4.  **Steps 9 to 11: quantification.** Estimate the six parameters for each scenario with rationale, evidence, and confidence (step 9); score the representative weakness with CVSS v4.0 (step 10); and calculate estimated and tolerated risk (step 11).

5.  **Steps 12 to 14: treatment.** Design mitigation measures (step 12); identify **shared controls** that reduce several scenarios and allocate their cost without double-counting (step 13); and recalculate mitigated and residual risk and the risk-to-tolerance ratio (step 14).

6.  **Steps 15 to 18: robustness, priorities, and money.** Run the sensitivity analysis (step 15); prioritize scenarios and treatments on multiple criteria (step 16); assemble the investment portfolio (step 17); and calibrate the cybersecurity budget against the appetite (step 18).

7.  **Step 19 and closing: monitoring and reporting.** Define **key risk indicators** with owners, sources, and thresholds (step 19); write the management report; and submit the deliverables to the quality control review before release.

### What each edition and the spreadsheet cover

The method is identical in every edition and every tool; what differs is how much of the work the tool automates. The **Community Edition** includes the AI-assisted analyst track, the calculation engine, the Excel model, the MediBec example, and a free teaching package of labs, assessment briefs, and cases. Its assistant automates steps 1 to 11 and produces a management summary; it can indicate obvious treatment directions, but marks them as indicative and uncosted. Students complete steps 12 to 19 themselves, with the Excel model and the labs, which cover treatment, shared controls, sensitivity, KRIs, and the management memorandum. The **Educational** edition, licensed to institutions, adds the full quantitative workflow—costed treatment portfolios with shared-control allocation, sensitivity analysis, the investment portfolio and budget calibration, a KRI catalogue with thresholds, a live-formula workbook, and a full board report—together with installation, customization, and instructor materials.

The **Excel model** is the most transparent of the tools and the one this chapter describes in detail. It covers the appetite and factor settings (step 4), the parameter sheets and the CVSS column (steps 9 and 10), the calculation of estimated and tolerated risk (step 11), the mitigation sheets and the residual calculation (steps 12 and 14), and the portfolio recommendation and its cost (step 17); the corrected template distributed with this edition adds the risk-to-tolerance bands, a cost-effectiveness column, and the budget calibration (step 18). The evidence register, scenario screening and selection, the sensitivity cases, the KRIs, and the report are prepared outside the workbook or in the application. Prioritization (step 16) is a matter of judgment in every tool.

### Evidence discipline and the three markers

Two habits distinguish a CRG assessment from an opinion with numbers attached. The first is the evidence register: every organizational fact cites an E-number, and anything that cannot be cited is labelled as an assumption. Exercise or simulation material, such as a tabletop inject, is indicative evidence to validate, never a confirmed incident; instructor or benchmark values are calibration aids, not organizational evidence. The second habit is the systematic use of three standard markers, written literally wherever they apply:

- **Analytical estimate — validation required** marks a value that the analyst, or an AI tool, estimated rather than obtained from the organization.

- **Information gap — organizational validation required** marks a fact that nobody has supplied and that the assessment needs.

- **Management decision required** marks a choice that only accountable management can make, such as approving an appetite, accepting a residual risk, or funding a portfolio.

These markers are the single most useful habit the method teaches. A student who uses them correctly has understood the difference between evidence, inference, and judgment. A deliverable without them is a warning sign that estimates, possibly generated by AI, have been presented as facts.

## The risk index and underlying formulas

CRG quantifies each scenario with a **calculated risk index**. The index is dimensionless: it is not a dollar amount and not a prediction of loss. It combines probabilities, a 0–10 severity score, normalized damage and criticality, and a scaling factor into a single relative indicator that supports comparison and prioritization. This design reduces the emotional and political reactions that monetary figures often trigger, and it keeps attention on two questions: which scenarios matter most, and whether what remains after treatment is within tolerance. Chapter 17 shows how the same scenarios can be expressed in monetary terms when an investment case requires it.

Earlier editions of this book called these calculated indices "KRIs". That usage is retired. In this edition, **risk indices** are what the formulas compute; **key risk indicators** are measured monitoring values with thresholds, defined in step 19 and discussed later in this chapter.

### The variables and their valid ranges

Each scenario is described by six parameters, a CVSS score, and two treatment parameters. The organization supplies the appetite and the multiplication factor, which are common to all scenarios in an assessment. The table below summarizes the variables and their validation rules.

| **Symbol** | **Meaning**                                                   | **Valid range**     | **Low to high anchors**                              |
|------------|---------------------------------------------------------------|---------------------|------------------------------------------------------|
| Pb(A)      | Probability the threat is present in 12 months                | 0–1                 | Rare actor → observed or constant                    |
| Pb(ψ,A)    | Probability it exploits the weakness, given existing controls | 0–1                 | Strong evidenced controls → weakness evidenced       |
| δe(ψ,A)    | Normalized expected damage                                    | 0–1                 | One site, few records → network-wide                 |
| δm(ψ,A)    | Normalized plausible maximum damage                           | 0–1, δm ≥ δe        | Material → safety or catastrophic loss               |
| θ(ψ,A)     | Organizational resilience                                     | above 0, up to 1    | No tested recovery → tested recovery, 24×7 detection |
| μ(E)       | Organizational criticality (expected utility)                 | 0–1                 | Supporting process → mission-critical                |
| CVSS       | CVSS v4.0 score of the representative weakness                | 0.0–10.0            | Computed, never estimated by eye                     |
| Appetite   | Approved or estimated risk appetite                           | 0.1–0.9 in practice | Very averse → very high appetite                     |
| rp, ri     | Probability and impact reduction of the treatment package     | 0–1 each            | Share of risk removed                                |
| Factor     | Multiplication factor                                         | constant, 1,000     | Scales indices, never ratios                         |

A few rules deserve emphasis. **Pb(A) and Pb(ψ,A) are estimated separately.** The first asks whether the threat source will act against the organization within the horizon; the second asks whether, if it acts, it will get through this weakness given the controls that can actually be evidenced. Merging them into a single "likelihood" counts the same uncertainty twice. **δm must be at least δe**, because the plausible worst case cannot be milder than the expected case. **θ measures demonstrated resilience**—the ability to prevent, detect, contain, respond, recover, and keep operating—not the number of security products the organization owns. **μ(E) is not a second damage estimate**: it expresses how important the affected service is to the mission, independently of how badly a given scenario would harm it.

CRG's moderate anchors sit around 0.4 to 0.6; low values run from about 0.2 to 0.35 and high values from about 0.65 to 0.85 and above. Values are rounded to two decimals, because more precision would be false precision. Every value carries an estimate, a qualitative reading, a rationale, an evidence reference, and a confidence level (High, Medium, or Low), together with the evidence that would raise that confidence.

### Why the extremes 0 and 1 are avoided

Resilience must be strictly greater than zero, because it divides the index: a θ of zero is a division by zero, which the CRG engine rejects and the spreadsheet reports as an error. A resilience of exactly 1 is permitted but should be rare, since it claims perfect demonstrated capability. For the other parameters, 0 and 1 are mathematically valid but analytically suspect. A probability of 0 declares a scenario impossible and silently removes it from the portfolio, which contradicts the decision to select it; a probability of 1 declares certainty over a 12-month horizon, which evidence rarely supports.

The appetite deserves particular care. CRG's anchors are 0.1–0.2 for very high risk aversion, 0.3 for a risk-averse or low appetite, 0.5 for a risk-neutral or moderate appetite, 0.7 for a relatively high appetite, and 0.8–0.9 for a very high appetite. Values of 0.1 and 0.9 are extremes that should not normally be used without strong evidence. An appetite of 0 would make tolerated risk zero and every ratio undefined; an appetite of 1 would declare almost any residual risk acceptable, because normalized damage cannot exceed 1. CRG's guidance adds that a health, public-safety, or critical-infrastructure organization rarely justifies an appetite above 0.3, and a small unregulated firm rarely justifies one below 0.4.

### Estimated and tolerated risk

For a given scenario, the **estimated risk (Re)** multiplies the two probabilities, the CVSS score, the average of expected and maximum damage, and the criticality, divides by resilience, and applies the multiplication factor. The **tolerated risk (Rt)** uses exactly the same terms, but replaces the average damage with the organization's appetite:

- **Estimated risk:** Re = Pb(A) × Pb(ψ,A) × CVSS × ((δe + δm) ÷ 2) × μ(E) ÷ θ(ψ,A) × Factor

- **Tolerated risk:** Rt = Pb(A) × Pb(ψ,A) × CVSS × Appetite × μ(E) ÷ θ(ψ,A) × Factor

These two indicators answer two questions: how risky is the scenario, taking both the expected and the plausible worst case into account (Re); and how much risk is the organization prepared to tolerate in that same scenario, given its appetite (Rt)? Earlier editions of the book computed a separate "maximum risk" with δm alone and excluded CVSS from the formula. CRG instead builds the worst case into the estimate through the average of δe and δm, and treats CVSS as a **mandatory multiplier**. The worst case remains visible in its own right through δm, which is one of the prioritization criteria discussed later.

The **multiplication factor** is a constant, set to 1,000 in CRG and in the MediBec assessment. It exists only to produce indices of a readable size. Because it multiplies every estimated and tolerated value equally, it scales every index but never changes any ratio or verdict. It must be the same for every scenario in an assessment; changing it between scenarios, or between assessments that will be compared, makes the numbers incomparable.

### Mitigated and residual risk

Treatment is expressed through two package-level parameters: the **probability reduction (rp)**, the share of the probability of exploitation that the treatment package removes, and the **impact reduction (ri)**, the share of the damage it removes. Both are reductions, that is, the share removed, not the share remaining. An impact reduction of 0.70 means the package is expected to remove 70% of the damage. This convention reverses the one used in earlier editions of this book, and readers comparing editions should take care.

- **Mitigated risk:** Mitigated = Re × rp × ri

- **Residual risk:** Residual = Re − Mitigated

Mitigation is **multiplicative**. A package with an 80% probability reduction and an 80% impact reduction removes 0.80 × 0.80 = 64% of the estimated risk, not 80%. Because both reductions are bounded between 0 and 1, the mitigated amount can never exceed the estimated risk and the residual can never be negative. A negative residual is therefore always an estimation error: it can only arise if a reduction above 1 has been entered, typically because several controls' coefficients were added together.

### The risk-to-tolerance ratio and its bands

The tolerance test compares residual risk with tolerated risk:

- **Risk-to-tolerance ratio** = Residual ÷ Rt

CRG reads the ratio in three bands. A ratio **below 0.90** is read as **below tolerance**; a ratio **from 0.90 to 1.10** as **approximately at tolerance**; and a ratio **above 1.10** as **above tolerance**. The middle band is an analyst convention reflecting the limited precision of the estimates: a ratio of 1.03 does not mean the scenario is "slightly unacceptable", and a ratio of 0.97 does not mean it is "safely acceptable". Both mean the residual is roughly where management said it should be, and the decision turns on evidence and judgment rather than on the second decimal. Before treatment, when rp and ri are both zero, the same ratio is called the **pre-treatment ratio** and equals Re ÷ Rt.

The bands replace the single threshold of 1.25 used in earlier editions of this book, which had no documented basis. They also mean that "within tolerance" is a range, not a line. A recommendation that brings a scenario to 0.95 has brought it approximately to tolerance; whether management accepts that result, asks for more treatment, or requests better evidence first is a **Management decision required**.

### What cancels in the tolerance test

The ratio has a property that CRG states openly and that every user must understand. Because Re and Rt share the probabilities, the CVSS score, the criticality, the resilience, and the factor, all of these cancel when one is divided by the other. The ratio simplifies to:

- **Pre-treatment ratio** = ((δe + δm) ÷ 2) ÷ Appetite

- **Post-treatment ratio** = ((δe + δm) ÷ 2) × (1 − rp × ri) ÷ Appetite

In words, **the tolerance verdict depends only on damage, treatment, and appetite**. Probabilities, CVSS, resilience, and criticality drive the magnitude of the estimated risk, and therefore the ranking of scenarios, but not the verdict. A rare scenario and a frequent one with the same damage estimates and the same treatment receive the same verdict. Two consequences follow. First, before treatment almost everything is above tolerance when appetite is low: with an appetite of 0.30, any scenario whose average damage exceeds about 0.33 starts above tolerance. Second, the ratio must always be read together with the estimated risk. The ratio says whether treatment is sufficient relative to appetite; the estimated risk says how much the scenario matters relative to the others.

The simplification also gives a quick design rule. For the residual to be no greater than tolerated risk, the package must satisfy rp × ri ≥ 1 − Appetite ÷ ((δe + δm) ÷ 2). For it to fall below the 0.90 band, rp × ri must exceed 1 − 0.90 × Appetite ÷ ((δe + δm) ÷ 2). These two thresholds let an analyst see, before choosing any control, how strong a package must be.

| **The ratio tells you whether treatment is enough; the estimated risk tells you whether the scenario matters.** Read the two together, never one alone. |
|---------------------------------------------------------------------------------------------------------------------------------------------------------|

### Cost-effectiveness

CRG measures the efficiency of a treatment package as the risk removed per thousand dollars spent:

- **Cost-effectiveness** = (Re − Residual) ÷ Cost × 1,000

Because Re − Residual equals the mitigated amount, the indicator reads as "risk units removed per \$1,000". It allows packages for different scenarios to be compared on a common basis, and it helps identify quick wins. It does not, on its own, select treatments: a mandatory control required by law or contract is funded regardless of its cost-effectiveness, and a highly cost-effective package for a minor scenario may still rank below a less efficient package for a crown-jewel scenario.

### Several measures, shared controls, transfer, and avoidance

A scenario is rarely treated by a single control. CRG handles combinations at the level of the **treatment package**: the analyst estimates one probability reduction and one impact reduction for the package as a whole, each between 0 and 1, with a rationale that explains how the measures interrupt the causal chain. Per-control coefficients must not simply be added together. Ten controls credited with 0.06 each would sum to 0.60, but twenty would exceed 1, which is impossible and produces a meaningless residual. The package estimate must reflect diminishing returns and overlap: a second detection tool adds less than the first.

A **shared control** is one initiative that reduces several scenarios, such as phishing-resistant multifactor authentication, privileged access management, or immutable backups. Its effect is credited in each scenario's package, but its cost is counted **once**. For scenario-level analysis, its cost is split across the scenarios it serves, and an equal split is an acceptable default; the portfolio total, however, comes from the list of initiatives, not from the sum of scenario allocations. A control that already exists cannot be recommended as a new measure: its effect is already reflected in the current Pb(ψ,A) and θ, and claiming it again counts the same reduction twice.

**Risk transfer** and **risk avoidance** are recorded as CRG's own documentation records them. Transfer, through cyber insurance or contract terms, is recorded as a costed line in the portfolio, for example an insurance premium with a placeholder value until a broker quotation is obtained, and as a management decision in the decision register; no probability or impact reduction is claimed for it in the index. This follows directly from the multiplicative convention. Insurance does not make an incident less likely, so its probability reduction is zero, and with a zero probability reduction the mitigated amount is zero whatever impact reduction is entered. CRG's documentation states this openly: a package that reduces only probability, or only impact, mitigates nothing under the workbook convention. The residual index therefore continues to show the risk the organization still faces; the insurance policy changes who pays for part of the loss, which is a financing decision, not a reduction in the scenario's risk. **Avoidance**—stopping or not starting the activity that creates the exposure—is likewise recorded as a management decision in the decision register, with the owner who must approve it, rather than as a coefficient. When the activity has actually stopped and that is evidenced, the scenario is closed or removed at the next reassessment. In CRG's MediBec assessment, the scenarios that remain above tolerance even under favourable assumptions, such as vendor compromise and privileged-insider sabotage, are addressed by transfer and formal acceptance rather than by more spending.

The figure below summarizes the procedure, from evidence to report, with the formulas and tolerance bands at the centre.

<span id="_Toc243635531" class="anchor"></span>Figure 32: The CyberRiskGuardian risk assessment procedure

## Scoring vulnerabilities with CVSS v4.0

The Common Vulnerability Scoring System (CVSS) is maintained by FIRST, the Forum of Incident Response and Security Teams. Version 4.0 was released on 1 November 2023 and is the version used throughout this book and in CRG unless a workbook explicitly specifies another. Because CRG uses CVSS as a mandatory multiplier, every student who performs a CRG assessment must be able to score a weakness correctly, read a vector string, and explain what the score does and does not mean.

### What CVSS measures and what role it plays in CRG

CVSS measures the **technical severity of one vulnerability**. It describes how the weakness can be reached and exploited and what the exploitation would do to the confidentiality, integrity, and availability of the vulnerable system and of systems beyond it. It does not measure organizational risk. FIRST's specification expects consumers of CVSS to enrich the Base metrics with Threat and Environmental metrics, and it leaves factors such as regulatory requirements outside the scope of the score. A high CVSS score never means, by itself, the highest business risk.

In CRG, CVSS has two firm rules. First, it is a **mandatory multiplier** in the estimated and tolerated risk formulas: a scenario without a CVSS score has no index. Second, it is **never a ranking or selection criterion**. Scenarios are not selected because their vulnerabilities are severe, and they are not prioritized by CVSS; selection rests on materiality, and prioritization rests on the multi-criteria judgment described later. Because CVSS appears in both Re and Rt, it cancels in the tolerance ratio; it shapes the size of the index and therefore the ranking by estimated risk, but not the verdict. CRG uses the **Base score (CVSS-B)** unless the workbook specifies otherwise, so that all scenarios in an assessment are scored on the same footing.

The score always describes one defined weakness, the **representative weakness** of the scenario: the specific condition the causal chain depends on, such as "macro-enabled documents execute without restriction on clinic workstations" or "push-based multifactor authentication without number matching on remote access". Scoring "the scenario" as a whole is a common error, because a scenario contains many events and consequences that CVSS was never designed to describe.

### The metric groups and their values

CVSS v4.0 organizes its metrics into four groups. The **Base** metrics are required and describe intrinsic characteristics of the vulnerability. The **Threat** metric adjusts for the current state of exploitation. The **Environmental** metrics let the consumer reflect the importance of the affected system and any local differences in the base characteristics. The **Supplemental** metrics add context but do not change the score. The table below lists the metrics and their values.

| **Metric**                        | **Values**                                                | **What it asks**                                                                                  |
|-----------------------------------|-----------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| Attack Vector (AV)                | Network, Adjacent, Local, Physical                        | How remote can the attacker be?                                                                   |
| Attack Complexity (AC)            | Low, High                                                 | Must the attacker evade security measures or obtain a target-specific secret?                     |
| Attack Requirements (AT)          | None, Present                                             | Does success depend on deployment or execution conditions, such as a race or an on-path position? |
| Privileges Required (PR)          | None, Low, High                                           | What access must the attacker already hold?                                                       |
| User Interaction (UI)             | None, Passive, Active                                     | Must a person other than the attacker participate, and how?                                       |
| VC, VI, VA                        | High, Low, None                                           | Impact on confidentiality, integrity, availability of the vulnerable system                       |
| SC, SI, SA                        | High, Low, None                                           | Impact on confidentiality, integrity, availability of subsequent systems                          |
| Exploit Maturity (E)              | Attacked, Proof-of-Concept, Unreported, Not Defined (X)   | Is the weakness being exploited?                                                                  |
| CR, IR, AR                        | High, Medium, Low, Not Defined (X)                        | How important is each security property of the affected asset?                                    |
| Modified base (MAV to MSA)        | As base, plus Safety for MSI and MSA, and Not Defined (X) | Do local conditions differ from the base assessment?                                              |
| Supplemental (S, AU, R, V, RE, U) | Metric-specific, each with Not Defined (X)                | Safety, Automatable, Recovery, Value Density, Vulnerability Response Effort, Provider Urgency     |

The Base metrics deserve precise reading. **Attack Vector** is Network when the weakness can be exploited at the protocol level from afar, possibly across the internet; Adjacent when the attack is limited to a logically adjacent network, such as a shared subnet or Bluetooth range; Local when the vulnerable system is not bound to the network stack and the attacker needs local or terminal access, or relies on a user to act locally, for example by opening a malicious document; and Physical when the attacker must physically touch or manipulate the system. **Attack Complexity** is High when the attacker must evade or circumvent security-enhancing mechanisms or obtain a target-specific secret, and Low otherwise. **Attack Requirements**, new in version 4.0, is Present when success depends on conditions of the deployment or execution that the attacker does not control, such as winning a race condition or achieving an on-path network position. **Privileges Required** is None when no prior authentication or access is needed, Low when basic user privileges suffice, and High when administrative-level control is required. **User Interaction** is None when no human other than the attacker is needed, Passive when a user's limited, involuntary interaction is needed, and Active when a user must take a deliberate action or bypass a protection.

The impact metrics are split between the **vulnerable system**, the system that contains the weakness, and **subsequent systems**, any systems affected beyond it; the Environmental metrics also allow effects on human safety to be recorded for subsequent systems. For each, High means a total loss of the property, or a limited loss with a direct, serious consequence; Low means some loss, with limited control over or scope of the damage; and None means no loss. When a weakness has no effect beyond the vulnerable system, the subsequent-system metrics are set to None. Analysts score the final impact an attacker can reasonably achieve, not merely the first step.

Of the other groups, **Exploit Maturity** is Attacked when attacks have been reported or tools that simplify exploitation exist, Proof-of-Concept when public proof-of-concept code exists without reported attacks, and Unreported otherwise; when it is not defined, it is scored as Attacked, the worst case. The **security requirements** CR, IR, and AR are High when the loss of the property would likely have a catastrophic adverse effect, Medium when serious, and Low when limited; when not defined, they are treated as High. The **modified base metrics** mirror the base metrics, and the modified subsequent-system integrity and availability metrics (MSI and MSA) accept an additional value, Safety, for exploitation that could cause injury. The **Supplemental** metrics—Safety, Automatable, Recovery, Value Density, Vulnerability Response Effort, and Provider Urgency—inform response but never change the score.

### The vector string and the nomenclature

A CVSS v4.0 result is recorded as a **vector string**. It begins with the prefix "CVSS:4.0/", followed by each metric as an abbreviation and a value separated by a colon, with metrics separated by slashes. All eleven Base metrics are mandatory; every metric that is included, Base or optional, must appear only once and in the order defined by the specification; optional metrics may be omitted, in which case they are treated as Not Defined (X). For example, the representative weakness of the MediBec ransomware scenario used later in this chapter—a macro-enabled document that executes code when a user opens it—is recorded as:

- CVSS:4.0/AV:L/AC:L/AT:N/PR:N/UI:P/VC:H/VI:H/VA:H/SC:H/SI:H/SA:H

Read aloud, the vector says that the weakness is exploited locally (the document is opened on the workstation; FIRST's user guide scores such document-based weaknesses as Local whatever the delivery channel, including e-mail), requires no special conditions or evasion, needs no prior privileges, requires only passive user interaction (opening the document), and produces high impact on the workstation and on the subsequent clinical systems the attacker can reach. Its Base score is **9.3, Critical**. Keeping the full vector, not just the score, matters: it shows why the score is what it is, lets a reviewer recompute it, and exposes errors that a bare number hides.

CVSS v4.0 also introduces a **nomenclature** that states which metric groups a score includes. **CVSS-B** is a Base-only score; **CVSS-BT** combines Base and Threat metrics; **CVSS-BE** combines Base and Environmental metrics; and **CVSS-BTE** uses all three. A report should always say which one it uses. Adding the Threat metric E:U (Unreported) to a Base vector, for instance, can lower the resulting score substantially, and the result is then a CVSS-BT score that must not be compared with CVSS-B scores of other scenarios. Because CRG uses CVSS-B by default, Threat and Environmental context belongs in the scenario parameters—exploitation activity in Pb(A) and Pb(ψ,A), asset importance in μ(E)—rather than in the score.

### Severity bands

FIRST defines a qualitative severity scale that is widely used in vulnerability reporting: **None** for 0.0, **Low** for 0.1–3.9, **Medium** for 4.0–6.9, **High** for 7.0–8.9, and **Critical** for 9.0–10.0. The bands are useful for communication and for vulnerability management service levels (see Chapter 8). In CRG they are descriptive only: the numeric score enters the formula, and the band never determines a scenario's priority.

### Using the NVD CVSS v4.0 calculator step by step

Scores are computed with a calculator, never estimated by eye. The U.S. National Vulnerability Database (NVD) publishes a CVSS v4.0 calculator on its website (nvd.nist.gov, under Vulnerability Metrics), and FIRST publishes its own; CRG's engine and the desktop application compute the same scores. Calculators group the metrics as the specification does—Base (exploitability, vulnerable-system impact, and subsequent-system impact), Threat, Environmental (security requirements and modified base metrics), and Supplemental—and selecting a value for each metric updates the vector string and the score. The exact layout differs from one calculator to another and may change over time. The procedure is as follows:

1.  **Define the representative weakness in one sentence** before opening the calculator. Write down what, exactly, the attacker exploits and on which system.

2.  **Open the calculator and start from a clean state**, so that no value from a previous scoring remains selected.

3.  **Select the five exploitability metrics** (AV, AC, AT, PR, UI) and write a one-line justification for each in the scenario record.

4.  **Select the vulnerable-system impacts** (VC, VI, VA) for the system that contains the weakness.

5.  **Select the subsequent-system impacts** (SC, SI, SA) for the systems the attacker can reach beyond it, or None if there are none.

6.  **Leave the Threat, Environmental, and Supplemental metrics Not Defined** when producing the CVSS-B score that CRG uses by default. If the workbook specifies another nomenclature, apply it consistently to every scenario and say so in the report.

7.  **Copy the full vector string and the score** into the scenario record and the Analyse (summary) sheet, and record the CVSS version and nomenclature.

8.  **Verify** by pasting the vector into a second tool, such as the CRG calculator, and confirming the same score.

## Non-technical vulnerabilities by equivalency

Many of the most material scenarios in any portfolio do not rest on a software flaw. A supplier changes its bank details by e-mail and nobody calls back; a departed administrator keeps a working account; a network closet in a clinic is left unlocked; a managed service provider connects with a shared password. These are **process, people, governance, physical, third-party, or configuration and policy weaknesses**. CRG still requires a CVSS score for each scenario, because CVSS is a mandatory multiplier. The method's answer is to score such weaknesses **by equivalency**: to map the weakness onto the CVSS metrics by analogy, document every choice, and label the result as having **limited applicability**.

### The equivalency principle

Each CVSS metric asks a question that can be asked of any weakness, technical or not. Attack Vector asks who can reach the weakness and from where. Attack Complexity asks whether the attacker must defeat a safeguard or obtain a target-specific secret. Attack Requirements asks whether success depends on a condition the attacker does not control, such as timing or a particular business cycle. Privileges Required asks what access, role, or insider status the attacker must already hold. User Interaction asks whether a victim—a clerk, a nurse, a help-desk agent—must act, and whether that action is deliberate. The impact metrics ask what happens to the confidentiality, integrity, and availability of the system or process that contains the weakness, and of the systems and processes downstream of it.

Applied consistently, these questions produce a vector that is honest about the weakness's reach and consequences. The score is not a measurement in the sense a vendor advisory is; it is a structured analogy. That is why CRG requires three things whenever the equivalency is used: a one-line justification for every metric, the label *limited applicability* next to the score, and the marker **Analytical estimate — validation required**. Where no defensible representative weakness exists at all, the analyst states the limited applicability explicitly rather than fabricating a vulnerability.

### An equivalency table

The table below gives, for each Base metric, the question to ask and example values for four kinds of non-technical weakness: a process weakness (supplier bank details changed by e-mail without call-back), a people weakness (a departed administrator's account left active), a physical weakness (an unlocked network closet in a clinic), and a third-party weakness (a supplier's remote-support account protected only by a shared password).

| **Metric** | **Question to ask**                                               | **Process / people example**                                 | **Physical / third-party example**                     |
|------------|-------------------------------------------------------------------|--------------------------------------------------------------|--------------------------------------------------------|
| AV         | Who can reach the weakness, and from where?                       | N: anyone who can send e-mail / N: remote access still works | P: must enter the room / N: supplier connects remotely |
| AC         | Must a safeguard be defeated or a target secret obtained?         | L: no verification step / L: account simply works            | L: door unlocked / L: no second factor                 |
| AT         | Does success depend on conditions outside the attacker's control? | P: a payment must be pending / N: none                       | N: none / N: none                                      |
| PR         | What role, access, or insider status is needed first?             | N: none / H: the old administrator rights                    | N: none / L: the supplier's own account                |
| UI         | Must a victim act, and deliberately?                              | A: clerk updates the record / N: none                        | N: none / N: none                                      |
| VC, VI, VA | What happens to the process or system with the weakness?          | N, H, N: payment data falsified / H, H, H: full control      | L, L, H: ports, outage / H, H, L: support access       |
| SC, SI, SA | What happens downstream?                                          | N, L, N: ledger misstated / H, H, H: all dependent systems   | L, L, H: clinic devices / H, L, L: patient records     |

Three cautions apply. The impact metrics describe what the weakness enables, not how likely it is to be exploited: that belongs in Pb(A) and Pb(ψ,A), which must not be inflated or deflated to compensate for the score. The equivalency must be applied the same way across the portfolio, so that a people weakness in one scenario is not scored more generously than a similar one in another. And a low equivalency score depresses the estimated risk of fraud and error scenarios whose real weight lies in the damage estimates; CRG's documentation flags this effect, and the analyst should mention it in the report rather than adjust the score to avoid it.

### Worked equivalency examples

**Example 1: a process weakness.** At a regional distributor, accounts payable accepts supplier bank-detail changes received by e-mail and updates the vendor master record without a call-back to a known number. The scenario is business e-mail compromise leading to a fraudulent payment. By equivalency, the weakness is reachable by anyone who can send e-mail (AV:N); no safeguard has to be defeated (AC:L); success depends on a payment being due to that supplier, a condition the attacker does not control (AT:P); no privileges are needed (PR:N); and a clerk must deliberately update the record (UI:A). On the vulnerable process, nothing is disclosed (VC:N), the payment data are falsified with a direct, serious consequence (VI:H), and the process keeps running (VA:N). Downstream, the general ledger is misstated (SI:L), with no confidentiality or availability effect (SC:N, SA:N). The vector CVSS:4.0/AV:N/AC:L/AT:P/PR:N/UI:A/VC:N/VI:H/VA:N/SC:N/SI:L/SA:N gives a score of **5.9, Medium**, recorded with the label *limited applicability*.

**Example 2: a people weakness.** At a clinic network, the privileged directory account of a database administrator who left three months ago was never disabled, because the leaver process depends on an e-mail from the manager. By equivalency, the account is usable over the remote-access service (AV:N); no safeguard stands in the way (AC:L, AT:N); the actor already holds administrative rights, the equivalent of high privileges (PR:H); and no one else needs to act (UI:N). Administrative control of the directory gives high impact on the directory itself (VC:H, VI:H, VA:H) and on every system that trusts it (SC:H, SI:H, SA:H). The vector CVSS:4.0/AV:N/AC:L/AT:N/PR:H/UI:N/VC:H/VI:H/VA:H/SC:H/SI:H/SA:H gives **9.4, Critical**. The high score reflects what the weakness enables; whether the former employee is hostile, or whether anyone else has the credentials, is a question for Pb(A), not for CVSS. Adding the Threat metric E:U to this vector would produce a CVSS-BT score of 7.6, which illustrates why the nomenclature must always be stated and why CRG keeps threat context in the probabilities.

**Example 3: a physical weakness.** In one clinic, the network closet that houses the access switch for the imaging devices is routinely left unlocked, and unused switch ports are live. By equivalency, the attacker must be physically present (AV:P); no safeguard or special condition stands in the way (AC:L, AT:N); no privileges or victim action are needed (PR:N, UI:N). Locally, the attacker could observe or alter some traffic (VC:L, VI:L) and could disconnect or disable the switch (VA:H); downstream, the imaging devices and the clinic's connection to central systems would be affected in the same way (SC:L, SI:L, SA:H). The vector CVSS:4.0/AV:P/AC:L/AT:N/PR:N/UI:N/VC:L/VI:L/VA:H/SC:L/SI:L/SA:H gives **7.0, High**. Adding the Supplemental metric S:P, to note that a safety consequence is present, does not change the score, because Supplemental metrics never do; the safety concern is carried into δm and the prioritization criteria instead.

The three examples show the purpose of the equivalency. It does not pretend that a process or a door has a vulnerability identifier. It forces the analyst to ask the same disciplined questions of every weakness, to record the answers, and to let a reviewer challenge each one. A reviewer who disagrees that the bank-detail change needs a pending payment can change AT to None and see the effect immediately; that transparency is the point.

| For a non-technical weakness, the CVSS vector is a documented analogy, not a measurement. Justify every metric, label the score *limited applicability*, and never adjust it to reach a preferred ranking. |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## Step-by-step procedure in the spreadsheet and the application

This section describes a reproducible sequence for completing the quantitative steps in the CRG Excel model, aligned with the corrected template distributed with this edition, and notes where the application differs. Scenario development (steps 1 to 8) is assumed to be complete: each selected scenario has a causal chain, a list of evidenced controls, and a representative weakness.

### How the workbook is organized

The workbook has three kinds of sheet. The **Analyse (summary)** sheet holds the appetite and multiplication factor cells and one row per scenario, showing the scenario name, the six parameters drawn from the scenario sheets, the CVSS score, estimated and tolerated risk, the package cost and reductions drawn from the mitigation sheets, mitigated and residual risk, the risk-to-tolerance ratio and its band, cost-effectiveness, and a recommendation column. Ten **scenario sheets (S1 to S10)** hold each scenario's number, name, date, and six parameters. Ten **mitigation sheets (M1 to M10)** hold the treatment options for the matching scenario: a few rows for custom measures and a catalogue of predefined controls drawn from ISO/IEC 27002:2022, CIS Controls v8.1, and NIST CSF 2.0 and NIST SP 800-53 Rev. 5. The template holds ten scenarios; the application and the Educational edition's workbook builder apply the same formulas to any number.

The workbook opens in **manual calculation mode**, because its many formulas would otherwise slow every keystroke. Nothing on the Analyse (summary) sheet is current until the user presses **Calculate Now** (Formulas, then Calculate Now, or F9). Reading numbers before recalculating is the most common source of student error.

### Step 1: Prepare the workbook and verify the engine

Save a copy of the template under the team's name, then verify that the calculation works before entering the case. Enter the worked example of this chapter in the first scenario row, press Calculate Now, and confirm the published results to the unit. In the application, the engine performs this check itself and displays whether it agrees with the reference MediBec results (total estimated risk 84,491 and residual risk 41,104); in the analyst track, the same check is run with the CRG calculator. If the engine does not agree with the reference, nothing produced on that installation should be trusted.

### Step 2: Set the risk appetite and the multiplication factor

On the Analyse (summary) sheet, enter the approved appetite, or the estimated one with its rationale, in the appetite cell. Do not leave the template's placeholder value: a placeholder silently turns a risk-averse case into a risk-neutral one, and every tolerance verdict that follows becomes inconsistent with the governance context. In real organizations, the same error leads to serious under- or over-investment. Confirm that the multiplication factor is 1,000, the corrected template's default, and keep it unchanged for the whole assessment.

### Step 3: Populate the scenario sheets (S1 to S10)

For each scenario, enter the number, a name that states the causal chain in short form (for example "S2 Ransomware disrupting EHR access across clinics"), the date, and the six parameters. Respect the validation rules: every value between 0 and 1, δm at least δe, θ strictly above zero. Record, in the scenario notes or the separate parameter table, the qualitative reading, rationale, evidence reference, and confidence for each value, together with its marker. In a professional setting, values come from workshops and structured consensus, ideally with anonymous slider scales to reduce social pressure, and from historical data and sector evidence. In a course, they come from the case, from instructor guidance, or from AI-assisted drafts that students critique and recompute; in every case, the numbers must be internally consistent and traceable.

### Step 4: Enter the CVSS scores

Enter each scenario's CVSS v4.0 score in the CVSS column of the Analyse (summary) sheet, and keep its vector string next to it. In the corrected template, these cells are **required** and start **blank, not zero**, with validation that accepts only values from 0 to 10. The blank matters: a default of zero multiplies every index to zero, and a student who forgets the column would otherwise see a portfolio in which nothing appears to be at risk.

### Step 5: Calculate and read estimated and tolerated risk

Press Calculate Now. Read estimated risk and tolerated risk for each scenario, and the pre-treatment ratio. Compare the values numerically; do not rely on cell colours alone, because colour rules are conveniences, not the verdict. Check that the pre-treatment ratio equals the average damage divided by the appetite; if it does not, an input is wrong. At this point, expect most scenarios with material damage to be above tolerance, as the cancellation property predicts.

### Step 6: Define treatment on the mitigation sheets (M1 to M10)

Treatment corresponds to the risk treatment activity of ISO/IEC 27005:2022 and clause 6.1.3 of ISO/IEC 27001:2022, and to the Select and Implement steps of the NIST RMF. For each scenario, open its mitigation sheet and build a package in two ways. **Custom measures** are entered in the rows at the top: a description, capital expenditure, software, implementation and operating costs, and additional staffing in full-time equivalents, which the template values at 1,950 hours × \$80, or \$156,000 per year. **Catalogue controls** are included by marking them in the include column; each carries indicative default costs that must be recalibrated, because they are generic placeholders, not quotations.

For a scenario involving macro-enabled documents, for example, a package might combine blocking macros from internet-sourced files (catalogue: ISO/IEC 27002:2022 control 8.7, Protection against malware), endpoint detection and response with 24/7 monitoring, and immutable, isolated backups with tested restoration (ISO/IEC 27002:2022 control 8.13, Information backup). Each measure is chosen because it interrupts a specific step of the causal chain and adds something the organization does not already have.

Then enter the **package-level** probability reduction and impact reduction, each between 0 and 1, with a rationale. If the sheet accumulates per-control coefficients, check the totals: each must stay within 0 to 1, and the package estimate should reflect overlap and diminishing returns. Record shared controls once in the portfolio and allocate their cost across scenarios as described earlier. Record insurance and contract terms as costed transfer lines and decisions, not as reductions.

### Step 7: Recalculate and read residual risk, the ratio, and cost-effectiveness

Return to the Analyse (summary) sheet and press Calculate Now again. For each scenario, read the mitigated and residual risk, the risk-to-tolerance ratio and its band, and the cost-effectiveness. In the corrected template, residual risk is computed as Re − Mitigated with a guard that flags an invalid reduction instead of hiding it; earlier versions used an absolute value, which made the residual *rise* when coefficients summed above 1. If one or more scenarios remain above tolerance, four options exist: strengthen the package, transfer part of the risk, avoid the activity, or accept the residual formally. Each is a **Management decision required**; the analyst's role is to set out the options and their consequences.

### Step 8: Recommend the portfolio and calibrate the budget

In the recommendation column, mark each package recommended (1) or not (0). The sheet totals the cost of the recommended packages. Check that shared controls appear once in the portfolio total. Then complete the budget calibration block, which compares the resulting cybersecurity spending with the appetite-consistent target described later in this chapter. Save the workbook, and keep it with the memo: the numbers in the report must agree with the workbook to the unit.

### The same steps in the application and the analyst track

The desktop application presents the same steps on screens rather than sheets: a quantification editor for parameters with rationale, evidence, and confidence; a built-in CVSS v4.0 calculator that saves the vector to the scenario; a risk calculator that shows each step of the formula; a mitigation screen with proposals and a control catalogue; and a quality review before export. In the AI-assisted analyst track, the assistant drafts parameter rationales and vectors, but every number is computed by the CRG calculator rather than by the model's own arithmetic, and the analyst validates each value before accepting it.

## Worked example: ransomware at MediBec

MediBec is a fictional private medical centre with a Montréal head office and ten clinics in Québec. Its complete CRG assessment is published with the Community Edition. This worked example uses the parameter values of the worked example in CRG's guide (section 16), applied to MediBec's ransomware scenario, except that the Attack Vector of the CVSS vector is scored Local, as FIRST's user guide requires for a malicious document. These values are illustrative and differ from those in MediBec's published 30-scenario register; every one of them is an **Analytical estimate — validation required**.

### The scenario and its inputs

*A ransomware group gains a foothold through a macro-enabled phishing attachment, moves laterally across an insufficiently segmented network, exfiltrates health data, and encrypts EHR support servers, shared folders, and scheduling systems, halting clinical and administrative operations at several clinics.* The representative weakness is the execution of macro-enabled documents on clinic workstations, scored with the vector CVSS:4.0/AV:L/AC:L/AT:N/PR:N/UI:P/VC:H/VI:H/VA:H/SC:H/SI:H/SA:H, which gives a CVSS-B score of 9.3. The other inputs are shown below.

| **Input**         | **Value**              | **Reading**                                   |
|-------------------|------------------------|-----------------------------------------------|
| Pb(A)             | 0.70                   | Healthcare is a sustained ransomware target   |
| Pb(ψ,A)           | 0.60                   | Several weak links; perimeter controls exist  |
| CVSS-B            | 9.3                    | Critical; macro document weakness             |
| δe / δm           | 0.60 / 0.80            | Multi-clinic disruption / network-wide outage |
| θ                 | 0.50                   | Partial; recovery not fully tested            |
| μ(E)              | 0.65                   | EHR and scheduling support care delivery      |
| Appetite / Factor | 0.35 / 1,000           | Risk averse / CRG constant                    |
| rp / ri / Cost    | 0.65 / 0.70 / \$62,000 | Treatment package estimate                    |

### Calculation step by step

The probabilities, CVSS score, criticality, resilience, and factor are common to Re and Rt, so it is convenient to compute their product first:

1.  Pb(A) × Pb(ψ,A) = 0.70 × 0.60 = 0.42.

2.  × CVSS: 0.42 × 9.3 = 3.906.

3.  × μ(E): 3.906 × 0.65 = 2.5389.

4.  ÷ θ: 2.5389 ÷ 0.50 = 5.0778.

5.  × Factor: 5.0778 × 1,000 = 5,077.8. This is the common term.

6.  **Estimated risk:** average damage = (0.60 + 0.80) ÷ 2 = 0.70, so Re = 5,077.8 × 0.70 = **3,554.46**.

7.  **Tolerated risk:** Rt = 5,077.8 × 0.35 = **1,777.23**.

8.  **Pre-treatment ratio:** 3,554.46 ÷ 1,777.23 = **2.00**, which equals 0.70 ÷ 0.35, as the cancellation property predicts. Before treatment, the scenario is well above tolerance.

9.  **Mitigated risk:** rp × ri = 0.65 × 0.70 = 0.455, so Mitigated = 3,554.46 × 0.455 = **1,617.28**.

10. **Residual risk:** 3,554.46 − 1,617.28 = **1,937.18**.

11. **Risk-to-tolerance ratio:** 1,937.18 ÷ 1,777.23 = **1.09**. Check: 0.70 × (1 − 0.455) ÷ 0.35 = 1.09.

12. **Cost-effectiveness:** (3,554.46 − 1,937.18) ÷ 62,000 × 1,000 = **26.09** risk units removed per \$1,000.

These values match the CRG calculation engine exactly.

### Reading the result

With a ratio of 1.09, the treated scenario is **approximately at tolerance**, at the upper edge of the band. The package removes 45.5% of the estimated risk. The design rule shows why it is not enough to go further: to bring the residual down to the tolerated level, rp × ri must reach 1 − 0.35 ÷ 0.70 = 0.50, and to fall below the 0.90 band it must exceed 1 − 0.90 × 0.35 ÷ 0.70 = 0.55. A stronger package, for example a probability reduction of 0.75 and an impact reduction of 0.80 (product 0.60), would bring the ratio to 0.80, below tolerance, at a higher cost that management would have to weigh.

The sensitivity analysis (step 15) tests whether this conclusion survives reasonable error. Under CRG's convention, the lower case shifts the probabilities and damages down by 0.10, resilience up by 0.10, and both reductions up by 0.10; the higher case does the opposite, with all values kept between 0.05 and 0.99. The lower case gives a ratio of 0.69 (below tolerance) and the higher case 1.53 (above tolerance). Because the verdict changes across the range, this is a scenario where **better evidence is worth paying for**—for example a restore test of the EHR support servers, which would firm up θ and the impact reduction—before or alongside the investment. Finally, the estimated risk of 3,554.46 is not a dollar figure; it matters because it places the scenario high in the ranking against MediBec's other scenarios, which is why the ratio and the index must be read together.

## Prioritization, portfolio, and budget calibration

Calculated indices and ratios are necessary but not sufficient for deciding what to do first. In most organizations, the cost of all reasonable measures exceeds the available budget, and the order of investment matters as much as its total.

### Prioritization criteria

CRG step 16 builds a **management attention list** on several dimensions, never on one score, and explains it as judgment rather than mathematical certainty. The criteria are:

- **Estimated and residual risk.** Scenarios with the highest indices, and those where treatment removes the most, are prime candidates. A very high estimated risk deserves attention even when proposed controls bring it within tolerance, because the organization then depends heavily on those controls not failing; they need stronger assurance and monitoring.

- **Risk-to-tolerance ratio.** How far above the band a scenario remains after treatment, read with the CRG bands rather than a single threshold.

- **Maximum consequence and safety.** Scenarios whose δm implies harm to people, environmental damage, insolvency, permanent loss of critical data, or severe regulatory action warrant priority even at moderate likelihood.

- **Business criticality.** Scenarios affecting crown jewels, essential services, core revenue, sensitive intellectual property, or transformation programs carry extra weight.

- **Legal, privacy, regulatory, and contractual obligations.** Measures required by law, contract, insurance conditions, or audit remediation are generally mandatory, whatever their cost-effectiveness.

- **Risk velocity and threat activity.** Exposures that are actively exploited, internet-facing, fast-spreading, or slow to recover from require urgency.

- **Control maturity.** Where existing controls are weak or unevidenced, the uncertainty itself argues for action or for collecting evidence.

- **Feasibility, cost, and time to benefit.** A technically ideal plan that cannot be staffed, procured, or sustained does not reduce loss.

- **Shared benefits.** Foundational controls that reduce several scenarios at once, such as phishing-resistant MFA, centralized logging, and tested backup and recovery, often deliver the best return.

- **Dependencies.** Some initiatives must be sequenced: an asset inventory precedes vulnerability management, and centralized logging precedes advanced detection.

CVSS is deliberately absent from this list. Where the attention list departs from the order of the scores, the report says so and explains why.

### From scenarios to an investment portfolio

Step 17 turns scenario packages into an **investment portfolio** of initiatives. Each initiative has a name, the scenarios it serves, an owner, a priority (Immediate, Near-term, or Planned), a one-time cost, a recurring cost, its expected risk reduction, dependencies, target start and completion dates, and success indicators. The portfolio avoids double-counting and states its financial basis: first-year cost, annual recurring cost, implementation cost, or multi-year total cost of ownership. Costs are labelled as indicative estimates, quotations, or approved budgets, because the three carry very different weight. Chapter 17 develops the investment governance that surrounds this portfolio.

### Calibrating the budget to risk appetite

Step 18 compares the portfolio, together with existing cybersecurity spending, with an appetite-consistent budget target. The best practice CRG applies expresses the target as a percentage of the **total IT budget, including salaries**: **12%** for a risk-averse organization (appetite of 0.30 or below), **7.8%** for a risk-neutral one (appetite of 0.50), and **4%** for a risk-seeking one (appetite of 0.70 or above), with linear interpolation between these points. Between 0.30 and 0.50, the target falls by 4.2 percentage points over 0.20 of appetite; between 0.50 and 0.70, by 3.8 points.

Consider an organization with a total IT budget of \$50 million and an appetite of 0.35. The target is 12% − (0.05 ÷ 0.20) × 4.2 points = 10.95%, or about \$5.48 million. Current cybersecurity spending is \$3.0 million (6.0%) and the recommended portfolio adds \$1.7 million in its first year, for a total of \$4.7 million, or 9.4% of the IT budget. The plan is therefore about \$0.78 million below the appetite-consistent target. Read in reverse, 9.4% corresponds to an **implied appetite** of 0.30 + ((12 − 9.4) ÷ 4.2) × 0.20, or about 0.42: the organization is funding cybersecurity as if it were noticeably less risk-averse than it says it is.

The calibration does not dictate a budget. A gap may be acceptable for a first-year program, may reflect a portfolio that genuinely needs less, or may reveal that the stated appetite and the actual funding are inconsistent. Whatever the explanation, the gap is reported, and closing it or revising the appetite is a **Management decision required**. Shared initiatives that serve both IT and security, such as a backup platform, are split between the cybersecurity and IT budgets with the basis stated. In MediBec's published assessment, current spending of about 0.9% of the IT budget sat well below the 4% floor, while the risk-driven budget implied an appetite of about 0.64, far from the 0.30 used in the assessment. The value of the calibration is that it makes such inconsistencies visible to the board.

## Key risk indicators for monitoring

An assessment is a snapshot. Risk changes as threats evolve, controls are implemented or decay, and the organization changes. Step 19 defines the **key risk indicators (KRIs)** that tell management, between assessments, whether a risk is moving.

### Indices are not indicators

The calculated risk indices (Re, Rt, Residual, and the ratio) are products of estimates made at a point in time. They are not measured, and they do not change unless someone reassesses the parameters. A KRI, by contrast, is a **measured value**, drawn from a named source on a schedule and compared with thresholds that trigger action. A KRI is useful only if someone can measure it, from an identified system, on a set frequency, against a target, a warning threshold, and a critical threshold. "Security is improving" is not a KRI; it has no measure, source, owner, frequency, or threshold.

### Designing KRIs from the assessment

Good KRIs are connected to the risk drivers of the scenarios they monitor. Examples include the percentage of privileged and remote-access accounts on phishing-resistant MFA; critical vulnerabilities remediated within their service level; mean time to detect and to contain; unsupported critical systems; the success rate of crown-jewel backup restore tests within the recovery time objective; supplier assessments completed; phishing simulation failure rates; open security exceptions past expiry; endpoint detection coverage; log-source coverage; and cybersecurity spending as a percentage of the IT budget. For each KRI, CRG records the name, the scenarios it addresses, the calculation method, the data source, the owner, the frequency, the target, the warning threshold, and the critical threshold. For an indicator where lower is better, such as a phishing click rate, thresholds are written as "greater than", because a breach occurs when the value rises.

Each KRI also needs a stated response: what management does when the warning threshold is crossed. A KRI that crosses its threshold is a trigger to revisit the scenario's parameters, for example Pb(ψ,A) or θ, at the next reassessment or sooner. That is the link between monitoring and the index, and it should be written into the risk management process (see Chapter 7 for the operational side of measurement).

## The management report and the quality control review

The final products of an assessment are a management report that obtains decisions and an independent review that confirms the report can be trusted.

### The management report

CRG's full management report has thirteen parts: an executive summary (exposure, the most material scenarios, residual versus tolerance, systemic weaknesses, investment themes, approximate cost, and decisions required); scope and methodology; organizational and technology context; risk appetite; the scenario portfolio table (ID, scenario, estimated, tolerated and residual risk, ratio, and key treatment); detailed scenarios; a cross-scenario analysis of root causes, common weaknesses, concentration, and shared mitigations; the treatment portfolio with costs, sequence, and owners; the cybersecurity budget; an implementation roadmap (0–90 days, 3–6 months, 6–12 months, and 12–24 months); the KRIs; assumptions and limitations; and the management decisions required. The shorter management summary produced in the Community Edition leads with the conclusion, gives the portfolio table and a tiered attention list, marks treatment directions as indicative, and lists each decision on its own line.

In a course, the report is often reduced to a one-page **memorandum** and a short recommendation. A strong memo leads with the decision management must take; says how many scenarios are above, approximately at, and below tolerance before and after treatment; gives the cost of the recommended portfolio against the budget context; states the residual risk that remains and who must formally accept it; names the two or three key uncertainties and the next steps; and confirms that IM, BCP, and DRP arrangements address what remains. It does not restate tables or re-explain the model, and it agrees with the workbook to the unit. Students attach the completed workbook to the memo.

### The quality control review

Before release, the deliverables are reviewed independently, in the spirit of the second line or internal audit; in CRG's analyst track, a dedicated read-only reviewer performs this check. The review asks whether scenarios are organization-specific causal chains rather than threat labels; whether threats and vulnerabilities are distinguished; whether existing controls are evidence-backed; whether every value is justified and the horizon is common; whether CVSS is appropriate, uses v4.0 unless specified, discloses limited applicability, and is not used for ranking; whether a sample of scenarios recomputes exactly with the CRG engine; whether the factor is constant and reductions stay within 0 to 1; whether residual risk is compared with tolerated risk and the bands are applied consistently; whether shared controls are counted once and the cost basis is stated; whether the budget is calibrated and any gap escalated; whether assumptions, evidence gaps, and AI-generated estimates carry their markers; and whether risk acceptance is left to management. Findings are ranked Critical, Major, or Minor, with a location and a suggested fix. A disagreement between the memo and the workbook is the most common finding, and often a sign of unverified AI output.

## Role of IM, BCP, and DRP in dealing with residual risk

No matter how sophisticated the assessment or how large the budget, some risk always remains. Some scenarios will be formally accepted; others will be treated to a level management is comfortable with; a few, as MediBec's vendor and insider scenarios show, cannot be brought within tolerance by the organization alone. For all of them, the organization must not simply do nothing. A well-governed program ensures that **incident management plans** define what happens when a cybersecurity event is detected, with roles, communication channels, and playbooks; that **business continuity plans (BCPs)** describe how mission-critical processes continue during disruption, including manual downtime procedures; and that **disaster recovery plans (DRPs)** restore infrastructure, data, and applications within defined objectives.

Exercises and drills, much like emergency simulations in aviation, train the organization to respond under stress without improvisation. In standards terms, these capabilities support the ISO/IEC 27001:2022 Annex A controls for incident management and continuity (A.5.24 to A.5.30), the NIST SP 800-53 Rev. 5 Incident Response (IR) and Contingency Planning (CP) families, and the Respond and Recover functions of NIST CSF 2.0. In the CRG formulas, resilience θ partly reflects the maturity of these arrangements: tested recovery and practised response justify higher θ values and therefore lower risk indices, while untested plans do not. Chapter 12 develops these capabilities.

## Ensuring compatibility with ISO/IEC 27001, ISO/IEC 27005, and NIST

It is useful to make the mapping to formal standards explicit, both for academic rigour and for practitioners who must defend their methods to auditors or regulators.

**ISO/IEC 27001:2022, Information security, cybersecurity and privacy protection — Information security management systems — Requirements.** Clause 4 (context of the organization) and clause 6 (planning) are reflected in the evidence register, context, crown jewels, scope, and appetite (steps 1 to 4). Clause 6.1.2 (information security risk assessment) is implemented by scenarios, parameters, CVSS, and the tolerance test (steps 5 to 11), and clause 6.1.3 (information security risk treatment) by treatment, shared controls, residual risk, and the decision register (steps 12 to 14). Clauses 8.2 and 8.3 are reflected in performing the assessment at planned intervals and implementing the treatment plan, and clause 9 in KRIs and periodic reassessment. Annex A controls are referenced in the mitigation catalogue.

**ISO/IEC 27005:2022, Information security, cybersecurity and privacy protection — Guidance on managing information security risks.** Context establishment corresponds to steps 1 to 4; risk identification to the candidate and selected scenarios; risk analysis to the parameters, CVSS, Re, and Rt; risk evaluation to the ratio, bands, and prioritization; risk treatment and acceptance to steps 12 to 18 and the management decisions; communication and consultation to the workshops and the report; recording and reporting to the workbook, register, and report; and monitoring and review to the KRIs and reassessment.

**NIST SP 800-30 Rev. 1, Guide for Conducting Risk Assessments; NIST SP 800-37 Rev. 2, Risk Management Framework for Information Systems and Organizations: A System Life Cycle Approach for Security and Privacy; NIST SP 800-53 Rev. 5, Security and Privacy Controls for Information Systems and Organizations; and the NIST Cybersecurity Framework (CSF) 2.0.** The assessment follows SP 800-30's risk factors and its prepare, conduct, communicate, and maintain sequence. In the RMF, it corresponds to the risk assessment tasks of the **Prepare** step, not to the **Assess** step, which evaluates controls; the selection of measures supports the Select and Implement steps, and KRIs and residual risk feed the Monitor step. The mitigation catalogue maps to SP 800-53 control families, so that students can see how controls such as AC-2 (Account Management), SI-2 (Flaw Remediation), and IR-4 (Incident Handling) contribute to risk reduction. In CSF 2.0, the appetite and tolerance work supports GV.RM, the assessment and prioritized responses support ID.RA, and preparedness for residual risk supports the Respond and Recover functions.

This compatibility does not mean that CRG is a complete implementation of any of these standards. It is a **pedagogical bridge** and a practical starting point: students learn the logic and language of risk-based cybersecurity management in a transparent, controlled environment, and practitioners obtain an assessment whose structure auditors will recognize.

## Using generative AI responsibly in risk assessment

Generative AI can now maintain long, contextual conversations about an organization, draft plausible scenarios from a system description, propose parameter values with rationales, suggest mitigation measures and indicative costs, and help draft reports in professional language. CRG uses these capabilities deliberately. Its analyst track runs on an approved AI assistant (built on Claude) that performs context intake, scenario generation, screening, and draft quantification; the desktop application can route AI requests to a **local model** for confidential data, show the exact payload before anything is sent, anonymize it, or switch AI features off entirely.

In this book's methodology, AI accelerates the early and drafting stages: seeding a large candidate pool from which accountable humans select, producing first-draft scenario descriptions, proposing estimates that analysts critique, and checking that markers are present. Several cautions are essential, whatever tool is used:

- **Use only approved, governed tools.** Organizations must not paste confidential information into an AI service that has not been approved under their data-protection and AI governance rules. A local or enterprise deployment may be required; Chapter 18 discusses the governance of AI tools in depth.

- **Verify everything.** AI tools invent control identifiers, CVSS vectors, legal obligations, statistics, and sources with great confidence. Every such item is checked against a primary source before use.

- **Let the engine calculate.** Final numbers come from the verified CRG engine or the recalculated workbook, never from a model's arithmetic.

- **Label estimates.** AI-proposed values carry **Analytical estimate — validation required** until a person has validated them, and facts the AI cannot support become information gaps.

- **Keep accountability human.** ISO/IEC 27001 and the NIST frameworks assume that the organization remains accountable for its risk decisions. Scenario selection, risk acceptance, and funding decisions stay with accountable people.

In a course, students disclose their AI use with an acknowledgement, are graded on their verification rather than on the fluency of their prose, and recompute every number with the engine. Used this way, generative AI is an effective assistant in risk assessment; it does not replace governance, expertise, or evidence.

## Bringing it together: from scenarios to defensible decisions

This chapter has shown how the conceptual elements of cybersecurity risk—threats, vulnerabilities, impacts, appetite, and scenarios—become a structured assessment with CyberRiskGuardian. The method's value lies less in any single formula than in its discipline: one horizon, one formula, one set of anchors, CVSS scored and documented for every scenario, every value traced to evidence or marked as an estimate, residual risk tested against tolerance in bands, and every choice that belongs to management flagged as such.

### Illustrative scenario: a credit union's risk committee reviews its assessment

A fictional Québec credit union with 40 branches presents its first CRG assessment to the board's risk committee. The appetite approved by the board is 0.30. Of ten scenarios, all are above tolerance before treatment, which surprises two directors until the chief risk officer explains that, at this appetite, any scenario with average damage above about 0.33 starts above tolerance. After treatment, six scenarios are below tolerance, two approximately at, and two above: a payment-fraud scenario built on a process weakness scored by equivalency at 5.9, and a scenario involving the core banking provider.

The committee's discussion turns on three points. First, a director proposes buying additional cyber insurance for the core banking scenario "to bring it within tolerance". The analyst explains that, under the CRG convention, insurance is recorded as a costed transfer and a decision, but removes no risk from the index: the residual still shows the exposure the credit union faces, and insurance changes who pays. The committee records a transfer decision and a formal acceptance with a six-month review, conditional on KRI evidence from the provider. Second, the budget calibration shows that existing spending plus the portfolio equals 8.5% of the IT budget, against a 12% target for an appetite of 0.30, implying an appetite near 0.47. The committee must either fund a second wave or revise its stated appetite, and it asks management for both options. Third, the payment-fraud scenario's sensitivity analysis shows that its verdict changes between the lower and higher cases; the committee approves a call-back control immediately and asks internal audit to test it within the quarter. Each of these outcomes is a decision, recorded with an owner and a date. None of them could have been made from a heat map.

### Questions managers and boards should ask

- **Which appetite was used, who approved it, and when?** If it was estimated, what evidence supports it, and what would change at the next anchor up or down?

- **Has the calculation engine been verified,** and do the numbers in the report agree with the workbook to the unit?

- **Which values are analytical estimates and which rest on evidence?** Which two or three estimates matter most to the conclusions?

- **How many scenarios are above, approximately at, and below tolerance before and after treatment,** and which remain above tolerance even in the favourable case of the sensitivity analysis?

- **Are we reading the ratio together with the estimated risk,** knowing that probabilities, CVSS, resilience, and criticality cancel in the verdict?

- **What vulnerability, exactly, was scored with CVSS for each scenario,** which nomenclature was used, and which scores rest on equivalency with limited applicability?

- **Are shared controls counted once in the portfolio,** and are costs labelled as estimates, quotations, or approved budgets?

- **Where have we relied on insurance or contracts,** and have we accepted that transfer does not reduce the residual risk shown?

- **How does our total cybersecurity spending compare with the appetite-consistent target,** and what appetite does our actual funding imply?

- **Which KRIs will tell us between assessments that a risk is moving,** who owns them, and what happens when a threshold is crossed?

- **Who formally accepts each residual risk that remains above or near tolerance,** under which delegation of authority, and when will it be reviewed?

- **Has an independent quality review been performed,** and were AI-generated content and estimates labelled and verified?

The next chapters build on this foundation. Chapter 7 examines how cybersecurity operations carry out and sustain treatment decisions, Chapter 12 develops incident management and resilience, and Chapter 17 returns to the portfolio and budget questions as matters of investment governance.

## Chapter summary

Many risk assessments are neither repeatable nor defensible: two analysts reach different results, numbers lack rationale, tolerance is asserted rather than tested, and reports end with controls instead of decisions. This chapter teaches one transparent method in depth, CyberRiskGuardian, in which every formula is published, every parameter has a range and meaning, every estimate carries a rationale and evidence, and every result is read against the approved appetite. Students learn the nineteen-step workflow, the risk index formulas, CVSS v4.0 scoring including equivalency for non-technical weaknesses, the spreadsheet procedure, a full worked example, prioritization, budget calibration, KRIs, reporting, quality control, and the responsible use of AI.

### Key points to retain

- A defensible assessment is one that another analyst, with the same evidence, appetite, and method, would substantially reproduce, with every number traced to a rationale; CRG maps to ISO/IEC 27001, ISO/IEC 27005, NIST SP 800-30, the RMF Prepare step, and CSF 2.0.

- CRG runs nineteen steps in a fixed order, from evidence, context, and appetite through scenarios, quantification, treatment, sensitivity, prioritization, portfolio, budget, and KRIs, ending with a report and an independent quality review.

- Re = Pb(A) × Pb(ψ,A) × CVSS × ((δe + δm) ÷ 2) × μ(E) ÷ θ × Factor, and Rt uses the same terms with the appetite in place of average damage; the factor is a constant 1,000.

- Reductions are shares removed: Mitigated = Re × rp × ri and Residual = Re − Mitigated, so mitigation is multiplicative and a negative residual always signals an estimation error.

- The risk-to-tolerance ratio (Residual ÷ Rt) reads below 0.90 as below, 0.90 to 1.10 as approximately at, and above 1.10 as above tolerance, replacing the old single 1.25 threshold.

- Probabilities, CVSS, criticality, resilience, and the factor cancel in the ratio, so the verdict depends only on damage, treatment, and appetite; the ratio and the estimated risk must always be read together.

- Transfer and avoidance are recorded as costed lines and management decisions, not as reductions, because a package that reduces only probability or only impact mitigates nothing under the convention.

- CVSS v4.0 scores one representative weakness with a full vector, using CVSS-B by default; it is a mandatory multiplier but never a ranking or selection criterion.

- Non-technical weaknesses are scored by equivalency, mapping each metric by analogy with a justification, the label limited applicability, and the marker Analytical estimate — validation required.

- Prioritization uses multiple criteria without CVSS, the portfolio counts shared controls once, and budget calibration compares spending with 12%, 7.8%, or 4% of the IT budget interpolated by appetite.

- Calculated indices are not indicators: KRIs are measured values with a source, owner, frequency, target, warning and critical thresholds, and a stated response that triggers reassessment.

- The management report obtains decisions, an independent quality review checks it, IM, BCP, and DRP handle residual risk, and AI drafts but never calculates final numbers or makes risk decisions.

### Key terms

- **Estimated risk (Re)** — A dimensionless index multiplying both probabilities, the CVSS score, average damage, and criticality, divided by resilience and scaled by the factor.

- **Tolerated risk (Rt)** — The same index as Re with the organization's appetite in place of average damage, expressing how much risk it tolerates in that scenario.

- **Pb(A) and Pb(ψ,A)** — The probability that the threat is present within 12 months, and the probability that it exploits the weakness given evidenced controls, estimated separately.

- **Organizational resilience (θ)** — Demonstrated ability to prevent, detect, contain, respond, recover, and keep operating; strictly above zero because it divides the index.

- **Organizational criticality μ(E)** — How important the affected service is to the mission, independently of how badly a scenario would harm it; also called expected utility.

- **Multiplication factor** — A constant of 1,000 that scales every index to a readable size but never changes any ratio or verdict.

- **Probability and impact reduction (rp, ri)** — The shares of probability of exploitation and of damage that a treatment package removes, each between 0 and 1.

- **Mitigated and residual risk** — Mitigated = Re × rp × ri, the risk removed by treatment; Residual = Re − Mitigated, the risk that remains.

- **Risk-to-tolerance ratio** — Residual ÷ Rt, read in bands: below 0.90, 0.90 to 1.10, and above 1.10; called the pre-treatment ratio when rp and ri are zero.

- **Cost-effectiveness** — (Re − Residual) ÷ Cost × 1,000, the risk units removed per \$1,000 spent on a treatment package.

- **Representative weakness** — The single, specific condition a scenario's causal chain depends on, which is scored with CVSS v4.0 instead of the scenario as a whole.

- **CVSS-B, CVSS-BT, CVSS-BE, CVSS-BTE** — CVSS v4.0 nomenclature stating whether a score uses Base metrics only or adds Threat, Environmental, or both.

- **Equivalency scoring** — Mapping a process, people, governance, physical, third-party, or configuration weakness onto CVSS metrics by documented analogy, labelled limited applicability.

- **Shared control** — One initiative that reduces several scenarios, credited in each package but counted once in the portfolio cost.

- **Budget calibration** — Comparing cybersecurity spending with an appetite-consistent share of the total IT budget (12%, 7.8%, 4%, interpolated) and reporting any gap for management decision.

## Review questions

### Essay questions

1.  Using the MediBec ransomware worked example, explain each step of the CRG calculation from the common term to cost-effectiveness. Then evaluate what the ratio of 1.09 and the sensitivity results (0.69 and 1.53) mean for management, and recommend what MediBec should do next.

2.  The tolerance ratio simplifies to ((δe + δm) ÷ 2) × (1 − rp × ri) ÷ Appetite. Analyze the implications of this cancellation property for how managers should read assessment results, and explain why the ratio and the estimated risk must be read together.

3.  A board director proposes buying more cyber insurance to bring a scenario 'within tolerance'. Using the credit union scenario and CRG's handling of transfer and avoidance, evaluate the proposal and recommend how the decision should be recorded.

4.  Choose a non-technical weakness in an organization you know, such as a process, people, physical, or third-party weakness. Score it with CVSS v4.0 by equivalency, justify every metric, and critically discuss the strengths and limits of this approach in CRG.

5.  An organization with an appetite of 0.35 plans cybersecurity spending of 6.5% of its total IT budget. Calculate the appetite-consistent target and the implied appetite, and recommend how the management report, KRIs, and quality control review should present and follow up this finding.

### Multiple-choice questions

Choose the best answer. Answers are listed in the appendix "Answers to multiple-choice questions".

**1.** According to the chapter, what is the most common way a CRG assessment loses its credibility during the workflow?

> A. Setting the risk appetite after the numbers are already known
>
> B. Using a 12-month horizon instead of a multi-year horizon
>
> C. Generating more than twenty candidate scenarios for ten
>
> D. Recording evidence identifiers for every organizational fact

**2.** In the NIST Risk Management Framework (SP 800-37 Rev. 2), which activity does a CRG risk assessment correspond to?

> A. The Assess step, which verifies that selected controls operate as intended
>
> B. The Authorize step, which accepts residual risk for a system
>
> C. The risk assessment tasks of the Prepare step, at organization and system levels
>
> D. The Monitor step, which tracks controls and risk on an ongoing basis

**3.** Why does CRG require Pb(A) and Pb(ψ,A) to be estimated separately rather than as a single likelihood?

> A. Because CVSS already contains the threat probability for each weakness
>
> B. Because merging them counts the same uncertainty twice
>
> C. Because Pb(A) applies only to insiders and Pb(ψ,A) to outsiders
>
> D. Because the workbook cannot store a combined likelihood value

**4.** A quality reviewer finds that a student memo reports a residual risk of 1,940 while the recalculated workbook shows 1,937.18. How does the chapter treat this?

> A. As acceptable rounding, since indices are dimensionless and approximate
>
> B. As a minor style issue that the instructor may ignore
>
> C. As evidence that the multiplication factor was changed mid-assessment
>
> D. As the most common finding, since the memo must agree with the workbook to the unit

**5.** A scenario has Pb(A) = 0.50, Pb(ψ,A) = 0.40, CVSS = 8.0, δe = 0.40, δm = 0.60, μ(E) = 0.80, θ = 0.50, and a factor of 1,000. What is its estimated risk (Re)?

> A. 640
>
> B. 1,280
>
> C. 1,536
>
> D. 320

**6.** Before treatment, a scenario has δe = 0.50 and δm = 0.70, and the organization's appetite is 0.40. Using the cancellation property, what is its pre-treatment ratio?

> A. 1.50
>
> B. 0.67
>
> C. 1.75
>
> D. 1.25

**7.** A scenario's estimated risk is 2,000. Its treatment package has a probability reduction (rp) of 0.80 and an impact reduction (ri) of 0.50. What are the mitigated and residual risk?

> A. Mitigated 1,300; residual 700
>
> B. Mitigated 1,000; residual 1,000
>
> C. Mitigated 800; residual 1,200
>
> D. Mitigated 1,200; residual 800

**8.** A scenario has δe = 0.50, δm = 0.70, appetite 0.40, and a package with rp = 0.60 and ri = 0.50. What is the post-treatment risk-to-tolerance ratio, and how is it read?

> A. 0.30, below tolerance
>
> B. 1.05, approximately at tolerance
>
> C. 1.50, above tolerance
>
> D. 1.20, above tolerance

**9.** A treatment package costing \$40,000 lowers a scenario's risk from an estimated 2,000 to a residual of 1,200. What is its cost-effectiveness under CRG's formula?

> A. 0.02 risk units per \$1,000
>
> B. 30.0 risk units per \$1,000
>
> C. 50.0 risk units per \$1,000
>
> D. 20.0 risk units per \$1,000

**10.** A scenario has an average damage ((δe + δm) ÷ 2) of 0.60, and the appetite is 0.30. Using the design rule, what minimum value must rp × ri reach for the residual to be no greater than tolerated risk?

> A. 0.50
>
> B. 0.55
>
> C. 0.30
>
> D. 0.70

**11.** An organization with an appetite of 0.60 has a total IT budget, including salaries, of \$20 million. What appetite-consistent cybersecurity budget target does CRG's best-practice benchmark give?

> A. 7.8%, or \$1.56 million
>
> B. 5.7%, or \$1.14 million
>
> C. 5.9%, or \$1.18 million
>
> D. 4.0%, or \$0.80 million

**12.** How does the chapter link incident management, business continuity, and disaster recovery to the CRG formulas?

> A. They are recorded as transfer lines that lower the impact reduction
>
> B. They replace the need to accept residual risk formally
>
> C. They are scored as the representative weakness for CVSS
>
> D. Tested recovery and practised response justify higher θ and lower indices

**13.** A scenario with Re = 1,500 is treated only by cyber insurance, entered with rp = 0 and ri = 0.90. Under CRG's convention, what happens?

> A. Mitigated 1,350 and residual 150, so the scenario falls below tolerance
>
> B. Mitigated 0 and residual 1,500; transfer is recorded as a costed decision
>
> C. Mitigated 750 and residual 750, as insurance halves the exposure
>
> D. Mitigated 1,500 and residual 0, because the loss is fully transferred

**14.** An analyst credits twenty controls with a probability reduction of 0.06 each and adds them together for the package. What does the chapter say about this practice?

> A. It is invalid: the sum of 1.20 exceeds 1 and produces a meaningless residual
>
> B. It is valid, because each control was assessed on its own merits
>
> C. It is valid if the result is rounded down to 0.99 before calculation
>
> D. It is invalid only because catalogue controls carry default costs

**15.** Two scenarios share the same δe, δm, appetite, and treatment package, but one has Pb(A) = 0.20 and the other Pb(A) = 0.80, all else equal. What does the cancellation property imply?

> A. The likelier scenario has a ratio four times higher
>
> B. The rarer scenario is automatically below tolerance
>
> C. Both get the same verdict, but the likelier one has four times the Re
>
> D. Neither ratio can be computed until CVSS scores differ

**16.** Which of the following is deliberately absent from CRG's step 16 prioritization criteria?

> A. Maximum consequence and safety
>
> B. Risk velocity and threat activity
>
> C. Shared benefits of foundational controls
>
> D. The CVSS score of the representative weakness

**17.** A representative weakness is a macro-enabled document that executes code when a user opens it after receiving it by e-mail. Which CVSS v4.0 Attack Vector value applies?

> A. Network, because the document arrived by e-mail
>
> B. Local, because the document is opened on the workstation
>
> C. Adjacent, because the user is on the internal network
>
> D. Physical, because the user must handle the device

**18.** When a non-technical weakness is scored with CVSS v4.0 by equivalency, which three things does CRG require?

> A. A vendor advisory, a CVE identifier, and a Threat metric
>
> B. A Supplemental Safety metric, an Environmental score, and a penetration test
>
> C. A justification per metric, the label limited applicability, and the estimate marker
>
> D. A CVSS-BTE score, a severity band of High, and board approval

**19.** Phishing-resistant MFA reduces three scenarios in a portfolio. How does CRG treat this shared control?

> A. Its effect is credited in each package, but its cost is counted once in the portfolio
>
> B. Its cost is counted in full in each of the three scenarios' packages
>
> C. It is credited to only one scenario to avoid double-counting its effect
>
> D. It is excluded from packages because existing controls cannot be recommended

**20.** Which statement correctly distinguishes a key risk indicator (KRI) from CRG's calculated risk indices?

> A. KRIs are Re and Rt values recalculated monthly by the workbook
>
> B. KRIs are risk indices expressed in dollars rather than units
>
> C. KRIs are the residual risks that exceed the 1.10 band
>
> D. KRIs are measured values from a named source compared with thresholds
