<!-- Introduction to Cybersecurity Governance, 3rd ed., v2.1g (Aug 2026) - (c) Marc-André Léger. Licensed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/). Third-party standards, frameworks and quotations remain under their owners' terms. -->

# Chapter 6: Performing a Cybersecurity Risk Assessment

This chapter explains how to move from the conceptual foundations of risk to a concrete, repeatable cybersecurity risk assessment, using an open-source Excel tool. The approach is intentionally simplified for teaching and introductory practice, yet it remains conceptually aligned with selected elements of the logic of ISO/IEC 27001, ISO/IEC 27005, COBIT and NIST risk management guidance.

In the execution of a risk assessment, the chapter assumes that:

- The organization has already defined its cybersecurity governance framework.

- The fundamentals of risk, risk appetite and KRIs are understood.

- A first set of cybersecurity risk scenarios has been created, whether manually or with the help of generative AI.

In a teaching context, this chapter is meant to be used together with:

- The open-source Excel risk analysis spreadsheet is supplied with the book.

- A structured assignment and business case (for example, the Atlantic Canada Regional Bank or MediBec case that are provided to students).

The goal is to show, step by step, how to go from scenarios to quantified risk indicators, to prioritized mitigation projects, and finally to a concise recommendation to senior management.

## From concepts to practice: why we assess risk

In previous chapters, risk was defined as the intersection of three components:

- A **threat** or threat agent.

- A **vulnerability** that can be exploited.

- A **potential exposure to loss or negative outcome**.

When a threat exploits a vulnerability in a way that produces a negative outcome for the organization, such as financial loss, operational disruption, regulatory breach, loss of life, or reputational damage, a potential exposure to loss or negative outcome has materialized into actual negative impact or damage. Prior to that, the situation posed a potential risk**,** but now it has become a risk**.**

Risk assessment helps organizations move from reactive to proactive behaviour. Instead of waiting for incidents to occur, management asks:

- What could realistically go wrong in our environment?

- How severe would the impacts be for finances, operations, compliance and strategy?

- How likely is each scenario, given known threats and vulnerabilities?

- Which combinations of mitigation, transfer, avoidance or acceptance make sense given our risk appetite and limited resources?

In the GRC continuum, risk assessment belongs to the **R**:

- **Governance** defines objectives, obligations and risk appetite.

- **Risk management** uses those parameters to identify, analyze and evaluate risk scenarios.

- **Compliance** and assurance functions provide independent confirmation that decisions and controls are appropriate and effective.

The practical risk assessment described here follows the **IPM** logic introduced earlier:

1.  **Identification** of threats, vulnerabilities, and potential adverse impacts is achieved through scenarios and business impact assessments.

2.  **Prioritization** of scenarios using risk indicators and the organization’s risk appetite.

3.  **Mobilization** of resources through chosen risk responses (mitigation projects, transfer mechanisms, risk acceptance and IM/BCP/DRP preparedness).

The Excel-based tool used in this chapter operationalizes this IPM cycle in a structured, traceable way that students, and later, practitioners, can apply.

## The CyberRiskGuardian Excel model: purpose and standards alignment

The spreadsheet provided with this book implements a simple yet powerful risk-assessment model, which this book calls the *CyberRiskGuardian* approach.

Its design objectives are:

- To illustrate an end-to-end risk-assessment workflow in one or two class sessions.

- To make explicit the link between qualitative judgement and quantitative indicators.

- To show how risk appetite influences treatment decisions.

- To remain conceptually aligned with ISO/IEC 2700x and NIST guidance.

- To provide a free and simple-to-use tool to prepare students to be able to use commercial tools and risk assessment methodologies used in the real world.

The spreadsheet supports this alignment in several ways.

- First, it assumes that a high-level **risk management framework** already exists (as required by ISO 27001 clause 6 and NIST RMF Prepare and Categorize steps). The spreadsheet does not replace that framework; it is simply a tool to implement the assess-and-treat phases within a defined scope.

- Second, it uses the **risk scenario** construct recommended by ISO 27005 to link threats, vulnerabilities and impacts in a way that is meaningful to the business. Each row in the summary sheet represents one scenario.

- Third, the mitigation sheets allow the user to select or reference controls that map directly to **ISO/IEC 27002** and **NIST SP 800-53** controls and other best-practice catalogues.

- Finally, the model incorporates a **vulnerability severity input** using **CVSS** scores. CVSS provides a standard way to quantify the severity of a vulnerability; this aligns well with both ISO 27005’s need for consistent impact/probability criteria and NIST’s emphasis on repeatable, evidence-based analysis.

The result is a teaching-oriented methodology that can be understood and executed quickly yet maps cleanly onto the expectations of standards-driven programs.

## Inputs to the assessment

Before touching the spreadsheet, certain inputs must be available. Some are produced in earlier chapters; others come from the business case or assignment used in class.

Typically, the following information is required:

1.  **Context and scope:** The business case describes the organization, its mission, regulatory environment, critical systems and data, and the approximate size of its IT and cybersecurity budgets.

2.  **Risk appetite and tolerance:** The governance framework (or the case itself) should indicate whether the organization is risk-averse, risk-neutral or risk-seeking, and may provide a numeric expression of **risk appetite** $A_{r}$ on a 0–1 scale. For example, a value of 0.3 indicates a relatively risk-averse posture.

> This is often complemented by comparison with benchmark ranges for cybersecurity spending, as mentioned in Chapter 4. These ratios will also be used to help students later verify whether their recommended mitigation portfolio is realistic.

3.  **Cybersecurity risk scenarios:** Ten scenarios are usually sufficient in an introductory exercise, as described in the previous chapter.

4.  **Scenario-level quantitative estimates (KRIs):** For each selected scenario, approximate values are needed for:

    - Probability that the hazard will be present $Pb(A)$.

    - Probability that the hazard will exploit the vulnerability $Pb(\psi,A)$.

    - Estimated damage in the most likely case $\delta_{e}(\psi,A)$.

    - Maximum possible damage in a worst-case variant $\delta_{m}(\psi,A)$.

    - Organizational resilience $\theta(\psi,A)$(ability to absorb and recover).

    - Expected utility $\mu(E)$(importance of assets and processes involved).

> All of these are expressed on a continuous scale between 0 and 1, using either historical data or structured judgement. In a real-world situation, the identification of these values would be based on evidence or on a consensus among stakeholders. In a classroom setting, this is facilitated by a large language model such as ChatGPT, as described in Chapter 5.

5.  **Initial vulnerability severity scores:** For the main vulnerabilities referenced in each scenario, a **CVSS Base score** between 0 and 10 should be estimated or taken from a vulnerability scanner, advisory or generative-AI-assisted calculation. The details of CVSS are not the focus here; it is sufficient to have a consistent numeric value for each scenario’s main vulnerability cluster.

With these inputs prepared, the Excel model can be used to complete the assessment.

## Structure of the Excel spreadsheet

Although file layouts may vary slightly between versions, the teaching spreadsheet generally has three main parts:

1.  **Summary sheet**
    The first worksheet presents a compact overview:

    - A cell for **risk appetite** $A_{r}$.

    - One row per scenario (typically up to ten), showing:

      - Scenario ID and name.

      - Key inputs (KRIs, CVSS score).

      - Computed estimated risk $R_{e}$.

      - Computed maximum risk $R_{m}$.

      - Computed tolerated risk $R_{t}$.

      - Computed residual risk with mitigation $R_{mm}$.

      - Total cost of mitigation measures associated with that scenario.

    - A column where the analyst indicates, for each mitigation measure, whether it is **selected** (1) or not (0) for the recommended portfolio.

> Conditional formatting is used to highlight scenarios where residual risk exceeds the tolerated level, drawing attention to unacceptable risk.

2.  **Scenario sheets (S1–S10)**
    Each scenario sheet (for example, S1) corresponds to one risk scenario. The number of sheets required must match the number of scenarios used. It typically includes:

    - Scenario number and label.

    - Date of creation or last update.

    - The six KRI inputs $Pb(A)$, $Pb(\psi,A)$, $\delta_{e}(\psi,A)$, $\delta_{m}(\psi,A)$, $\theta(\psi,A)$, $\mu(E)$.

    - Optional textual notes summarizing the scenario, stakeholders and key assumptions.

    - Space for the **CVSS** score of the main vulnerability cluster.

3.  **Mitigation sheets (M1–M10)**
    For each scenario sheet, there is a corresponding mitigation sheet (for example, M1). This sheet is used to document risk-treatment options. It usually contains:

    - A set of rows where the analyst may define **custom mitigation measures**, with:

      - A free-text description of the measure.

      - Capital expenditure (Capex) and recurring operational expenditure (Opex).

      - Additional staffing requirements.

      - Expected impact reduction $\delta_{r}(\psi,A,MM_{n})$(0–1).

      - Expected reduction in probability of exploitation $Pb(\psi,A,MM_{n})$(0–1).

    - A set of rows pre-populated with standard controls mapped to:

      - **ISO/IEC 27002** references (supporting Annex A of ISO 27001).

      - **NIST SP 800-53** control identifiers.

      - **CIS Controls v8.1.**

> For these predefined controls, indicative cost estimates are already filled in, so students are not required to obtain actual vendor quotes for an introductory assignment.

On the technical side, the spreadsheet includes many formulas. To avoid performance problems, automatic recalculation is typically disabled; the user must explicitly trigger it (for example, by selecting Calculate now in the menus).

## The risk index and underlying formulas

The spreadsheet uses a **Key Risk Indicator (KRI)** to quantify risk for each scenario. The indicator is dimensionless; it is not directly expressed in monetary terms, which helps reduce the emotional and subjective reactions that often arise when discussing money.

For a given scenario $Z$, the **estimated risk** $R_{e}(Z)$is computed as:

$$R_{e}(Z) = \frac{Pb(A) \times Pb(\psi,A) \times \delta_{e}(\psi,A) \times \mu(E)}{\theta(\psi,A)}$$

The **maximum risk** $R_{m}(Z)$uses the maximum damage instead of estimated damage:

$$R_{m}(Z) = \frac{Pb(A) \times Pb(\psi,A) \times \delta_{m}(\psi,A) \times \mu(E)}{\theta(\psi,A)}$$

The **tolerated risk** $R_{t}(Z)$is derived from the organization’s risk appetite $A_{r}$:

$$R_{t}(Z) = \frac{Pb(A) \times Pb(\psi,A) \times A_{r} \times \mu(E)}{\theta(\psi,A)}
$$These three indicators correspond to:

- How risky is the scenario in the **most likely** case ($R_{e}$)?

- How bad could it be in a **worst-case** variant ($R_{m}$)?

- How much risk is the organization willing to **tolerate** in that area ($R_{t}$), given its risk appetite?

When mitigation measures are added, a **mitigated risk index** $R_{mm}$is computed by applying the expected reduction factors for probability and impact:

$$R_{mm}(Z) = R_{e}(Z) \times \delta_{r}(\psi,A,MM_{n}) \times Pb(\psi,A,MM_{n})$$

The CVSS score does not directly enter these formulas; rather, it serves as an additional **decision aid** and tiebreaker. When two scenarios have similar$R_{e}$values, the one with a higher CVSS score may reasonably be prioritized because it is more exposed to well-known, easily exploitable vulnerabilities.

<img src="media/image32.png" style="width:6in;height:4in" />

<span id="_Toc235166914" class="anchor"></span>Figure 32: step-by-step risk assessment procedure

## Step-by-step procedure in the spreadsheet

This section describes a practical, reproducible sequence of steps for using the spreadsheet in a course or introductory consulting engagement.

### Step 1: Set the risk appetite

On the summary sheet, the first task is to enter the **risk appetite** $A_{r}$in its dedicated cell, on a scale from 0 to 1, keeping in mind that values of 0 and 1 are now allowed.

- A value close to 0.1 (for example, 0.2–0.3) corresponds to a **risk-averse** organization.

- A value of 0.5 is **risk-neutral**.

- A value of 0.7 or higher would reflect a **risk-seeking** posture.

The business case or assignment usually specifies the risk appetite explicitly (for example, the board has become strongly risk-averse after a major data breach; use $A_{r} = 0.3$).

A common mistake in class is to forget to change the default value in the spreadsheet (often 0.5). Doing so silently transforms a risk-averse case into a risk-neutral one, and all subsequent calculations become inconsistent with the governance context. In real organizations, misaligning analysis with risk appetite can lead to serious under- or over-investment in cybersecurity.

### Step 2: Populate scenario sheets (S1–S10)

For each of the chosen scenarios (typically ten in an introductory assignment), the analyst moves to the corresponding scenario sheet and enters:

- A unique Scenario ID and name (for example, S3 Ransomware affecting core banking system).

- Creation or update date.

- The six KRI inputs, on a 0–1 scale:

  - $Pb(A)$: probability that the hazard (e.g., the presence of ransomware campaigns in the sector) is present.

  - $Pb(\psi,A)$: probability that this hazard will exploit the specific vulnerability in the organization (for example, unpatched servers).

  - $\delta_{e}(\psi,A)$: estimated damage in the most likely case.

  - $\delta_{m}(\psi,A)$: maximum damage in a plausible worst case.

  - $\theta(\psi,A)$: resilience level (how well incident management, BCP and DRP would limit the damage).

  - $\mu(E)$: expected utility (value of the assets and processes involved in the scenario).

In a professional setting, these values should be obtained through **workshops and structured consensus-building** with stakeholders, ideally using visual slider scales and anonymity to reduce social pressure and bias. Another approach would be to use evidence found based on historical data, industry statistics or research.

In a classroom setting, one of two strategies is often used:

- The case or instructor provides suggested numeric values, based on a prior expert assessment.

- Students derive initial values from a detailed scenario description generated by a large language model, such as ChatGPT, but then adjust them using their own judgement and the hints in the scenario narrative.

In both cases, it is important to document the reasoning briefly in the scenario sheet or in a separate note. The numbers are approximations; what matters pedagogically is that they be **internally consistent** and **traceable**.

### Step 3: Enter CVSS scores

For each scenario, the analyst estimates a **CVSS Base score** for the primary vulnerability or set of vulnerabilities involved.

In more advanced courses or professional practice, this is done using:

- Output from vulnerability scanners or penetration tests.

- Vendor cybersecurity advisories and public CVE entries.

- The official CVSS calculator.

In an introductory context, the scenario description itself, whether written by the instructor or generated with ChatGPT, often includes a proposed CVSS score and a brief justification, which students can either accept or adjust.

The CVSS value is entered in the corresponding column on the summary sheet. It does not change the core KRI formulas, but it adds useful context and helps justify why some scenarios are more urgent than others.

### Step 4: Define mitigation measures (M1–M10)

Once all scenarios are entered and risk appetite is set, the next step is to define **risk treatment options**. This corresponds to the **risk treatment** step in ISO 27005 and to the Select and Implement phases of the NIST RMF.

For each scenario, the analyst goes to the corresponding mitigation sheet and either:

1.  **Defines custom measures**

> For example, in a scenario involving unpatched servers, a custom mitigation measure might be:

- Implement a centralized patch-management platform with automated deployment and exception handling.

> The analyst would then enter:

- Capex (for licenses and implementation services).

- Annual Opex (for maintenance and support).

- Additional staffing effort (for administration and governance).

- Expected impact reduction $\delta_{r}(\psi,A,MM_{n})$(for example, 0.6, meaning a 40 % residual impact).

- Expected probability reduction $Pb(\psi,A,MM_{n})$(for example, 0.4, meaning the probability of exploitation is cut by 60 %).

> These reduction factors must be reasonable and justified, but they are inevitably approximate. In a real engagement, they would be refined with vendor data, benchmarks, and experience.

2.  **Select predefined controls from standards**

> Many versions of the spreadsheet include a catalogue of controls that are already mapped to:

- ISO/IEC 27002 (identified by control ID and title).

- NIST SP 800-53 (identified by control ID).

- CIS Controls v8.1.

> When you select one of these rows, indicative cost estimates are already present. This makes it easy to show, for example, that:

- Implementing ISO 27002 control A.8.21‘Protection of records’ and NIST SP 800-53 control AU-9 ‘Protection of audit information’ will reduce the impact and likelihood of log tampering scenarios at a moderate cost.

Both strategies can be combined. For instance, a scenario may be treated by:

- Selecting one or two standard controls from the catalogue.

- Adding one or two organization-specific measures (such as awareness training or process redesign).

Finally, risk transfer and avoidance measures can be captured in the same sheets:

- **Risk transfer** may be represented by a row describing cyber insurance coverage for certain incident types, with modest probability reduction but very significant impact reduction (because the financial loss is partially reimbursed).

- **Risk avoidance** may be represented by a row describing the cancellation of a high-risk project or the discontinuation of a product line that cannot be made compliant, which may eliminate the scenario in question.

Once the analyst has entered (or selected) all relevant measures for all scenarios, they return to the summary sheet.

### Step 5: Run the calculations

On the summary sheet, after populating all input fields, the analyst triggers a recalculation (for example, using Calculate Now). The spreadsheet then:

- Reads the KRI inputs, risk appetite and CVSS scores from the scenario sheets.

- Reads the costs and reduction factors from the mitigation sheets.

- Computes:

  - Estimated risk $R_{e}$for each scenario.

  - Maximum risk $R_{m}$for each scenario.

  - Tolerated risk $R_{t}$for each scenario.

  - Residual risk $R_{mm}$after applying the selected mitigation measures.

- Sums the costs of all selected measures to yield:

  - A per-scenario mitigation cost.

  - A total recommended cybersecurity investment.

The first thing to check after calculation is whether **residual risk indices remain above tolerated levels**:

- If, $R_{mm}(Z) \leq R_{t}(Z)$for all scenarios, the proposed set of measures brings all considered scenarios into alignment with risk appetite.

- If one or more scenarios still have $R_{mm}(Z) > R_{t}(Z)$, additional or stronger mitigation measures must be designed for those scenarios, or risk appetite must be reconsidered (for example, if requirements are unrealistically strict).

### Step 6: Prioritize scenarios and mitigation measures

In many real organizations, the total cost of all possible mitigation measures will **exceed** the available budget. Even in the classroom, it is useful to explore trade-offs.

Two perspectives guide prioritization:

1.  **Risk-based prioritization**

> Scenarios should be examined in descending order of residual risk $R_{mm}$. Those with the highest indices, especially where the difference between $R_{e}$and $R_{mm}$is large (indicating strong benefit from mitigation), are prime candidates for treatment.
>
> For example, if:

- Scenario 9 (unpatched software) has a high estimated risk and a large risk reduction for a moderate cost; it is likely a top priority.

- Scenario 4 (minor website defacement risk) has a very low estimated risk and small risk reduction for a relatively high cost; it may be deferred.

2.  **Cost-effectiveness**

> For each scenario and measure, an informal risk reduction per unit of cost can be considered. Even if the spreadsheet does not compute this explicitly, it is straightforward to look at:

- The change from $R_{e}$to $R_{mm}$(risk reduction).

- The total cost of measures for that scenario.

> Measures that produce large risk reductions at modest cost are especially attractive early in a program.

In addition to residual risk and cost-effectiveness, the cybersecurity risk assessment methodology outlines several other key criteria for prioritizing risk scenarios and selecting a mitigation portfolio. These criteria ensure that prioritization aligns with business realities, legal constraints, and operational feasibility:

- **Legal, Regulatory, and Contractual Obligations:** Measures required by privacy laws, payment-card requirements (e.g., PCI DSS), contracts, insurance policies, or audit remediations are generally mandatory and must be prioritized regardless of their apparent cost-effectiveness.

- **Strategic Importance:** Additional weight should be given to scenarios that affect the organization's critical products, essential services, core revenue-generating activities, sensitive intellectual property, or digital transformation programs.

- **Maximum Damage and Catastrophic Consequences:** Scenarios that could lead to safety issues, environmental harm, insolvency, permanent loss of critical data, or severe regulatory actions warrant high priority, even if their likelihood is only moderate.

- **Risk Velocity and Threat Urgency:** Prioritization should account for how quickly a threat could cause harm. Exposures that are actively exploited, internet-facing, escalate rapidly, or have long recovery times require urgent treatment.

- **Quick Wins and Shared Benefits:** Prioritizing foundational controls (such as multi-factor authentication, centralized logging, or backup and recovery) is highly effective because they reduce the risk of multiple scenarios simultaneously. A low-cost control that quickly brings a scenario within tolerance is often chosen over a highly expensive package with limited marginal benefit.

- **Current Estimated Risk:** A scenario with a very high current risk score (before any proposed mitigations) may warrant priority attention, even if proposed controls reduce it below tolerance. Because the organization relies heavily on those controls not failing, they require stronger assurance, testing, and monitoring.

- **Feasibility and Time to Benefit:** A technically perfect mitigation plan that cannot be executed will not reduce loss. Prioritization must account for budget, personnel availability, procurement lead times, technological compatibility, and the ongoing maintenance required to sustain the measure.

- **Control Dependencies:** Certain initiatives must be sequenced because one control is a prerequisite for another. For example, creating an asset inventory is necessary before implementing vulnerability management, and centralized logging is required before advanced monitoring can succeed.

- **Risk-to-Tolerance Ratio:** Beyond the absolute residual risk score, organizations should consider how far above tolerance a scenario is. A scenario with a risk-to-tolerance ratio greater than 1.25 is materially above tolerance and typically demands prioritized treatment, exposure reduction, or formal escalation.

At this stage, the analyst also needs to compare total spending against **benchmark ranges**. The recommended benchmarks are that the cybersecurity budget should be between 4% and 12% of the total IT budget, based on the organization’s risk appetite. For example, if the organization’s IT budget is 50 million and it is strongly risk-averse, it may be appropriate to allocate 8-12% of that amount to cybersecurity (4–6 million). If the recommended portfolio costs only 1.7 million and current spending is already 3 million, for a total of 4.7 million, that may be acceptable for a first-year program, but it also suggests that some additional strengthening could be considered (1.3 million remaining).

### Step 7: Formulate recommendations

The final step is to translate the spreadsheet outputs into a concise, decision-oriented recommendation, typically a one-page memo or executive summary.

Such a memo should:

- Briefly recall the **scope and method**:

  - Scenario-based risk assessment aligned with ISO 27005 and NIST RMF.

  - Use of the CyberRiskGuardian Excel model and KRIs.

  - Risk appetite used and benchmark spending range.

- Summarize the **key findings**:

  - The ten scenarios analyzed and the most critical ones (for example, unpatched systems, ineffective incident response, ransomware on core banking).

  - Which scenarios currently exceed tolerated risk and by how much.

- Present the **recommended portfolio of mitigation measures**:

  - Which projects and controls should be implemented in the next budgeting period.

  - The total estimated cost (Capex and Opex).

  - The expected effect on risk indices, bringing all scenarios at or below $R_{t}$.

- Address **residual risk and preparedness**:

  - Confirm that some level of residual risk remains (zero risk does not exist).

  - Explain how incident management (IM), business continuity planning (BCP) and disaster recovery planning (DRP) will address unforeseen or low-probability, high-impact events.

- Highlight **compliance and standards alignment**:

  - Show that the selected measures support relevant ISO 27001 controls and NIST SP 800-53 families and contribute to regulatory and contractual obligations.

In a teaching assignment, students are typically asked to attach their completed spreadsheet and to focus the memo on the recommendation logic, not on re-explaining the entire model. Students are usually provided with a sample memo and recommendation report to guide them.

## Role of IM, BCP and DRP in dealing with residual risk

No matter how sophisticated the assessment and how large the budget, some risk always remains. Some scenarios may be explicitly accepted; others may be mitigated to a level management is comfortable with. For scenarios that remain within appetite, organizations must not simply do nothing. A well-governed program ensures that:

- **Incident management plans** define what happens when a cybersecurity event is detected: roles, responsibilities, communication channels, and technical playbooks.

- **Business continuity plans (BCPs)** outline how mission-critical processes will be maintained during disruptive events, including cyber incidents.

- **Disaster recovery plans (DRPs)** focus on restoring IT infrastructure, data and applications after major disruptions.

Exercises and drills, much like emergency simulations in aviation, are crucial. They train the organization to respond under stress without improvisation. In terms of standards, these capabilities support ISO 27001 controls for incident management and continuity (Annex A) as well as NIST SP 800-53 families such as IR (Incident Response) and CP (Contingency Planning).

In the spreadsheet logic, resilience$\theta(\psi,A)$partly reflects the maturity of these arrangements: better IM/BCP/DRP usually correspond to higher resilience values and, therefore, lower risk indices. These will be discussed further in Chapter 12.

## Ensuring compatibility with ISO 27001, ISO 27005 and NIST

It is useful to make the mapping to formal standards explicit, both for academic rigour and for practitioners who may later need to defend their methods to auditors or regulators.

A simplified alignment is as follows:

- **ISO/IEC 27001**

  - Clause 4 (context) and clause 6 (planning) are reflected in the business case and governance framework, which define scope, risk appetite, and objectives.

  - Clauses 6.1.2 (information security risk assessment) and 6.1.3 (risk treatment) are implemented through scenarios, KRI estimation, spreadsheet analysis, and mitigation selection.

  - Clause 8 (operation) is reflected in the actual implementation of selected measures and in the IM/BCP/DRP capabilities, which are also referenced in the resilience parameter $\theta$.

  - Annex A controls are directly referenced in the mitigation sheets.

- **ISO/IEC 27005**

  - **Context establishment**: business case, asset and process descriptions, regulatory environment, risk appetite.

  - **Risk identification**: creation of risk scenarios (Chapter 5), identification of threats, vulnerabilities and impacts.

  - **Risk analysis**: estimation of KRI parameters, CVSS scores, computation of $R_{e}$and $R_{m}$.

  - **Risk evaluation**: comparison of $R_{e}$and $R_{mm}$to $R_{t}$, ranking and prioritization.

  - **Risk treatment**: selection and costing of mitigation measures, decisions on avoidance, transfer and acceptance.

  - **Monitoring and review**: interpretation of residual risk and planning of periodic reassessments; ability to rerun the spreadsheet with updated scenarios.

  - **Communication and consultation**: stakeholder workshops for scenarios and KRIs, presentation of results in a management-oriented memo.

- **NIST** (SP 800-30, SP 800-37, SP 800-53)

  - The spreadsheet’s assessment phase corresponds to the **Assess** step in the NIST Risk Management Framework (RMF), and its scenario-based approach aligns with NIST SP 800-30’s emphasis on threat, vulnerability, likelihood, and impact.

  - The use of a control catalogue mapped in the mitigation sheets mirrors NIST SP 800-53’s cybersecurity and privacy control families. Students can, for example, see how controls such as AC-2 (Account Management), SI-2 (Flaw Remediation), or IR-4 (Incident Handling) contribute to risk reduction.

  - The selection of measures based on risk indices and appetite supports the **Select** and **Implement** steps of the RMF, while the discussion of IM/BCP/DRP and residual risk contributes to **Respond** and **Monitor** in broader NIST cybersecurity frameworks.

This compatibility does not mean the Excel model is a complete implementation guide for any of these standards. Rather, it serves as a **pedagogical bridge**: students learn the logic and language of risk-based cybersecurity management in a controlled, accessible environment.

## Using generative AI responsibly in risk assessment

Large language models such as ChatGPT are now capable of:

- Maintaining long, contextual conversations about a given organization or case.

- Generating plausible cybersecurity risk scenarios when provided with a system description.

- Proposing mitigation measures, approximate costs and even rough KRI and CVSS values.

- Helping to draft risk-assessment reports and recommendation memos in professional language.

In this book’s methodology, generative AI is used primarily to **accelerate the early stages** of the assessment:

- Seeding a large pool of candidate scenarios from which a subset is selected.

- Producing first-draft detailed scenario descriptions for classroom use.

- Providing approximate quantitative estimates that students can critique and refine.

However, several cautions are essential:

- **Confidentiality**: real organizations must not paste sensitive internal data into public AI tools without appropriate legal and technical safeguards. Internal, self-hosted models or vendor-supplied enterprise versions may be required.

- **Validation**: AI-generated scenarios and numbers are suggestions, not truth. They must be validated by human experts and, where possible, by empirical data.

- **Bias and hallucination**: AI outputs can reflect training-data biases and invent details that are not true. The risk-assessment process must remain anchored in real evidence and sound professional judgement.

- **Accountability**: ISO 27001 and NIST frameworks assume that organizations remain accountable for their risk decisions; delegating this responsibility to an AI system is not acceptable. The AI tool should be used to provide guidance only in a real-world setting, the final decisions remaining with humans.

Used responsibly, generative AI is an effective **assistant** in risk assessment, but it does not replace governance, expertise or evidence.

## Conclusion

This chapter has shown how the conceptual elements of cybersecurity risk, including threats, vulnerabilities, impacts, risk appetite, scenarios and KRIs, can be integrated into a practical, spreadsheet-based assessment aligned with ISO 27001, ISO 27005 and NIST guidance.

Students and practitioners who work through the exercise using the provided Excel tool and assignment will experience the full cycle:

- Translating qualitative scenarios into quantitative indicators.

- Comparing estimated and tolerated risk to identify unacceptable adverse outcomes.

- Exploring mitigation options, costs and residual risk.

- Formulating a concise, defensible recommendation that balances risk reduction, cost and compliance.

The next chapters will build on this foundation, exploring in greater depth how to design specific controls, integrate risk assessment into a continuous improvement cycle, and link cybersecurity risk management to broader enterprise risk and strategy.
