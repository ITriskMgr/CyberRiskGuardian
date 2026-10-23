<!-- Léger, M.-A. (2026). Introduction to Cybersecurity Governance for Business Technology Management (3rd ed., v2.51). Chapter 5: Creating Cybersecurity Risk Scenarios. CC BY-NC 4.0 -->

# Chapter 5: Creating Cybersecurity Risk Scenarios

## Learning outcomes

After completing this chapter, you should be able to:

1.  Explain why scenarios, rather than risk lists or threat names, are the unit of analysis for cybersecurity risk, and what they make estimable, comparable, treatable, and accountable.

2.  Write a cybersecurity risk scenario as a CyberRiskGuardian causal chain, and distinguish a threat label from a decision-useful scenario.

3.  Relate the artifacts of scenario creation to ISO/IEC 27001:2022, ISO/IEC 27005:2022, NIST SP 800-30 Rev. 1, NIST CSF 2.0, and ISO 31000:2018.

4.  Apply the context work that precedes scenario writing: the evidence register, crown jewels and dependencies, scope, the 12-month horizon, and appetite as materiality.

5.  Generate a population of candidate scenarios using the five generation strategies and appropriate public sources, without treating generic threats as organizational evidence.

6.  Evaluate candidates with the twelve screening criteria, escalation flags, consolidation into families, weighting schemes, a coverage matrix, and the 'why this scenario?' test.

7.  Design a validation and participation plan using the examine, interview, and test methods and a cross-functional group of stakeholders.

8.  Recommend governance conditions for using generative AI in scenario work, and apply CRG's three standard markers correctly.

Chapter 4 presented cybersecurity risk management as a governance discipline: risk exists where a threat can exploit a vulnerability to produce an impact on something the organization values, and managers must decide which of those risks to accept, reduce, transfer, or avoid. That definition is correct, but it is not yet usable. A board cannot approve a budget for "ransomware," and a risk owner cannot accept "phishing." Before risks can be estimated, compared, and treated, they must be described in a form that people can examine, challenge, and agree on. In this book, that form is the **cybersecurity risk scenario**.

This chapter presents a scenario-based method designed to help students learn to assess cybersecurity risk through a structured, teachable approach. It is intentionally lighter than the full methods used in large organizations, because formal risk methods require time, training, and organizational data that are rarely available in a course setting. The underlying hypothesis is that students who understand this approach will be well prepared to learn the more robust methodologies used by the organizations that will hire them. The approach is the one implemented in CyberRiskGuardian (CRG), the scenario-driven risk assessment method and toolset used throughout Chapters 5 and 6; where earlier editions of this book and CRG differed, this edition follows CRG.

The approach is not anti-standard. It is deliberately structured so that students learn risk thinking in a way that is compatible with mainstream governance and risk standards. The scenario work in this chapter can be viewed as a practical implementation of the **information security risk assessment process that ISO/IEC 27001:2022 (Clause 6.1.2) requires organizations to define and apply**, including setting risk criteria and producing consistent, valid, and comparable results. It draws on the scenario concepts of ISO/IEC 27005:2022, the risk factors of NIST SP 800-30 Rev. 1, the Govern and Identify functions of the NIST Cybersecurity Framework (CSF) 2.0, and the enterprise risk principles of ISO 31000.

The chapter also addresses a core problem: risk assessment is vulnerable to bias and subjectivity. Optimism, anchoring on the last incident, and the loudest voice in the room all distort judgment. Scenarios reduce these biases because they force participants to describe concrete event sequences, enabling conditions, and business consequences, rather than debating vague risks in the abstract. A well-written scenario also makes its own assumptions visible, so that they can be tested instead of silently inherited.

The chapter proceeds in the order in which CyberRiskGuardian builds a scenario portfolio: why scenarios matter and how the method maps to recognized standards; CRG's scenario standard—a causal chain, never a threat name; the context work that must precede any scenario; candidate generation, screening, consolidation, weighting, coverage, and the final selection of ten scenarios, illustrated with the MediBec teaching case; and finally validation, participation, the responsible use of generative AI, and the hand-off to Chapter 6.

## Why scenarios: the unit of analysis for cyber risk

Scenarios have long been used to help decision-makers reflect on plausible futures, stress-test assumptions, and consider alternatives. In cybersecurity, they are especially useful because threats evolve quickly, organizations have complex dependencies, and purely historical loss data can be misleading. Historical data often fails to capture recent architectural changes, new vendors, cloud migrations, emerging attack patterns, or new regulatory exposure. ISO/IEC 27005:2022 frames scenarios as a structured way to connect risk sources, threats, vulnerabilities, events, and consequences, and it encourages analysts to combine historical evidence with forward-looking judgment.

### A scenario is a story with a causal chain

In this book, scenarios are the central unit of analysis for learning cybersecurity risk assessment. **A scenario is a story.** Following the logic of the risk triangle introduced in Chapter 4, it describes how a threat source exploits one or more vulnerabilities so that the organization's exposure materializes as an adverse outcome—one that affects the organization's mission, strategy, legal and regulatory obligations, and confidentiality, integrity, and availability requirements.

A story, however, can be told well or badly. CyberRiskGuardian therefore imposes a discipline on the story: every scenario must be written as a **causal chain** that links a threat source or event, a vulnerability or predisposing condition, an affected asset or process, a cybersecurity event, and a business consequence. A scenario that cannot be written this way is not yet a scenario; it is a topic. The discipline matters because each link in the chain is a place where a control can intervene, where evidence can be gathered, and where an estimate can be challenged.

Scenarios become most useful when they are explicit about event sequences, triggers, stakeholders, and consequences, because that is precisely where hidden assumptions and optimism bias tend to live. A statement such as "our backups protect us against ransomware" sounds reassuring until a scenario asks whether the backups share an administrative domain with the production systems an attacker would already control.

### What scenarios do that risk lists cannot

Many organizations begin with a risk register that lists items such as "cyberattack," "data breach," or "third-party failure." Such lists are easy to produce and almost impossible to manage: nothing in them says how likely or severe the event would be, which weakness a control should remove, or who should own the outcome. Scenarios solve these problems in four ways. First, they make risk **estimable**: a specific chain allows separate judgments about whether the threat will act, whether it will succeed, and how much harm it would cause. Second, they make risk **comparable**: when every scenario uses the same structure, horizon, and scales, a ransomware scenario and a supplier scenario can be ranked on the same footing. Third, they make risk **treatable**: the chain shows where a control breaks the sequence, which turns a vague concern into a defensible investment proposal. Fourth, they make risk **accountable**: each scenario has an affected process, a business owner, and a consequence that someone must either accept or treat.

| A threat is not a scenario. **"Ransomware" or "Phishing" alone names a threat; a scenario explains who exploits which weakness, in which asset or process, to cause which business consequence.** |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

### Scenarios as a managerial instrument

For managers, scenarios are not a technical artifact owned by the security team. They are the means by which a board's risk appetite is applied to concrete situations, the means by which investment proposals are justified, and the record that shows a regulator or auditor how risk decisions were reached. A scenario portfolio that is current, evidence-based, and owned by the business is one of the strongest signals of real cybersecurity governance, as opposed to compliance theatre.

There is no universally correct number of scenarios. In an educational setting, ten detailed scenarios are typically sufficient for students to learn the mechanics and reasoning, and CRG's default portfolio is ten. In practice, a reasonable maturity progression is to start with about ten scenarios and gradually increase to twenty or more over a few assessment cycles, as stakeholders become comfortable with the method and the organization becomes more data-driven. The MediBec worked example published with CRG illustrates both: a ten-scenario assessment and a thirty-scenario assessment built on the same context. In the long term, the right number depends on the organization's size, complexity, and analytical capacity.

## Standards compatibility

Because this is an educational method, compatibility does not mean that a student assignment satisfies an auditor. It means that the method produces artifacts that map cleanly onto what recognized standards expect, so that the skills transfer directly to professional practice. Five references matter most, and the book uses their formal names.

### ISO/IEC 27005:2022 and ISO/IEC 27001:2022

ISO/IEC 27001:2022 (as amended by Amd 1:2024) requires an organization to define and apply an information security risk assessment process that establishes risk criteria, identifies risks associated with the loss of confidentiality, integrity, and availability, identifies risk owners, analyses consequences and likelihood, and evaluates the results against the criteria (Clause 6.1.2). It then requires a risk treatment process that determines the necessary controls, compares them with Annex A, and produces a Statement of Applicability and a risk treatment plan (Clause 6.1.3). A CRG scenario portfolio provides the identified risks, owners, consequences, and likelihood inputs; the treatment work of Chapter 6 provides the treatment plan.

ISO/IEC 27005:2022, *Information security, cybersecurity and privacy protection — Guidance on managing information security risks*, is the more direct reference. It describes risk identification through two complementary approaches: an **event-based approach**, which reasons from risk sources and the events they can cause at a strategic level, and an **asset-based approach**, which reasons from assets, their threats, and their vulnerabilities at an operational level. It also treats risk scenarios as the bridge among risk sources, events, consequences, and controls, and it encourages the use of both approaches together. CRG's top-down and bottom-up generation strategies, described later in this chapter, are practical implementations of these two approaches.

### NIST SP 800-30 Rev. 1 and the NIST CSF 2.0

NIST SP 800-30 Rev. 1, *Guide for Conducting Risk Assessments* (2012), remains NIST's reference for the risk assessment step of the NIST risk management process. It decomposes risk into risk factors: threat sources, threat events, **vulnerabilities and predisposing conditions**, likelihood, and impact. CRG borrows its vocabulary directly. The phrase "vulnerability or predisposing condition" in CRG's scenario standard comes from SP 800-30, and it is important because many material weaknesses are not software flaws at all: a flat network, a single data centre, a two-person security team, or a supplier with standing remote access are predisposing conditions rather than vulnerabilities in the narrow technical sense. SP 800-30 also separates the likelihood that a threat event is initiated from the likelihood that it results in adverse impact, a distinction CRG preserves with two separate probability parameters (see Chapter 6).

The NIST Cybersecurity Framework 2.0 (February 2024) places risk assessment within its Identify function (category ID.RA, Risk Assessment), which expects organizations to identify threats, vulnerabilities, likelihoods, and impacts and to use them to understand inherent risk and prioritize risk responses. CSF 2.0's Govern function supplies the context that scenarios depend on: organizational context (GV.OC), risk management strategy including appetite and tolerance (GV.RM), and roles and responsibilities (GV.RR). A scenario portfolio is therefore an Identify output that only makes sense when Govern inputs exist.

Scenarios also support NIST-aligned control work. The treatments proposed for a scenario can be mapped to the control families of NIST SP 800-53 Rev. 5, tailored using the baselines in SP 800-53B, and later assessed using the assessment objectives and methods of SP 800-53A Rev. 5. Chapter 9 explains these catalogues; the point here is that scenario documentation should make such mapping possible.

### ISO 31000 and the enterprise view

ISO 31000:2018, *Risk management — Guidelines*, supplies the enterprise-level principles within which cybersecurity risk sits: risk management should be integrated, structured, customized, inclusive, dynamic, based on the best available information, attentive to human and cultural factors, and continually improved. ISO 31000 is under revision at the time of writing, with a committee draft circulated in 2026, but the 2018 edition remains the published standard. Two of its principles shape CRG directly. "Best available information" is why CRG insists on an evidence register and on labelling every estimate. "Inclusive" is why scenario creation must involve the business, not only technical specialists.

CRG also draws on concepts from Factor Analysis of Information Risk (FAIR), the quantitative risk model maintained as an Open Group standard: threat event frequency, threat capability, resistance strength, and loss magnitude. The separation of "will the threat act?" from "will it succeed?" from "how much harm?" is shared across all these references.

The table below summarizes how the main artifacts of this chapter correspond to these references.

| **CRG artifact**              | **ISO/IEC 27005 and 27001**                           | **NIST**                              | **ISO 31000**              |
|-------------------------------|-------------------------------------------------------|---------------------------------------|----------------------------|
| Evidence register and context | Context establishment; documented information         | CSF GV.OC; SP 800-30 preparation step | Best available information |
| Crown jewels and dependencies | Asset-based approach; risk owners                     | CSF ID.AM; SP 800-30 impact           | Scope and context          |
| Causal-chain scenarios        | Risk scenarios; event-based approach                  | SP 800-30 risk factors                | Risk identification        |
| Screening and selection       | Risk analysis and evaluation criteria                 | CSF ID.RA prioritization              | Risk evaluation            |
| Validation and evidence plan  | Risk assessment results that are valid and comparable | SP 800-53A examine, interview, test   | Monitoring and review      |

## The CyberRiskGuardian approach to scenario creation

CyberRiskGuardian is a scenario-driven method for assessing cybersecurity risk, supported by an AI assistant, a desktop application, and an Excel model. It is a **decision-support tool**: it structures the analysis and speeds it up, while the analyst remains accountable for every estimate and management remains accountable for every risk decision. Its estimates are initial analytical hypotheses, not facts about the organization.

### How this book uses CyberRiskGuardian

The book follows CRG's method and terminology throughout Chapters 5 and 6, and the teaching activities associated with the book are built on it. **All teaching activities should use CyberRiskGuardian's Community Edition, which is free for students and non-profit organizations under a Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0) licence, or its Educational edition, which is licensed to colleges and universities as institutions and includes installation and customization.** The Community Edition includes an education package with an instructor guide, a student lab workbook built on the MediBec teaching case, an assessment kit, and a library of five fictional business cases; these are the sources for the examples in this chapter.

The method is the same whether students work with CRG's AI assistant, its desktop application, or its Excel model; many courses draft scenarios with the assistant and quantify them in the application or workbook, where every number is recomputed by the verified engine.

### The process at a glance

CRG's workflow runs from organizational context to crown jewels, risk appetite, candidate scenarios, a selected portfolio, quantification, treatment, and recommendations. Scenario creation occupies the first part of that sequence and follows five broad stages, shown in the figure below: establishing the facts and materiality criteria; generating about 25 to 40 candidates for a portfolio of ten; screening every candidate against twelve criteria, with escalation flags that protect severe but improbable scenarios; consolidating variants into families and testing the ranking under different weighting schemes; and selecting the final portfolio, normally ten scenarios, with a coverage review and preparation for quantification.

<span id="_Toc243635529" class="anchor"></span>Figure 30: Scenario creation strategy in CyberRiskGuardian

The sequence matters. Candidates written before the context exists tend to be generic, and a selection made before screening and consolidation tends to be dominated by whichever technique the analyst knows best. The goal is not to list every conceivable threat, but to produce a manageable, decision-useful portfolio that represents the organization's material exposure.

### Principles that hold throughout

Several principles apply at every stage and are worth stating once. **Never invent organizational facts.** Conclusions rest on the evidence supplied; where evidence is missing, the gap is recorded rather than filled with a plausible guess. **Use one horizon and one scoring method.** Every scenario is assessed over the same period, 12 months by default, with the same scales. **Treat scores as decision-support indicators,** not as loss probabilities or dollar predictions. **Preserve traceability** from objectives to assets, threats, vulnerabilities, scenarios, consequences, controls, scores, and recommendations. **Keep accountability human:** an AI tool or an analyst may propose, but the final selection and every risk acceptance belong to accountable people.

To make these principles visible in every deliverable, CRG uses three standard markers, written literally wherever they apply:

- **Analytical estimate — validation required** marks a value that an analyst or an AI tool estimated rather than observed.

- **Information gap — organizational validation required** marks a fact that nobody has supplied and that the assessment needs.

- **Management decision required** marks a choice that only management can make, such as approving an appetite value, accepting a residual risk, or excluding a scenario that carries an escalation flag.

These markers are the single most useful habit the method teaches: a student who uses them correctly has understood the difference between evidence, inference, and judgment.

## The scenario standard: causal chains, not threat names

CRG's scenario construction standard is short and strict. A scenario must never be "Ransomware," "Phishing," "Data breach," or "Cloud compromise" alone. It must be written as a causal chain:

**threat source or event → vulnerability or predisposing condition → affected asset or process → cybersecurity event → business consequence.**

### The five links of the chain

The **threat source or event** identifies who or what initiates the scenario: a ransomware affiliate, an initial-access broker, a malicious or careless insider, a compromised supplier, a state-linked actor, or a non-adversarial event such as hardware failure. Naming the source matters because capability, motivation, and frequency differ widely between an opportunistic scanner and a targeted extortion group.

The **vulnerability or predisposing condition** identifies the weakness that the source exploits. It may be technical (an unpatched edge device, push-based multifactor authentication susceptible to fatigue), procedural (help-desk password resets without identity verification), organizational (a two-person security team without after-hours monitoring), architectural (backups reachable from the production domain), or contractual (a supplier with standing, unbrokered remote access). It should be specific enough that a control could remove it and an auditor could test for it.

The **affected asset or process** identifies what is harmed, ideally both the primary business asset (patient records, the payment process) and the supporting technical assets (the health record application, the directory, the data centre), connecting the scenario to the crown jewels identified during context work. The **cybersecurity event** describes what actually happens: unauthorized access, exfiltration, encryption, manipulation of data, or loss of service or recoverability. The **business consequence** states why management should care: interrupted care, a reportable privacy incident, a regulatory sanction, a material financial loss, a safety risk, or damage to public trust. Without this last link, the scenario remains a technical observation.

### From weak to strong statements

The difference between a threat label and a scenario is easiest to see side by side. "Ransomware" is a threat. "A ransomware affiliate uses stolen VPN credentials and the absence of multifactor authentication on remote access to encrypt the clinical scheduling servers, stopping appointments at all sites for several days" is a scenario. The second statement can be estimated (how likely is a ransomware affiliate to obtain VPN credentials in the next 12 months?), challenged (is MFA really absent on remote access?), and treated (enforce phishing-resistant MFA, segment the scheduling servers, test restoration).

CRG's lab workbook offers a sentence pattern that students find helpful: *Because \[threat source or event\] can exploit \[weakness\] affecting \[asset or process\], \[undesirable event\] may occur, resulting in \[consequences\] and impairing \[objective\].* The pattern is a scaffold, not a requirement; experienced analysts write the same content in more natural prose. A useful test of any statement is to ask, as the CRG instructor guide suggests, "Which step of your chain would a control interrupt?" If no answer is possible, the chain is incomplete.

### Multi-stage and hybrid scenarios

Real attacks often unfold in stages: initial access, privilege escalation, lateral movement, reaching a critical system, interruption or exfiltration, notification, and third-party or reputational effects. CRG allows multi-stage and hybrid scenarios where the stages genuinely belong to one exposure, because analysing them together produces a stronger causal account and avoids counting the same consequence several times. A ransomware scenario that includes backup destruction and data theft is usually more realistic, and more decision-useful, than three separate scenarios for encryption, backup loss, and exfiltration.

The opposite error, combining unrelated events artificially, must also be avoided: the test is whether each stage plausibly causes or enables the next.

### What a developed scenario records

Once a scenario has been selected for the final portfolio, it is developed in full. CRG records, for each selected scenario, an identifier and name; a concise statement; the business objective and process affected; the stakeholders and risk owner; background and scoping assumptions; the threat source and initiating event; the vulnerabilities or predisposing conditions; the affected assets and processes, primary and supporting; the dependencies; the existing controls, recorded only when evidence supports them; the event sequence; the consequences across confidentiality, integrity, availability, operations, finance, regulatory obligations, reputation, privacy, and safety; the relevance to risk appetite; the selection rationale; the evidence basis; the confidence level; and the validation needs.

Three additional elements strengthen alignment with ISO/IEC 27001 and ISO/IEC 27005:

- **Risk criteria and acceptance references.** The scenario should reference the risk criteria in use, including the organization's appetite and acceptance thresholds, because ISO/IEC 27001 requires these criteria to exist before risk assessment results are meaningful and comparable. In CRG, the appetite value and any escalation thresholds stated by the organization play this role.

- **Asset linkage.** Each scenario identifies at least one primary business asset and one supporting asset, because controls apply to specific assets and ISO/IEC 27005 links threats, vulnerabilities, events, and consequences to assets.

- **Control mapping.** Proposed treatments should later be mapped to ISO/IEC 27001 Annex A control themes (organizational, people, physical, and technological) and, where useful, to NIST SP 800-53 Rev. 5 control families, so that control selection and assessment planning are disciplined rather than improvised. Chapter 6 and Chapter 9 develop this mapping.

## Context first: evidence, crown jewels, and scope

A risk assessment starts from the business, not from a list of threats. CRG makes this explicit by placing context work before any scenario is written. Its context-intake step produces four things: an evidence register, an organizational context summary, a crown-jewel and dependency map, and a scope statement with a list of gaps. Everything later in the assessment traces back to this step.

### The evidence register

The **evidence register** lists every source the assessment relies on. Each row receives an identifier (E1, E2, and so on), a title, a type, and a note of what the source supports. CRG distinguishes documented evidence (policies, architecture diagrams, audit reports, contracts), exercise or simulation evidence (tabletop injects, sample logs), input from subject-matter experts, and external evidence (threat reports, sector statistics). For each important assertion later in the assessment, the analyst cites the evidence identifier or labels the assertion as an assumption.

Two rules prevent common errors. Exercise or simulation material is indicative evidence to be validated, never a confirmed incident: a tabletop inject showing an approved MFA prompt from an unusual location suggests a weakness worth testing, not proof that an attacker succeeded. Instructor, template, or benchmark values are calibration aids, not organizational evidence.

The register also supports a finer classification used during scenario selection. Each item is classified as verified organizational evidence, external or sector evidence, expert judgment, assumption, or unknown requiring validation. This classification later drives the "evidence strength" criterion in screening and the confidence rating of each selected scenario.

### Organizational context and crown jewels

The context summary covers, where evidence exists, the organization's mission and objectives, sector, products and services, customers or beneficiaries, critical business processes, information assets and their sensitivity, critical systems, infrastructure and cloud, identity and remote access, operational technology, Internet of Things, or medical devices, third parties, geography and jurisdictions, legal, regulatory, and contractual obligations, availability and continuity requirements, existing controls, incident history and audit findings, transformation initiatives, and cybersecurity maturity. Where a topic has no evidence, the analyst writes **Information gap — organizational validation required** rather than filling it in.

From this context, the analyst identifies the **crown jewels**: the assets or services whose loss or compromise would most harm the mission. For each crown jewel, CRG records its business purpose, owner, confidentiality, integrity, and availability requirements, technical dependencies, and third-party dependencies. The analyst then traces the chain **Objective → Process → Primary asset → Supporting assets → Dependencies → Consequences**. For a clinic network, one such chain runs from continuity of care to clinical documentation, to the electronic health record, to the directory, data centre, and wide-area network, to the consequence of delayed or unsafe care if any link fails.

Crown jewels are not always the systems that hold the most sensitive data. Identity and remote access are typically crown jewels even though they store little business information, because every other crown jewel depends on them.

### Dependencies, single points of failure, and peripheral assets

CRG then analyses dependencies explicitly, because many material scenarios arise not from the crown jewels themselves but from what they depend on. The analysis covers identity, DNS and network, cloud, virtualization, backup, telecommunications, privileged administration, critical software-as-a-service, managed service providers, the software supply chain, payment providers, specialized suppliers, and outsourced processes. The aim is to find **single points of failure** (a single data centre, a single identity plane, a single integration engine) and **concentration risk** (many crown jewels relying on one supplier or one administrative team).

The analysis also looks beyond centrally managed IT. **Peripheral assets**—clinic Wi-Fi, building IoT devices, medical equipment, unmanaged local storage, forgotten administrative interfaces, staff use of external AI services—often sit outside IT oversight yet can still reach crown jewels. CRG flags them explicitly so that bottom-up and dependency-driven scenario generation can use them.

### Scope, the 12-month horizon, and appetite as materiality

The scope statement defines what the assessment covers and excludes, the risk perspectives considered (confidentiality, integrity, availability, privacy, operations, finance, compliance, reputation, safety, and strategy), the assumptions made, and the material information gaps. It also fixes the **assessment horizon**, which in CRG is **12 months** unless the organization specifies otherwise. A common horizon is essential: a probability estimated over one year cannot be compared with one estimated over five, and scenarios that mix horizons produce rankings that mean nothing.

Before candidates are generated, CRG also establishes the **risk appetite**, expressed on a 0-to-1 scale with anchors ranging from 0.1–0.2 (very high risk aversion) through 0.3 (risk averse), 0.5 (risk neutral), and 0.7 (relatively high appetite) to 0.8–0.9 (very high appetite). The organization's approved value is used where it exists; otherwise the analyst proposes one, labelled **Analytical estimate — validation required** and **Management decision required**. CRG's guidance notes that a health, public-safety, or critical-infrastructure organization rarely justifies more than 0.3, and a small unregulated firm rarely justifies less than 0.4. Chapter 4 explains appetite and tolerance in governance terms, and Chapter 6 shows how the appetite becomes the tolerated risk for each scenario.

For scenario creation, the appetite has a second role: it is translated into **materiality criteria** that tell the analyst which consequences matter enough to justify a scenario. For a healthcare organization with low appetite for patient-safety and privacy risk, any credible threat to clinical decisions, any exposure of health information for many patients, and any loss of clinical systems at several sites beyond a working day are material. These criteria become selection criteria: a plausible scenario that could exceed tolerance must stay on the list even when its probability is low.

## Generating candidate scenarios

With the context in place, the analyst generates a broad population of **candidate scenarios**. Candidates are short: each is a one-sentence causal chain with an identifier, a name, the primary asset or process, the threat, the vulnerability, and the main consequence, tagged with the confidentiality, integrity, or availability dimension it mainly affects and with the generation strategy that produced it. Candidates are deliberately numerous, because the quality of the final portfolio depends on the breadth of the population from which it is selected.

### How many candidates

CRG's scenario-selection instructions call for about **25 to 40 candidates for a final portfolio of ten scenarios**, scaled proportionally for other portfolio sizes, and never fewer than **20**. The lower bound is the quick version used in the student labs: 20 candidates, from which the 10 most material are developed. The upper range is preferable whenever the context is rich, because a larger population makes it more likely that unusual but material exposures surface. In the published MediBec example, the ten-scenario assessment drew on 20 candidates and the later re-selection under the full procedure drew on 45.

Each candidate must still be a real causal chain grounded in the context. Duplicates are expected and are dealt with during consolidation rather than suppressed during generation, because suppressing them early hides which exposures recur across several perspectives.

### Five generation strategies

Scenario creation should be treated as a project, typically led by a cybersecurity risk analyst, using interviews, brainstorming, and consensus-building. CRG combines five complementary generation strategies, and a good candidate population uses at least three of them; a candidate produced by more than one strategy deserves attention.

**Top-down generation** begins with organizational strategy, objectives, governance requirements, and undesirable end states, then asks how deficiencies, nonconformities, or control breakdowns could realistically occur and cause business harm. It corresponds to the event-based approach of ISO/IEC 27005 and aligns naturally with risk-based planning, because it requires the scenario to link business objectives, obligations, and risk criteria before discussing technical fixes. A hospital's top-down question might be: "What would have to happen for a cyber event to cause unsafe care?"

**Bottom-up generation** starts with operational value creation: critical systems, processes, data flows, and third parties, and the weaknesses the evidence reveals in them. It corresponds to the asset-based approach of ISO/IEC 27005. Scenario development links events, consequences, threats, and vulnerabilities to assets, and controls are then applied to subsets of those assets, which is precisely what the bottom-up method trains students to do. A bottom-up question might be: "The case shows backup failures and a shared administrative domain; what scenario does that enable?"

**Threat-driven generation** starts from the threat actors and techniques that are active against the organization's sector and region, and asks which of them could succeed given the organization's weaknesses. It uses threat intelligence and sector reports, discussed below, and protects the portfolio against an inward-looking view that ignores what adversaries are actually doing.

**Dependency-driven generation** starts from the dependency analysis, single points of failure, concentration risks, and peripheral assets identified during context work. It is the strategy most likely to surface supplier, cloud, telecommunications, and shared-infrastructure scenarios that neither the business nor the security team would think of first.

**Localized generic generation** starts from published generic scenarios and adapts them to the organization. Generic scenarios are useful scaffolding, especially early in an organization's maturity. The standards-consistent discipline is to localize each one: anchor it to real or plausible assets and events, to an explicit undesirable outcome, and to the organization's risk criteria and appetite. CRG's rule is that a generic scenario is never selected merely because it is common; it earns its place only when the organization's context makes it material. As an organization matures, generic sources become a coverage check rather than a primary source of candidates.

Generative AI is not a sixth strategy. It is a drafting aid that can apply any of the five strategies quickly, and it is discussed separately later in this chapter because its use must be governed.

### Sources for generic and threat-driven candidates

Several reputable public sources support threat-driven and localized generic generation. Each describes threats in general; none is organizational evidence, and CRG's rule that "generic threats are not organizational evidence" applies to all of them.

- **ENISA Threat Landscape.** The European Union Agency for Cybersecurity's annual analysis of prime threats, actors, and trends (the 2025 edition was published in October 2025).

- **Canadian Centre for Cyber Security, National Cyber Threat Assessment.** The Cyber Centre's biennial assessment (the 2025–2026 edition at the time of writing) of the threats most likely to affect Canadian organizations, including ransomware, state-sponsored activity, and threats to critical infrastructure.

- **MITRE ATT&CK.** A knowledge base of adversary tactics and techniques based on real-world observations, useful for writing realistic event sequences; it describes how attacks unfold, not how likely they are.

- **ISACA risk scenario resources.** ISACA's risk scenario guidance, including its 2022 *Risk Scenarios Starter Pack* of example IT risk scenarios, provides a structured checklist of scenario families.

- **ISO/IEC 27005:2022 annexes.** Illustrative examples of risk sources, threats, and vulnerabilities provide a neutral, standards-based checklist.

- **Case scenario libraries.** In a course, each business case includes a scenario library or list of scenario prompts. CRG's case library is explicit that these are candidates, not answers: each library entry a team keeps must be rewritten as a complete causal chain grounded in the case.

## Screening, consolidating, and selecting

Candidate generation deliberately produces more scenarios than the assessment can analyse in detail. The next stages narrow the population to a portfolio, and they are where most of the analytical judgment in scenario creation lies. CRG treats selection as strategic filtering and prioritization, not as a mechanical ranking.

### Twelve screening criteria

Every candidate is scored from 1 (very low) to 5 (very high) on twelve criteria, with a recorded rationale. For evidence strength, 5 means direct organizational evidence and 1 means assumption only. The criteria, with the question each answers, are as follows.

- **Business criticality:** does the scenario threaten an important objective or critical process?

- **Asset criticality:** does it affect high-value information, technology, services, or infrastructure?

- **Threat relevance:** is the threat credible for this organization and sector?

- **Exposure or vulnerability:** are there plausible weaknesses or predisposing conditions?

- **Potential impact:** could the consequences materially affect the organization?

- **Regulatory and legal:** could it create important compliance, privacy, contractual, or legal consequences?

- **Operational resilience:** could it disrupt critical services or create cascading failures?

- **Dependency and concentration:** does it involve important third parties or single points of failure?

- **Evidence strength:** is there organizational, technical, historical, sector, or threat-intelligence evidence?

- **Risk velocity:** could material consequences develop rapidly?

- **Control uncertainty:** is it uncertain whether existing controls would prevent or contain it?

- **Decision usefulness:** would analysing it potentially change a management decision?

The total, out of 60, is a **screening aid, not a mechanical rule**. Two criteria are often misunderstood. Control uncertainty rewards scenarios where nobody knows whether controls work, because those are exactly the scenarios that analysis can clarify. Decision usefulness asks whether better understanding could change a decision about investment, controls, acceptance, transfer, architecture, suppliers, or governance; a scenario that cannot change any decision adds little, however dramatic. In a first course, CRG's lab workbook uses a simplified screen of four criteria (business criticality, threat relevance, exposure, and potential impact); the full twelve-criterion screen is used for graded and professional work.

### Escalation flags and confidence

Scores alone would allow a severe but improbable scenario to drop out of the portfolio. CRG prevents this with **mandatory escalation flags**: a scenario that could affect human safety, interrupt an essential service, cause catastrophic data loss, trigger systemic organizational failure, produce severe privacy consequences, cause a major regulatory breach, cause existential financial loss, compromise highly privileged infrastructure, destroy backup or recovery capabilities, cause simultaneous failure of primary and recovery environments, expose critical supply-chain concentration, or spread across multiple units or jurisdictions cannot be excluded for low probability alone. A flagged scenario may still be left out of the final ten, but only after an explicit review, and its exclusion is a **Management decision required**.

Each candidate also receives a **selection confidence** rating of high, medium, or low, together with its dominant evidence basis. Confidence is not the same as risk. A scenario with low confidence and high potential consequence may need investigation rather than exclusion; CRG's guidance is explicit that uncertainty must never be confused with low risk. Where information allows, FAIR-inspired reasoning helps test whether the scores are coherent.

Each candidate ends the screen with a **disposition**: Select, Consolidate (into another scenario), Consider (explicit review), or Defer.

### Consolidating variants into families

Candidate populations always contain variants of the same exposure: three phishing scenarios that differ only by department, or separate candidates for the initial access, privilege escalation, and encryption stages of one ransomware attack. CRG consolidates these into **families**, each represented by a stronger causal scenario. Consolidation serves two purposes. It prevents double-counting, in which the same exposure is estimated and treated several times, inflating total risk and the apparent case for investment. It also prevents any single technique from dominating the portfolio.

The rule for keeping scenarios separate is practical: two candidates remain distinct when their assets, threat actors, consequences, controls, owners, or treatment decisions differ materially. Phishing of clinic staff and phishing of billing staff are usually one family, because they share the same weakness, owner, and treatment. Phishing that leads to health-record access and business e-mail compromise that diverts supplier payments may be separate families, because the consequence, owner, and control set differ.

### Testing the selection under different weighting schemes

Equal weighting of the twelve criteria embodies an assumption: that every criterion matters equally. CRG tests that assumption by scoring each family under at least four weighting schemes. The **equal** scheme weights all criteria at 1. An **impact-led** scheme gives extra weight to business criticality, potential impact, regulatory consequences, and operational resilience. An **evidence-led** scheme gives extra weight to evidence strength, threat relevance, and exposure. A **decision-led** scheme gives extra weight to decision usefulness and control uncertainty. Each family takes the highest score of any of its members, so that variants do not add up.

The output of this sensitivity check is not a new ranking but two kinds of flag. A family that is selected but does not rank in the top ten under every scheme is **selected by judgment**, and the rationale must say why. A candidate that is not selected but ranks in the top ten under some scheme requires **explicit review**. These flags make the analyst's judgment visible instead of hiding it inside the arithmetic, which is what a reviewer, an auditor, or a board needs to see.

### Coverage and the final ten

The final selection is normally **ten scenarios**, unless evidence justifies otherwise, and it is emphatically not the ten highest totals. CRG selects on a balance of materiality, potential to exceed tolerance, evidence, uncertainty that needs attention, regulatory and safety significance, critical dependencies, coverage, and decision usefulness.

Coverage is checked with a **coverage matrix** that maps each selected scenario to the categories it addresses: confidentiality, integrity, and availability; identity; ransomware; cloud and SaaS; third parties; insiders; applications and APIs; infrastructure; endpoints; OT, IoT, or medical devices; privacy; fraud and business e-mail compromise; resilience; AI; and safety. A category is included only where the exposure is credible and material; the matrix is a test for blind spots, not a quota to be filled.

Each selected scenario must then pass the **"why this scenario?" test**, which asks six questions: Why is it relevant to this organization? What organizational evidence supports it? Which asset or objective does it threaten? Which weakness or condition does it exploit? What material consequence could it cause? Why was it selected over the strongest candidates left out? The last question is the hardest and the most useful, because it forces a comparison that a simple ranking avoids.

Finally, CRG reviews the portfolio as a whole: whether it represents material exposure without duplication, overlooks major dependencies, addresses low-frequency, high-impact risks and third-party and systemic risks, reflects regulatory and safety considerations, and supports meaningful management decisions. Candidates that were not selected but remain material go onto a **watch list** with explicit promotion triggers, and the review ends with an **evidence-collection plan** that says how each selected scenario will be validated, by whom, and by when.

| The final portfolio is a management instrument, not a league table. **Ten scenarios that cover distinct assets, actors, consequences, and decisions are worth more than the ten highest scores.** |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## Illustrative scenario: building MediBec's portfolio

MediBec is a fictional private medical centre with a Montreal head office, a Montreal data centre, and ten clinics in the Montreal, Québec City, Gatineau, and Saguenay regions. It is the lab case of CRG's education package, and its full assessment is published with the Community Edition so that students can check their work. The case describes a lean IT department of fifteen people, two of them dedicated to cybersecurity; an electronic health record (EHR), radiology, laboratory, pharmacy, billing, and scheduling systems; cloud storage used to transfer medical images; connected medical equipment; and multiple unsuccessful attack attempts since 2020, increasing recently, with no confirmed breach.

### Context and evidence

The analyst's evidence register lists nine sources. E1 to E5 are sections of the business case, including the technology environment, threat context, and risk appetite statement. E6 is the case's incident injects, classified as exercise evidence to be validated: a macro-enabled phishing attachment from a look-alike domain, a suspicious PowerShell alert closed as low severity, a remote login from an unusual location approved through an MFA push prompt, an external download of a medical-image export from cloud storage, unusual query volumes from a service account, and backup failures at several clinics. E7 is the audit universe, E8 the case's scenario library, and E9 the instructor annex's illustrative values, recorded as calibration only.

The crown-jewel analysis identifies eight assets or services, including the integrated EHR and clinical record data, diagnostic systems, identity and remote access, the Montreal data centre with its WAN and backups, cloud storage for image transfer, and connected medical equipment. Identity is recognized as a crown jewel because every clinical and administrative service depends on it. The dependency analysis flags the single identity plane and the Montreal hub as concentration points and the clinic networks, medical devices, and staff use of external AI services as peripheral assets.

The case states a normalized tolerance of 0.40, with executive review above 0.25 and board acceptance above 0.40. The analyst proposes an appetite of 0.30 because of patient-safety and health-data sensitivity, labelled **Analytical estimate — validation required** and **Management decision required**. The materiality criteria derived from this appetite include any credible threat to patient care, exposure of health information of many patients, and loss of clinical systems at several clinics beyond a working day. The horizon is 12 months.

### Three candidates written as causal chains

Among the 45 candidates in the full selection exercise, three illustrate the method well. Each is written as a causal chain and tagged with its generation strategy.

- **C01 — Phishing-led compromise of clinic administrator credentials** (threat-driven and bottom-up). A financially motivated external actor phishes clinic administrators and billing staff with an EHR-themed lure, captures credentials, and obtains approval of an MFA push prompt, exploiting push-based MFA susceptible to fatigue and the delivery of macro-enabled attachments, to gain remote access to the EHR, billing, and administrative systems, enabling unauthorized access to patient records and follow-on intrusion.

- **C20 — MFA push-fatigue bypass of remote access** (bottom-up). An attacker holding a stolen password floods a clinician with MFA push requests until one is approved, exploiting push approval without number matching, taking over a remote-access session and reaching the EHR, with unauthorized access to patient information as the consequence.

- **C23 — Cloud tenant takeover through token theft or consent phishing** (threat-driven and dependency-driven). An attacker steals a session token or tricks a user into granting consent to a malicious application, exploiting tokens that are not bound to devices and a tenant setting that lets users consent to applications, to obtain control of MediBec's cloud tenant, with tenant-wide exposure of health data and a foothold for destructive attacks.

Each statement is a complete chain that could be estimated, challenged, and treated. Each is also, on inspection, a route into the same identity plane.

### Screening and consolidation

The table below shows the screening results for these three candidates and for three others discussed in this example. Totals are out of 60 under equal weighting; the escalation flags are those recorded in the MediBec selection workbook.

| **Candidate**                                | **Total (/60)** | **Escalation flags**                                                     | **Disposition**       |
|----------------------------------------------|-----------------|--------------------------------------------------------------------------|-----------------------|
| C01 Phishing of clinic administrators        | 52              | Severe privacy; privileged infrastructure; multiple clinics              | Select (family P01)   |
| C20 MFA push-fatigue bypass                  | 52              | Privileged infrastructure                                                | Consolidate into P01  |
| C23 Cloud tenant takeover                    | 48              | Privileged infrastructure; severe privacy; multiple clinics              | Consolidate into P01  |
| C03 Backup compromise before encryption      | 54              | Catastrophic data loss; backup destruction; primary and recovery failure | Consolidate into P02  |
| C31 Privileged administrator sabotage        | 48              | Privileged infrastructure; backup destruction; systemic failure          | Consider (watch list) |
| C29 Patient data pasted into public AI tools | 44              | Severe privacy; major regulatory                                         | Select (family P10)   |

C01 and C20 score identically, which is a clue rather than a coincidence: they describe the same exposure. C20 is a technique within C01's chain, and both are treated by the same decision (phishing-resistant MFA and conditional access) under the same owner. C23 differs in its technique and its immediate asset, but it exploits the same identity plane, shares the treatment family, and has the same owner. The analyst therefore consolidates C01, C20, C23, and two further identity variants (help-desk reset fraud and mailbox compromise) into a single family, **P01: identity takeover of clinical, privileged, or cloud-administrator accounts**. The consolidated statement reads: *A cybercriminal obtains valid credentials or session tokens for clinic administrators, clinicians, or cloud administrators—through EHR-themed phishing, MFA push fatigue, help-desk reset fraud, or token theft—and exploits push-only MFA, absent conditional access, and weak reset verification to access the EHR, mail, and the cloud tenant, leading to health-data exposure and a foothold for destructive attacks.*

The same logic applies to the ransomware chain. C03, backup compromise before encryption, has one of the highest totals in the population, but it is a stage of the ransomware scenario rather than an independent exposure. It is consolidated with ransomware encryption, detection and response gaps, untested incident response, hypervisor encryption, and Active Directory escalation into **P02: ransomware with data theft and backup destruction halting multi-clinic operations**. CRG's own notes on the example observe that a raw top ten by total score would have contained four ransomware-chain variants and three identity variants—exactly the variant domination the method warns against.

### Selection, judgment, and the watch list

After consolidation, the analyst scores each family under the four weighting schemes. Eight of the ten selected families rank in the top ten under every scheme. Two do not and are **selected by judgment**. P05, manipulation of clinical data leading to unsafe care, is retained as a low-frequency, high-consequence patient-safety scenario: without it, the portfolio would treat integrity only as a side effect. P10, uncontrolled disclosure of patient information by staff through unapproved AI tools, is retained for coverage and decision usefulness, because no other scenario addresses negligent insiders or AI, and it drives a distinct decision on AI policy and sanctioned tools.

The strongest unselected candidate is C31, sabotage by a malicious or departing IT administrator, which outranks P10 under every weighting scheme and carries several escalation flags. The analyst does not select it, because its treatments (privileged access management, immutable backup, and rigorous offboarding) are delivered through P01 and P02, but places it on the watch list for explicit management review. The reliance on P01 and P02 to treat C31 is a **Management decision required**.

The coverage matrix shows that the final ten address confidentiality in seven scenarios and integrity and availability in five each. The only coverage gap is fraud and business e-mail compromise, deferred because its consequence is bounded and treated by low-cost finance controls; it will be promoted if Finance cannot confirm that payment-verification procedures exist. The evidence-collection plan assigns twelve validation activities, from an identity and cloud configuration review within 30 days to a cross-functional validation workshop within 45 days.

The lesson of the example is that the method's value lies less in the scores than in the decisions they force: which variants are one exposure, which improbable scenarios must stay, which judgment calls must be explained, and which gaps must close before the numbers in Chapter 6 can be trusted.

## Scenario validation

Scenario validation is where standards compatibility becomes tangible. It helps ensure that the risk assessment process yields results that are consistent, valid, and comparable, as ISO/IEC 27001 requires, and it supports risk treatment planning and documentation. In CRG, validation is not a single event at the end of the process. It begins when evidence is classified, continues through the "why this scenario?" test, and culminates in the evidence-collection plan and the cross-functional review of the portfolio.

<span id="_Toc243635530" class="anchor"></span>Figure 31: Scenario validation

### What validation checks

Validation confirms that each scenario correctly links its threat source, vulnerabilities, affected assets, consequences, and controls, as the figure above shows, and that it supports the later definition of key risk indicators for monitoring. In practice, the validation questions follow the causal chain. Is the threat source credible for this organization, and on what evidence? Does the weakness actually exist, or was it inferred? Is the affected asset correctly identified, with its owner and its dependencies? Is the consequence stated in business terms and consistent with the materiality criteria? Are the existing controls recorded only where evidence supports them? Are assumptions labelled, and are information gaps marked for organizational validation?

CRG's quality-control checklist adds that each scenario must be organization-specific, use the common horizon, keep threats and vulnerabilities distinct, and label AI-generated assumptions. A scenario that fails any of these tests returns to the analyst before it is quantified.

### Examine, interview, test

Validation also serves as the bridge to control assessment. Even when risk scenarios are used in a classroom, or when a full control assessment is not performed, scenarios should be written so that a real organization could later assess the existence and effectiveness of controls using recognized methods and evidence. NIST SP 800-53A Rev. 5 describes three assessment methods: examining documents and artifacts, interviewing personnel, and testing mechanisms or processes. Scenario documentation should be written so that, later, someone could reasonably answer three questions:

- What would we examine to confirm that the weakness exists and the claimed controls are in place?

- Whom would we interview to confirm how the process actually works, and who owns it?

- What would we test to validate the control claim, and what result would show that it works?

The MediBec evidence-collection plan illustrates the point. The identity scenario is validated by reviewing MFA methods and coverage, conditional-access policies, the help-desk reset procedure, and sign-in logs; the ransomware scenario by reviewing detection coverage, log retention, backup isolation, the last restore test, and the incident response plan's test history. Each activity has an owner and a deadline, so that validation becomes a managed task rather than a good intention.

## Who to include in scenario creation

Cybersecurity risk scenarios should be developed collaboratively, not by a single technical specialist. Involving a representative cross-section of stakeholders reduces individual bias, improves the scenarios' completeness and realism, and increases organizational acceptance of the resulting assessment and treatment decisions. CRG makes cross-functional validation an explicit step: for each selected scenario, the analyst maps who is responsible and who must be consulted, and records the disagreements the analyst anticipates.

At a minimum, the scenario-development group should include the business owner of the affected asset or process, who can explain its strategic and operational importance and may ultimately serve as the risk owner. IT and cybersecurity personnel with knowledge of the relevant systems, dependencies, vulnerabilities, and existing controls should also participate. In addition, a representative from enterprise risk management, compliance, legal, or governance should contribute expertise on the organization's risk criteria, regulatory obligations, risk appetite, and formal acceptance practices. An executive sponsor gives the exercise authority and ensures that its conclusions reach decision-makers.

Depending on the scenario, other participants may include privacy, human resources, finance, procurement and vendor management, business continuity and incident response, physical security, architecture, internal audit, and the managers of key suppliers or outsourced services. When safety, operational technology, medical devices, or critical infrastructure are involved, engineering, clinical, and operational personnel should be included as well. In the MediBec example, biomedical engineering validates the medical-device scenario and clinical operations validates the data-integrity scenario, because nobody else can judge their consequences credibly.

This multidisciplinary participation is not meant to achieve unanimous agreement on every estimate; it ensures that the scenario reflects business, technical, legal, and control realities. CRG asks the analyst to surface disagreements rather than smooth them over: if the CIO believes backups are isolated and the infrastructure team is unsure, that disagreement is itself a finding, and it belongs in the evidence-collection plan. Differences in professional judgment should be documented and resolved through the organization's established risk-governance and decision-making processes.

## Using generative AI responsibly to draft scenarios

Generative AI can accelerate scenario creation considerably. It can draft context summaries from a business case, propose candidates across all five generation strategies, rewrite threat labels as causal chains, suggest consolidations, and critique a draft portfolio for blind spots. In course projects with limited stakeholder access, a well-instructed model can produce plausible first drafts comparable to early-stage human brainstorming; in organizations, it allows more scenarios to be explored with fewer resources. These benefits come with risks that must be governed: invented facts, fabricated sources and control identifiers, overconfident estimates, and disclosure of confidential information.

CyberRiskGuardian is itself an AI-assisted tool, and its approach to AI is a useful model. CRG's AI assistant track is built on Claude, packaged as skills that automate the context-intake, appetite, scenario-building, and quantification steps described in this chapter and Chapter 6, including the full screening and selection procedure. Its desktop application can route AI features to a locally hosted model or switch them off entirely, shows the exact content of any AI request before it is sent, and can anonymize it. Whatever tool is used, the principles below apply; they are tool-neutral by design.

### Governance conditions for AI-assisted scenario work

Four conditions should be met before AI is used to draft scenarios. First, **use an approved tool**. In a course, this is the tool designated by the instructor; in an organization, it is an AI service approved under the organization's AI governance, as discussed in Chapter 18. Second, **protect confidentiality**. Fictional teaching cases are safe to use with AI tools. Real organizational information—architecture diagrams, vulnerability reports, incident records, audit findings—is sensitive, because a scenario portfolio is effectively a map of how to attack the organization. It should not be submitted to an external AI service unless the organization has authorized it in writing and applicable data-protection rules allow it (see Chapter 16); a locally hosted model or an enterprise service with contractual data-protection commitments is the appropriate option.

Third, **keep the markers**. Every value or assertion that an AI tool proposes and that the case or evidence does not support carries **Analytical estimate — validation required** or **Information gap — organizational validation required**, and every choice reserved for management carries **Management decision required**. CRG's assistant includes a label check that lists missing markers; a deliverable without markers is a warning sign that AI output was taken as fact. Fourth, **keep humans accountable**. The analyst remains accountable for the final selection and every estimate, and management remains accountable for every risk decision. In a course, students disclose their use of AI according to course rules, using the AI acknowledgement provided in CRG's assessment kit.

### What AI may and may not do

CRG's education package draws a clear line that applies equally to professional practice. AI may draft scenarios, propose parameter values with rationales, critique estimates, suggest controls, check language, and summarize sources. AI may not state facts about the organization that the evidence does not contain; produce a final number that has not been recomputed with a verified calculation engine; or invent standards, control identifiers, legal obligations, statistics, or sources. AI tools are particularly prone to inventing control identifiers, CVSS vectors, and laws with great confidence. Verification is therefore the skill that matters: a fabricated or untraceable source is both an evidence problem and, in a course, an academic-integrity concern.

### A tool-neutral prompt template

The following template replaces the product-specific prompts of earlier editions. It is aligned with CRG's scenario fields and works with any capable approved model. Where CRG's assistant is available, its scenario-builder skill performs the same steps automatically and produces the screening workbook; the template remains useful for understanding what the automation does and for working in other tools. Replace the bracketed values before use.

*Part 1 — context and rules.* "You are assisting a cybersecurity risk analyst with a scenario-driven risk assessment of \[ORGANIZATION\], following the CyberRiskGuardian method. Treat the attached \[business case or documents\] as the authoritative organizational source. Do not invent systems, incidents, controls, legal duties, statistics, or sources that the documents or verifiable primary sources do not support. Build an evidence register (E1, E2, and so on) and cite an evidence identifier for every organizational assertion. Use a 12-month horizon and 0-to-1 scales. Distinguish verified organizational evidence, external or sector evidence, expert judgment, assumptions, and unknowns. Mark estimates 'Analytical estimate — validation required', missing facts 'Information gap — organizational validation required', and management choices 'Management decision required'. Identify the crown jewels with owners, C/I/A requirements, and dependencies, and trace Objective → Process → Primary asset → Supporting assets → Dependencies → Consequences."

*Part 2 — candidates.* "Generate \[25–40\] candidate scenarios for a final portfolio of \[10\]. Write each as a causal chain: threat source or event → vulnerability or predisposing condition → affected asset or process → cybersecurity event → business consequence. Never use a threat name alone. For each candidate give: ID, name, one-sentence causal statement, primary asset or process, threat source, vulnerability or condition, main consequence, C/I/A dimension, and generation strategy (top-down, bottom-up, threat-driven, dependency-driven, or localized generic). Cover identity, ransomware, cloud and SaaS, third parties, insiders, applications and APIs, infrastructure, endpoints, OT, IoT or medical devices, privacy, fraud, resilience, and AI only where the exposure is credible for this organization."

*Part 3 — screening and selection.* "Score every candidate from 1 to 5 on business criticality, asset criticality, threat relevance, exposure or vulnerability, potential impact, regulatory and legal, operational resilience, dependency and concentration, evidence strength, risk velocity, control uncertainty, and decision usefulness, with a one-line rationale. Add mandatory escalation flags where applicable, a selection confidence (high, medium, low), and a disposition (Select, Consolidate, Consider, Defer). Consolidate variants into families. Recompute family rankings under equal, impact-led, evidence-led, and decision-led weights and flag selections that rest on judgment. Propose ten scenarios with a coverage matrix, answer the six 'why this scenario?' questions for each, and list the strongest unselected candidates for a watch list. Do not rank on CVSS."

*Part 4 — development.* "For scenario \[ID\], record: name, statement, objective and process, stakeholders and risk owner, background, threat source and initiating event, vulnerabilities or predisposing conditions, affected assets and processes, dependencies, existing controls (evidence-backed only), event sequence, consequences (confidentiality, integrity, availability, operations, finance, regulatory, reputation, privacy, safety), appetite relevance, selection rationale, evidence basis, confidence, and validation needs. Prepare quantification inputs for Pb(A), Pb(ψ,A), δe, δm, θ, and μ(E), each as an estimate with rationale, confidence, and the evidence required to validate it, and identify a representative weakness for CVSS v4.0 scoring or state its limited applicability. Do not calculate final risk values."

### Reviewing AI output

Before AI-assisted scenarios are used, they must be reviewed as one would review a junior analyst's work. Check that each organizational fact is supported by the evidence register; confirm that cited sources exist and say what the model claims; remove invented systems, incidents, controls, and obligations; check that every candidate is a genuine causal chain; and revise any estimate the team cannot justify in its own words. A practical test, used in CRG's labs, is to ask of each scenario: "Which step of this chain does the evidence support?"

| AI can draft a scenario portfolio in minutes, but **only evidence, markers, and accountable human review turn a draft into an assessment that management can rely on**. |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## From scenarios to quantification: preparing Chapter 6

Scenario creation ends where quantification begins. CRG's final step in scenario selection is to prepare, for each selected scenario, the inputs that Chapter 6 will use. Each input is recorded as an estimate with a rationale, a confidence level, and the evidence required to validate it, without false precision.

### The parameters each scenario must support

Chapter 6 uses six parameters, each on a 0-to-1 scale over the 12-month horizon, and each maps onto a link of the causal chain:

- **Pb(A), the probability that the threat is present:** will this threat source act against the organization during the period? It is informed by the threat-source link and by threat intelligence.

- **Pb(ψ,A), the probability of exploitation:** if the threat acts, will it succeed against this weakness given existing controls? It is informed by the vulnerability link and by the evidence about controls. CRG estimates it separately from Pb(A) because merging the two double-counts.

- **δe, the expected damage:** how much harm is reasonably expected if the scenario occurs? It is informed by the consequence link.

- **δm, the maximum damage:** how much harm is plausible at worst? It must be at least equal to δe, and it is where escalation flags such as safety or catastrophic data loss are reflected.

- **μ(E), organizational criticality:** how important is the affected service or process to the mission? It is informed by the crown-jewel analysis and is not a second damage estimate. Some sources call this parameter expected utility; this book, like CRG, estimates it as criticality.

- **θ, organizational resilience:** how well can the organization prevent, detect, contain, respond, recover, and keep operating? Higher values mean stronger resilience, and the value is never zero.

A scenario that cannot support separate judgments about presence, exploitation, and damage is usually missing a link in its chain.

### CVSS v4.0: a multiplier, never a selection criterion

Each selected scenario also needs a **CVSS v4.0** score for a defined, representative weakness in its chain, with the full vector string. In CRG, the CVSS score is a mandatory multiplier in the risk index calculated in Chapter 6, but it is never a ranking or selection criterion. CRG's critical rules for scenario selection are explicit: do not auto-select technically severe vulnerabilities, and do not rank on CVSS. A critical software flaw in an isolated test system may matter far less than a moderate weakness in the identity plane that gates every crown jewel.

During scenario development, the analyst's task is limited to naming the weakness that will be scored. For technical scenarios this is usually straightforward. For process, people, governance, physical, third-party, and configuration weaknesses—a missing payment-verification procedure, a supplier's standing remote access—CVSS does not apply directly. Chapter 6 explains how such weaknesses are scored by equivalencies with the CVSS v4.0 metrics, labelled with CRG's "limited applicability" caution and the appropriate evidence marker.

Chapter 6 combines these inputs into the estimated and tolerated risk for each scenario and compares them using the risk-to-tolerance ratio; the quality of every number there depends on the scenarios produced here.

### Questions managers and boards should ask

- Is every scenario in our portfolio written as a causal chain, or do we still have entries such as "ransomware" or "data breach" that cannot be estimated or treated?

- Which crown jewels did the context work identify, and does every crown jewel appear in at least one selected scenario?

- Does our evidence register distinguish documented evidence from exercise material, expert opinion, external sources, and assumptions, and can each scenario be traced to it?

- How many candidates were generated, by which strategies, and why were the remaining candidates consolidated, considered, or deferred?

- Which scenarios carry escalation flags for safety, essential services, backup destruction, or catastrophic data loss, and were any of them excluded? If so, who approved that decision?

- Which selections rest on judgment rather than score, and is the rationale convincing?

- What is the strongest scenario we did not select, and what would cause us to promote it from the watch list?

- Does the coverage matrix reveal any blind spot, such as third-party, insider, integrity, or AI-related exposure?

- Which information gaps most affect our conclusions, and when will the evidence-collection plan close them?

- Were business owners, risk, legal, privacy, and operational staff involved, and where did they disagree?

- If generative AI was used, was the tool approved, was confidential information protected, and was every AI-generated assertion verified and labelled?

- Is anyone using CVSS scores to rank or select scenarios rather than to describe a specific weakness?

## Bringing it together: scenarios as the foundation of defensible risk decisions

Cybersecurity risk scenarios are the point at which governance meets analysis. Risk appetite, risk ownership, and accountability are abstract until they are applied to concrete situations, and scenarios are those situations. The CyberRiskGuardian approach begins with evidence and context, the crown jewels and their dependencies, a 12-month horizon, and materiality criteria derived from appetite. It then generates a broad population of causal-chain candidates, screens them while protecting severe scenarios with escalation flags, consolidates variants, tests the selection under different weightings, and selects about ten scenarios that cover the organization's material exposure and support real decisions.

The method's strength lies in its discipline rather than its arithmetic. Each step is designed to expose a common failure of risk assessment: inventing facts, mistaking threats for scenarios, confusing probability with impact, letting one familiar technique dominate, discarding the improbable catastrophe, ranking on technical severity, or presenting AI-generated assumptions as findings. The three markers—**Analytical estimate — validation required**, **Information gap — organizational validation required**, and **Management decision required**—carry that discipline into every deliverable, making it clear what is known, what is estimated, what is missing, and what only management can decide.

For business technology managers, the practical lesson is that a scenario portfolio is a governed artifact. It should have an owner, a refresh cycle, a documented evidence base, validation by the people who own the affected processes, and a clear path to the decisions it informs. Generative AI can make it faster to build, but not more trustworthy unless the same discipline is applied to its output. The selected scenarios now carry everything Chapter 6 needs to turn them into estimated and tolerated risk and to show how treatment changes the picture; a portfolio built with care here is what makes those numbers worth acting on.

## Chapter summary

Chapter 4 defined risk; this chapter makes it usable. A board cannot approve a budget for 'ransomware', and a risk owner cannot accept 'phishing'. Risks must first be described as scenarios: causal chains linking a threat source, a weakness, an affected asset or process, a cybersecurity event, and a business consequence. Following the CyberRiskGuardian method, the chapter shows how context and evidence come first, how a broad population of candidates is generated, screened, consolidated, and narrowed to about ten scenarios that cover material exposure, and how validation, broad participation, governed use of AI, and three standard markers make the resulting portfolio a defensible basis for Chapter 6.

### Key points to retain

- Scenarios make risk estimable, comparable, treatable, and accountable, which risk lists and threat names cannot do; a threat is not a scenario.

- The method is compatible with ISO/IEC 27001 Clause 6.1.2, the event-based and asset-based approaches of ISO/IEC 27005:2022, the risk factors of NIST SP 800-30 Rev. 1, CSF 2.0 Govern and ID.RA, and ISO 31000 principles.

- Every CRG scenario is a causal chain: threat source or event, vulnerability or predisposing condition, affected asset or process, cybersecurity event, and business consequence.

- Three markers make knowledge status visible: Analytical estimate — validation required, Information gap — organizational validation required, and Management decision required.

- Context comes first: an evidence register, a context summary, crown jewels with dependencies, and a scope with a common 12-month horizon and appetite-derived materiality criteria.

- About 25 to 40 candidates, never fewer than 20, are generated for a portfolio of ten, using at least three of the top-down, bottom-up, threat-driven, dependency-driven, and localized generic strategies.

- Candidates are scored 1 to 5 on twelve criteria (total out of 60) as a screening aid, and mandatory escalation flags prevent severe but improbable scenarios from dropping out for low probability alone.

- Variants are consolidated into families to prevent double-counting and technique domination, and four weighting schemes flag selections that rest on judgment.

- The final ten are chosen for materiality, coverage, and decision usefulness, not as the ten highest totals, and pass the 'why this scenario?' test, with a watch list and an evidence-collection plan.

- Validation follows the causal chain and prepares later control assessment through examine, interview, and test, while cross-functional participation surfaces disagreements as findings.

- Generative AI is a drafting aid, not a sixth strategy: use approved tools, protect confidential information, keep the markers, verify everything, and keep humans accountable.

- Each selected scenario prepares six parameters for Chapter 6 and a representative weakness for CVSS v4.0, which is a mandatory multiplier but never a ranking or selection criterion.

### Key terms

- **Cybersecurity risk scenario** — A structured story describing how a threat source exploits one or more weaknesses so that the organization's exposure materializes as an adverse business outcome.

- **Causal chain** — CRG's required scenario structure: threat source or event → vulnerability or predisposing condition → affected asset or process → cybersecurity event → business consequence.

- **Predisposing condition** — A weakness that is not a software flaw, such as a flat network, a single data centre, or a supplier with standing remote access, that makes a scenario possible.

- **Evidence register** — The numbered list (E1, E2, and so on) of every source the assessment relies on, classified by type and strength, which every organizational assertion must cite.

- **Crown jewels** — The assets or services whose loss or compromise would most harm the mission, recorded with owners, C/I/A requirements, and dependencies.

- **Materiality criteria** — Criteria derived from risk appetite that state which consequences matter enough to justify a scenario and must keep plausible high-impact scenarios on the list.

- **Generation strategies** — The five complementary ways of producing candidates: top-down, bottom-up, threat-driven, dependency-driven, and localized generic generation.

- **Screening criteria** — The twelve criteria, each scored 1 to 5 with a rationale, whose total out of 60 is a screening aid rather than a mechanical rule.

- **Escalation flag** — A mandatory flag, such as human safety or backup destruction, that prevents a scenario from being excluded for low probability alone without a management decision.

- **Scenario family** — A group of variants of the same exposure consolidated into one stronger causal scenario to prevent double-counting and technique domination.

- **Coverage matrix** — A map of selected scenarios to risk categories used to test the portfolio for blind spots, not as a quota to be filled.

- **Watch list** — Material candidates that were not selected, recorded with explicit triggers that would promote them into the portfolio.

- **Analytical estimate — validation required** — The CRG marker for a value that an analyst or AI tool estimated rather than observed.

- **Management decision required** — The CRG marker for a choice only management can make, such as approving appetite, accepting residual risk, or excluding a flagged scenario.

## Review questions

### Essay questions

1.  Apply the CyberRiskGuardian context and generation steps to an organization of your choice. Describe its evidence register, three crown jewels and their dependencies, its materiality criteria, and five candidate scenarios written as causal chains, each tagged with the generation strategy that produced it.

2.  Using the MediBec example, analyze why the final portfolio should not simply be the ten candidates with the highest screening totals. Discuss consolidation, escalation flags, weighting schemes, coverage, and the treatment of C31 on the watch list.

3.  Evaluate the claim that CRG's scenario method is 'compatible with standards' while remaining an educational method. Explain how its artifacts map to ISO/IEC 27001:2022, ISO/IEC 27005:2022, NIST SP 800-30 Rev. 1, NIST CSF 2.0, and ISO 31000:2018, and what compatibility does not mean.

4.  A colleague proposes to generate the organization's entire scenario portfolio with a public AI chatbot by pasting in internal documents. Critically assess this proposal and recommend governance conditions, review practices, and markers that would make AI-assisted scenario work defensible.

5.  Design a validation and participation plan for one scenario of your choice. Identify who must be involved, what disagreements you anticipate, and what you would examine, whom you would interview, and what you would test to confirm the weakness and the claimed controls.

### Multiple-choice questions

Choose the best answer. Answers are listed in the appendix "Answers to multiple-choice questions".

**1.** Which of the following statements meets CRG's standard for a scenario rather than a threat label?

> A. Ransomware attacks against clinical systems, which could cause serious disruption to operations at all sites
>
> B. Phishing campaigns by financially motivated actors targeting staff across the organization during the next year
>
> C. A ransomware affiliate uses stolen VPN credentials and absent MFA to encrypt scheduling servers, halting appointments for days
>
> D. A data breach involving patient information, with regulatory and reputational consequences for the organization

**2.** Which sequence correctly states the five links of CRG's causal chain?

> A. Threat source or event, vulnerability or predisposing condition, affected asset or process, cybersecurity event, business consequence
>
> B. Asset, threat actor, existing control, likelihood of occurrence, and expected financial impact
>
> C. Vulnerability, threat source, cybersecurity event, compensating control, and residual risk after treatment
>
> D. Business objective, crown jewel, threat name, CVSS score, and recommended treatment package

**3.** Which ISO/IEC 27005:2022 approach does CRG's bottom-up generation strategy implement?

> A. The event-based approach, reasoning from risk sources at a strategic level
>
> B. A quantitative approach, reasoning from historical loss data and frequencies
>
> C. A threat-based approach, reasoning from adversary techniques catalogued in ATT&CK
>
> D. The asset-based approach, reasoning from assets, their threats, and their vulnerabilities

**4.** Why does CRG use the phrase 'vulnerability or predisposing condition', borrowed from NIST SP 800-30 Rev. 1?

> A. Because only flaws with CVE identifiers can be scored and must be separated
>
> B. Because many material weaknesses, such as a flat network, are not software flaws
>
> C. Because predisposing conditions remove the need for an evidence register
>
> D. Because NIST requires every weakness to be recorded as a condition

**5.** What is CRG's default assessment horizon, and why must the same horizon apply to every scenario?

> A. Five years, because cyber risks evolve slowly and need long-range planning
>
> B. 12 months, because probabilities estimated over different horizons cannot be compared
>
> C. Three years, to match the certification cycle of an ISO/IEC 27001 management system
>
> D. 12 months, because cyber insurance policies are renewed on an annual basis

**6.** Which statement about crown jewels is consistent with the chapter?

> A. Crown jewels are always the systems that store the most sensitive data
>
> B. Crown jewels are identified after the scenario portfolio has been selected
>
> C. Identity systems are excluded because they store little business information
>
> D. Identity and remote access are typically crown jewels because other crown jewels depend on them

**7.** A healthcare organization has no approved risk appetite and asks the analyst to propose one. Which response follows CRG guidance?

> A. Propose about 0.30 or less, marked as an analytical estimate and a management decision
>
> B. Use 0.50 as a neutral default without labels, since none was supplied
>
> C. Propose 0.70, because healthcare faces sustained high threat activity
>
> D. Leave appetite blank and define materiality later, after scenarios are selected

**8.** How many candidate scenarios do CRG's scenario-selection instructions call for when building a final portfolio of ten?

> A. Exactly ten, one candidate for each final scenario
>
> B. About 12 to 15, allowing a few spares
>
> C. About 25 to 40, and never fewer than 20
>
> D. At least 100, covering every ATT&CK technique

**9.** Which generation strategy is most likely to surface supplier, cloud, telecommunications, and shared-infrastructure scenarios that neither the business nor the security team would think of first?

> A. Dependency-driven generation
>
> B. Top-down generation
>
> C. Localized generic generation
>
> D. Threat-driven generation

**10.** In CRG's screening, why does the 'control uncertainty' criterion give high scores to scenarios where nobody knows whether existing controls work?

> A. Because uncertain controls are presumed to have already failed
>
> B. Because such scenarios automatically receive mandatory escalation flags
>
> C. Because analysis can clarify exactly those uncertain scenarios
>
> D. Because uncertainty serves as a proxy for high threat probability

**11.** A candidate scenario could cause simultaneous failure of primary and recovery environments, but its probability appears low. What does CRG require?

> A. Exclude it, because its total screening score falls outside the top ten
>
> B. Merge it automatically into the highest-scoring ransomware family
>
> C. Select it only if its representative weakness has a critical CVSS score
>
> D. Keep it visible; excluding it needs explicit review and a management decision

**12.** In the MediBec example, candidates C01 (phishing of clinic administrators) and C20 (MFA push-fatigue bypass) both scored 52 out of 60. Why were they consolidated into one family?

> A. Because CRG requires consolidating any candidates with identical totals
>
> B. Because they describe the same exposure, share a treatment, and have one owner
>
> C. Because C20 carried fewer escalation flags than C01 in the workbook
>
> D. Because consolidation always keeps the candidate with the highest CVSS

**13.** What is the main purpose of scoring scenario families under equal, impact-led, evidence-led, and decision-led weighting schemes?

> A. To produce a final ranking that replaces the analyst's judgment
>
> B. To compute the CVSS multiplier that each family will need
>
> C. To add the scores of all variants within each family
>
> D. To flag selections resting on judgment and unselected candidates needing review

**14.** In the MediBec example, P10 (staff disclosure of patient information through unapproved AI tools) did not rank in the top ten under every scheme. Why was it selected?

> A. For coverage and decision usefulness, since no other scenario addressed negligent insiders or AI
>
> B. Because it had the highest total score in the entire candidate population
>
> C. Because its representative weakness had a critical CVSS score requiring selection
>
> D. Because generic scenario libraries list AI misuse as a mandatory scenario family

**15.** Why was C31, sabotage by a malicious or departing IT administrator, placed on the watch list instead of being selected?

> A. Its total score was too low for any further consideration
>
> B. It carried no escalation flags in the MediBec selection workbook
>
> C. Its treatments come through P01 and P02, a reliance marked for management decision
>
> D. CRG excludes insider scenarios from portfolios by default

**16.** Which three assessment methods from NIST SP 800-53A Rev. 5 should scenario documentation make possible?

> A. Plan, do, check, and act
>
> B. Examine, interview, and test
>
> C. Identify, protect, and detect
>
> D. Scan, patch, and verify

**17.** During scenario creation, the CIO states that backups are isolated, but the infrastructure team is unsure. How should the analyst handle this?

> A. Record the disagreement as a finding and add it to the evidence-collection plan
>
> B. Accept the CIO's view, since the CIO holds the most senior authority
>
> C. Average the opinions and record the backups as partially isolated
>
> D. Remove ransomware scenarios until both teams reach unanimous agreement

**18.** A team wants to paste real architecture diagrams and audit findings into a public AI chatbot to draft scenarios. What does the chapter advise?

> A. Proceed, because scenario drafting does not involve personal information
>
> B. Proceed, provided each AI-generated scenario carries an estimate marker
>
> C. Remove the organization's name, then use any available AI service
>
> D. Do not, unless authorized in writing and lawful; prefer a local or contracted service

**19.** Under the rules in CRG's education package, which of the following may an AI tool do?

> A. State organizational facts that the evidence register does not contain
>
> B. Produce final risk numbers without recomputation by a verified engine
>
> C. Propose parameter values with rationales for analysts to critique
>
> D. Supply control identifiers and legal obligations without verification

**20.** Which Chapter 6 parameter answers the question 'if the threat acts, will it succeed against this weakness given existing controls?'

> A. Pb(A), the probability that the threat is present
>
> B. Pb(ψ,A), the probability of exploitation
>
> C. δe, the expected damage if the scenario occurs
>
> D. μ(E), the organizational criticality of the process
