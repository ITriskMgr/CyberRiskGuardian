<!-- Léger, M.-A. (2026). Introduction to Cybersecurity Governance for Business Technology Management (3rd ed., v2.51). Chapter 1: Basic Concepts of Cybersecurity. CC BY-NC 4.0 -->

# Chapter 1: Basic Concepts of Cybersecurity

## Learning outcomes

After completing this chapter, you should be able to:

1.  Explain why cybersecurity is first a governance and risk issue, using the definition of security as the absence of unacceptable risk.

2.  Distinguish security, information security, and cybersecurity, and analyze how subjectivity and cognitive biases distort risk judgments.

3.  Apply the CIA triad and related objectives to express protection objectives in business terms, distinguishing privacy from confidentiality.

4.  Design a simple data classification scheme that links business value, information ownership, and CIA requirements to proportionate handling rules.

5.  Analyze how insider and external cyber risks emerge using the fraud triangle and the risk triangle.

6.  Compare the risk treatment options and justify how residual risk should be documented, accepted, and reviewed.

7.  Evaluate an organization's cybersecurity maturity and assurance arrangements using a 0-to-5 maturity scale, the five lines of accountability, and the distinction between audit and risk assessment.

8.  Analyze the governance lessons of the Ashley Madison and Target breaches.

Cybersecurity is often introduced as a technical domain made up of firewalls, encryption, anti-malware, and access controls. In practice, it is first and foremost a governance and risk topic, because security is not an absolute state that can be proven once and for all. Security exists when an organization operates under conditions in which the remaining risks are considered acceptable, given its mission, regulatory obligations, stakeholders’ expectations, and its capacity to absorb losses.

This chapter introduces cybersecurity from strategic and managerial perspectives. It is written for current and future managers who may not be IT practitioners but will inevitably be accountable for decisions that shape an organization’s exposure to cyber risk. In many organizations, cybersecurity leaders come from business, audit, risk management, operations, or governance backgrounds rather than purely technical roles. That is not a weakness. Many cybersecurity failures are not caused by a lack of technology but rather by unclear priorities, weak decision rights, insufficient accountability, underinvestment in the right areas, and inconsistent execution across business units.

| **The goal is managerial literacy:** enough conceptual competence to ask the right questions, interpret risk information, govern trade-offs responsibly, and maintain organizational resilience under uncertainty. |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## Clarifying the cybersecurity domain

Before delving into cybersecurity governance, it is important to define the basic cybersecurity concepts that managers need to master.

**Security** is best treated as a general condition. It refers to the protection of valued assets against unacceptable risk, whatever the domain. Framing the topic matters because it immediately shifts the discourse from tools to objectives, and from tasks to management decisions. Security is better defined as the absence of unacceptable risks.

**Information security focuses on protecting information, technology, and systems that store**, process, and transmit information. The National Institute of Standards and Technology (NIST), in its glossary, captures this compactly by defining information security as “the protection of information and systems from unauthorized access, use, disclosure, disruption, modification, or destruction in order to provide confidentiality, integrity, and availability.” Information security is often used interchangeably with cybersecurity.

**Cybersecurity** is information security applied to cyberspace, the connected world in which organizations operate, including networks, the Internet, cloud platforms, third-party integrations, remote work environments, and increasingly cyber-physical systems. With e-business, remote work, and collaborative software becoming increasingly important, most business processes depend on interconnected systems. This has made cybersecurity central to organizational functioning. Once a niche technical concern, it has become a strategic one. A practical working definition of cybersecurity for managers is therefore the following:

| **Cybersecurity** is the set of organizational governance **approaches, actions, policies, safeguards, technologies, training, and risk management processes** used to protect relevant information and business technologies in cyberspace, thereby **maintaining an acceptable risk state for the organization and its stakeholders.** |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

This definition deliberately emphasizes governance. Cybersecurity is not merely technology; it is the management of risk in a highly interconnected environment.

## Security is the absence of unacceptable risk

As stated, a widely used definition describes security as the absence of unacceptable risk. This definition is helpful, but it hides two practical difficulties that matter for cybersecurity governance.

- First, **absence is subjective**. The presence of risk is not always observable. Individuals perceive risk in different ways. A risk can exist even when it is not seen, not measured, or not understood. An organization may believe it is secure because it has not yet experienced incidents, but a lack of incidents can reflect luck, levels below measurable thresholds, limited detection capabilities, willful blindness, or undetected adversaries. Therefore, a perceived absence doesn’t mean that this absence is real.

- Second, **unacceptability is inherently subjective**. Different individuals can evaluate the same risk differently using different criteria that may be inconsistent or poorly defined. Similarly, organizations in the same industry can reach different conclusions. This is because their risk appetites, regulatory constraints, and business dependencies differ. The same activity can be acceptable in one context and unacceptable in another as conditions change. In organizations, this subjectivity is amplified by organizational culture, history, geography, industry norms, stakeholder expectations, and strategy.

Governance exists precisely to help organizations frame their reality and convert subjective preferences into explicit decisions, policies, and accountability structures that can be applied consistently.

Cognitive biases further complicate these two difficulties because risk judgments are made by humans under uncertainty, time pressure, and incomplete information. Individuals may overweight recent or vivid events (availability bias), underestimate familiar exposures because nothing has happened in the past (normalization of deviance), or seek confirming evidence for an initial view rather than disconfirming signals (confirmation bias). Optimism bias can lead teams to assume that strong controls are in place, while authority bias and groupthink can suppress technical dissent when senior stakeholders prioritize speed or cost savings. These distortions affect what risks are noticed, how severity is perceived, and what is deemed acceptable, reinforcing the need for governance mechanisms, structured assessment, independent challenge, evidence requirements, and documented risk acceptance to counterbalance human judgment errors and maintain consistency over time.

## The CIA triad: defining security objectives in business terms

A manager cannot govern cybersecurity without understanding what the organization is trying to protect and why it matters. The starting point is the CIA triad, a triangle with the three faces being **confidentiality, integrity, and availability**, as shown in Figure 1. These three faces represent the security properties that can help organizations specify the primary security objectives they seek to maintain for their informational assets (data, technology, business processes). They also provide a shared language for expressing potential consequences of undesired outcomes and establishing priorities.

<span id="_Toc243635500" class="anchor"></span>Figure 1: the CIA triad

- **Confidentiality** is concerned with the prevention of unauthorized disclosure of information. When confidentiality is compromised, sensitive data is divulged. Then divulged information becomes accessible to individuals or entities who should not have this access or is used in an unauthorized context. Confidentiality breaches can cause direct harm through fraud, extortion, identity theft, and loss of trade secrets, and they can create legal and regulatory exposure when personal information is involved. Even when confidentiality is not legally mandated, it can still be strategically critical because competitive advantage often depends on information others do not possess.

- **Integrity** is about protecting information and systems from unauthorized or accidental modification or destruction, and ensuring information remains accurate, complete, and trustworthy. Integrity failures, or alteration of information, can be subtle and, in some ways, more dangerous than confidentiality breaches because they can remain undetected while decisions continue to be made using corrupted data. If information cannot be trusted, then planning, reporting, financial controls, customer service, supply chain operations, and decision-making degrade, sometimes without immediate awareness of the root cause.

- **Availability** means ensuring that authorized users can access the information and systems required to accomplish a business process when needed. Availability failures include downtime, service degradation, and inaccessibility caused by technical failures, sabotage, denial-of-service attacks, or ransomware encryption. Availability is often the most visible security objective because its failure immediately disrupts operations. In many sectors, such as healthcare, finance, manufacturing, transportation, and education, availability is also tied to safety, service continuity, and public trust.

CIA is not the only set of objectives. There can be many others depending on the situation. A few frequently mentioned security objectives are listed here:

- **Authenticity**, as there is a need to ensure that information is genuinely from the claimed source.

- **Non-repudiation**, to ensure that an actor can’t credibly deny taking an action.

- **Accountability**, allowing actions to be traced to the responsible parties.

- **Privacy, which goes beyond confidentiality to ensure that the collection, use, and disclosure of personal information are lawful and appropriate.**

Nevertheless, CIA is a durable baseline commonly found in best practices, literature and in practice. It is particularly useful to cybersecurity actors because it translates cybersecurity into clear managerial outcomes: keep information appropriately secret, keep it correct, and keep it accessible when needed.

## Privacy and confidentiality: related but not identical

Privacy is typically framed as a legal and ethical requirement grounded in individual rights. Confidentiality is a cybersecurity objective focused on restricting access to authorized users. In practice, organizations operationalize privacy obligations largely through confidentiality mechanisms such as access control, encryption, logging, minimization, and secure handling. Yet privacy requires more than secrecy. A system can keep personal data confidential while still violating privacy through excessive collection, unclear purposes, inappropriate retention, or unfair processing. The governance implication is that privacy must be designed into information practices, not treated as a mere access-control problem.

NIST, notably in its Privacy Framework, explicitly stresses that security and privacy are distinct but overlapping disciplines, and that protecting individuals’ privacy cannot be achieved solely by securing personally identifiable information.

## Data classification: translating business value into protection requirements

Because information security aims to reduce risk to an acceptable level, organizations must determine how much protection different types of information require. Not all information has the same value, sensitivity, or operational importance. Data classification provides a structured method for translating business value, legal obligations, and potential harm into proportionate protection requirements.

Classification begins with an accurate inventory of the information the organization creates, receives, processes, stores, transmits, and disposes of. The process should start with business activities rather than technology. Managers should identify the processes that support organizational objectives, the information those processes require, the systems and third parties that handle it, and the consequences if the information is disclosed, altered, lost, or unavailable. Information becomes an organizational asset when it supports decision-making, operations, compliance, customer service, revenue, institutional knowledge, or competitive advantage.

<span id="_Toc243635501" class="anchor"></span>Figure 2: data classification

Each significant information asset should have a business owner. This is not necessarily the person who administers the system. The information owner is the manager who understands the business purpose of the information and has authority to approve access, determine its classification and retention requirements, and accept or escalate associated risks. Technology teams and custodians implement the safeguards, but the business remains accountable for determining the required level of protection.

Classification should reflect confidentiality, integrity, and availability requirements. For confidentiality, the organization should consider whether the information contains personal data, financial records, credentials, intellectual property, strategic plans, legal advice, or other sensitive content. It should assess who could be harmed by disclosure and whether the consequences could be legal, contractual, competitive, financial, or reputational.

For integrity, the organization should determine how strongly business decisions and activities depend on the accuracy, completeness, authenticity, and timeliness of the information. Data used in financial reporting, regulatory submissions, automated decisions, payroll, procurement, access control, or safety-related operations may require strong protection against unauthorized modification. Altered or inaccurate information could enable fraud, mislead management, disrupt operations, or produce unsafe outcomes.

For availability, the organization must determine how long operations can continue without the information. This includes considering recovery time and recovery point objectives, backup requirements, system dependencies, alternative processes, and the consequences of prolonged unavailability. Information supporting essential or time-sensitive services may require high availability even when it is not highly confidential.

A practical classification model normally uses a small number of levels, such as Public, Internal, Confidential, and Restricted. Each level should be linked to minimum requirements for access, storage, transmission, encryption, sharing, retention, backup, and secure disposal. Labels must be understandable and consistently applied across documents, databases, cloud services, collaboration platforms, and physical records. Overly complex schemes often fail because users cannot distinguish among categories or apply the associated controls correctly.

Classification must also consider aggregation and context. Individual data elements may appear harmless but become sensitive when combined. For example, a name, employee number, location, and work schedule may collectively pose privacy, fraud, or physical security risks.

Governance ensures that classification is not treated as an isolated IT activity. The framework should be approved by management, aligned with strategy and risk appetite, and supported by legal, privacy, records-management, compliance, and cybersecurity functions. Responsibilities for assigning, reviewing, changing, and enforcing classifications should be clearly defined, and exceptions should be formally authorized.

Classification is not permanent. Information may become more sensitive when combined with other data, less sensitive after public release, obsolete after its retention period, or more critical when business dependence increases. Classifications should therefore be reviewed periodically and following significant changes such as new legal requirements, system migrations, acquisitions, incidents, or new uses involving artificial intelligence and analytics.

An effective classification program connects business judgment to security action. It helps the organization protect its most important information, establish defensible handling requirements, improve compliance, and avoid both inadequate protection and unnecessary control costs.

## The fraud triangle: explaining why insiders misbehave

Cybersecurity outcomes are deeply influenced by human behaviour. Individual actors are the most difficult variable to control. People can be victims (through manipulation), unintentional contributors (through error), or threat actors (through malicious insider behaviour). In governance, it is useful to understand not only technical vulnerabilities but also human vulnerabilities, often referred to as the human factor. Because attention tends to focus on external threat actors, such as so-called hackers and cybercriminals, the fraud triangle is a useful reminder that insider threats must also be taken seriously. Decision-makers need to understand why wrongdoing often emerges within organizations.

<span id="_Toc243635502" class="anchor"></span>Figure 3: the fraud triangle

The fraud triangle is a long-standing model, derived from criminologist Donald R. Cressey’s research on embezzlers in the 1950s and widely used in fraud examination, to explain why otherwise normal individuals sometimes commit fraud. It proposes that three elements often converge when fraud occurs: opportunity, pressure (or need), and rationalization.

- **Opportunity** refers to the ability to commit wrongdoing due to access, weak controls, insufficient oversight, poor process design, or inadequate segregation of duties. In cybersecurity terms, opportunity arises from excessive privileges, shared accounts, lack of monitoring, weak identity controls, unmanaged third-party access, and informal practices that bypass official processes. Opportunity is not merely technical; it is a governance design problem.

- **Pressure** (or need) refers to a motivating driver. It can be financial distress, addiction, coercion, fear of job loss, resentment, or perceived injustice. Pressure does not justify wrongdoing, but it helps explain why an individual might exploit an existing opportunity.

- **Rationalization** is the intellectual process by which an individual transforms pressure into action acceptable to them. It is how they convince themselves that it’s OK, the internal story that makes wrongdoing feel acceptable:

  - I deserve this,

  - They owe me,

  - It’s only temporary,

  - No one will be harmed, or

  - Everyone does it.

A core governance implication is that managers cannot govern cybersecurity by trusting people alone. Insiders are most aware of possible opportunities, such as inadequate technical mitigation measures, absent, inadequate, or dysfunctional internal controls, or even organizational complacency. Organizations must reduce opportunities for abuse and establish oversight that discourages wrongdoing, detects it early, and creates evidence and accountability. This can be achieved in many ways, such as through logging, monitoring, audit trails, clear policies and a well-structured awareness program.

## The risk triangle: how cyber risk emerges 

Risk is the central concept in cybersecurity governance because it links human and technical realities to business consequences and provides a rational basis for allocating resources. A useful way to conceptualize cybersecurity risk is the risk triangle, which connects threats, vulnerabilities, and exposure.

- **Threats** include threat actors and threat events: cybercriminal groups, malicious insiders, hacktivists, and state-sponsored actors, as well as non-malicious hazards such as human error, technology failures, and process breakdowns. Threats represent potential causes of harm. The fraud triangle can help understand the motivations of many threat actors.

- **Vulnerabilities** are weaknesses that can be exploited or triggered. They may be technical (software flaws, misconfigurations, insecure interfaces), procedural (weak approvals, poor change management, missing backups), or human (lack of training, susceptibility to manipulation, excessive trust). Vulnerabilities create the conditions under which threats succeed. In the fraud triangle, weak controls and excessive access create opportunity, which is itself a vulnerability.

- **Exposure** refers to the magnitude of harm if a threat successfully exploits a vulnerability. When a potential impact of damage becomes a reality, the risk has moved from an exposure to a materialized risk. Exposure is tied to what the organization values: data, service continuity, reputation, financial stability, legal compliance, and strategic position. Exposure maps naturally to the CIA triad: confidentiality breaches create privacy and competitive harms; integrity failures degrade reporting and decision-making; availability failures disrupt operations.

<span id="_Toc243635503" class="anchor"></span>Figure 4: the risk triangle4

Risk materializes when a plausible threat event interacts with a vulnerability and produces an adverse outcome. One way this interaction can occur is through a threat actor **exploiting** a vulnerability, resulting in a negative outcome for the organization, such as financial loss. Many cybersecurity frameworks formalize this as a function of likelihood and impact. Managers thus have two broad levers to mitigate risk. They can reduce likelihood, for example by reducing the number of vulnerabilities or by increasing deterrence and detection. They can also reduce impact by strengthening resilience, for example through regular and tested backups, information system redundancy, incident response readiness, and crisis communications capability. Because no single control is reliable on its own, mature organizations combine these levers in layers, an approach known as defence in depth: preventive, detective, and recovery controls are arranged so that the failure of one does not leave a critical asset unprotected.

## Risk treatment: accept, avoid, transfer, mitigate, and manage residual risk

Once risks are identified and assessed, governance requires intentional decision-making regarding treatment. The classic options are to accept the risk (when within appetite and justified), avoid the risk (change the activity so the risk no longer exists), transfer or share the risk (insurance, contractual allocation, outsourcing), or mitigate the risk (reduce likelihood and/or impact through controls).

Risk treatment does not eliminate risk. Residual risk remains even after controls are implemented. Mature governance, therefore, treats residual risk as an explicit subject of decision-making: it is documented, approved by accountable risk owners, and reviewed periodically as conditions change.

This logic is operationalized in modern standards. For example, ISO/IEC 27005:2022 provides guidance, aligned with the requirements of ISO/IEC 27001, on selecting treatment options, determining necessary controls, producing a Statement of Applicability, and developing a risk treatment plan that risk owners approve, together with their acceptance of the residual risks.

<span id="_Toc243635504" class="anchor"></span>Figure 5: risk treatment

## The human factor: usable policies, continuous awareness, and the zero-trust mindset

In practice, people are involved in most cybersecurity outcomes. Humans can be manipulated through social engineering; they can make mistakes that expose sensitive information; they can reuse passwords; they can bypass policies to get work done; and, in some cases, they act maliciously. Cybersecurity governance must therefore treat human behaviour as a core risk domain.

Effective governance approaches to the human factor tend to share several characteristics.

- First, **policies must be usable**. A long policy document filled with legal wording rarely changes behaviour; it often encourages avoidance. Policies that shape behaviour are clear, short, role-specific, and reinforced through leadership example, reminders, and training.

- Second, **senior leadership must visibly own cybersecurity**. When executives treat cybersecurity as optional, the culture follows. Empowering a cybersecurity leader—commonly a CISO—with credible authority, independence, and access to decision-makers is therefore not a cosmetic choice; it is a structural control.

- Third, **awareness must be continuous**. Training is not a single annual ritual. Organizations benefit from ongoing, low-friction education, targeted campaigns, and onboarding training that reflects realistic threats. Governance should also prioritize testing reality rather than assuming compliance: phishing simulations, tabletop exercises, and control testing reveal gaps early, when they are cheaper to fix than after an incident.

- Fourth, **leaders should internalize a zero-trust mindset:** trust is not a control. Access and privileges should be managed as if compromise is possible, because it is. In practical terms, this is expressed through least privilege (only the access required for legitimate role tasks) and segregation of duties (no single person can execute end-to-end high-risk actions without oversight). These principles reduce both external attack blast radius and insider opportunity, directly connecting back to the fraud triangle.

- Finally, **culture matters**. Culture is the set of unwritten rules that shape behaviour under pressure. A culture that rewards speed at any cost will produce insecure workarounds. A culture that punishes reporting will hide incidents. Governance must shape incentives and accountability so that secure behaviour aligns with what the organization rewards.

## Cybersecurity maturity: from ad hoc practices to embedded capability

Cybersecurity maturity refers to the degree to which an organization's cybersecurity practices are structured, consistent, measured, and continuously improved. A mature organization does not merely claim to be secure or rely on the absence of past incidents as proof of safety; rather, it demonstrates a deep understanding of its risks, implements coherent controls, monitors performance, learns from incidents and near-misses, and continuously evolves as threats and technologies change.

### The multi-dimensional nature of maturity 

Maturity is fundamentally multi-dimensional. It extends far beyond technical tools to include governance structures, policy discipline, training effectiveness, incident response capabilities, risk management routines, and the ability to coordinate securely across business units and third-party suppliers. Organizations with higher maturity levels shift their mindset from a passive state of "we have security tools" to an evidence-based state of "we can show our security works" through continuous operational validation, testing, and auditing. This discipline strongly influences cyber resilience: mature organizations are equipped to detect incidents sooner, contain them more effectively, and recover their critical business functions more reliably under stress.

Maturity models make this progression operational by describing distinct levels of capability. Different frameworks use different scales. NIST Cybersecurity Framework 2.0, for example, describes four Implementation Tiers (Partial, Risk Informed, Repeatable, and Adaptive) that characterize the rigour of an organization’s cybersecurity risk governance and management practices; NIST does not present them as a maturity model in the strict sense. A widely used progression in information systems governance is the six-level scale (0 to 5) derived from the Capability Maturity Model (CMM) and popularized by the generic maturity model of COBIT 4.1 (ISACA, 2007), from which the level descriptions below are adapted for cybersecurity:

- **Level 0 (Non-existent):** The organization lacks basic security processes, does not perform risk assessments, and has no assigned accountability for protecting information assets.

- **Level 1 (Initial / Ad Hoc):** Cybersecurity practices are unplanned, informal, and highly reactive. Responses to threats are managed on a case-by-case basis, often relying on individual heroism rather than institutional strategy.

- **Level 2 (Repeatable but Intuitive):** A pattern of security activities begins to emerge, and staff tend to handle the same tasks in similar ways, but nothing is formally documented, standardized, or taught. Practice depends on what particular individuals happen to know, so results vary and mistakes are common.

- **Level 3 (Defined Process):** Security procedures are standardized, documented, and actively communicated through training programs. Risk assessment follows a defined methodology, and responsibilities are formally assigned, allowing the organization to proactively manage its security posture.

- **Level 4 (Managed and Measurable):** Management actively monitors and measures compliance with security procedures using quantitative metrics and key performance indicators. Security risks are clearly aligned with enterprise risk management, and policy deviations are detected and corrected efficiently.

- **Level 5 (Optimized):** Processes are continuously refined and improved based on measurable feedback, audits, and external benchmarking. The organization is agile, using automated tools and threat intelligence to dynamically adapt its security posture in response to emerging threats and structural changes.

### Strategic value and investment governance

The purpose of a maturity model is not to grade an organization for its own sake, but to provide a clear roadmap for identifying vulnerabilities and a governance mechanism for prioritizing investments.

Used well, a maturity assessment helps leadership set a target level for each capability that is consistent with the organization’s risk appetite and dependencies, rather than aiming for the highest level everywhere. A small organization with modest exposure may reasonably target a defined, repeatable level for most processes and reserve higher targets for the few capabilities that protect its most critical services. The gap between current and target maturity then becomes a prioritized, costed roadmap that can be defended to executives and the board. Maturity scores should nevertheless be interpreted with care: they measure the discipline of processes, not risk itself, and an organization can score well on paper while remaining exposed to the scenarios that matter most. Chapter 9 returns to control frameworks, and Chapter 17 to the investment decisions that maturity roadmaps inform.

<span id="_Toc243635505" class="anchor"></span>Figure 6: cybersecurity maturity model

## Governance assurance: the Three Lines Model, audit, and continuous improvement

A useful governance framing distinguishes lines of responsibility for risk. The Institute of Internal Auditors (IIA) originally described three lines of defence and, in 2020, updated them as the Three Lines Model, which places these roles beneath a governing body and within management’s overall responsibility (see Chapter 21). For teaching purposes, this book extends the model into five lines of accountability, naming executive management and the board as distinct lines so that their oversight roles are explicit:

- **First Line (Operations and Management):** Business owners, IT, and cybersecurity operations professionals (including the CISO where the CISO runs security operations; a CISO who mainly sets policy and challenges IT acts in a second-line role, as Chapter 21 explains) own the risks and are responsible for deploying, executing, and maintaining security controls in daily work.

- **Second Line (Risk Management and Compliance):** These functions provide tactical oversight by defining frameworks, formulating security policies, monitoring the first line's controls, and ensuring that security practices align with the organization's risk appetite.

- **Third Line (Internal Audit):** Operating independently from management, internal audit evaluates the effectiveness of the first two lines and provides objective assurance directly to the board of directors. Due to the highly technical nature of cybersecurity, this line often co-sources or leverages external assurance providers to benchmark practices against industry standards.

- **Fourth Line (Executive Management):** Executives are responsible for managing the organization overall, allocating the resources necessary for cyber risk management, and ensuring that cybersecurity aligns with broader Enterprise Risk Management (ERM).

- **Fifth Line (Board of Directors):** The board provides ultimate strategic oversight, endorses the organization's risk appetite, participates in refining key strategic objectives, and ensures that executive management operates within acceptable risk tolerances.

Audit and risk assessment are complementary but not interchangeable. Audit is largely retrospective and evidence-based: it examines what exists, verifies compliance, and assesses how controls performed against established policies. Risk assessment is forward-looking: it estimates plausible threat scenarios, evaluates future exposure, and identifies where an organization's digital footprint may be vulnerable. Mature governance tightly integrates them so that audits objectively validate control execution, while risk assessments guide which controls should exist and where security investments are most justified.

This continuous discipline is well reflected in international standards such as ISO/IEC 27001:2022 (information security management) and ISO 37301:2021 (compliance management systems). Both follow ISO’s harmonized structure for management system standards, whose clauses are commonly mapped to the Plan-Do-Check-Act (PDCA) cycle, moving away from treating risk assessments and audits as static, "annual rituals". Instead, they mandate a continuous feedback loop encompassing governance context, leadership commitment, planning, support, operations, performance evaluation (including internal audits and management reviews), and continual improvement. By integrating these mechanisms, an organization ensures that its cybersecurity posture remains defensible, resilient, and adaptive as threats and business conditions change.

<span id="_Toc243635506" class="anchor"></span>Figure 7: governance assurance

## Case study integration: why governance failures amplify cyber incidents

Two well-known incidents illustrate why governance matters as much as technology.

The 2015 breach of Ashley Madison, a service operated by Toronto-based Avid Life Media, is commonly treated as a privacy and governance failure. A joint 2016 investigation by the Office of the Privacy Commissioner of Canada and the Office of the Australian Information Commissioner found that the company lacked an adequate overall information security framework, including documented security policies and a documented risk management process, and that it retained the personal information of users with deactivated or inactive accounts indefinitely. The sensitivity of the service meant that the risk of predictable harm from disclosure was exceptionally high. Governance lessons in such cases are not limited to the use of stronger tools. They include minimizing the collection of sensitive data, enforcing disciplined retention and deletion, ensuring transparency about data practices, and implementing controls proportionate to foreseeable harm. When an organization’s business model requires collecting highly sensitive data, it inherits a duty to protect that data through robust management practices and credible assurances.

The 2013 Target breach is widely discussed as a third-party risk and segmentation failure pattern. Attackers used network credentials stolen from a heating, ventilation, and air-conditioning (HVAC) vendor, moved laterally within internal networks, and ultimately impacted payment environments. Governance lessons include treating supplier relationships as first-class risks with contractual and technical controls; constraining vendor access; enforcing segmentation so that compromise does not propagate; and ensuring monitoring is operationally connected to decisive response rather than treated as a passive tool.

## Concluding integration: risk appetite as the managerial core

Cybersecurity governance is ultimately a question of strategic balance. Organizations cannot eliminate all vulnerabilities, nor should they try, as resources are finite and excessive controls impose operational friction that can stifle innovation. Instead, mature governance aims to achieve a defensible posture in which the organization does "enough" to keep exposure aligned with its documented **risk appetite** and **tolerance**. In this light, unmanaged cyber risk is simply an implicit management decision; effective governance exists to ensure those decisions become conscious, explicit, and justifiable.

This is the central takeaway: **cybersecurity is a continuously managed state of acceptable risk, not a permanent technical accomplishment**. The concepts introduced thus far form the ecosystem of this managerial discipline:

- The **CIA triad** (Confidentiality, Integrity, Availability) provides the fundamental business language for protection objectives.

- **Data classification** translates the language of the CIA triad into business-specific requirements and protection levels, helping to identify key informational assets.

- The **fraud triangle** clarifies the motivations behind insider threats and highlights the need for structural oversight.

- The **risk triangle** structures scenario-based reasoning by connecting threats, vulnerabilities, and business impacts.

- **Attack lifecycle models** (such as the Cyber Kill Chain or MITRE ATT&CK) reveal multiple points for intervention, detection, and disruption; they are examined in Chapter 4.

- The **human factor** and organizational culture emphasize why usable policies, continuous awareness, and a "human firewall" are critical to minimizing errors and security bypasses.

- **Defence-in-depth** and **maturity models** illustrate why sustainable cybersecurity requires layered capabilities and continuous operational validation rather than ad hoc, reactive measures.

- **Assurance mechanisms**—the IIA’s Three Lines Model, extended in this book into five lines of accountability (operations, risk and compliance, internal audit, executive management, and the board of directors)—establish structured oversight, ensuring that formal authority translates into real, verified protection.

The chapters that follow—covering standards, compliance frameworks, policy design, risk assessment methodologies, security architecture, and control selection—should not be viewed as mere technical checklists. Rather, they are vital **governance instruments**. Together, they form the continuous feedback loops necessary to maintain an acceptable risk level over time, navigate cognitive biases in human decision-making, and ensure resilience under uncertainty in a highly connected digital world.

## Chapter summary

Chapter 1 argues that cybersecurity is first a governance and risk topic, not a collection of technologies. Security is never proven once and for all; it exists when the risks that remain are acceptable given the organization's mission, obligations, stakeholders, and capacity to absorb losses. Managers therefore need literacy rather than technical depth: the ability to ask the right questions, interpret risk information, and govern trade-offs. The chapter builds that literacy through a set of connected models, from the CIA triad and data classification to the fraud and risk triangles, risk treatment, the human factor, maturity, and assurance, and concludes that unmanaged cyber risk is simply an implicit management decision that governance must make explicit.

### Key points to retain

- Security is best defined as the absence of unacceptable risk, and cybersecurity is information security applied to the interconnected world of cyberspace.

- Both absence and unacceptability are subjective, and cognitive biases such as availability, normalization of deviance, and confirmation bias distort risk judgments, which governance must counterbalance.

- The CIA triad expresses protection objectives in business terms: keep information appropriately secret, correct, and accessible when needed.

- Privacy overlaps with confidentiality but requires more: a system can keep personal data secret while still collecting, using, or retaining it inappropriately.

- Data classification translates business value, legal obligations, and potential harm into proportionate protection requirements, with each significant asset assigned a business owner.

- The fraud triangle explains insider wrongdoing as the convergence of opportunity, pressure, and rationalization, and opportunity is a governance design problem.

- The risk triangle shows that risk arises when threats exploit vulnerabilities and create exposure, and managers can reduce either likelihood or impact, layering controls through defence in depth.

- Risk can be accepted, avoided, transferred or shared, or mitigated, but residual risk always remains and must be documented, approved by risk owners, and reviewed.

- Human behaviour is a core risk domain that requires usable policies, visible leadership, continuous awareness, a zero-trust mindset, and a culture that rewards secure behaviour.

- Maturity models describe a progression from non-existent to optimized practices; targets should follow risk appetite, and maturity scores measure process discipline, not risk itself.

- The Three Lines Model, extended in this book to five lines of accountability, and the integration of retrospective audit with forward-looking risk assessment make governance verifiable.

- The Ashley Madison and Target breaches show that governance failures, such as missing policies, indefinite retention, and unconstrained vendor access, amplify cyber incidents.

### Key terms

- **Security** — The protection of valued assets against unacceptable risk, best understood as the absence of unacceptable risk.

- **Cybersecurity** — The set of governance approaches, actions, policies, safeguards, technologies, training, and risk management processes used to protect information and business technologies in cyberspace and maintain an acceptable risk state.

- **CIA triad** — The three core security objectives of confidentiality, integrity, and availability used to express what must be protected and why.

- **Confidentiality** — The prevention of unauthorized disclosure of information.

- **Integrity** — The protection of information and systems from unauthorized or accidental modification or destruction, so that information remains accurate, complete, and trustworthy.

- **Availability** — The assurance that authorized users can access the information and systems a business process requires when needed.

- **Non-repudiation** — A security objective ensuring that an actor cannot credibly deny having taken an action.

- **Data classification** — A structured method for translating business value, legal obligations, and potential harm into proportionate protection requirements for information.

- **Information owner** — The manager who understands the business purpose of information and has authority to approve access, set its classification and retention, and accept or escalate its risks.

- **Fraud triangle** — A model, derived from Donald R. Cressey's research, explaining wrongdoing as the convergence of opportunity, pressure (or need), and rationalization.

- **Risk triangle** — A model connecting threats, vulnerabilities, and exposure to explain how cyber risk emerges.

- **Exposure** — The magnitude of harm that would result if a threat successfully exploited a vulnerability.

- **Residual risk** — The risk that remains after treatment, which must be documented, approved by accountable risk owners, and reviewed periodically.

- **Defence in depth** — Layering preventive, detective, and recovery controls so that the failure of one does not leave a critical asset unprotected.

- **Cybersecurity maturity** — The degree to which an organization's cybersecurity practices are structured, consistent, measured, and continuously improved.

## Review questions

### Essay questions

1.  The chapter defines security as the absence of unacceptable risk but warns that both absence and unacceptability are subjective. Analyze these two difficulties, explain how cognitive biases amplify them, and evaluate which governance mechanisms best counterbalance them.

2.  Choose an organization you know, such as an employer, a university, or a local business. Identify three significant information assets and design a four-level classification approach for them, justifying each asset's confidentiality, integrity, and availability requirements, its owner, any aggregation concerns, and the events that should trigger a review.

3.  Using both the fraud triangle and the risk triangle, analyze how an insider in an accounts payable or payroll function could cause a loss. Recommend controls that reduce likelihood and controls that reduce impact, and explain how they combine as defence in depth.

4.  A board member argues that the organization should aim for maturity level 5 in every cybersecurity capability. Evaluate this position using the chapter's discussion of maturity models, risk appetite, investment governance, and the limits of maturity scores.

5.  Apply the chapter's governance lessons to the Ashley Madison and Target breaches. For each case, identify the governance failures, explain where in the five lines of accountability they should have been detected or challenged, and recommend the decisions managers should have made.

### Multiple-choice questions

Choose the best answer. Answers are listed in the appendix "Answers to multiple-choice questions".

**1.** Which definition of security does Chapter 1 adopt as the most useful for managers?

> A. The complete elimination of all known technical vulnerabilities
>
> B. The absence of unacceptable risk
>
> C. Full conformity with every applicable security standard
>
> D. A period of at least one year without reported incidents

**2.** Why does the chapter suggest that integrity failures can be more dangerous than confidentiality breaches?

> A. They always become public and damage reputation immediately
>
> B. They can only be caused by sophisticated state-sponsored actors
>
> C. They mainly affect backup systems rather than live operations
>
> D. They can remain undetected while decisions continue to rely on corrupted data

**3.** A customer platform encrypts personal data and restricts access tightly, but collects far more data than its stated purpose requires. What does this illustrate?

> A. Privacy can be violated even when confidentiality is maintained
>
> B. Confidentiality failures are the main source of privacy harm
>
> C. Encryption fully satisfies the organization's privacy obligations
>
> D. Data minimization is a purely technical access-control matter

**4.** According to the chapter, who should normally determine the classification of a significant information asset?

> A. The system administrator who maintains the hosting platform
>
> B. The CISO, because classification is a security function
>
> C. The business information owner who understands its purpose
>
> D. The external auditor who reviews the classification scheme

**5.** An employee list containing names, employee numbers, work locations, and schedules is labelled Internal because each field seems harmless alone. Which classification principle has been overlooked?

> A. Labels must be consistent across cloud and physical records
>
> B. Classification should begin with technology, not business activities
>
> C. Combining data elements can make information more sensitive
>
> D. Classification schemes should use more than four levels

**6.** Which three elements make up the fraud triangle?

> A. Opportunity, pressure (or need), and rationalization
>
> B. Threat, vulnerability, and exposure to harm
>
> C. Confidentiality, integrity, and availability of data
>
> D. Motive, means, and detection of the wrongdoing

**7.** Which element of the fraud triangle can governance influence most directly through design choices?

> A. Pressure, by monitoring employees' personal finances closely
>
> B. Rationalization, by requiring annual ethics declarations
>
> C. Pressure and rationalization, through stricter discipline alone
>
> D. Opportunity, through least privilege, segregation of duties, and monitoring

**8.** An employee who copies client files before leaving says, “They owe me after all my unpaid overtime.” Which fraud triangle element does this statement represent?

> A. Opportunity created by weak controls
>
> B. Rationalization of the wrongdoing
>
> C. Pressure from financial distress
>
> D. Exposure of the organization

**9.** In the risk triangle, how should missing backups and weak change approvals be classified?

> A. As procedural vulnerabilities
>
> B. As threat events
>
> C. As exposure
>
> D. As residual risks

**10.** A firm invests in regularly tested backups, system redundancy, and crisis communications capability. Which risk lever is it mainly using?

> A. Reducing likelihood by increasing deterrence
>
> B. Transferring risk to an external party
>
> C. Reducing impact by strengthening resilience
>
> D. Avoiding risk by stopping the activity

**11.** What does the chapter mean by defence in depth?

> A. Concentrating the budget on the single strongest preventive control
>
> B. Layering preventive, detective, and recovery controls so one failure is not fatal
>
> C. Requiring every user to pass several authentication steps for each system
>
> D. Placing the most sensitive systems physically deep within the data centre

**12.** A company stops offering a service in a market where it cannot meet data protection expectations, so the risk no longer exists. Which treatment option is this?

> A. Accepting the risk within appetite
>
> B. Transferring the risk through insurance
>
> C. Mitigating the risk with new controls
>
> D. Avoiding the risk by changing the activity

**13.** How does mature governance handle residual risk after controls are implemented?

> A. It assumes residual risk is negligible once controls are tested
>
> B. It delegates residual risk decisions to the security operations team
>
> C. It records residual risk only when an incident has occurred
>
> D. It documents it, has risk owners approve it, and reviews it periodically

**14.** In the Three Lines Model as presented in the chapter, what is the distinctive role of internal audit?

> A. Deploying and maintaining security controls in daily operations
>
> B. Providing independent, objective assurance directly to the board
>
> C. Formulating security policies and monitoring first-line controls
>
> D. Allocating the resources needed for cyber risk management

**15.** A team argues that a long-standing exposure is acceptable because nothing has gone wrong in years. Which bias does the chapter associate with this reasoning?

> A. Availability bias, overweighting vivid recent events
>
> B. Authority bias, deferring to senior stakeholders
>
> C. Normalization of deviance, underestimating familiar exposures
>
> D. Confirmation bias, seeking evidence for an initial view

**16.** Staff perform security tasks in similar ways, but nothing is documented, standardized, or taught, so results depend on individuals. Which maturity level is this?

> A. Level 2, repeatable but intuitive
>
> B. Level 1, initial or ad hoc
>
> C. Level 3, defined process
>
> D. Level 4, managed and measurable

**17.** A small organization with modest exposure is setting maturity targets. Which approach is most consistent with the chapter?

> A. Target level 5 everywhere to demonstrate leadership commitment
>
> B. Set no targets until a serious incident reveals the main gaps
>
> C. Target defined, repeatable levels broadly and higher levels for critical services
>
> D. Copy the targets of the largest firm in its industry

**18.** How does the chapter describe the zero-trust mindset that leaders should internalize?

> A. Trusted employees should receive broader access to reduce friction
>
> B. Trust is earned once at hiring through thorough background checks
>
> C. Zero trust applies only to external partners and remote users
>
> D. Trust is not a control, so access is managed as if compromise is possible

**19.** Which statement best captures the chapter's distinction between audit and risk assessment?

> A. Audit is retrospective and evidence-based; risk assessment is forward-looking
>
> B. Audit estimates future threats; risk assessment verifies past compliance
>
> C. Audit and risk assessment are interchangeable terms for the same activity
>
> D. Risk assessment replaces audit once an organization reaches maturity level 4

**20.** Which governance lesson does the chapter draw from the 2013 Target breach?

> A. Retain customer data only for inactive accounts to reduce exposure
>
> B. Treat suppliers as first-class risks, constrain their access, and enforce segmentation
>
> C. Rely on monitoring tools as passive records for later investigation
>
> D. Prioritize confidentiality controls over all third-party contracts
