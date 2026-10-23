<!-- Léger, M.-A. (2026). Introduction to Cybersecurity Governance for Business Technology Management (3rd ed., v2.51). Chapter 12: Incident Management, Business Continuity, and Resilience. CC BY-NC 4.0 -->

# Chapter 12: Incident Management, Business Continuity, and Resilience

## Learning outcomes

After completing this chapter, you should be able to:

1.  Explain why major cybersecurity incidents become executive crises and identify the decision rights and escalation thresholds that must exist before an incident begins.

2.  Distinguish the roles of the incident response team and the crisis management team and design the interface that makes them interlock.

3.  Apply the five executive phases of the incident lifecycle, including an after-action review that identifies managerial as well as technical causes.

4.  Compare the notification obligations of Québec Law 25, PIPEDA, OSFI, the SEC, NIS2, and DORA, and recommend how to govern several clocks at once.

5.  Evaluate ransomware readiness as recoverability readiness and justify a board-approved approach to the extortion decision.

6.  Distinguish MTPD, RTO, RPO, and MBCO and explain how recovery objectives should cascade from business tolerances to disaster recovery capabilities.

7.  Assess disaster recovery and operational resilience plans for restoration realism, clean recovery, identity reconstitution, and systemic dependency, using OSFI Guideline E-21 as a reference.

8.  Recommend resilience metrics, exercises, and a governance-first implementation sequence that give boards evidence that recovery capability is real.

Every organization that depends on digital services will, sooner or later, experience a cybersecurity incident serious enough to disrupt its operations. Earlier chapters treated this as a matter of residual risk: no combination of controls, however well chosen, reduces the likelihood of compromise to zero. Chapter 4 explained that risk owners must accept some residual risk, Chapters 5 and 6 showed how scenario-based assessment makes that residual risk visible, and Chapter 8 described the operational discipline of incident response—preparation, identification and analysis, containment, eradication, recovery, and post-incident improvement. This chapter picks up where those discussions leave off. It asks what happens when residual risk materializes at a scale that the security operations centre cannot absorb alone.

At that scale, an incident is no longer a technical event. It becomes a business crisis that may affect customers, patients, employees, regulators, suppliers, insurers, investors, law enforcement, the media, and the public. A ransomware attack can turn into a liquidity problem, a patient-safety problem, a production stoppage, a breach of contract, a privacy notification, a securities disclosure, and a reputational crisis, all within the same week. A cloud provider outage can reveal that a supposedly diversified organization was concentrated on a single platform. A supplier compromise can expose dependencies that no one had mapped. In each case, the decisive questions are managerial: who decides, on what evidence, within what authority, and with what accountability.

The governance problem is that most organizations prepare for incidents in fragments. The security team owns an incident response plan, IT owns a disaster recovery plan, a business continuity coordinator maintains a plan that few people have read, legal counsel has a privacy breach procedure, and communications has a crisis template. Each document may be competent on its own terms. Yet when a major incident arrives, these plans must operate as one system under extreme time pressure, and the seams between them are where organizations fail. **Incident management, business continuity, and resilience must therefore be governed as a single enterprise capability**, with predefined decision rights, escalation thresholds, recovery objectives, notification processes, evidence discipline, and post-incident accountability.

This view is reinforced by current guidance. NIST SP 800-61 Rev. 3 (April 2025) frames incident response as part of cybersecurity risk management and organizes its recommendations around the six functions of the NIST Cybersecurity Framework (CSF) 2.0, including **Govern**, as discussed later in this chapter. ISO 22301:2019 (amended in 2024) defines requirements for a business continuity management system, and the ISO/IEC 27035 series provides a management-system view of information security incident management. For federally regulated financial institutions in Canada, OSFI Guideline E-21 on operational risk management and resilience reached its full-adherence date on September 1, 2026, making critical operations and tolerances for disruption a supervisory expectation rather than a voluntary good practice.

This chapter takes the executive, crisis, and continuity view. It does not repeat the operational mechanics covered in Chapter 8. Instead, it examines why major incidents must be managed as executive crises, how the operational incident response team and the executive crisis management team should interlock, how to govern the incident lifecycle, how to manage legal and regulatory obligations that run on overlapping clocks, why ransomware has become the defining stress test of recoverability, and how business continuity, disaster recovery, and operational resilience fit together. A worked scenario follows a Québec distributor through a ransomware crisis, and the chapter closes with metrics, an implementation sequence, and questions boards should ask.

## Why major incidents are executive crises

Organizations often describe incident response as a technical function: detect, contain, eradicate, and recover. The description is correct but incomplete. The incidents that matter most to governance unfold under time pressure, incomplete information, legal uncertainty, operational disruption, and public scrutiny. In those moments, the organization must answer questions that are fundamentally managerial. Three such questions define the executive view, and each corresponds to a governance capability that must exist before the incident begins.

### What business harm is occurring or about to occur?

The first question connects the incident to what the organization exists to do. Executives are rarely helped by knowing which servers are encrypted or which accounts are compromised; they need to know which services are interrupted, which customers or patients may be harmed, which contractual commitments may be missed, which regulatory obligations may be triggered, and how quickly the harm will escalate if decisions are delayed. This is the language of business impact, and it is the same language used in the risk triangle introduced earlier in the book: the threat has materialized and exploited a vulnerability, so the governance focus shifts decisively to impact.

Translating technical facts into business harm is not automatic. It requires a current map of critical services and their dependencies, a business impact analysis that describes how harm grows over time, and people who can bridge the technical and business vocabularies. Organizations that lack these assets spend the first hours of a crisis reconstructing them under pressure. Organizations that have them can move almost immediately from "what is broken?" to "what does this mean for our customers, our obligations, and our viability?"

The harm question also looks forward. In a ransomware incident, encrypted systems are only the visible harm; the attacker may also have exfiltrated data, may still hold privileged credentials, and may be preparing a second stage. Leaders must govern the harm that could still occur, not only the harm that has occurred.

### Who has authority to make decisions that bind the organization?

Major incidents require decisions that affect the whole enterprise. Shutting down a production network, suspending online banking, disconnecting a key supplier, invoking a disaster declaration, notifying a regulator, engaging outside counsel, activating cyber insurance, contacting law enforcement, or responding to an extortion demand are not technical decisions. They bind the organization legally, financially, and reputationally, and they must be made by people with the authority to make them.

The governance failure is not usually that the wrong person decides; it is that no one is sure who should. When authority is unclear, decisions are delayed while people seek permission, or they are made by whoever happens to be available and then contested afterwards. **Decision rights for crisis actions must be defined, documented, and rehearsed before the incident occurs**, with named alternates for when the primary decision-maker is unavailable, travelling, or personally affected. This is a direct application of the accountability principles discussed in Chapter 2 and of the Govern function of NIST CSF 2.0, which expects roles, responsibilities, and authorities for cybersecurity risk management to be established and communicated. At a minimum, five kinds of authority should be assigned:

- **Containment authority:** who may isolate networks, disable remote access, or take customer-facing systems offline when the evidence justifies it, even at significant short-term cost.

- **Disclosure authority:** who approves notifications to regulators, affected individuals, customers, investors, and the public, and who signs them.

- **Financial authority:** who may commit emergency spending for forensic firms, legal counsel, temporary infrastructure, or customer remediation outside the normal procurement cycle.

- **Extortion authority:** who, if anyone, may engage with a threat actor, and which body must approve any decision related to payment.

- **Resumption authority:** who determines that systems are trustworthy enough to return to production and that the organization can stand down from crisis mode.

### How will the organization demonstrate diligence afterwards?

The third question is about defensibility. An organization can recover operationally and still fail in governance if it cannot show that its decisions were reasonable, evidence-based, proportionate, timely, and properly authorized. Regulators, insurers, courts, auditors, customers, and boards will later ask what the organization knew, when it knew it, what options it considered, who approved key decisions, what evidence it preserved, and whether its conduct was consistent with its own policies and stated risk posture.

This is why the book repeatedly emphasizes defensible decisions. In a crisis, defensibility is not achieved by being right in hindsight; it is achieved by deciding through a sound process and recording that process. A decision to delay notification while facts are verified, for example, may be entirely reasonable, but only if the organization can show that it was actively verifying facts, that counsel was consulted, and that the delay was bounded. The same decision, made silently and recorded nowhere, looks like concealment.

| In a crisis, an organization performs at the level of its **governance preparation**, not at the level of its technical tools. |
|--------------------------------------------------------------------------------------------------------------------------------|

### Standing crisis structures and escalation thresholds

These three questions lead to a practical conclusion: crisis management requires standing structures, not ad hoc gatherings. The organization should know in advance who leads the crisis, who provides technical facts, who represents legal and regulatory interests, who communicates externally, who coordinates with suppliers, who briefs the board, who manages continuity, and who records decisions. It should also define **escalation thresholds**—the criteria that move an incident from technical handling to executive crisis management.

Escalation thresholds work best when they are expressed in business terms rather than technical ones. Examples include the interruption of a critical service beyond a defined period, credible evidence that personal information of more than a defined number of individuals has been accessed, any incident with potential safety consequences in operational technology environments (see Chapter 11), any incident that may trigger a regulatory reporting obligation, any extortion demand, and any incident likely to attract media attention. These thresholds should be linked to the organization's risk appetite and tolerance statements, so that crossing a tolerance in a live incident automatically engages the people who own that tolerance.

## Two systems that must interlock: the incident response team and the crisis management team

A frequent failure in major cyber incidents is expecting one group, often a CISO-led operational team, to run the technical response and manage the enterprise consequences at the same time. In practice, these are two distinct systems with different purposes, decision horizons, information needs, and success criteria. Mature incident governance does not merge them into one overloaded structure, nor does it separate them so completely that they lose contact. It makes them interlock.

### The operational incident response team

The **incident response team (IRT)** exists to reduce the attacker's advantage and stabilize the technical environment. It performs triage, investigation, containment, eradication, restoration, and evidence collection. It coordinates the security operations centre, IT operations, cloud, network, endpoint, and identity teams, operational technology engineers where relevant, managed service providers, and external forensic specialists. Its working methods—playbooks, severity levels, war rooms, and the response lifecycle—are described in Chapter 8 and are not repeated here.

From a governance perspective, what matters is what the IRT produces for the rest of the organization: timely, reliable facts. The IRT should be able to state what has happened, what is affected, what is still at risk, what containment options exist and what each would cost operationally, which recovery dependencies must be resolved, and how confident it is in each of these conclusions. It is evaluated on speed, accuracy, containment effectiveness, evidence quality, and discipline under pressure.

### The executive crisis management team

The **crisis management team (CMT)** exists to govern the incident as an enterprise risk event. It sets business priorities, allocates resources, directs legal and regulatory response, oversees communications, manages stakeholder relationships, and determines how the organization will handle consequences beyond the technical environment. Its decisions may include activating business continuity plans, suspending or maintaining services, notifying customers and regulators, engaging insurers and law enforcement, managing suppliers, briefing the board, and making or refusing extortion-related decisions.

The CMT is usually chaired by the chief executive, the chief operating officer, or another executive with enterprise-wide authority. It normally includes the CISO and CIO, general counsel, the privacy officer, the chief risk officer, communications, finance, human resources where employees are affected, and the business owners of the most affected critical services. It is evaluated on decisiveness, defensibility, stakeholder management, legal discipline, communication quality, continuity protection, and long-term consequence management. Its members do not need to understand every technical detail, but they must understand enough to challenge the facts they receive and to recognize when a technical recommendation carries business consequences.

The figure below contrasts the two teams. The IRT asks how to regain control of the environment; the CMT asks how to protect the enterprise while that happens. Neither question can be answered well without the other, and the liaison between them is as important as either team.

<span id="_Toc243635555" class="anchor"></span>Figure 56: Two interlocking teams: incident response and crisis management

### Two failure modes: blending and fragmentation

When the two systems are blended into one, the organization tends to over-focus on technical detail and underperform on everything else. Senior technical leaders are pulled away from containment to brief executives, draft statements, and manage vendors, while executives receive a flood of technical detail and little decision-ready analysis. Legal readiness, notification, and stakeholder management are handled late and inconsistently.

When the two systems are separated but not integrated, the opposite problem appears. Technical teams act without business alignment, restoring the systems they consider important rather than those the business needs first. Executives communicate without reliable facts and later have to retract statements. Legal review slows urgent containment, or containment destroys evidence that counsel needed. Business units demand recovery actions that conflict with forensic preservation. Suppliers receive inconsistent instructions, and customers, regulators, and employees hear contradictory messages.

**Governance maturity is the ability to run both systems in parallel, with clear hand-offs, synchronized decision gates, and disciplined information flow.** The IRT provides facts, impact assessments, confidence levels, options, and evidence updates. The CMT translates those facts into enterprise decisions: what to prioritize, what to disclose, which services to suspend or restore, which obligations are triggered, and which residual risks leadership is prepared to accept.

### Designing the interface before the incident

The interface between the two teams is itself a governance artifact and should be designed, documented, and exercised in advance. Several elements deserve explicit attention:

- **A liaison role:** a designated person, often a deputy CISO or senior incident manager, who sits in both teams, translates between them, and protects the IRT from interruptions.

- **A battle rhythm:** a fixed schedule of situation reports (for example, every four hours in the first two days, then twice daily), with a standard format that distinguishes confirmed facts, working hypotheses, unknowns, and decisions required.

- **Immediate-escalation criteria:** events that bypass the schedule, such as evidence of data exfiltration, a ransom note, a safety concern, a media inquiry, or a regulator contact.

- **Reserved decisions:** a short list of decisions that only the CMT may take, such as taking a customer-facing service offline, issuing an external statement, or engaging with a threat actor.

- **A single source of truth:** one incident record and one decision log, maintained by a named scribe, that both teams use.

- **Out-of-band communications:** pre-arranged channels that do not depend on the possibly compromised corporate email, chat, or identity platform.

A simple maturity test is whether, within the first hours of a major incident, the organization can answer without hesitation: Who leads the technical response? Who leads the enterprise response? Who may shut down systems or suspend services? Who speaks to regulators, customers, employees, and the media? Who keeps the decision log? Who sets recovery priorities? Who decides when the organization can return to normal operations? If any of these questions produces a debate, the interface has not been designed.

## A governance view of the incident lifecycle

Chapter 8 presented the operational incident response lifecycle. For governance purposes, the same lifecycle can be reframed into five executive phases that align more closely with board, executive, legal, and risk-management decision-making. These phases do not replace technical activities. They provide a leadership structure for governing the business consequences of a major incident, and they help executives recognize what kind of decision is expected of them at each point.

### Five executive phases

- **Stabilization.** The immediate objective is to prevent uncontrolled expansion of harm. This is when decisive containment may be necessary: isolating networks, disabling access paths, suspending remote access, halting sensitive workflows, blocking supplier connections, or taking systems offline. These actions disrupt operations, sometimes severely. Governance requires leadership to support containment when the evidence justifies it, even when containment is inconvenient, expensive, or visible to customers, because an uncontained incident almost always becomes more expensive.

- **Situational clarity.** Executives rarely receive perfect information early in a crisis. Governance requires a disciplined, staged understanding: what is confirmed, what is suspected, what remains unknown, what evidence supports each conclusion, and which hypotheses are being tested. This phase establishes a single source of truth with versioned updates, documented assumptions, and explicit confidence levels. The objective is not certainty; it is coherent decision-making under uncertainty.

- **Enterprise decision-making.** This is where incident response becomes crisis leadership. Decisions may include activating continuity plans, declaring a disaster for recovery purposes, suspending or maintaining services, notifying customers, partners, and regulators, coordinating with insurers and counsel, managing employee communications, and handling extortion demands. These decisions must be authorized, documented, and justified, because they will later be reviewed by boards, regulators, insurers, auditors, customers, and possibly courts.

- **Recovery and reconstitution.** Restoring service is not the same as restoring trust. Executives must govern the sequence of restoration: what comes back first and why, which dependencies must be validated, and which risks are accepted during temporary workarounds. Recovery choices include clean rebuilds versus rapid restoration, credential resets, restoration from backups, reconnection of third parties, manual operations, and phased return to normal. Recovery must be driven by evidence and by business priorities agreed in advance, not by whoever applies the most pressure.

- **Accountability and improvement.** The post-incident period determines whether the organization becomes stronger or simply returns to its previous vulnerability. Governance requires an after-action review, root-cause analysis that includes managerial causes, assigned accountability for remediation, and executive tracking of corrective actions. The outcome should be a funded remediation roadmap tied to risk appetite and resilience objectives.

The figure below summarizes these phases, their key governance actions and outputs, and the principles that run through all of them: authority and accountability, evidence and documentation, stakeholder alignment, proportionality, and continuous improvement.

<span id="_Toc243635556" class="anchor"></span>Figure 57: A governance view of the incident lifecycle

Read together, the phases form a leadership lens. Stabilization prevents harm from expanding; situational clarity gives leaders a reliable basis for decisions; enterprise decision-making aligns technical action with legal, operational, financial, and reputational consequences; recovery restores trusted operations in a controlled sequence; and accountability converts the incident into organizational learning. A mature organization does not wait for certainty before acting, but it does not act blindly either.

### Root cause includes managerial causes

The fifth phase deserves particular emphasis because it is where many organizations quietly fail. After-action reviews often stop at the immediate technical cause: an unpatched server, a phished credential, a misconfigured storage bucket. These findings are real but rarely sufficient. A governance-quality review also asks why the condition existed and persisted. Was ownership of the asset unclear? Had the control been underfunded? Was there a risk exception that had expired without review? Did vendor governance fail to require a security commitment? Was escalation delayed because thresholds were ambiguous? Were continuity plans untested? Was logging inadequate to support the investigation?

These managerial causes are uncomfortable because they implicate decisions made by people who may be in the room. That is why the review should be facilitated by someone independent of the response and reported to a governance body with authority to act. A review that identifies only technical causes produces a patch; one that identifies managerial causes produces change. Internal audit, as the third line, can verify that corrective actions are completed and actually effective.

### Aligning with NIST SP 800-61 Rev. 3 and CSF 2.0

NIST SP 800-61 Rev. 3, *Incident Response Recommendations and Considerations for Cybersecurity Risk Management: A CSF 2.0 Community Profile*, superseded the 2012 Rev. 2 in April 2025. Its most significant change for managers is conceptual. The earlier version presented a self-contained four-phase lifecycle; Rev. 3 instead maps incident response recommendations onto the six CSF 2.0 functions. Preparation activities sit largely under **Govern**, **Identify**, and **Protect**; incident response proper sits under **Detect**, **Respond**, and **Recover**; and lessons learned feed improvement activities across all functions.

The governance message is that incident response is not a separate program that starts when an alert fires. It depends on decisions made long before: risk strategy, roles and authorities, supplier requirements, asset inventories, logging policy, and recovery planning. For executives, this alignment has a practical benefit. An organization that already reports its cybersecurity posture using CSF 2.0 (see Chapter 9) can report its incident readiness in the same language, so that the board sees incident capability as part of the overall risk picture rather than as a separate technical topic. The CSF Recover function, in particular, covers incident recovery plan execution and recovery communication, which are the areas where the CMT and continuity functions take the lead.

### The ISO/IEC 27035 series

Organizations that run an ISO/IEC 27001 information security management system often look to the ISO/IEC 27035 series for incident management guidance. At the time of writing, the series comprises ISO/IEC 27035-1:2023 (principles and process), ISO/IEC 27035-2:2023 (guidelines to plan and prepare for incident response), ISO/IEC 27035-3:2020 (guidelines for ICT incident response operations), and ISO/IEC 27035-4:2024, which addresses coordination among multiple organizations responding to the same incident. Part 4 is especially relevant to the systemic-dependency issues discussed later in this chapter, because many serious incidents now involve a cloud provider, a managed service provider, a software supplier, and several customers at once.

For managers, the value of the ISO approach lies less in its technical content than in its management-system logic: plan and prepare, detect and report, assess and decide, respond, and learn. This logic fits naturally with the incident management controls of ISO/IEC 27002:2022 and with the internal audit and management review cycle of an ISMS, which means that incident capability can be audited, measured, and improved like any other management process.

## Legal and regulatory response discipline

This chapter is not a substitute for legal advice. Organizations must rely on qualified counsel who understand their jurisdiction, sector, contracts, and the facts of the incident. Nevertheless, executives must understand the governance dimensions of legal response, because legal discipline shapes what can be said, what must be done, what evidence must be preserved, and how the organization's conduct will be judged afterwards. Legal and regulatory response is not a phase that begins after technical containment; it is part of incident governance from the first hour.

### Evidence discipline and legal hold

The first principle is **evidence discipline**. During a significant incident, the organization should assume that its actions, communications, and decisions will be scrutinized by regulators, insurers, auditors, customers, litigants, or law enforcement. Logs, forensic images, access records, tickets, communications, and decision records must be preserved through controlled procedures, so that the evidence is reliable and explainable rather than contaminated by well-intentioned but uncontrolled investigation. Reimaging an infected server to restore service quickly, for example, may destroy the only evidence of how the attacker entered and what data was taken.

Evidence discipline also includes legal-hold thinking. A major incident is a record-creation event: emails, chat messages, meeting notes, draft statements, customer-impact analyses, and vendor communications all become part of the record. Poorly controlled records, contradictory timelines, and casual speculation in writing can create avoidable legal exposure. The governance response is not to stop writing things down, but to write them down in the right place, in factual terms, through designated channels. Retention rules for incident records should be defined in advance and should account for regulatory requirements such as Québec's five-year retention period for the confidentiality incident register discussed below.

### Privilege and controlled investigation design

The second principle concerns how the investigation is structured. In many jurisdictions, organizations engage forensic firms through external counsel so that certain communications and work product may attract legal privilege. The rules vary considerably between jurisdictions, including between common-law provinces and Québec's civil-law tradition, and courts have not always upheld privilege claims over forensic reports that served business purposes as well as legal ones. This is precisely why counsel must be involved early, before reports are commissioned and before findings are circulated.

Regardless of the legal specifics, the governance point is operational. Investigations must be directed by a named person, with clear authority to engage external specialists, validated findings, and controlled distribution. The objective is not to hide facts. It is to establish facts through a reliable process before the organization makes legal, regulatory, contractual, or public commitments that it may later have to retract.

### Communications governance

The third principle is **communications governance**. Many organizations are damaged less by the incident than by how they talk about it. Common failures include premature assurances that "no personal data was affected", inconsistent statements to different audiences, minimization that later collapses, technical jargon that confuses stakeholders, and silence that allows others to define the narrative. Crisis communication must therefore be governed as a controlled process with pre-approved holding statements, a clear approval path, and designated spokespeople.

A disciplined communications model communicates what is known and verified, distinguishes facts from ongoing assessment, acknowledges uncertainty, explains what the organization is doing for those affected, avoids speculation about attribution, and commits to further updates. Messages to employees, customers, regulators, suppliers, media, and the board should be consistent but not identical, because each audience needs different detail. Employees deserve particular attention: they are often the first to be asked questions by customers and friends, and they may be targeted by attackers impersonating the incident response team. **Credibility over the life of an incident is worth more than reassurance in its first hours.**

### Third parties, insurers, and law enforcement

Significant incidents require coordination with external counsel, forensic firms, cyber insurers, crisis-communications advisers, cloud and managed service providers, critical suppliers, regulators, and law enforcement. Cyber insurance policies frequently require prompt notice and may specify panel vendors for legal, forensic, and negotiation services; engaging a non-panel firm in the first hours can jeopardize coverage. These relationships should be established before a crisis, with retainers, contact details, and engagement procedures documented and tested. The organization should also decide in advance who may contact external parties, under what circumstances, and through which channels.

Law-enforcement engagement deserves a deliberate policy rather than a case-by-case debate. In Canada, the Canadian Centre for Cyber Security (CCCS) encourages incident reporting, and its ransomware guidance advises contacting police and reporting to the Cyber Centre and the Canadian Anti-Fraud Centre. Law enforcement may hold decryption keys or intelligence about the threat actor, and reporting does not mean surrendering control of the response.

### Decision logging

The final principle is the simplest and the most frequently neglected. During a major incident, the organization should maintain a **decision log** recording each material decision: what was decided, when, by whom, on what facts, which alternatives were considered, which risks were accepted, and which follow-up actions were assigned. Decision logs are not administrative overhead. They demonstrate diligence, support after-action review, preserve institutional memory when people rotate off the response, and provide evidence that leadership acted reasonably under uncertainty. A dedicated scribe, appointed before the first CMT meeting, is one of the cheapest and most valuable controls in crisis governance.

| A decision that cannot be reconstructed afterwards cannot be defended afterwards. The decision log is the CMT's most important deliverable after the decisions themselves. |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## Notification obligations: governing several clocks at once

Modern cyber incidents create multiple notification surfaces at once. Privacy regulators must be considered whenever personal information is involved. Sector regulators may impose operational incident-reporting expectations. Securities regulators may require disclosure of material incidents. Contracts with enterprise customers, public-sector clients, payment processors, and suppliers often include their own notice periods, sometimes shorter than any statutory deadline. Insurers require timely notice as a condition of coverage. Each of these obligations has its own trigger, threshold, recipient, content, and clock, and they rarely start at the same moment.

### Québec Law 25: confidentiality incidents

For organizations subject to Québec's *Act respecting the protection of personal information in the private sector*, as amended by Law 25, a **confidentiality incident** includes unauthorized access to, use of, communication of, or loss of personal information. The enterprise must take reasonable measures to reduce the risk of injury and prevent recurrence. When an incident presents a **risk of serious injury**, the enterprise must promptly notify the Commission d'accès à l'information (CAI) and the persons concerned. The assessment of serious injury considers the sensitivity of the information, the anticipated consequences of its use, and the likelihood that it will be used for injurious purposes.

The *Regulation respecting confidentiality incidents*, in force since December 2022, specifies the content of notices to the CAI and to individuals and the content of the **register of confidentiality incidents**. Every incident must be entered in the register, not only those presenting a risk of serious injury, and the register must be kept for at least five years after the enterprise becomes aware of the incident; the CAI may request a copy. Since September 2023, failure to report can expose an enterprise to administrative monetary penalties of up to \$10 million or 2% of worldwide turnover, and penal fines of up to \$25 million or 4%. Chapter 16 discusses Law 25 in more depth; the governance point here is that the register and the serious-injury assessment are recurring operational duties, not one-off crisis tasks.

### PIPEDA: breaches of security safeguards

For private-sector organizations subject to the federal *Personal Information Protection and Electronic Documents Act* (PIPEDA), the breach provisions and the *Breach of Security Safeguards Regulations* (SOR/2018-64), in force since November 1, 2018, require organizations to report to the Office of the Privacy Commissioner of Canada any breach that creates a **real risk of significant harm** to individuals, as soon as feasible after determining that the breach occurred. Affected individuals must also be notified, directly where possible, and other organizations or government institutions that can reduce the risk of harm may need to be informed. Organizations must keep a record of every breach of security safeguards, whether reportable or not, for 24 months.

Many Canadian organizations are subject to both regimes, for example a Québec-headquartered company with commercial activity in other provinces. Their notification process must apply both tests and produce consistent, coordinated notices, which is easier when one privacy officer owns the analysis and one register satisfies both record-keeping requirements.

### OSFI expectations for federally regulated financial institutions

Federally regulated financial institutions (FRFIs) face additional expectations. OSFI's *Technology and Cyber Security Incident Reporting* advisory, in effect since August 2021, requires FRFIs to report qualifying incidents to OSFI within 24 hours, or sooner if possible, and to provide regular updates (for example, daily) until the incident is resolved, followed by a post-incident review and lessons learned. Reportable incidents include, among others, those with potential consequences for other institutions or the financial system, those affecting customer information, those that activate disaster recovery plans, those reported to the Privacy Commissioner or law enforcement, those leading to a cyber insurance claim, and those assigned a high or critical internal severity. OSFI encourages precautionary reporting when an institution is unsure.

These reporting expectations sit within OSFI Guideline B-13 on technology and cyber risk management, which expects FRFIs to maintain incident management, recovery, and resilience capabilities, and within Guideline E-21, discussed later in this chapter, which extends the lens from individual incidents to the resilience of critical operations.

### SEC disclosure for U.S.-listed companies

Canadian organizations listed in the United States, and Canadian subsidiaries of U.S. registrants, must consider the U.S. Securities and Exchange Commission's cybersecurity disclosure rules adopted in 2023. Under Form 8-K Item 1.05, a domestic registrant that determines a cybersecurity incident to be **material** must disclose its nature, scope, timing, and material or reasonably likely material impact within four business days of that determination, and the materiality determination itself must be made without unreasonable delay after discovery. A delay is possible only if the U.S. Attorney General determines that disclosure would pose a substantial risk to national security or public safety. Annual disclosures of cybersecurity risk management, strategy, and governance are required under Regulation S-K Item 106. At the time of writing, these requirements remain in force, although the SEC invited public comment on reforming Regulation S-K disclosure requirements in early 2026 and industry associations have asked for Item 1.05 to be rescinded; managers should check their current status.

The governance lesson from the SEC regime applies well beyond the United States. **Materiality is a business judgment, not a technical one**, and organizations need a pre-defined process, typically involving a disclosure committee, counsel, finance, and the CISO, for assessing incident materiality quickly and documenting the reasoning.

### NIS2 and DORA in the European Union

Organizations operating in the European Union face two significant regimes. The NIS2 Directive (Directive (EU) 2022/2555) requires essential and important entities to submit an early warning of a significant incident within 24 hours of becoming aware of it, an incident notification within 72 hours, and a final report no later than one month after the notification. The Digital Operational Resilience Act (DORA, Regulation (EU) 2022/2554), which applies to EU financial entities from January 2025, requires an initial notification of a major ICT-related incident within four hours of classifying it as major and no later than 24 hours after becoming aware of it, an intermediate report within 72 hours, and a final report within one month, with content and timelines set by Commission Delegated Regulation (EU) 2025/301. A Canadian group with EU subsidiaries or EU financial clients may therefore be drawn into these timelines through its operations or its contracts.

| **Regime**             | **Trigger**                                          | **Recipient**                        | **Clock**                                                   |
|------------------------|------------------------------------------------------|--------------------------------------|-------------------------------------------------------------|
| Québec Law 25          | Confidentiality incident with risk of serious injury | CAI and persons concerned            | Promptly; register every incident, keep 5 years             |
| PIPEDA                 | Breach with real risk of significant harm            | Privacy Commissioner and individuals | As soon as feasible; keep records 24 months                 |
| OSFI advisory          | Reportable technology or cyber incident (FRFIs)      | OSFI                                 | Within 24 hours; regular updates                            |
| SEC Form 8-K Item 1.05 | Material cybersecurity incident                      | Investors (public filing)            | 4 business days after materiality determination             |
| NIS2                   | Significant incident                                 | CSIRT or competent authority         | 24 h early warning; 72 h notification; final in 1 month     |
| DORA                   | Major ICT-related incident                           | Financial supervisor                 | 4 h after classification (max 24 h); 72 h; final in 1 month |

### Governing the overlap

These obligations overlap, interact, and sometimes conflict. One regulator expects notice within 24 hours while the investigation is still establishing whether data left the network. A contract requires notice to a customer before the organization has finished assessing materiality for securities purposes. A securities filing becomes public and effectively notifies the attacker, other customers, and journalists at the same moment. Law enforcement may ask for discretion while privacy law requires notice to individuals.

Governance maturity is the ability to reconcile these pressures quickly, with counsel, through a documented rationale. Executives should not be discovering notification obligations for the first time during a crisis. The organization should maintain a **notification matrix** that lists each applicable regime and material contract, its trigger, threshold, recipient, deadline, and owner; pre-drafted templates; a process for assessing thresholds; and a log of what was communicated, when, to whom, by whom, and on what factual basis. Canada's Critical Cyber Systems Protection Act, enacted by Bill C-8, which received royal assent in June 2026, will add mandatory reporting for designated operators in federally regulated critical sectors once its provisions and regulations take effect; organizations in those sectors should already be extending their matrices accordingly.

## Ransomware as a governance stress test

Ransomware has transformed the resilience problem. It is not simply an availability incident; it is an adversarial disruption designed to maximize leverage over the victim while destroying confidence in the integrity of its systems. Most modern ransomware operations combine encryption with data theft and the threat of publication, so that availability harm and confidentiality harm arrive together. Many also target identity systems, privileged accounts, and administrative tools, which means that recovery can be blocked if the identity fabric remains compromised. Some groups skip encryption altogether and rely purely on data-theft extortion. For these reasons, ransomware tests almost every element of governance discussed in this book at the same time.

### Recoverability is the core capability

For executives, the essential shift is to treat ransomware readiness as **recoverability readiness**. The central question becomes: can we restore our critical services and data to an acceptable operating state, within our stated tolerances, without relying on the attacker? If the honest answer is uncertain, the organization has not built resilience; it has built hope.

Recoverability is not the same as having backups. It is the demonstrated ability to restore within defined objectives, at enterprise scale, with validated integrity, under adverse conditions. Three governance decisions are routinely neglected. First, the organization must define which services are critical and set explicit tolerances for downtime and data loss for each; this is a business continuity decision, not a technical one. Second, it must invest in restoration realism: backups that are protected from the attacker, tested at the scale required, and restorable fast enough to matter. Third, it must treat **identity recovery** as a first-class recovery domain, because the attacker's advantage is often anchored in credentials, tokens, and privileged access (see Chapter 10). A recovery plan that does not address credential resets, privileged account re-establishment, secure administrative pathways, and the safe reconnection of third parties is incomplete.

### The extortion decision

Whether to pay a ransom is one of the most difficult decisions an executive team can face, and it should never be made for the first time under pressure. Canadian and allied government guidance consistently emphasizes the risks of payment. The CCCS ransomware playbook (ITSM.00.099, effective December 11, 2025, current release January 28, 2026) states that the decision rests with the organization but stresses that payment does not guarantee recovery of data or systems, that attackers may leak or sell stolen data anyway, that payment may encourage retargeting and further demands, and that payment may be unlawful under anti-terrorism, money-laundering, criminal-organization, or sanctions laws. In 2023, members of the International Counter Ransomware Initiative, including Canada, endorsed a joint statement that their national governments should not pay ransoms. In the United States, the Treasury's Office of Foreign Assets Control (OFAC) has advised since 2020 (updated in 2021) that facilitating payments to sanctioned persons can violate sanctions regardless of whether the payer knew the recipient's status, while treating prompt reporting and cooperation with law enforcement as mitigating factors.

For governance purposes, the payment question should be broken into components that can be prepared in advance:

- **Policy stance:** the board should approve a documented position on extortion payments, recognizing that the final decision may be context-dependent but setting a strong default and the conditions under which it may be reconsidered.

- **Decision authority:** the body that may authorize any engagement with a threat actor or any payment, typically the CMT with board notification, never a technical team or an individual negotiator.

- **Legal screening:** a process, led by counsel, for assessing sanctions exposure in Canada and in any other relevant jurisdiction, as well as criminal-law and anti-money-laundering risks.

- **Insurance conditions:** a clear understanding of what the cyber policy covers, which vendors it requires, and what consent the insurer must give.

- **Decision criteria:** the factors leadership will weigh, including safety and patient or customer harm, the realistic alternative recovery path and its duration, the likelihood that a decryptor will work, the nature of the stolen data, the ethical and reputational consequences, and the incentive effects of payment.

- **Documentation:** a complete record of the reasoning, alternatives, and approvals, regardless of the outcome.

A central insight emerges once these components are laid out. **The pressure to pay is greatest when the organization cannot recover any other way within a tolerable period.** The most effective way to avoid an impossible extortion decision is therefore to invest, well before the incident, in the recoverability that makes payment unnecessary. Payment for the suppression of stolen data is even harder to justify, because the organization cannot verify that the data has been deleted, and the notification obligations described earlier generally remain regardless of whether payment is made.

## Business continuity management: the managerial spine of resilience

Technology recovery alone is not continuity. Even with a flawless disaster recovery plan, there are gaps that technology cannot fill: manual processes, alternative workflows, staffing constraints, customer communications, supply-chain dependencies, and decisions about what to prioritize when not everything can be delivered. **Business continuity management (BCM)** addresses these gaps. It is the managerial spine that connects a cyber event to business survival, and it is the discipline through which leadership decides, in advance, what the organization must keep doing no matter what.

### ISO 22301 and the business continuity management system

ISO 22301:2019 specifies requirements for a **business continuity management system (BCMS)**. Like ISO/IEC 27001, it follows the harmonized management-system structure: context, leadership, planning, support, operation, performance evaluation, and improvement. Its 2024 amendment (ISO 22301:2019/Amd 1:2024) added climate-change considerations to the analysis of the organization's context and interested parties, in line with other ISO management-system standards; at the time of writing, ISO has a revision of ISO 22301 under development.

The management-system structure matters because it makes continuity auditable: top management must lead, set objectives, provide resources, review performance, and drive improvement. Organizations already running an ISMS can integrate the two systems rather than duplicate them. ISO/IEC 27002:2022 builds the bridge explicitly through its controls on information security during disruption and on ICT readiness for business continuity.

### Business impact analysis that reflects real dependencies

A credible continuity program begins with a **business impact analysis (BIA)**, but the BIA must reflect modern systemic dependency. It cannot be a static spreadsheet listing applications and their owners. It must map critical business processes, the services and data that enable them, essential third parties, peak periods, and the consequences of interruption over time—for example at two hours, one day, three days, and two weeks. This time-phased view matters because harm is non-linear. Many organizations can tolerate a short outage, but the same outage becomes existential if it spans a payroll run, a regulatory reporting deadline, a month-end close, or a seasonal revenue peak.

The BIA is also where business owners, not IT, state what they can tolerate. This is where risk appetite becomes concrete. A board that has declared a low appetite for service disruption must eventually translate that statement into specific tolerances for specific services, and the BIA is the instrument for doing so. When business owners are asked to sign off on these tolerances, they frequently discover that the tolerances they want are much tighter than the recovery capabilities they have funded. That discovery is valuable: it turns a hidden gap into a visible, decidable investment question.

### Recovery objectives: MTPD, RTO, RPO, and MBCO

Chapter 8 introduced recovery time and recovery point objectives from an operational standpoint. At the governance level, four related concepts, defined in the vocabulary of ISO 22301 and its companion terminology standard ISO 22300, should be understood as a set:

- **Maximum tolerable period of disruption (MTPD):** the time after which the impact of not resuming a product, service, or activity becomes unacceptable to the organization. It is a business judgment, derived from the BIA and anchored in risk appetite; Chapter 8 uses the closely related term maximum tolerable downtime.

- **Recovery time objective (RTO):** the period within which the activity or its supporting resources must be resumed. The RTO must be shorter than the MTPD, leaving a margin for the unexpected.

- **Recovery point objective (RPO):** the point in time to which data must be restored, which determines how much data loss is acceptable and therefore how frequently data must be protected.

- **Minimum business continuity objective (MBCO):** the minimum acceptable level of service during the disruption, which recognizes that a degraded service may be acceptable for a time when no service is not.

The figure below positions these objectives within the broader relationship between incident response, disaster recovery, business continuity, and operational resilience. **The objectives cascade from the business downward**: the MTPD is set by leadership, business continuity strategies are designed to stay within it, and disaster recovery capabilities are engineered to meet the RTO and RPO that the continuity strategy requires. When the cascade is reversed—when IT announces what it can recover and the business is expected to accept it—continuity becomes a technical outcome rather than a governance decision.

<span id="_Toc243635557" class="anchor"></span>Figure 58: How business continuity, disaster recovery, and resilience fit together

### Continuity strategies beyond technology

Executives must also govern continuity options beyond IT. A continuity plan that simply assumes systems will be back soon offers little protection. Real continuity strategies include manual workarounds, degraded services, restricted customer-facing features, alternative communication channels, alternative suppliers, temporary policy adjustments such as pausing certain transaction types, and pre-arranged mutual aid with peers. These options carry their own risks: manual processes are slower and more error-prone, temporary policies may create fraud opportunities, and alternative suppliers may not meet the same security standards.

For that reason, continuity options must be pre-defined, with acceptable risk boundaries and pre-approved decision rights. A hospital that intends to revert to paper charting must have printed forms, trained staff, and a plan for reconciling paper records once systems return. A distributor that intends to take orders by phone must know which customers to prioritize and what credit limits to apply without its normal systems. Making these choices during a crisis causes delays and inconsistent risk-taking; making them in advance turns them into governed exceptions with known owners and known expiry.

## Disaster recovery: from restoring servers to reconstituting trust

Disaster recovery planning is often misrepresented as an infrastructure plan. In modern cyber resilience, it must be treated as the controlled reconstitution of reliable services, not merely the restoration of computing capacity. In ransomware-driven incidents a dangerous illusion appears: systems can be restored quickly, but the restored systems may still be compromised, or the attacker may still hold valid credentials. **Speed without integrity becomes a path to repeated compromise.** Mature disaster recovery therefore includes four dimensions that executives should expect to see as explicit commitments.

### Guidance: NIST SP 800-34 and ISO/IEC 27031

Two references are particularly useful. NIST SP 800-34 Rev. 1, *Contingency Planning Guide for Federal Information Systems* (May 2010, with errata updates in November 2010), remains NIST's principal contingency planning guide at the time of writing. It describes a seven-step contingency planning process, the relationship between the BIA and recovery strategies, and a family of related plans, from continuity of operations to information system contingency plans. Although it predates the ransomware era and was written for U.S. federal systems, its structure remains widely used. ISO/IEC 27031:2025, published as a second edition in May 2025, provides guidance on **ICT readiness for business continuity**: the preparedness of information and communication technology to support continuity objectives, including dependencies on cloud and other third-party services.

### Tiered recovery and restoration realism

Not all services are equal. Disaster recovery must classify services by criticality and define recovery priorities tied to BIA outcomes, so that recovery sequencing does not become a political negotiation during a crisis. A typical approach defines three or four tiers, each with its own RTO, RPO, and recovery architecture, and assigns every critical service to a tier with the explicit approval of its business owner. Dependencies must be honoured: a tier-one customer portal is of little use if the identity service, database, and network on which it depends are in tier three.

The second dimension is **restoration realism**. Backup and redundancy strategies matter, but the governance requirement is tested restoration at the required scale. Executives should insist on evidence of restoration tests, not declarative assurances. Restoring a single virtual machine in a test environment proves very little about restoring several hundred systems simultaneously, with limited staff, while the network is partially isolated. In ransomware scenarios, backups that are immutable, offline, or otherwise isolated from the production identity domain become essential, because attackers specifically seek out and destroy backups to remove the victim's alternatives.

### Clean recovery and identity reconstitution

The third dimension is **clean recovery design**. Organizations increasingly maintain segregated recovery environments—sometimes called clean rooms or isolated recovery environments—in which systems can be rebuilt from trusted images, validated, and only then reconnected. Whether or not this specific architecture is adopted, the governance point remains: recovery must include integrity validation and a plan to prevent reinfection. Leadership must accept that clean recovery is slower than simply restoring the last backup, and that this trade-off is deliberate.

The fourth dimension is **identity and key recovery**. Disaster recovery has traditionally focused on infrastructure and data. In cyber incidents, the ability to restore identity services, reset privileged access, rotate cryptographic keys and certificates (see Chapter 20), re-establish trust relationships, and re-onboard third parties becomes central. If the directory service is compromised, every system that trusts it is compromised too. Executives should treat identity recovery as part of disaster recovery scope and should ask to see it tested, not merely documented.

| Restoring service is not the same as restoring trust. Recovery is complete only when the organization can show that the systems it brought back are free of the attacker. |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## Operational resilience and systemic dependency

**Cyber resilience** is often described as the ability to withstand and recover from attacks. Governance requires a more operational definition: it is the organization's ability to maintain acceptable levels of critical service under adversarial disruption and to restore those services within pre-defined tolerances, while controlling legal, regulatory, and reputational harm. This definition implies that resilience is not a plan but a continuous posture that spans governance, architecture, operations, suppliers, and people. Incident response, business continuity, and disaster recovery are its instruments; resilience is the outcome they are meant to produce.

### Three realities that shape resilience

The first reality is that compromise is plausible. Mature resilience rests on **assume-breach thinking**. This is not pessimism; it is risk realism. An organization that assumes it will detect every attacker early will build fragile systems. An organization that assumes some attackers will get in will invest in segmentation, blast-radius reduction, rapid containment, and recovery discipline.

The second reality is that dependency is systemic. Organizations now depend on ecosystems as much as on the systems they operate themselves. Cloud platforms, identity providers, software suppliers, payment processors, managed service providers, and telecommunications carriers create concentration risks that no single organization controls. Widely reported events in recent years, from compromised managed-service and file-transfer platforms to a faulty security-software update that disabled millions of computers worldwide in July 2024, have shown how a single supplier failure can disrupt thousands of organizations at once.

The third reality is that ransomware and similar attacks are designed to destroy confidence. Recovery is therefore about re-establishing trust—in data integrity, in identity systems, in administrative control, and in the organization's own statements—as much as about uptime. In governance terms, resilience can be summarized as reducing three intervals: the time to detect, the time to contain, and the time to recover to a trusted state.

### OSFI E-21: critical operations and tolerances for disruption

Regulators have begun to codify this view of resilience. OSFI Guideline E-21, *Operational Risk Management and Resilience*, finalized in August 2024, expects FRFIs to identify their **critical operations**, map the people, processes, technology, information, facilities, and third parties that support them, and set **tolerances for disruption** for each. Full adherence to the guideline, including mapping and tolerance-setting and the development of a scenario-testing methodology, was expected by September 1, 2026, with scenario testing of all critical operations to be completed by September 1, 2027. Scenario testing in this sense examines whether critical operations can remain within their tolerances under severe but plausible circumstances, end to end, across business lines.

The shift is subtle but important. Traditional continuity planning asks whether each system can be recovered within its RTO. Operational resilience asks whether the *service* that customers and the financial system depend on can stay within tolerance, regardless of which component fails. That question forces organizations to look across technology, people, facilities, and suppliers at once, and it gives boards a clearer basis for oversight. DORA takes a comparable approach for EU financial entities. Even organizations outside financial services can borrow the logic: identify the handful of services whose disruption would cause intolerable harm, set explicit tolerances, and test against severe scenarios.

### Mapping and governing concentration risk

Executives should govern resilience with an explicit dependency and concentration lens. This includes maintaining a living dependency map for each critical service, identifying single points of failure outside the organization, and setting explicit tolerance for those dependencies. For some dependencies, the organization may decide that tolerance is high and accept the risk. For others, it may require contractual commitments, architectural redundancy, alternative providers, or a documented exit plan. Chapter 15 discusses cloud concentration in depth, and Chapter 19 addresses third-party and supply-chain governance more broadly.

Governance should also insist on **third-party incident integration**. Many incidents are amplified because organizations have no clear expectations for supplier notification, evidence sharing, joint investigation, recovery coordination, and restoration sequencing. Contracts should therefore cover not only availability service levels, but also incident notification timelines, cooperation with forensic investigations, alignment of crisis communications, participation in joint exercises, and support during recovery. A useful test is to ask each critical supplier how it would notify the organization of an incident, to whom, and within what period—and then to check whether the organization's own notification matrix could meet its regulatory deadlines on that basis.

### Exercising: from tabletop to full recovery

A plan that has not been exercised under stress is a plan that has not been verified. Exercises are the principal means by which governance obtains evidence that incident, continuity, and recovery capabilities work together. They range from discussion-based tabletop exercises, in which the CMT works through a scenario around a table, to functional exercises that test specific capabilities such as notification or failover, to full-scale exercises that restore real systems under realistic constraints.

The most valuable exercises are deliberately uncomfortable. They inject conflicting information, remove key people, disable the primary communication channel, force a trade-off between evidence preservation and rapid recovery, and add a ransom deadline or a journalist's call. They draw on the risk scenarios developed through the methods of Chapters 5 and 6, so that the organization rehearses the events it has assessed as most significant rather than generic disasters. Board members benefit from participating occasionally, because it gives them a realistic sense of the decisions they may be asked to support.

Findings from exercises must be treated as risk treatment actions: recorded, owned, funded, tracked, and closed. An exercise program that produces the same findings year after year is an indicator of governance failure, not of operational weakness, because it shows that leadership is receiving evidence and not acting on it.

## A worked crisis: ransomware at a Québec distributor

The following scenario is fictional, but its decisions reflect situations real organizations have faced. It is presented in managerial terms; the parallel technical response appears only where it shapes an executive decision.

### Illustrative scenario: the organization and its preparation

Saint-Laurent Medical Supply is a privately held Québec company with about 1,100 employees that distributes pharmaceutical and medical supplies to hospitals, clinics, and pharmacies in Québec and eastern Ontario. It also runs a small home-delivery service for patients of partner pharmacies. Its critical services, identified in a business impact analysis two years earlier, are hospital order fulfilment, pharmacy order fulfilment, and payroll. For hospital fulfilment, the BIA set an MTPD of 72 hours, because several hospital clients hold only three days of stock for some critical items, and an MBCO of fulfilling at least the hospitals' designated critical-item orders within 24 hours. The documented RTO for the warehouse management system was 48 hours, with an RPO of 24 hours.

On paper, Saint-Laurent was reasonably prepared. It had an incident response plan, a disaster recovery plan, a crisis management team chaired by the chief operating officer, a cyber insurance policy, and immutable backup copies stored with a cloud provider. The board had approved an extortion policy two years earlier that set a strong default against payment. Some weaknesses were known but not prominent: the restore procedure had been tested on about twenty servers but never at scale, the home-delivery service had been launched after the last BIA, and a risk exception allowing a remote-access appliance to run an outdated firmware version had expired four months earlier without anyone renewing, remediating, or escalating it.

### Hours 0–6: stabilization and escalation

At 02:40 on a Friday, endpoint detection alerts showed mass encryption on file servers and the warehouse management system at the main distribution centre. By 04:00, ransom notes had appeared on dozens of machines, and the attacker's leak site claimed to hold 400 GB of Saint-Laurent data. Two escalation thresholds had been crossed at once: an extortion demand and the interruption of a critical service. The incident manager therefore called the CISO, who activated the crisis management team. The COO convened it by telephone at 06:00 using the out-of-band conference bridge and contact list specified in the crisis plan, because corporate email and chat ran on the identity platform the attacker was suspected of controlling.

The CISO's first recommendation was to isolate the distribution centre network and disable all remote access. This would halt automated order picking and stop every remote employee from working. The COO, with the CFO and the vice-president of operations, approved the containment in less than fifteen minutes, because the crisis plan gave the CMT chair containment authority and the decision criteria had been rehearsed in an exercise the previous year. A scribe opened the decision log and recorded the decision, the alternatives considered, and the accepted consequence: a full stop in automated fulfilment.

### Day 1: continuity, counsel, and first notifications

With automated systems offline, operations invoked the continuity plan for hospital fulfilment. Supervisors printed pick lists from a standalone reporting copy refreshed nightly, and staff fulfilled the hospitals' designated critical-item orders manually. By the evening, about 70% of critical-item orders had shipped, above the MBCO. Pharmacy fulfilment was deliberately suspended, and pharmacies were told to expect delays. This trade-off had been decided in principle during the BIA; the CMT only needed to confirm it.

The general counsel notified the insurer at 07:30, which allowed the panel law firm and a forensic firm to be engaged under the policy's terms. External counsel retained the forensic firm and directed the investigation. Several hospital contracts required notice within 24 hours of any incident affecting service, so communications issued a pre-approved holding statement to hospital clients by noon, stating what was known, what Saint-Laurent was doing to protect supply, and when the next update would come. Employees received instructions by text message through an emergency notification service, including a warning that attackers sometimes impersonate IT staff during incidents. The CISO reported the incident to the Sûreté du Québec and to the Canadian Centre for Cyber Security, after confirming the approach with counsel.

### Days 2–4: facts, the extortion decision, and notification

By the second day, the forensic firm had established that the attacker entered through the outdated remote-access appliance, obtained domain administrator credentials, and spent nine days in the environment before encrypting. Data had been exfiltrated from the human resources file share, including employee social insurance numbers and banking details, and from the home-delivery system, including patient names, addresses, and medication types. The primary on-premises backup server had been encrypted, but the immutable cloud copies were intact.

The IRT then delivered unwelcome news: the 48-hour RTO for the warehouse management system could not be met. Because domain administrator credentials were compromised, the identity service had to be rebuilt in a clean environment before anything else could be trusted, and restoration at scale had never been tested. The realistic estimate was six days for the warehouse system and twelve days for broad normal operations. With manual fulfilment covering the hospitals' critical items, the six-day estimate exceeded the RTO but, crucially, did not breach the MTPD for critical hospital supply, because the continuity strategy was holding.

The attacker demanded the equivalent of 2.4 million dollars for a decryptor and the deletion of stolen data. The CMT worked through the decision criteria set out in the board-approved policy. A viable recovery path existed without payment. The reliability of the decryptor was unknown. Payment would not undo the data theft, and the notification obligations would remain whatever the outcome. Counsel's sanctions screening of the threat group was inconclusive. The CMT decided not to pay, the chair of the board was informed the same day, and the full board was briefed at a special meeting on Day 3. The reasoning, the alternatives, and the approvals were recorded in the decision log.

In parallel, the privacy officer assessed the data theft. The employee data and the patient health information both presented a risk of serious injury under Québec's private-sector Act, so Saint-Laurent notified the Commission d'accès à l'information and the affected individuals, and entered the incident in its register of confidentiality incidents. Because the home-delivery service also handled personal information of Ontario patients in the course of commercial activity, counsel concluded that the PIPEDA real-risk-of-significant-harm test was also met and reported to the Office of the Privacy Commissioner of Canada. The notices were drafted together so that their content was consistent, and each recorded what was known at the time and committed to updates.

### Days 5–12: recovery and the pressure to cut corners

Recovery followed the tiering approved before the incident: identity services first, rebuilt in an isolated recovery environment; then the warehouse management and ordering systems; then finance and payroll; and finally email, collaboration, and the sales CRM. On Day 5 the vice-president of sales asked for the CRM to be restored ahead of finance so that account managers could reassure customers. The CMT declined, citing the approved tiering and the fact that payroll was due on Day 8. The decision and its rationale were logged.

The warehouse management system returned to production on Day 7, after validation in the recovery environment and the reset of all privileged credentials. Payroll ran on time. Most services were back to normal by Day 12, and the CMT formally stood down on Day 14, with the IRT continuing to monitor for signs of re-entry.

### After-action: the managerial causes

The after-action review, facilitated by an external adviser and reported to the board's audit and risk committee, found that the technical cause was the outdated appliance. It also identified the managerial causes that had allowed the condition to persist and the incident to grow:

- **An expired risk exception with no escalation:** the exception register recorded expiry dates but triggered no review, so an accepted temporary risk silently became a permanent one.

- **An untested recovery objective:** the 48-hour RTO had been documented but never validated at scale, so leadership had been relying on an assumption it had not verified.

- **An outdated BIA:** the home-delivery service, which handled the most sensitive data, had never been included in the impact analysis or the notification matrix.

- **Logging gaps:** limited retention of remote-access logs lengthened the forensic investigation and delayed the notification assessment.

The board approved a funded remediation roadmap with named owners and quarterly reporting. It included automatic escalation of expiring exceptions, quarterly restoration tests at increasing scale, an annual BIA refresh triggered by any new service, twice-yearly CMT exercises, and extended log retention. The managerial lesson was not that Saint-Laurent had been unprepared; its continuity strategy and crisis structure had held. **It was that the gaps it had tolerated in quiet times became the critical path in the crisis.**

## Metrics, evidence, and a governance-first implementation sequence

Resilience must be measurable enough to govern. The objective is not to produce vanity dashboards that report the number of plans written or training sessions delivered, but to create credible indicators that correlate with real recovery capability. In the language of earlier chapters, leadership needs key risk indicators that warn when resilience is eroding and evidence that supports assurance that it is real. Three categories of evidence matter.

### Preparedness, recovery, and governance evidence

- **Preparedness evidence** includes the frequency, realism, and participation of tabletop exercises, crisis simulations, and recovery drills, the proportion of critical services exercised in the past year, and the proportion of exercise findings closed on time.

- **Operational recovery evidence** includes restore test success rates and durations measured against RTO and RPO, the largest scale at which restoration has actually been demonstrated, the time to rebuild critical systems from known-good images, the results of identity recovery tests, and the coverage of immutable or isolated backups for critical data.

- **Governance evidence** includes documented and current decision rights, escalation criteria, contact lists verified within a defined period, retained external specialists, insurance engagement procedures, a maintained notification matrix, a confidentiality incident register, and defined communication approval paths. In a crisis, governance evidence often determines whether the organization can move quickly without losing control of its narrative or violating its obligations.

### Key risk indicators for resilience

Several indicators are particularly useful as KRIs because they reveal drift before an incident exposes it. They include the gap between documented and demonstrated RTO for each critical service; the number of critical services whose BIA is older than the agreed refresh cycle; the number of expired or overdue risk exceptions affecting critical services; the proportion of critical suppliers with tested incident-notification commitments; and the age of the last successful full-scale restore. Each KRI should have a threshold tied to the organization's tolerance and an owner who must act when it is crossed. Reporting these indicators alongside the CSF 2.0 profile gives the board a single, coherent picture of whether the organization can withstand the scenarios it has assessed as most significant.

| The most honest resilience metric is the gap between the recovery time the organization has promised and the recovery time it has demonstrated. |
|-------------------------------------------------------------------------------------------------------------------------------------------------|

### A governance-first implementation sequence

Organizations often try to build resilience by buying technology first: a new backup platform, an isolated recovery environment, a crisis-management application. These investments may be worthwhile, but without governance decisions to anchor them they tend to be sized for the wrong services, tested against the wrong objectives, and owned by no one in the business. A governance-first sequence is more durable. The figure below presents it as six steps, each producing evidence that the next step relies on.

<span id="_Toc243635558" class="anchor"></span>Figure 59: A governance-first sequence for building resilience

1.  **Define critical services and tolerances.** Leadership identifies the services whose disruption would cause intolerable harm and sets the MTPD, MBCO, RTO, and RPO for each, making the trade-offs visible and linking them to risk appetite.

2.  **Map dependencies.** For each critical service, the organization maps the people, processes, technology, data, facilities, and third parties it relies on, because hidden dependencies determine actual recoverability.

3.  **Establish crisis governance.** Decision rights, escalation thresholds, the CMT and IRT interface, the notification matrix, the extortion policy, and the decision log are defined and approved before they are needed.

4.  **Design recovery and continuity strategies.** Backup architecture, clean recovery, identity recovery, tiered sequencing, manual workarounds, and supplier commitments are designed to meet the objectives set in the first step.

5.  **Exercise and validate.** Tabletop, functional, and full-scale exercises, built on the organization's own risk scenarios, test whether the strategies work together; findings become tracked risk treatment actions.

6.  **Institutionalize and report.** Resilience is integrated into management review, internal audit scope, KRI reporting, investment planning (see Chapter 17), and board oversight, so that it becomes part of the managerial operating system rather than an emergency program.

### Questions boards and executives should ask

Directors do not need to master recovery architectures, but they should be able to ask questions that reveal whether resilience is real or aspirational. The following questions are a practical starting point:

- **Critical services:** Which of our services, if disrupted, would cause intolerable harm, and who has formally approved the tolerances for each?

- **Demonstrated recovery:** For each critical service, what recovery time have we actually demonstrated, at what scale, and how does it compare with what we have promised?

- **Recovery without the attacker:** If ransomware encrypted our core systems tomorrow, could we recover without paying, and how long would it take?

- **Identity recovery:** Have we tested rebuilding our identity platform from a trusted state, and how long did it take?

- **Decision rights:** Who has authority to take critical systems offline, notify regulators, and engage with a threat actor, and who are their alternates?

- **Extortion policy:** Has the board approved a position on ransom payments, including the legal screening and decision criteria that would apply?

- **Notification readiness:** Do we maintain a current matrix of our regulatory and contractual notification obligations, and when was it last tested?

- **Supplier dependency:** Which third parties could disrupt our critical services, what have they committed to in an incident, and what is our plan if one of them fails?

- **Exercises:** When did the executive team last exercise a severe cyber scenario, what did it reveal, and which findings remain open?

- **Communication:** How would we communicate with employees, customers, and the board if our email and collaboration platforms were unavailable?

- **Learning:** After our last significant incident or exercise, which managerial causes were identified, and what has changed as a result?

- **Assurance:** What independent evidence—internal audit, external assessment, or regulatory review—supports management's statements about our resilience?

## Bringing it together: resilience as the governance outcome that survives imperfect security

The core thesis of cybersecurity governance is that organizations cannot eliminate uncertainty, but they can manage it diligently and defensibly. Incident management, business continuity, disaster recovery, and operational resilience are therefore not ancillary operations topics. They are the governance mechanisms that prevent residual risk from becoming existential failure. Preventive controls reduce the likelihood that harm occurs; resilience determines how much harm an incident can cause and how quickly the organization recovers its ability to serve.

This chapter has argued that these capabilities must be governed as one system. A major incident requires two interlocking teams, one that regains control of the environment and one that protects the enterprise. It requires a lifecycle that leaders can navigate—stabilize, clarify, decide, recover, and learn—with decision rights and evidence discipline established in advance. It requires a notification process that can run several regulatory and contractual clocks at once. And it requires recovery objectives that cascade from business tolerances set by leadership, not from technical capabilities that leadership is asked to accept. For business technology managers, this means translating tolerances into recovery requirements, challenging objectives that have never been demonstrated, and insisting that exercise findings and expired exceptions are closed rather than tolerated.

Ransomware has made these requirements concrete. It has shown that backups are not recoverability, that restoring service is not restoring trust, that identity systems are part of the recovery critical path, and that most extortion decisions are forced by recovery gaps created long before the attack. It has also shown that dependency is systemic: the resilience of an organization is bounded by the resilience of the suppliers, platforms, and providers on which it relies. Regulators have responded by shifting attention from plans to outcomes, as OSFI Guideline E-21 illustrates with its focus on critical operations, tolerances for disruption, and severe but plausible scenario testing.

In an environment of persistent ransomware and systemic dependency, an organization's credibility is increasingly measured not by whether it experiences incidents—often outside its control—but by how it prepares, decides, communicates, restores, and learns. When resilience is treated as a continuous executive discipline rather than a binder of plans, cybersecurity becomes governable under crisis conditions, and the organization gains the capacity to absorb shocks without losing its mission, its trust, or its strategic viability.

## Chapter summary

Every organization that depends on digital services will eventually face an incident too large for the security operations centre to absorb alone. At that scale, an incident becomes a business crisis whose decisive questions are managerial: who decides, on what evidence, within what authority, and with what accountability. The chapter argues that incident management, business continuity, disaster recovery, and resilience must be governed as one enterprise capability, with two interlocking teams, predefined decision rights, notification processes that can run several clocks at once, and recovery objectives that cascade from business tolerances. Ransomware has made this concrete: backups are not recoverability, and restoring service is not restoring trust.

### Key points to retain

- Major incidents are executive crises that turn on three questions: what business harm is occurring, who has authority to bind the organization, and how diligence will be demonstrated afterwards.

- Containment, disclosure, financial, extortion, and resumption authority should be assigned with named alternates, and escalation thresholds should be expressed in business terms linked to risk appetite.

- The incident response team regains control of the environment while the crisis management team protects the enterprise; a designed interface prevents both blending and fragmentation.

- The executive lifecycle runs through stabilization, situational clarity, enterprise decision-making, recovery and reconstitution, and accountability and improvement, aligned with NIST SP 800-61 Rev. 3 and CSF 2.0.

- After-action reviews must identify managerial causes, such as unclear ownership or expired exceptions, and be reported to a body with authority to act.

- Legal discipline starts in the first hour: evidence preservation, privilege-aware investigation design, governed communications, pre-arranged third parties, and a decision log kept by a named scribe.

- Notification regimes such as Law 25, PIPEDA, OSFI, SEC Item 1.05, NIS2, and DORA have different triggers and clocks, so organizations need a maintained notification matrix and documented rationale.

- Ransomware readiness is recoverability readiness; the most effective way to avoid an impossible extortion decision is to invest early in recovery that makes payment unnecessary.

- Business continuity is the managerial spine of resilience: the BIA states tolerances, and MTPD, RTO, RPO, and MBCO cascade from the business down to technology.

- Disaster recovery must reconstitute trust through tiered recovery, tested restoration at scale, clean recovery, and identity and key recovery.

- Operational resilience assumes compromise, governs systemic and concentration dependency, and, as in OSFI E-21, keeps critical operations within tolerances under severe but plausible scenarios.

- Resilience must be measured and exercised: the most honest metric is the gap between promised and demonstrated recovery time, and exercise findings must be funded and closed.

### Key terms

- **Incident response team (IRT)** — The operational team that triages, investigates, contains, eradicates, restores, and collects evidence, providing reliable facts to the rest of the organization.

- **Crisis management team (CMT)** — The executive team that governs a major incident as an enterprise risk event, setting priorities, directing legal and communications response, and taking reserved decisions.

- **Escalation threshold** — A predefined business-level criterion, such as a critical service outage or an extortion demand, that moves an incident from technical handling to executive crisis management.

- **Decision log** — A record of each material crisis decision: what was decided, when, by whom, on what facts, which alternatives and risks were considered, and which actions followed.

- **Confidentiality incident** — Under Québec Law 25, unauthorized access to, use of, communication of, or loss of personal information; every one must be entered in the register.

- **Real risk of significant harm (RROSH)** — The PIPEDA threshold that makes a breach of security safeguards reportable to the Office of the Privacy Commissioner of Canada and to affected individuals.

- **Notification matrix** — A maintained list of each applicable regime and material contract with its trigger, threshold, recipient, deadline, and owner.

- **Recoverability** — The demonstrated ability to restore critical services and data within defined objectives, at scale, with validated integrity, under adverse conditions and without relying on the attacker.

- **Business impact analysis (BIA)** — The analysis that maps critical processes, dependencies, and the time-phased harm of interruption, and where business owners state their tolerances.

- **Maximum tolerable period of disruption (MTPD)** — The time after which not resuming a product, service, or activity becomes unacceptable; a business judgment anchored in risk appetite.

- **Recovery time objective (RTO)** — The period within which an activity or its supporting resources must be resumed, set shorter than the MTPD to leave a margin.

- **Recovery point objective (RPO)** — The point in time to which data must be restored, which defines acceptable data loss and how often data must be protected.

- **Minimum business continuity objective (MBCO)** — The minimum acceptable level of service during a disruption, recognizing that degraded service may be tolerable when no service is not.

- **Clean recovery** — Rebuilding systems from trusted images in a segregated environment and validating them before reconnection, to prevent reinfection.

- **Tolerance for disruption** — Under OSFI Guideline E-21, the level of disruption a critical operation can withstand, against which severe but plausible scenarios are tested.

## Review questions

### Essay questions

1.  The chapter argues that 'in a crisis, an organization performs at the level of its governance preparation, not at the level of its technical tools.' Evaluate this claim using the three executive questions, the five kinds of crisis authority, and the Saint-Laurent Medical Supply scenario.

2.  Analyze the two failure modes the chapter identifies, blending and fragmentation of the incident response and crisis management teams. Design an interface between the two teams for a mid-sized organization and explain how you would test it.

3.  A Québec-headquartered company with Ontario customers and a U.S. listing suffers a data-theft incident. Using the chapter, explain which notification regimes may apply, how their triggers and clocks differ, and how the organization should govern the overlap.

4.  Apply the chapter to an organization of your choice. Identify two critical services, set an MTPD, RTO, RPO, and MBCO for each, and evaluate whether its current recovery capabilities could meet them in a ransomware scenario.

5.  Evaluate the chapter's claim that 'most extortion decisions are forced by recovery gaps created long before the attack.' Propose the board-level policy, decision process, and recovery investments that would reduce the pressure to pay.

### Multiple-choice questions

Choose the best answer. Answers are listed in the appendix "Answers to multiple-choice questions".

**1.** According to the chapter, at what point does a cybersecurity incident become an executive crisis rather than a technical event?

> A. When residual risk materializes at a scale the security operations centre cannot absorb alone
>
> B. When the incident response team has been working on it for more than one shift
>
> C. When the attacker is confirmed to be a state-sponsored threat actor
>
> D. When the incident has been assigned a CVSS v4.0 score above nine

**2.** What does the chapter identify as the usual governance failure regarding crisis decision-making?

> A. The wrong executive consistently makes decisions that should belong to the CISO
>
> B. No one is sure who should decide, so decisions are delayed or later contested
>
> C. Boards insist on taking every operational decision during the incident
>
> D. Decision-makers rely too heavily on documented authority and rehearsed criteria

**3.** Which of the following is an escalation threshold expressed in business terms, as the chapter recommends?

> A. More than fifty endpoint detection alerts in one hour
>
> B. Detection of malware with a known signature on a laptop
>
> C. Interruption of a critical service beyond a defined period
>
> D. A firewall rule change made outside the change window

**4.** Which statement best describes the different purposes of the IRT and the CMT?

> A. The IRT handles external communications; the CMT performs forensic investigation
>
> B. The IRT sets business priorities; the CMT carries out containment and eradication
>
> C. The IRT reports to the board; the CMT reports to the security operations centre
>
> D. The IRT regains control of the environment; the CMT protects the enterprise while that happens

**5.** A technical team restores the systems it considers important while the business needed others first, and executives issue statements they later retract. Which failure mode does this illustrate?

> A. Blending of the two teams into one overloaded structure
>
> B. Excessive reliance on a single source of truth
>
> C. Over-use of out-of-band communication channels
>
> D. Separation of the two teams without integration

**6.** Which element of the IRT–CMT interface ensures that some decisions, such as issuing an external statement, can be taken only by the executive team?

> A. Battle rhythm
>
> B. Liaison role
>
> C. Reserved decisions
>
> D. Out-of-band communications

**7.** In the chapter's five executive phases, what is the main objective of situational clarity?

> A. Complete certainty about the attacker's identity before any decision is taken
>
> B. Coherent decision-making under uncertainty, using confirmed facts, hypotheses, and confidence levels
>
> C. Restoring every affected system before informing executives of the incident
>
> D. Producing the funded remediation roadmap approved by the board

**8.** Why does the chapter insist that after-action reviews identify managerial causes and be facilitated by someone independent of the response?

> A. Managerial causes implicate people in the room, and reviews finding only technical causes produce a patch
>
> B. Regulators require all after-action reviews to be performed and signed by external auditors
>
> C. Technical causes are rarely real and should therefore be left out of the review report
>
> D. Independent facilitators are the only people who can assign disciplinary penalties to staff

**9.** What is the most significant conceptual change in NIST SP 800-61 Rev. 3 for managers?

> A. It adds a seventh incident phase dedicated to ransom negotiation
>
> B. It maps incident response recommendations onto the six CSF 2.0 functions, including Govern
>
> C. It replaces the CSF with ISO/IEC 27035 as the basis for incident response
>
> D. It limits its scope to U.S. federal agencies and their contractors

**10.** Why might reimaging an infected server immediately to restore service conflict with legal response discipline?

> A. It may destroy the only evidence of how the attacker entered and what was taken
>
> B. It automatically triggers a public disclosure under SEC Form 8-K Item 1.05
>
> C. It voids all cyber insurance coverage regardless of the policy's terms
>
> D. It is prohibited under Québec Law 25 for any system holding personal data

**11.** Under Québec Law 25 as described in the chapter, which confidentiality incidents must be entered in the register?

> A. Only incidents that present a risk of serious injury to the persons concerned
>
> B. Only incidents involving more than a defined number of affected individuals
>
> C. Only incidents that the Commission d'accès à l'information has requested
>
> D. Every confidentiality incident, whether or not it presents a risk of serious injury

**12.** A Canadian company with a U.S. listing determines on Monday that an incident is material. Assuming no holidays, by when must it file Form 8-K Item 1.05 disclosure?

> A. Monday, the same day as the materiality determination
>
> B. Thursday, 72 hours after the incident was first discovered
>
> C. Friday, four business days after the materiality determination
>
> D. The following Monday, seven calendar days after discovery

**13.** An EU financial entity classifies an incident as major six hours after becoming aware of it. Under DORA, when is the initial notification due?

> A. Within 24 hours of classification, with no limit based on awareness
>
> B. Within 72 hours of awareness, as for NIS2 incident notifications
>
> C. Within four hours of classification, and no later than 24 hours after awareness
>
> D. Within four business days of determining that the incident is material

**14.** According to the chapter, what is the most effective way to avoid an impossible extortion decision?

> A. Pre-negotiate a payment ceiling with ransomware brokers before any attack
>
> B. Purchase cyber insurance that reimburses any ransom paid by the organization
>
> C. Delegate the payment decision to an experienced external negotiator
>
> D. Invest, well before the incident, in recoverability that makes payment unnecessary

**15.** Which statement about the relationship between MTPD and RTO is consistent with the chapter?

> A. The RTO must be shorter than the MTPD, leaving a margin for the unexpected
>
> B. The RTO must be longer than the MTPD to allow full forensic investigation
>
> C. The RTO and MTPD should be equal so that resources are not wasted
>
> D. The MTPD is set by IT after the RTO has been technically validated

**16.** A hospital accepts that it can operate for a time on paper charting with reduced services rather than full electronic records. Which recovery objective expresses this acceptable degraded level?

> A. Recovery point objective (RPO)
>
> B. Minimum business continuity objective (MBCO)
>
> C. Maximum tolerable period of disruption (MTPD)
>
> D. Recovery time objective (RTO)

**17.** Why does the chapter treat identity and key recovery as part of disaster recovery scope?

> A. If the directory service is compromised, every system trusting it is compromised too
>
> B. Identity systems are the cheapest component to restore after an incident
>
> C. Regulators prohibit restoring data before identity services are replaced
>
> D. Identity platforms never contain backups and must always be purchased again

**18.** How does OSFI Guideline E-21's operational resilience question differ from traditional continuity planning?

> A. It asks whether each individual system can recover within its documented RTO
>
> B. It asks whether backup copies are stored in at least two separate data centres
>
> C. It asks whether a critical operation can stay within tolerance, regardless of which component fails
>
> D. It asks whether the institution has purchased sufficient cyber insurance coverage

**19.** In the Saint-Laurent scenario, the BIA set a 72-hour MTPD for hospital fulfilment and a 48-hour RTO for the warehouse system, but recovery took six days. Why was the MTPD for critical hospital supply not breached?

> A. The attacker supplied a working decryptor that restored the warehouse system early
>
> B. Manual fulfilment of critical-item orders kept service above the MBCO while systems were rebuilt
>
> C. The hospitals agreed to extend the MTPD to six days during the first CMT meeting
>
> D. Pharmacy orders were rerouted through the hospitals' own warehouse systems

**20.** Which statement does the chapter present as the most honest resilience metric?

> A. The number of continuity plans written and approved during the year
>
> B. The percentage of employees who completed incident awareness training
>
> C. The total budget spent on backup and recovery technology this year
>
> D. The gap between the recovery time promised and the recovery time demonstrated
