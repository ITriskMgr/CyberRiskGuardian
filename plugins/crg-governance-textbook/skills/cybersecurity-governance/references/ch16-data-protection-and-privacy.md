<!-- Léger, M.-A. (2026). Introduction to Cybersecurity Governance for Business Technology Management (3rd ed., v2.51). Chapter 16: Data Protection and Privacy. CC BY-NC 4.0 -->

# Chapter 16: Data Protection and Privacy

## Learning outcomes

After completing this chapter, you should be able to:

1.  Explain what counts as personal and sensitive information and distinguish the roles of controllers, processors, and service providers.

2.  Distinguish privacy risk from cybersecurity risk and explain why a secure system can still be a privacy failure.

3.  Compare the main obligations of Québec's Law 25, PIPEDA, and the GDPR, including breach notification thresholds, timing, and enforcement.

4.  Apply the five-phase personal information lifecycle to identify privacy controls, accountabilities, and investment priorities.

5.  Evaluate minimization, retention, residency, encryption, and data loss prevention measures as privacy engineering controls.

6.  Design a privacy impact assessment process that covers triggers, proportionality, cross-border transfers, and residual risk acceptance.

7.  Recommend how to integrate privacy into an ISMS, control library, and operating model with clear roles, evidence, and key risk indicators.

In contemporary organizations, privacy risk does not live in legal texts. It lives in data flows, system configurations, logs, backups, application programming interfaces, analytics pipelines, supplier platforms, and everyday business processes. Many of the incidents that reach the news, the regulator's desk, or the board agenda are, in substance, privacy failures expressed through technology: personal information that was collected without a clear purpose, copied into a cloud environment no one inventoried, kept long after it was needed, exposed through weak access controls, or shared with a partner on terms nobody reviewed. Cybersecurity governance that ignores data protection is therefore incomplete, and privacy governance that ignores cybersecurity is little more than a set of policies waiting to be contradicted by reality.

The governance problem is that privacy obligations are expressed in the language of law and principle, while the controls that satisfy them are expressed in the language of architecture, configuration, and operations. A statute may require that personal information be destroyed once its purpose is fulfilled; somebody must still decide which systems hold that information, which retention period applies, how deletion propagates to replicas and backups, and what evidence will show that it actually happened. A regulator may require an assessment before information leaves the province; somebody must still decide who performs it, what it examines, who accepts the residual risk, and what happens when a cloud provider changes its sub-processors. Managers sit at precisely this junction. They do not need to become privacy lawyers or cryptographers, but they must ensure that obligations become decisions, that decisions have owners, and that owners can produce evidence.

This chapter builds on several earlier ones. Chapter 3 treated compliance as an evidence discipline rather than a box-ticking exercise; Chapters 4 to 6 developed scenario-based risk assessment, which applies directly to privacy risk; Chapter 10 covered the identity and access controls on which most privacy safeguards depend; Chapter 12 addressed incident response, which now carries legal notification deadlines; and Chapter 15 examined the cloud environments in which much personal information now resides. Chapters 19 and 20 later extend the discussion to third-party governance and cryptography.

The chapter treats data protection and privacy as a managerial control domain. It begins with core concepts and actors, then surveys the regulatory landscape from a Québec and Canadian vantage point, with the European Union's General Data Protection Regulation (GDPR) and international standards as essential reference points. It then follows personal information through its lifecycle, examining minimization and purpose limitation, retention and deletion, residency and cross-border transfers, encryption as a privacy control, and the monitoring and logging that make data protection observable. It devotes particular attention to privacy impact assessments, which in Québec are a legal obligation and everywhere are the most practical bridge between privacy law and system design, and it works through an illustrative scenario of a Québec organization assessing a new cloud customer-relationship platform hosted outside the province. It closes by showing how privacy obligations can be absorbed into an information security management system (ISMS) and an operating model with clear roles, metrics, and evidence.

The central thesis is simple. **Privacy risk becomes manageable when the organization explicitly governs the lifecycle of personal information and the safeguards that apply at each phase, and when it can demonstrate that it does so.** Privacy engineering is the discipline that connects lifecycle decisions to system design, configuration, monitoring, and evidence; privacy governance is what ensures those decisions are made deliberately, by the right people, within an approved risk appetite.

## Core concepts: personal information, sensitivity, and the actors

Although different instruments use different vocabularies, their underlying concepts are remarkably convergent. A manager who understands a handful of core ideas—what counts as personal information, what makes some information sensitive, who decides and who executes, and how privacy risk differs from security risk—can navigate most regimes with the help of specialists. Without these ideas, every new law or standard looks like a separate universe, and the organization ends up with overlapping programs that duplicate effort and still leave gaps.

### What counts as personal information

Under Canada's federal *Personal Information Protection and Electronic Documents Act* (PIPEDA), **personal information** is information about an identifiable individual, regardless of its form. Québec's *Act respecting the protection of personal information in the private sector*, as modernized by Law 25, uses a similar concept: any information that relates to a natural person and allows that person to be identified, directly or indirectly. The GDPR speaks of *personal data* and makes explicit that online identifiers, location data, and factors specific to a person's physical, economic, cultural, or social identity are included. ISO/IEC 29100:2024, the international privacy framework, uses the neutral expression *personally identifiable information* (PII).

The practical consequence is that the category is far broader than most business units assume. It includes names and email addresses, but also device identifiers, customer numbers, IP addresses in many contexts, call recordings, badge-swipe records, vehicle telematics, photographs, inferences produced by analytics models, and free-text notes typed by a sales representative. Information does not stop being personal because it is stored in a log file or because the organization considers it operational rather than customer data. A useful test is whether the information, alone or combined with other information reasonably available, could be linked back to a person; if so, it should be governed as personal information until a defensible analysis shows otherwise.

### Sensitive information and the reasonable expectation of privacy

Most regimes recognize that some information warrants more protection than other information. Québec's law, as amended, treats information as **sensitive** if, in the words of section 12 of the private-sector Act, "due to its nature, in particular its medical, biometric or otherwise intimate nature, or the context of its use or communication, it entails a high level of reasonable expectation of privacy." Sensitive information generally requires express consent for secondary uses and weighs heavily in privacy impact assessments and incident severity analyses. The GDPR lists "special categories" of data—including health, biometric data used for identification, genetic data, racial or ethnic origin, religious beliefs, political opinions, trade union membership, and sex life or sexual orientation—whose processing is prohibited unless a specific condition applies. PIPEDA does not define a closed list but expects safeguards and consent to be proportionate to sensitivity, and the Office of the Privacy Commissioner of Canada (OPC) has long treated health, financial, and biometric information as generally sensitive.

For governance, sensitivity is the bridge between privacy and the impact dimension of the risk triangle introduced in Chapter 1 and developed in Chapter 4. A dataset's sensitivity affects the harm that individuals could suffer, the legal consequences for the organization, and the controls that the organization's risk appetite should require. **Sensitivity should therefore be a mandatory attribute in the data classification scheme, not a judgment made informally by each project team.** When the classification scheme already distinguishes "confidential" from "restricted," managers should ensure that those labels map explicitly to privacy sensitivity, rather than to commercial confidentiality alone.

### Controllers, processors, and service providers

ISO/IEC 29100 distinguishes three actors: *PII principals* (the individuals the information is about), *PII controllers* (those who determine the purposes and means of processing), and *PII processors* (those who process information on behalf of a controller). The GDPR's controller–processor model uses essentially the same logic, adding the notion of joint controllers when two organizations jointly determine purposes and means. Québec's law speaks of the enterprise that holds the information and of the persons or bodies to which it entrusts information under a written mandate or service contract; PIPEDA expresses the same idea through the principle that an organization remains responsible for information transferred to a third party for processing.

Three managerial implications follow. First, the organization must know when it is acting as a controller (or equivalent) and when it is acting as a processor, because obligations differ: a controller must justify purposes, obtain consent where required, and answer to individuals, while a processor must follow instructions and protect what it receives. Second, the same system may involve both roles depending on whose data is processed and under which contract; a software-as-a-service provider is a processor for its customers' data but a controller for its own billing and marketing data. Third, **outsourcing processing does not outsource accountability.** Cloud and service arrangements add an actor whose controls must be governed, contracted, and evidenced, but the controller-level obligations remain. ISO/IEC 27018:2025, the code of practice for protecting PII in public clouds acting as PII processors, is built precisely around this relationship, and Chapter 19 develops the supplier-assurance mechanisms that make it work.

### Privacy risk and security risk: overlapping but distinct

Privacy and cybersecurity risk overlap heavily, but they are not identical. The NIST Privacy Framework expresses the relationship well: cybersecurity risk arises from loss of confidentiality, integrity, or availability, while privacy risk arises from data processing that can cause problems for individuals. The two intersect in what NIST calls cybersecurity-related privacy events, such as a breach that exposes customer records. But privacy problems can also arise from processing that is perfectly secure: an analytics project that infers health conditions from purchasing patterns, a monitoring tool that tracks employees beyond what is reasonable, a retention practice that keeps information for decades, or a transfer that is technically encrypted but legally unjustified.

This distinction matters for governance because a security team can be fully effective and the organization can still be non-compliant. A firewall does not make an unjustified collection lawful, and a well-run vulnerability management program does not fulfil an access request within the legal deadline. Conversely, a privacy program that relies on policy without technical safeguards will fail the first time an attacker reaches a database. The managerial answer is integration with clear boundaries: shared risk scenarios, shared inventories, and shared incident processes, but distinct accountability for the questions each discipline is best placed to answer.

| A system can be perfectly secure and still be a privacy failure; **privacy governance asks not only whether data is protected, but whether it should be processed at all, for this purpose, in this place, for this long.** |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## The regulatory landscape: Québec, Canada, the EU, and beyond

From a Canadian and Québec vantage point, managers navigate a layered regime. Provincial law, federal law, sector-specific rules, foreign laws with extraterritorial reach, contractual obligations, and voluntary standards all apply at once, often to the same dataset. The aim of this section is not to enumerate every obligation—that is the work of counsel and the privacy office—but to give managers an accurate map of the main instruments as of October 2026 and of how they converge. Because privacy law changes frequently, any program built on this map should include a mechanism for monitoring legislative developments and updating obligations registers, as discussed in Chapter 3.

### Québec: Law 25 and its phased modernization

The *Act to modernize legislative provisions as regards the protection of personal information*, assented to in September 2021 and widely known as **Law 25** (formerly Bill 64), substantially amended Québec's private-sector privacy statute as well as the law governing public bodies. Its private-sector obligations came into force in three phases, which matter because they explain why many Québec organizations are still maturing programs that were legally required years ago.

- **September 22, 2022.** The first phase made the person exercising the highest authority in the enterprise—typically the chief executive—the **person in charge of the protection of personal information** by default, with the ability to delegate the function in writing, and required the person's title and contact information to be published on the enterprise's website. It also introduced mandatory notification of **confidentiality incidents** presenting a risk of serious injury to the Commission d'accès à l'information (CAI) and to affected persons, and the obligation to keep a register of all confidentiality incidents.

- **September 22, 2023.** The second and largest phase brought in governance policies and practices that must be published in clear terms on the website; mandatory privacy impact assessments for projects to acquire, develop, or overhaul information systems or electronic service delivery involving personal information, and before communicating personal information outside Québec; confidentiality by default for technological products and services offered to the public; transparency obligations for technologies that can identify, locate, or profile a person and for decisions based exclusively on automated processing; strengthened consent rules, including express consent for sensitive information; rules on destruction and anonymization; a right to de-indexation; and a new enforcement regime of administrative monetary penalties and higher penal fines.

- **September 22, 2024.** The third phase gave individuals a right to data portability: on request, computerized personal information collected from them must be communicated in a structured, commonly used technological format, to them or to a person or body they designate.

Two regulations complete the picture. The *Regulation respecting confidentiality incidents* specifies the content of notices and of the incident register, which must be kept for five years after the enterprise becomes aware of an incident. The *Regulation respecting the anonymization of personal information*, in force since May 30, 2024, sets the process an enterprise must follow before it can treat information as anonymized, including a re-identification risk analysis, supervision by a qualified person, periodic reassessment, and a register of anonymized information.

The enforcement regime deserves attention from boards. Since September 2023 the CAI can impose administrative monetary penalties of up to \$10 million or 2% of worldwide turnover for the preceding fiscal year, whichever is greater, on enterprises; penal fines on prosecution can reach \$25 million or 4% of worldwide turnover, and directors and officers who order, authorize, or consent to an offence can be held liable as parties to it. The CAI published a general framework for applying monetary penalties in 2023, which lists factors such as the seriousness and duration of the breach, the sensitivity of the information, the number of persons affected, the remedial measures taken, and the enterprise's cooperation. Individuals also have a private right of action: where an unlawful infringement causes injury and results from an intentional or gross fault, a court must award punitive damages of at least \$1,000. **For managers, the significance of the regime is less the maximum amounts than the fact that the CAI's framework rewards demonstrable diligence—documented assessments, prompt remediation, and cooperation—which is exactly what a well-governed program produces as a by-product.**

### Federal law: PIPEDA and the long road to reform

PIPEDA continues to govern the collection, use, and disclosure of personal information in the course of commercial activities in provinces without substantially similar legislation, and in federally regulated works and businesses such as banks, telecommunications, and interprovincial transportation throughout Canada. Its obligations are expressed in ten fair information principles: accountability; identifying purposes; consent; limiting collection; limiting use, disclosure, and retention; accuracy; safeguards; openness; individual access; and challenging compliance. Since November 2018, organizations must report breaches of security safeguards that create a real risk of significant harm to the OPC and notify affected individuals as soon as feasible, and they must keep records of all breaches for at least 24 months. Québec, British Columbia, and Alberta have private-sector laws recognized as substantially similar, so PIPEDA generally does not apply to purely intra-provincial commercial activity in those provinces, though it still applies to interprovincial and international flows.

The federal reform effort has been long and uneven. Bill C-11 died on the Order Paper in 2021. Its successor, Bill C-27, would have replaced PIPEDA's private-sector rules with a *Consumer Privacy Protection Act*, created a privacy tribunal, and enacted an *Artificial Intelligence and Data Act*; it died on the Order Paper when Parliament was prorogued in January 2025. On June 15, 2026, the government introduced **Bill C-36**, which would enact a *Protecting Privacy and Consumer Data Act* to replace Part 1 of PIPEDA. According to published legal analyses (for example, by Borden Ladner Gervais in June 2026), it carries forward much of the substance of the earlier consumer privacy proposal, drops the separate tribunal in favour of a commission structure, contains no stand-alone AI statute, and proposes administrative penalties of up to the greater of \$10 million or 3% of global revenue. At the time of writing, the bill had not been enacted; PIPEDA therefore remains the operative federal law. Managers should treat the bill as a signal of direction—stronger enforcement, more explicit de-identification rules, and transparency for automated decision systems—rather than as current obligation, and should ask counsel to track its progress.

### Other provinces in brief

Outside Québec, the provincial picture is uneven, and managers of national organizations should avoid assuming that one province's rules apply everywhere. British Columbia's *Personal Information Protection Act* (PIPA) and Alberta's *Personal Information Protection Act* govern private-sector organizations in those provinces; Alberta's PIPA has required notification to its Information and Privacy Commissioner of incidents presenting a real risk of significant harm since 2010. Alberta also replaced its public-sector *Freedom of Information and Protection of Privacy Act* in June 2025 with two statutes, the *Access to Information Act* and the *Protection of Privacy Act*. Ontario has no general private-sector privacy statute, so PIPEDA applies to most commercial activity there, but its *Personal Health Information Protection Act* (PHIPA) governs health information custodians, and amendments introduced by Bill 194 brought new obligations for provincial public institutions under the *Freedom of Information and Protection of Privacy Act*—including privacy impact assessments and mandatory breach reporting to the Information and Privacy Commissioner—into force on July 1, 2025. Several provinces, including Québec, also have health-information statutes that apply regardless of the private-sector regime.

For federally regulated financial institutions, privacy obligations intersect with prudential expectations. OSFI Guideline B-10 on third-party risk management and Guideline B-13 on technology and cyber risk management do not create privacy rules as such, but they require governance over data held by third parties and over technology risk, which in practice covers much of the same ground. A bank's privacy office and its technology risk function should therefore work from shared inventories and shared third-party assessments.

### The European Union: GDPR and its global reach

The GDPR applies not only to organizations established in the EU but also to organizations outside the EU that offer goods or services to individuals in the EU or monitor their behaviour there. Many Canadian organizations—universities recruiting international students, software firms with European customers, manufacturers with EU distributors, tourism operators—fall within its scope for at least part of their processing. The GDPR combines a set of principles (lawfulness, fairness and transparency; purpose limitation; data minimization; accuracy; storage limitation; integrity and confidentiality; and accountability) with risk-based duties such as records of processing activities (Article 30), data protection impact assessments for high-risk processing (Article 35), data protection by design and by default (Article 25), and breach notification to the supervisory authority within 72 hours where feasible (Article 33). Administrative fines can reach €20 million or 4% of worldwide annual turnover for the most serious infringements.

Canada benefits from an EU adequacy decision for organizations subject to PIPEDA, which the European Commission reconfirmed in its January 2024 review of existing adequacy decisions; this simplifies transfers from the EU to Canadian commercial organizations, though it does not relieve those organizations of GDPR obligations when the GDPR applies to them directly. The EU continues to adjust its framework: in November 2025 the European Commission proposed a "digital omnibus" package that would amend certain GDPR provisions, and at the time of writing those proposals were still under negotiation. Managers should treat the GDPR's core principles as stable and its details as subject to change.

### Standards and frameworks as governance scaffolding

Laws state obligations; standards and frameworks help organizations organize themselves to meet them. Several references are particularly useful, and the figure below positions them alongside the main laws.

- **ISO/IEC 29100:2024** provides a privacy framework: terminology, actors, and eleven privacy principles (consent and choice; purpose legitimacy and specification; collection limitation; data minimization; use, retention, and disclosure limitation; accuracy and quality; openness, transparency, and notice; individual participation and access; accountability; information security; and privacy compliance).

- **ISO/IEC 29134:2023** gives guidelines for privacy impact assessment, including when to conduct one, how to plan it, and what the report should contain; it aligns well with Québec's assessment requirement and the GDPR's DPIA.

- **ISO/IEC 27701:2025** specifies requirements for a privacy information management system (PIMS). Its 2025 edition is a stand-alone management system standard, no longer an extension that presupposes ISO/IEC 27001 certification, although it remains designed to integrate with an ISMS.

- **ISO/IEC 27018:2025** provides a code of practice for public cloud providers acting as PII processors, extending ISO/IEC 27002 controls with cloud-specific privacy guidance.

- **The NIST Privacy Framework** organizes privacy risk management into functions—Identify-P, Govern-P, Control-P, Communicate-P, and Protect-P—that parallel the NIST Cybersecurity Framework. Version 1.0 dates from 2020; NIST released an initial public draft of Version 1.1 (NIST CSWP 40) in April 2025 to realign it with CSF 2.0 and add a discussion of AI-related privacy risk, and at the time of writing NIST had indicated that the final version was forthcoming.

<span id="_Toc243635568" class="anchor"></span>Figure 69: The layered privacy regulatory landscape for Québec organizations

Managers need not master every instrument; what matters is that they converge on a common operational reality: **governed data lifecycles, documented risk assessments, and demonstrable controls.** An organization that builds its program around that reality, rather than around any single statute, will find that most new obligations require adjustment rather than reinvention.

## The data lifecycle as the spine of privacy governance

To make privacy tangible, it is useful to view personal information as moving through a lifecycle. ISO/IEC 29100 explicitly advocates lifecycle thinking, urging organizations to establish privacy controls throughout the life of the information, and every major regime attaches obligations to particular phases: consent and notice at collection, purpose limitation at use, safeguards during storage, conditions on disclosure and transfer, and storage limitation at the end. The lifecycle also aligns naturally with how systems are built and operated, which makes it the most practical common language between legal, privacy, security, and engineering teams.

### Five phases, five sets of decisions

A simple model distinguishes five phases, each carrying distinct obligations and control needs.

1.  **Collection and creation.** Information enters the organization through forms, applications, sensors, partners, or is created internally by employees and systems. Decisions here concern purpose, necessity, notice, consent, and the sources from which information is obtained.

2.  **Use and internal sharing.** Information is processed for the purposes identified and shared among teams and systems. Decisions concern access rights, secondary uses, analytics, profiling, and automated decision-making.

3.  **Storage and replication.** Information is stored in databases, file shares, collaboration platforms, logs, caches, data lakes, and backups, often in several copies. Decisions concern classification, encryption, location, and the number and placement of copies.

4.  **Disclosure and transfer.** Information is communicated to service providers, partners, regulators, or affiliates, sometimes across borders. Decisions concern contracts, transfer assessments, sub-processors, and monitoring.

5.  **Retention, archiving, and disposal.** Information is retained for defined periods, archived, and eventually destroyed or anonymized. Decisions concern retention schedules, deletion mechanisms, legal holds, and evidence of disposal.

The figure below summarizes the lifecycle and the principal privacy controls that apply at each phase. The cycle is not merely a teaching device: retention decisions shape collection, and storage-phase incidents often reveal collection-phase failures.

<span id="_Toc243635569" class="anchor"></span>Figure 70: The personal information lifecycle and its privacy controls

For governors and managers, the lifecycle framing supports several tasks. It provides a checklist for *where* risk can appear, such as a backup regime that silently undermines deletion or an analytics extract that escapes the controls of its source system. It clarifies *who* is accountable at each stage: project owners at collection, data and system owners during use and storage, procurement and the privacy office at disclosure, and records management at disposal. And it allows more precise investment decisions, for example whether to prioritize data loss prevention at the disclosure stage or automated deletion at end of life, because each investment can be tied to a specific phase and a specific risk scenario.

### Privacy by design and by default

Québec's Law 25 and the GDPR both require that privacy be considered from the outset of projects and reflected in default settings. The GDPR's Article 25 requires data protection by design and by default; Québec's law requires that technological products and services offered to the public be configured, by default, to provide the highest level of confidentiality without any intervention by the person concerned (browser cookies being a recognized exception), and requires that functions allowing a person to be identified, located, or profiled be disclosed and, in practice, activated only by the person's choice.

**Privacy by design** is often presented as a philosophy, but for managers it is better understood as a set of gates in the project and change lifecycle. A project charter should identify whether personal information is involved; an architecture review should examine data flows, minimization, and residency; a privacy impact assessment should be completed before significant commitments are made; and a go-live checklist should confirm that defaults, retention settings, logging, and notices are in place. **Privacy by default** is a configuration decision: the default sharing setting, the default retention period, the default visibility of a profile, and the default state of tracking features. Configuration baselines (Chapter 14) are the natural vehicle for enforcing these defaults consistently across systems.

### Inventories, maps, and records of processing

None of this works without knowing where personal information is. Records of processing activities under GDPR Article 30, the inventories implicit in Québec's governance policy requirement, and the CAI's recommendation that organizations map their information as part of impact assessments all point to the same foundation: a maintained inventory of personal information, the systems that hold it, the purposes it serves, the parties it is shared with, its location, and its retention period.

Many organizations build such an inventory once, as a large spreadsheet, and then let it decay. A more durable approach links the privacy inventory to existing governance artifacts: the configuration management database and asset inventory (Chapter 7), the cloud resource inventory (Chapter 15), the supplier register (Chapter 19), and the application portfolio. The governance test is simple: for any system that holds personal information, can the organization name its owner, its purposes, its locations, and its retention rule within a day? If not, the inventory is not yet a governance instrument.

## Data minimization and purpose limitation in practice

Most modern privacy regimes converge on two core principles. Organizations should collect and process only what is necessary, and they should use information only for explicit, legitimate purposes that individuals can reasonably expect. ISO/IEC 29100 expresses these as purpose legitimacy and specification, collection limitation, and data minimization. Québec's law requires that an enterprise collect only the information necessary for the purposes determined before collection, and inform persons of those purposes; the GDPR requires that personal data be adequate, relevant, and limited to what is necessary; PIPEDA's limiting collection principle says much the same. These principles sound modest, but they are among the most powerful risk-reduction levers available to managers, because information that was never collected cannot be breached, misused, or subpoenaed.

### From principle to design constraint

From a privacy engineering perspective, minimization and purpose limitation translate into concrete design and configuration choices. The following are among the most common, and each can be written into architecture standards and procurement criteria.

- **Lean collection interfaces.** Forms, application programming interfaces, and event schemas should be designed so that unnecessary fields are never collected, and optional fields should be clearly marked as optional rather than silently required.

- **Purpose-bound datasets.** Analytics and reporting should use aggregated, pseudonymized, or sampled views rather than "just in case" copies of entire production databases, and extracts should carry metadata about their purpose and expiry.

- **Segregation of sensitive attributes.** Data models should store particularly sensitive attributes, such as health information, government identifiers, or financial details, separately from general profile data, with stricter access controls and logging.

- **Minimal logging of content.** Application and security logs should capture what is needed for troubleshooting and investigation without recording full payloads, passwords, or unnecessary personal details—a frequent and avoidable source of exposure.

- **Free-text discipline.** Free-text fields in customer-service and case-management tools attract sensitive information that no one intended to collect; guidance, field-level warnings, and periodic review reduce the problem.

For governance, the key message is that **minimization is not a slogan but a design constraint** that should appear in solution architecture reviews, procurement criteria, data modelling decisions, and the acceptance criteria of development teams. Every new collection should be traceable to a documented purpose in the privacy inventory, so that "why do we collect this?" always has an owner and an answer.

### Consent, purposes, and secondary use

Purpose limitation becomes most difficult when an organization wants to reuse existing information for a new purpose: training a machine-learning model on customer interactions, combining loyalty data with web analytics, or sharing information with a partner for joint marketing. Québec's law permits use for a new purpose without consent only in limited circumstances, such as when the new purpose is consistent with the original one, when the use is clearly for the benefit of the person, or when it is necessary for study, research, or statistics and the information is de-identified; otherwise, consent is required, and it must be express for sensitive information. The GDPR requires a lawful basis for each purpose and a compatibility assessment for further processing. PIPEDA requires meaningful consent appropriate to the sensitivity of the information and the reasonable expectations of the individual, as elaborated in the OPC's guidance on meaningful consent.

The managerial implication is that new uses of existing data should pass through the same governance gate as new collections. Data science and AI initiatives are the most frequent source of unreviewed secondary use, and Chapter 18 discusses the specific governance of AI systems. Here, the essential point is that **access to data for experimentation should not be confused with authorization to use it for a new purpose.** A sandbox with real customer data is a processing activity like any other.

### Pseudonymization, de-identification, and anonymization

Three related techniques are often confused, and the confusion has legal consequences. **Pseudonymization** replaces direct identifiers with tokens or codes while preserving the ability to re-link records using additional information held separately. It is a valuable safeguard, but under the GDPR pseudonymized data remain personal data, and Québec's law treats de-identified information—information that no longer allows the person to be directly identified—as personal information still subject to the law. **Anonymization**, by contrast, means that the information irreversibly no longer allows the person to be identified, directly or indirectly; in Québec it must be carried out according to generally accepted best practices and the criteria set by the 2024 anonymization regulation, and only then does the information leave the scope of most obligations.

The anonymization regulation is a good example of how privacy law now prescribes process rather than outcome alone. Before anonymizing, the enterprise must establish serious and legitimate purposes for using the anonymized information; the process must be supervised by a qualified person; the enterprise must assess re-identification risk against criteria of individualization, correlation, and inference; it must select techniques and protective measures, verify that residual risk is very low, reassess periodically, and keep a register. "We anonymized it" is therefore a claim that must be backed by documented analysis. Standards such as ISO/IEC 27559:2022 on privacy-enhancing data de-identification frameworks can help structure that analysis.

## Retention, archiving, and deletion: ending the "keep everything forever" culture

Many organizations struggle with retention. Legal and business requirements sometimes demand long retention—tax records, financial statements, employment files, clinical records, evidence required for litigation—while other requirements, such as storage limitation under PIPEDA, Québec's law, and the GDPR, demand that information be destroyed or anonymized once its purposes are fulfilled. The default in most organizations is neither: information simply accumulates because storage is cheap, because deletion is technically awkward, and because no one is accountable for removing it. That default is expensive: hoarded information increases the impact of every breach, slows access requests, enlarges litigation discovery, and undermines the credibility of privacy statements.

Québec's law (section 23) requires that, once the purposes for which personal information was collected or used have been achieved, the enterprise destroy it or anonymize it "to use it for serious and legitimate purposes, subject to any preservation period provided by an Act"; and governance policies must address retention and destruction. **For many organizations, the most significant privacy risk reduction available in the short term is not a new security tool but the disciplined deletion of information they no longer need.**

### Retention schedules as governance instruments

Practical retention governance requires four elements.

1.  **Retention schedules.** A documented mapping between categories of records—including records held in digital systems, not just paper files—and retention periods grounded in legal, regulatory, contractual, and business requirements. Schedules should be owned by records management or legal, approved at an appropriate level, and reviewed periodically.

2.  **System-level implementation.** Applications, databases, collaboration platforms, and log systems should be configured to enforce retention where technically feasible, through time-based deletion jobs, lifecycle policies on storage buckets, automatic log rotation, and archiving mechanisms that move information out of production while preserving access controls.

3.  **Disposal and de-identification design.** Clear criteria should state when information must be permanently destroyed and when robust anonymization is acceptable, recognizing that pseudonymized or merely de-identified information remains personal information.

4.  **Evidence and auditability.** The organization should be able to show, for critical categories of personal information, that retention rules are applied in practice, including reports of deletion jobs, exceptions, and their approval.

ISO/IEC 29134 emphasizes that impact assessments should consider retention and the feasibility of deletion or anonymization as mitigations. In practice, this means that a project should not be approved with an undefined retention period; "to be determined" is a common entry in project documentation and a reliable predictor of indefinite retention.

### Backups, replicas, and the deletion problem

Deletion is harder than it appears because modern systems copy information widely. A customer record may exist in the production database, a read replica, a data warehouse, an analytics extract, a search index, a cache, a customer-service tool, an email thread, a file export on a laptop, and several generations of backups. Deleting the primary record does not delete the copies, and backups are usually designed precisely to resist deletion.

The pragmatic approach is to define, for each class of system, what deletion means and how quickly it propagates. For backups, many organizations adopt a position that deleted information will age out of backups within a defined period, that restored backups will be re-purged before use, and that backups are protected by strong access controls in the meantime; this position should be documented and reflected in notices. For derived datasets, lineage metadata and expiry dates help. For unstructured repositories, periodic review campaigns supported by discovery tools are often the only realistic option. **The objective is not perfection but a defensible, documented, and consistently applied approach.**

### Legal holds, exceptions, and shared accountability

Retention schedules must accommodate exceptions. Litigation, regulatory investigations, and audits may require that deletion be suspended for specific information, and the organization must be able to apply and release such holds without suspending deletion everywhere. Holds should be documented, scoped, approved by legal, and reviewed—the same discipline the book recommends for risk exceptions with expiry dates.

In governance terms, retention is a shared responsibility among legal, records management, IT, data and system owners, the person in charge of the protection of personal information, and the CISO. It often requires investment in data discovery, classification, and lifecycle automation, which competes with other priorities. Framing that investment in terms of risk scenarios helps: a scenario in which an attacker exfiltrates a decade of inactive customer records has a materially higher impact than one involving two years of active records, and the difference is a quantifiable argument for deletion that Chapter 17's investment methods can support.

## Data residency and cross-border transfers

Personal information rarely stays inside one jurisdiction. Cloud services replicate data across regions, support teams work from several countries, software providers rely on sub-processors around the world, and even a modest organization may use dozens of software-as-a-service applications hosted outside its province. Yet both Québec and the EU place particular emphasis on communications outside their territory, and federal law expects organizations to remain accountable wherever information travels. Residency is therefore one of the areas where privacy governance and cloud architecture (Chapter 15) are most tightly coupled.

### Québec: an assessment before information leaves the province

Under Québec's law, before communicating personal information outside Québec, an enterprise must conduct a privacy impact assessment. The assessment must take into account, in particular, the sensitivity of the information, the purposes for which it will be used, the protection measures—including contractual ones—that will apply, and the legal framework applicable in the destination jurisdiction, including its personal information protection principles. Under section 17, the information may be communicated only if the assessment establishes that it "would receive adequate protection, in particular in light of generally recognized principles of personal information protection," and the communication must be the subject of a written agreement that takes into account the results of the assessment and, where applicable, the terms agreed to mitigate the risks identified. The same logic applies when the enterprise entrusts a service provider outside Québec with collecting, using, communicating, or keeping information on its behalf.

Two features of this rule matter for managers. First, "outside Québec" includes the rest of Canada: a transfer to a data centre in Ontario triggers the assessment as surely as one to Virginia or Dublin, though the analysis of the destination's legal framework will differ. Second, the CAI's guide ties the obligation to communications that take place after September 22, 2023, and recommends an assessment as good practice even for projects finalized before that date; continuing transfers under older arrangements are therefore widely treated as within scope. Many Québec organizations therefore have a backlog of existing software-as-a-service relationships that should have been assessed. The proportionality principle applies—the scope of the assessment should be proportionate to the sensitivity, purpose, quantity, distribution, and medium of the information—so a low-risk transfer of business contact information can be assessed briefly, while a transfer of health or financial information warrants depth.

### The EU: adequacy, safeguards, and the Data Privacy Framework

The GDPR permits transfers of personal data to countries benefiting from an adequacy decision, such as Canada for PIPEDA-covered organizations. Elsewhere, transfers require appropriate safeguards such as the European Commission's standard contractual clauses or binding corporate rules, supplemented, since the *Schrems II* judgment of the Court of Justice of the EU in 2020, by a case-by-case assessment of whether the destination's laws, particularly on government access, undermine those safeguards.

For transfers to the United States, the **EU–U.S. Data Privacy Framework** has provided an adequacy basis since the European Commission's decision of July 10, 2023, for U.S. organizations that self-certify to it. On September 3, 2025, the EU General Court dismissed a challenge to that decision in *Latombe v Commission* (Case T-553/23); the applicant lodged an appeal to the Court of Justice in October 2025, which remained pending at the time of writing. The framework is therefore in force but not beyond legal challenge, and its two predecessors—Safe Harbour and the Privacy Shield—were both invalidated. **Prudent organizations relying on the framework maintain a fallback, typically standard contractual clauses and a transfer assessment, so that a future judgment does not leave critical transfers without a legal basis.**

### Federal accountability and architecture choices

PIPEDA does not prohibit cross-border transfers. It requires organizations to remain accountable for information transferred to third parties for processing, to use contractual or other means to provide a comparable level of protection, and to be transparent with individuals that their information may be processed in another jurisdiction and may be accessible to that jurisdiction's courts and authorities.

From a privacy engineering perspective, residency and cross-border governance translate into a set of architecture and contract decisions.

- **Region selection.** Primary and backup data should be hosted deliberately, using Canadian or other appropriate cloud regions where requirements or risk appetite warrant, and the decision should be recorded.

- **Metadata, telemetry, and support access.** Many services replicate logs, telemetry, or support tickets globally, or allow support staff abroad to access customer environments, unless configured or contracted otherwise; these flows are transfers too.

- **Sub-processor governance.** Contracts should require notice of sub-processor changes and give the organization a meaningful opportunity to object, and the privacy office should have a process for reviewing those notices.

- **Segregation by population.** Systems should be designed so that datasets subject to different regimes—Québec residents, EU residents, health information—can be located, restricted, and reported on separately when their conditions differ.

- **Encryption as a risk reducer, not a legal substitute.** Encryption with keys controlled by the organization reduces practical exposure, but it does not by itself establish adequate protection where the provider can access keys or where the information is processed in clear text.

ISO/IEC 27018 stresses transparency about where PII is stored and processed and encourages cloud providers to support customer requirements for geographic restrictions and transfer controls. For managers, **the governance challenge is to ensure that cross-border decisions are made consciously, during design and procurement, rather than discovered after deployment** when a contract has been signed and a migration completed.

## Encryption and key management as privacy controls

Encryption is one of the most visible technical safeguards in data protection, and regulators frequently refer to it. Properly encrypted information whose keys were not compromised generally reduces the risk of harm in breach assessments, and encryption is a recurring expectation in security safeguards provisions. Yet encryption is also widely misunderstood: it protects information in particular states and against particular threats, and its value depends almost entirely on how keys are managed. Chapter 20 treats cryptography and key management in managerial terms; this section highlights only the privacy-specific points.

### What encryption does and does not achieve for privacy

Three questions determine the privacy value of encryption. The first is *where* it is applied: at rest (databases, file systems, storage buckets, backups, laptops), in transit (TLS for web and application traffic, secured interconnections), and, for some specialized workloads, in use through confidential computing. The second is *who controls the keys*: a cloud provider's default encryption protects against theft of physical media but not against access by the provider or by an attacker who compromises an administrative account, whereas customer-managed or externally held keys change the threat model and the organization's own responsibilities. The third is *which threats remain*: encryption at rest does nothing against an authorized user who exports records or an application compromised while data is decrypted.

**Encryption is therefore a necessary layer but never a substitute for minimization, access control, monitoring, or a legitimate purpose.** A dataset that should not exist is not made acceptable by encrypting it.

### Governance expectations for managers

From a privacy standpoint, a small number of expectations cover most needs; Chapter 20 develops the key management practices behind them. Encryption requirements should be tied to data classification, so that the most sensitive datasets receive stronger and more granular protection, including field-level encryption or tokenization where appropriate. Separation of duties should prevent administrators of storage systems from unilaterally accessing key material. For cross-border and cloud workloads, the decision to require customer-managed or externally held keys should be recorded in the privacy impact assessment, because it changes the residual risk of the transfer. Finally, retention and encryption decisions interact: information collected today and kept for decades may outlive the algorithms that protect it, which is one reason the post-quantum transition discussed in Chapter 20 is also a privacy concern.

## Data loss prevention, monitoring, and logging for personal information

Data protection is as much about where information *goes* as about where it is stored. Many privacy incidents involve no sophisticated attack at all: a spreadsheet emailed to the wrong recipient, a file-sharing link set to "anyone with the link," a customer list uploaded to a personal cloud account, or an employee browsing records out of curiosity. Controls that make the movement and use of personal information observable are therefore central to privacy governance. At the same time, those controls themselves process personal information—about employees, customers, and correspondents—and must be designed with proportionality in mind.

### Data loss prevention as a layered capability

**Data loss prevention** (DLP) tools attempt to detect and prevent unauthorized transmission or copying of sensitive information across email, web traffic, endpoints, collaboration platforms, and cloud services. Comprehensive DLP programs can be costly and noisy, and many organizations have purchased tools whose rules were never tuned beyond the default. A layered approach usually delivers more value for the investment.

- **Classification first.** Clear policies and labels that identify what counts as restricted or highly confidential personal information give DLP rules something meaningful to enforce; without them, rules default to pattern matching on card or social insurance numbers and miss most of the information that matters.

- **Native controls in core platforms.** Email and collaboration suites can warn before external sharing, block sharing of labelled content outside the organization, expire external links, and detect common sensitive patterns; these controls are often already licensed.

- **Visibility into cloud use.** Cloud access security brokers and the posture management capabilities discussed in Chapter 15 help detect unsanctioned applications and misconfigured storage that expose personal information.

- **Coaching before blocking.** User-facing prompts that explain why an action is risky change behaviour and reduce false positives; hard blocks are best reserved for high-confidence, high-impact rules.

- **Measured tuning.** DLP alerts should feed a triage process with service levels, and the false-positive rate should be tracked as a metric, because a tool that generates thousands of ignored alerts is compliance theatre rather than risk reduction.

### Logging access to personal information

Access to sensitive datasets and to administrative functions must be logged in a way that cannot easily be altered by the users being monitored, and logs must be retained long enough to support incident investigations, regulatory inquiries, and, where relevant, requests from individuals who want to know who has accessed their information. In health care, for example, auditing of access to electronic health records is a primary control against "snooping" by authorized staff.

Logs are also personal information. They should be minimized (recording who accessed which record and when, without copying the record's content), protected with access controls and integrity mechanisms, and subject to retention rules like any other dataset. The security operations practices described in Chapters 7 and 8 should therefore include privacy requirements for the logging pipeline itself.

### Proportionate monitoring of employees

Monitoring employees raises distinct issues. Indiscriminate surveillance—recording keystrokes, capturing screens continuously, reading personal communications, or tracking location outside working hours—may itself create privacy, labour-law, and trust risks, and in Québec it engages the *Charter of human rights and freedoms* as well as the private-sector privacy statute. Monitoring programs should therefore be justified by documented purposes, limited to what is necessary, disclosed to employees, and subject to an impact assessment when they involve new technology. Ontario, for its part, requires employers with 25 or more employees to maintain a written electronic monitoring policy. The governance principle is consistent with the rest of this chapter: **monitoring is a processing activity that must itself pass the tests of necessity, proportionality, and transparency.**

### Incident registers and breach notification

Québec's requirement to keep a register of all confidentiality incidents—not only those that are notified—is a concrete example of privacy obligations intersecting with incident response. The register must record, among other things, a description of the information involved, the circumstances, the date or period of the incident and when the enterprise became aware of it, the number of persons concerned, the assessment of the risk of serious injury, and the measures taken; it must be kept for five years and provided to the CAI on request. Incidents presenting a risk of serious injury must be notified promptly to the CAI and to affected persons. The assessment of that risk considers the sensitivity of the information, the anticipated consequences of its use, and the likelihood that it will be used for injurious purposes, and the person in charge of the protection of personal information must be consulted.

Comparable obligations apply under other regimes, with different thresholds and timelines, as the table below summarizes. The practical consequence is that the incident response plan described in Chapter 12 must include a privacy assessment track running in parallel with technical containment, with pre-agreed criteria, templates, decision rights, and a clock.

| **Regime**                | **Notification threshold**    | **Timing**                                   | **Record-keeping**                 |
|---------------------------|-------------------------------|----------------------------------------------|------------------------------------|
| Québec private-sector law | Risk of serious injury        | Promptly, to CAI and persons                 | Register of all incidents, 5 years |
| PIPEDA                    | Real risk of significant harm | As soon as feasible, to OPC and individuals  | All breaches, 24 months            |
| GDPR                      | Risk to rights and freedoms   | Within 72 hours to authority, where feasible | All breaches documented            |

Two governance points follow. First, an organization operating in several jurisdictions should not run separate incident processes for each; it should run one process with jurisdiction-specific decision points, so that a single investigation produces the facts that each regulator requires. Second, the incident register is not merely a compliance artifact. Analyzed over time, it is one of the richest sources of key risk indicators available to the privacy program: recurring misdirected emails, repeated access violations in a particular department, or incidents traced to a single supplier reveal where controls are failing.

## Privacy impact assessments: structuring privacy risk analysis

Risk-based privacy laws do not simply tell organizations to be careful; they require structured analysis before certain decisions are made. **Privacy impact assessments** (PIAs)—called *évaluations des facteurs relatifs à la vie privée* (EFVP) in Québec and data protection impact assessments (DPIAs) under the GDPR—are the mechanism through which that analysis is performed, documented, and acted upon. Done well, a PIA is the single most effective bridge between privacy law and system design, because it forces the project team, the privacy office, security, and the business owner to examine the same data flows together, early enough to change them.

### Triggers: when an assessment is required

The triggers differ by regime, and an internal methodology should encode all that apply to the organization.

- **Québec projects.** An enterprise must conduct an assessment for any project to acquire, develop, or overhaul an information system or electronic service delivery involving the collection, use, communication, keeping, or destruction of personal information. The person in charge of the protection of personal information must be consulted from the outset of the project, and the project must allow computerized information collected from individuals to be communicated in a structured, commonly used format.

- **Québec communications outside the province.** As discussed above, an assessment is required before personal information is communicated outside Québec or entrusted to a service provider there.

- **Québec research disclosures.** An assessment is also required before communicating personal information without consent for study, research, or statistical purposes.

- **GDPR high-risk processing.** A DPIA is required where processing is likely to result in a high risk to individuals' rights and freedoms, including systematic and extensive profiling with significant effects, large-scale processing of special categories, and systematic monitoring of publicly accessible areas, with supervisory authorities publishing lists of further examples.

- **Federal and other contexts.** PIPEDA does not mandate PIAs for private-sector organizations, but the OPC recommends them, and federal and many provincial public-sector bodies are required to conduct them under government policy or statute.

The CAI's guide on privacy impact assessments, most recently updated in April 2024 (version 3.1), goes beyond the statutory triggers by recommending an assessment for any project involving personal information, scaled to its risk. For most organizations that is sound advice. A lightweight screening questionnaire applied to every project, followed by a full assessment only where screening identifies material risk, keeps the burden proportionate while ensuring that nothing significant escapes review.

### Method: ISO/IEC 29134 and the CAI guide

ISO/IEC 29134:2023 provides guidance on the PIA process: preparing the assessment, describing the processing and its context, consulting stakeholders, identifying privacy risks, analyzing and evaluating them, treating them, and documenting and following up on the results. The CAI's guide follows a compatible structure and provides a report template; it recommends that risks be evaluated according to their severity for individuals and their likelihood, that each residual risk be assigned an owner, and that the assessment be revisited throughout the life of the project rather than filed once at approval. Because the Québec requirements and ISO/IEC 29134 are closely aligned, organizations can use a single internal methodology to satisfy several regimes, provided it includes jurisdiction-specific triggers, the transfer factors required by Québec's law, and the documentation the GDPR expects. The figure below summarizes the steps of such a methodology.

<span id="_Toc243635570" class="anchor"></span>Figure 71: Privacy impact assessment steps from screening to follow-up

The proportionality analysis at the heart of Québec's approach deserves attention, because it is where privacy assessments differ most from security assessments. The question is not only whether the information is protected, but whether the processing is justified: whether there is a rational link between the information and the project's objectives, whether the intrusion is minimal or no less intrusive alternative would be effective, and whether the benefits outweigh the consequences for the persons concerned. A project can fail this test even with excellent safeguards.

### Integrating PIAs with cybersecurity risk assessment

An effective PIA process is integrated with cybersecurity risk assessment rather than running in a separate silo. The threats, vulnerabilities, and mitigations are often shared; what differs is the focus on impact to individuals alongside impact to the organization. The scenario-based method developed in Chapters 5 and 6 adapts well: a privacy risk scenario names a threat source (an external attacker, an over-privileged employee, a supplier, or the organization's own processing), an event (unauthorized access, excessive collection, unexpected secondary use, unlawful transfer), the affected population and information, and the consequences for individuals and the organization. Tools such as CyberRiskGuardian can carry these scenarios alongside security scenarios, so that a single register shows both.

An effective process also has clear roles. The business owner is accountable for the project and for accepting residual risk within delegated authority; the person in charge of the protection of personal information, or a delegate in the privacy office, leads the methodology; the CISO or security architect assesses technical safeguards; legal counsel interprets obligations; procurement manages contractual terms; and, where appropriate, representatives of affected users contribute. The assessment produces a written report with a risk-based justification, including the conditions under which a project proceeds with significant residual risk and who approved it. **A PIA whose conclusions are never reflected in the design, the contract, or the risk register is documentation, not governance.**

### Illustrative scenario: a transfer assessment for a cloud CRM

Chaudière Services Résidentiels is a fictional Québec company with 600 employees that installs and maintains heating, ventilation, and air-conditioning systems for about 90,000 residential customers. Its legacy customer-relationship system, hosted on premises, is reaching end of support. The sales director proposes a widely used software-as-a-service CRM platform whose primary data centres are in the United States, with support teams in the United States, Ireland, and India and an AI assistant that summarizes customer calls. The project will migrate customer names, addresses, telephone numbers, service histories, payment status, technician notes, call recordings, and, for customers in a subsidized energy-efficiency program, household income bands used to establish eligibility.

Because the project overhauls an information system and will communicate personal information outside Québec, two assessment triggers apply. The person in charge of the protection of personal information—the chief executive, who has delegated the function in writing to the director of legal affairs—assigns a privacy analyst to lead the assessment with the security architect, the CRM product owner, and procurement. Screening classifies the project as high risk because of the volume of records, the household income information, the call recordings, and the transfer outside Canada.

The assessment produces several findings. The data inventory reveals that technician notes contain occasional health information ("customer uses oxygen equipment; prioritize heat restoration") and access codes for homes, neither of which anyone had classified. The vendor offers a Canadian data residency option for primary storage at additional cost, but support access, telemetry, and the AI assistant's processing would still occur outside Canada. The vendor is certified under the EU–U.S. Data Privacy Framework, which is irrelevant to a Québec transfer; the analysis must instead consider U.S. law on government access, the vendor's contractual commitments, its ISO/IEC 27001 and ISO/IEC 27018 certifications and SOC 2 Type II report, and its sub-processor list, which includes a speech-to-text provider. The legacy system holds records going back nineteen years, although the company's retention schedule calls for seven.

The proposed treatment combines design, contract, and governance measures. The company will purchase the Canadian residency option; migrate only records within the seven-year retention period and securely destroy the remainder after confirming no legal holds apply; move household income bands into a separate restricted object visible only to the program eligibility team; create a structured "safety flag" field to replace free-text health notes and train technicians accordingly; prohibit storing home access codes in the CRM; disable the AI call-summary feature until a separate assessment of the speech-to-text sub-processor is completed; configure field-level encryption with customer-managed keys for the income object; enable access logging exported to the company's own security monitoring platform; and negotiate a written agreement that addresses purposes, confidentiality, sub-processor notification with a right to object, incident notification within 24 hours, assistance with access and portability requests, return and deletion at termination, and audit rights.

Residual risks remain. Support personnel abroad may access customer data under the vendor's controls; U.S. legal processes could compel disclosure; and the company depends on the vendor's notification of incidents. The assessment rates these as moderate given the mitigations and recommends acceptance. Because the company's risk appetite statement requires executive approval for moderate privacy risks involving more than 50,000 individuals, the decision goes to the executive committee, which accepts the residual risk for three years with a review at contract renewal, approves the additional residency cost, and asks for a quarterly report on deletion progress and vendor incidents. The report, the agreement, the approval, and the conditions are filed as the evidence package for the system.

The assessment created value well beyond legal compliance: it removed twelve years of unnecessary records, eliminated two categories of information that should never have been collected, and kept an unreviewed AI feature from going live. Its transfer analysis depended on facts—support locations, sub-processors, telemetry flows—that only procurement and the security architect could obtain. And the final decision was a defensible risk acceptance by the right authority, with an expiry date and conditions, rather than an implicit acceptance by a project team under schedule pressure.

## Turning privacy obligations into an ISMS and cloud-security reality

Cybersecurity managers often operate within an ISO/IEC 27001-aligned ISMS and, increasingly, within cloud-centric control environments structured by the NIST Cybersecurity Framework 2.0, the CSA Cloud Controls Matrix, CIS Benchmarks, and CIS Controls v8.1. Privacy programs, meanwhile, are often run by legal or compliance teams with their own registers, policies, and tools. The result in many organizations is two parallel universes that describe the same systems in different vocabularies. The opportunity is to make privacy an explicit dimension within existing management systems and control frameworks, as Chapter 9 recommends for frameworks generally.

### ISO/IEC 27701:2025 and the privacy information management system

ISO/IEC 27701 specifies requirements and guidance for a **privacy information management system** (PIMS). The 2019 edition was written as an extension of ISO/IEC 27001 and ISO/IEC 27002, so certification presupposed an ISMS. The 2025 edition, published in October 2025, is a stand-alone management system standard that follows the harmonized structure of ISO management system standards; organizations can now be certified to ISO/IEC 27701 without holding ISO/IEC 27001 certification, although the two remain designed to integrate. Accreditation bodies were still establishing transition arrangements for certificates issued against the 2019 edition at the time of writing, so certified organizations should confirm timelines with their certification body.

For organizations with an ISMS, the practical path is integration: extend the ISMS scope and context analysis to privacy obligations, add privacy-specific roles and objectives, incorporate PIAs into risk assessment, map controller and processor controls into the Statement of Applicability, and include privacy metrics in management review. For organizations without one, the 2025 edition now offers a certifiable route in its own right; and even without certification, it structures roles, records, and accountability in a way auditors and customers recognize.

### Mapping privacy into security controls and cloud baselines

Many ISO/IEC 27002:2022 controls are directly privacy-relevant: classification of information, access control, identity management, use of cryptography, information deletion, data masking, data leakage prevention, logging, secure development, supplier relationships, and, explicitly, privacy and protection of PII. The NIST CSF 2.0 Govern function provides a home for privacy obligations within organizational context, risk strategy, roles, policy, and supply-chain risk management. An integrated control library maps each privacy requirement—from Law 25, PIPEDA, the GDPR, and contracts—to the controls that satisfy it, the owners of those controls, and the evidence they produce, so that a single test of a control supports several obligations.

For public cloud workloads, organizations can adopt ISO/IEC 27018 expectations in supplier requirements, use cloud-specific benchmarks to define baseline configurations as described in Chapters 14 and 15, and overlay privacy-specific controls: residency constraints, lifecycle and deletion policies, logging of access to personal information, restrictions on public sharing, and customer-managed encryption for sensitive datasets. Configuration policies enforced through infrastructure as code are particularly valuable, because they turn privacy-by-default requirements into controls that are tested automatically on every deployment.

### The NIST Privacy Framework alongside CSF 2.0

The NIST Privacy Framework provides a structure for organizations that prefer an outcome-based approach. Its five functions map onto privacy program activities: **Identify-P** (inventory and mapping, business environment, risk assessment), **Govern-P** (policies, roles, risk strategy, awareness), **Control-P** (data management, disassociated processing, managing individuals' requests), **Communicate-P** (transparency and data processing awareness), and **Protect-P** (security safeguards for data processing). Because Version 1.1, in draft at the time of writing, realigns the framework with CSF 2.0, organizations can use common profiles and tiers to describe current and target states across cybersecurity and privacy, and report both to the board in a consistent format. As with the CSF, the value lies in identifying gaps and prioritizing investment, not in claiming conformance.

| **Privacy should not be a parallel management system.** The most efficient programs add privacy requirements to the inventories, risk registers, control libraries, and assurance cycles the organization already operates. |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## Operating model: roles, responsibilities, and evidence

Privacy is not solely the domain of specialists. A credible operating model distributes responsibility across the organization while keeping accountability clear, and it produces evidence as a by-product of normal work. Chapter 21 develops organizational design and accountability mechanisms in general; this section applies them to privacy.

### Roles and decision rights

The core roles are consistent across most organizations, even if titles differ.

- **The person in charge of the protection of personal information.** In Québec, this is by default the person exercising the highest authority in the enterprise, who may delegate the function in writing; the role approves governance policies, is consulted on assessments and incidents, and is the public point of contact. Where the function is delegated, the delegate needs sufficient authority, independence, and resources to perform it, and the chief executive remains accountable.

- **The Data Protection Officer, where required.** Under the GDPR, certain controllers and processors must designate a DPO with expert knowledge and independence; Canadian organizations subject to the GDPR should determine whether this obligation applies to them.

- **The CISO and security architecture.** They ensure that technical safeguards—access control, encryption, logging, DLP, monitoring, secure development, and cloud configuration—are aligned with privacy risk and obligations, and they contribute the security analysis in PIAs and incident assessments.

- **Data and system owners.** They are accountable for the lifecycle of specific datasets and systems, including minimization, retention, classification, access reviews, and adherence to approved designs; they are the natural risk owners for privacy scenarios affecting their systems.

- **Procurement and vendor management.** They integrate privacy and security requirements into due diligence, contracts, and ongoing assurance, including review of audit reports, sub-processor changes, and incident notification terms (Chapter 19).

- **Legal, records management, and human resources.** They interpret obligations, own retention schedules, and govern employee information and monitoring.

The three lines model provides a useful frame. Business and technology management in the first line own and operate privacy controls; the privacy office and risk and compliance functions in the second line set methodology, challenge assessments, and monitor; internal audit in the third line provides independent assurance. A common failure is a privacy office that performs first-line work—completing assessments for project teams or maintaining inventories owners never consult—and so lacks capacity to challenge. **The privacy office should own the method and the challenge; the business should own the risk.**

### Evidence and privacy audit readiness

Privacy obligations are fundamentally evidentiary. Regulators, courts, auditors, and customers care not only about what the organization claims but about what it can show. A well-governed program can produce, for any significant system, a coherent evidence package that includes the following.

- Records of processing or inventory entries showing purposes, categories of information, recipients, locations, and retention rules.

- Privacy impact assessments, their decisions, the approvals of residual risk, and evidence that mitigations were implemented.

- Contracts and transfer assessments for service providers and cross-border communications, with sub-processor lists and review records.

- The confidentiality incident register, breach notifications, and post-incident improvements.

- Published governance policies and notices, and records of how individuals' access, correction, de-indexation, and portability requests were handled and within what time.

- Policies, standards, training records, configuration baselines, and control test results showing that privacy by design and by default operate in practice.

Managers should think of this as **privacy audit readiness**: if asked to show how a particular system complies with Law 25, PIPEDA, or the GDPR, the owner can produce the package within days rather than assembling it under pressure after a complaint or an incident.

### Metrics and key risk indicators

A privacy program, like any other governance domain, needs indicators that tell leadership whether risk is within appetite. Useful measures combine activity, coverage, and outcome.

- **Assessment coverage.** The percentage of in-scope projects and existing cross-border service relationships with a current PIA, and the age of the oldest unassessed transfer.

- **Inventory currency.** The percentage of systems holding personal information with a named owner, documented purpose, location, and retention rule reviewed within the past year.

- **Retention compliance.** The volume of records past their retention date, and the trend in deletion completed against plan.

- **Request performance.** The number of access, correction, and portability requests, the percentage completed within the legal deadline, and the number escalated as complaints.

- **Incident indicators.** Confidentiality incidents by category and root cause, the time from detection to privacy assessment, and the percentage notified within target.

- **Supplier assurance.** The percentage of high-risk processors with current assurance evidence and reviewed sub-processor changes.

As Chapter 17 emphasizes, metrics should drive decisions: an indicator that has never caused anyone to change a plan or a budget should be replaced with one that will.

## Bringing it together: privacy as a dimension of cybersecurity governance

This chapter has treated data protection and privacy as a bridge between high-level obligations and the concrete design of systems and operations. Several cross-cutting conclusions follow, and they connect privacy firmly to the governance themes developed throughout the book.

### Privacy is risk-based, lifecycle-based, and evidentiary

First, privacy is fundamentally risk-based and lifecycle-based. Laws and standards increasingly expect organizations to identify and manage the risks arising from their processing activities, not merely to publish policies. Privacy engineering provides the toolkit for doing so—minimization, purpose-bound design, retention automation, residency choices, encryption, logging, DLP, and impact assessment—and the lifecycle provides the structure that tells managers where each tool applies. Second, privacy obligations in Québec, Canada, and the EU are converging on a common vocabulary: accountability, transparency, privacy by design and by default, risk assessment, proportionality, and demonstrable safeguards. Québec's Law 25, now fully in force, is among the most demanding expressions of that vocabulary in North America; federal reform remains in progress with Bill C-36; and the GDPR continues to set the international reference point. ISO/IEC 29100, 29134, 27701, and 27018, together with the NIST Privacy Framework and CSF 2.0, provide recognized structures for expressing and evidencing that vocabulary inside the organization.

Third, privacy is evidentiary. The enforcement frameworks that now carry significant penalties reward organizations that can show what they decided, why, and with what result. The incident register, the impact assessment, the transfer agreement, the retention report, and the risk acceptance signed by the right authority are not bureaucratic overhead; they are the means by which an organization demonstrates that its decisions were defensible.

### Questions managers and boards should ask

Boards and senior managers do not need to master the details of every statute, but they should be able to ask questions that reveal whether the program is real. The following questions are a practical starting point.

- **Accountability.** Who is our person in charge of the protection of personal information, to whom has the function been delegated, and does that person have the authority and resources the role requires?

- **Inventory.** Can we identify, for every system holding personal information, its owner, purposes, locations, and retention rule, and how current is that inventory?

- **Sensitive information.** Where do we hold sensitive information—health, biometric, financial, information about minors—and are the controls on those datasets stronger than on the rest?

- **Assessments.** What percentage of new projects and existing cross-border service relationships have a current privacy impact assessment, and which material transfers have never been assessed?

- **Residual risk.** Which significant privacy risks have been accepted, by whom, on what conditions, and when do those acceptances expire?

- **Retention.** How much personal information do we hold beyond its retention period, and what is our plan and budget to reduce it?

- **Transfers.** For our most important cloud and service providers, where is our information stored and accessed, which sub-processors are involved, and what is our fallback if a transfer mechanism is invalidated?

- **Incidents.** How many confidentiality incidents did we record last year, what were their root causes, how many were notified, and how quickly did we complete the risk-of-serious-injury assessment?

- **Individuals' rights.** Are access, correction, de-indexation, and portability requests handled within legal deadlines, and what do complaints tell us?

- **New technologies.** How are AI features, analytics projects, and monitoring tools reviewed before personal information is used for a new purpose?

- **Evidence.** If the CAI, the OPC, or a major customer asked tomorrow how a particular system complies, could its owner produce the evidence package within a week?

- **Integration.** Are privacy risks recorded in the same risk register, reported through the same governance channels, and tested by the same assurance cycle as cybersecurity risks?

### The managerial challenge

The managerial challenge is not to become a privacy lawyer or a cryptography expert, but to ensure that governance mechanisms exist so that projects with privacy risk are identified and analyzed early, data lifecycles are understood and documented, technical safeguards are aligned with classification and risk, cross-border transfers and supplier relationships are governed explicitly, and evidence of compliance and good faith is generated as part of normal operations rather than as a scramble after a breach. These are the mechanisms the book has advocated throughout: clear risk owners, a defined risk appetite, scenario-based assessment, exceptions with expiry dates, metrics that drive decisions, and independent assurance.

| **The organizations that manage privacy well are rarely those with the longest policies; they are those that know where their personal information is, why they hold it, who decided, and what they can prove.** |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

In the broader arc of this book, data protection and privacy complement the preceding chapters on governance, risk, architecture, identity, and cloud security, and they prepare the ground for the chapters that follow on investment, AI, third parties, cryptography, and accountability. Cybersecurity governance that treats privacy as an integral design constraint—rather than as a separate or adversarial function—is better positioned to protect individuals, satisfy regulators, and maintain the trust on which a data-driven organization ultimately depends.

## Chapter summary

Chapter 16 argues that privacy risk lives in data flows, configurations, suppliers, and everyday processes rather than in legal texts, so privacy governance must turn legal obligations into decisions with owners and evidence. Privacy risk becomes manageable when the organization explicitly governs the lifecycle of personal information and the safeguards at each phase, and can demonstrate that it does so. The chapter maps the layered regime facing Québec organizations—Law 25, PIPEDA, other provinces, the GDPR, and ISO and NIST frameworks—then examines minimization, retention, residency, encryption, monitoring, and privacy impact assessments, and closes with ISMS integration and an operating model built on roles, metrics, and evidence.

### Key points to retain

- Personal information is any information about an identifiable individual, which includes identifiers, logs, recordings, inferences, and free-text notes far beyond what most business units assume.

- Sensitivity links privacy to the impact dimension of the risk triangle and should be a mandatory attribute of the data classification scheme.

- Outsourcing processing does not outsource accountability: controllers keep their obligations when processors and cloud providers handle information for them.

- Privacy risk and security risk overlap but differ; a system can be perfectly secure and still process data unlawfully, excessively, or for too long.

- Law 25 came into force in three phases (2022, 2023, 2024) and introduced mandatory PIAs, incident registers, portability, and penalties of up to \$10 million or 2% of worldwide turnover.

- PIPEDA remains the operative federal law while Bill C-36 is pending, and the GDPR applies to Canadian organizations that offer goods or services to, or monitor, individuals in the EU.

- The five-phase lifecycle—collection, use, storage, disclosure, and retention and disposal—is the most practical common language between legal, privacy, security, and engineering teams.

- Minimization and purpose limitation are design constraints, and information that was never collected cannot be breached, misused, or subpoenaed.

- Disciplined deletion, with retention schedules, system-level enforcement, evidence, and managed legal holds, is often the largest short-term privacy risk reduction available.

- Under Québec law, communicating personal information outside Québec, including to the rest of Canada, requires a prior privacy impact assessment and a written agreement.

- Encryption, DLP, and logging are necessary layers but never substitutes for minimization, access control, or a legitimate purpose, and they process personal information themselves.

- Privacy should be absorbed into existing inventories, risk registers, control libraries, and assurance cycles, with the privacy office owning the method and the business owning the risk.

### Key terms

- **Personal information** — Information about an identifiable individual, regardless of its form, that alone or combined with other reasonably available information could be linked back to a person.

- **Sensitive information** — Under Québec law, information that, due to its nature or the context of its use, entails a high level of reasonable expectation of privacy.

- **PII controller** — The party that determines the purposes and means of processing personal information and must justify purposes, obtain consent, and answer to individuals.

- **PII processor** — The party that processes personal information on behalf of a controller and must follow instructions and protect what it receives.

- **Person in charge of the protection of personal information** — Under Law 25, by default the person exercising the highest authority in the enterprise, who may delegate the function in writing.

- **Confidentiality incident register** — The Québec register of all confidentiality incidents, notified or not, kept for five years and provided to the CAI on request.

- **Privacy by default** — A configuration decision ensuring that products and services offered to the public provide the highest level of confidentiality without any action by the person.

- **Data minimization** — The principle of collecting and processing only the personal information necessary for purposes determined before collection.

- **Pseudonymization** — Replacing direct identifiers with tokens while preserving the ability to re-link records using separately held information; the data remain personal information.

- **Anonymization** — Processing after which information irreversibly no longer allows a person to be identified, directly or indirectly, following Québec's 2024 regulation.

- **Retention schedule** — A documented, approved mapping between categories of records and retention periods grounded in legal, regulatory, contractual, and business requirements.

- **Data residency** — The deliberate choice of where personal information is stored, replicated, and accessed, including telemetry, support access, and sub-processors.

- **Data loss prevention (DLP)** — Tools and practices that detect and prevent unauthorized transmission or copying of sensitive information across email, web, endpoints, and cloud services.

- **Privacy impact assessment (PIA)** — A structured, documented analysis of privacy risks and their treatment before a decision, called EFVP in Québec and DPIA under the GDPR.

- **Privacy information management system (PIMS)** — A management system for privacy specified by ISO/IEC 27701, which since 2025 can be certified without ISO/IEC 27001.

## Review questions

### Essay questions

1.  Using the Chaudière Services Résidentiels scenario, evaluate how the privacy impact assessment changed the CRM project. Which findings depended on information from procurement and the security architect, and why was the final risk acceptance defensible?

2.  The chapter states that a system can be perfectly secure and still be a privacy failure. Analyze this claim using examples from the chapter, and explain how an organization should integrate privacy and security while keeping their accountabilities distinct.

3.  Choose an organization you know. Apply the five-phase personal information lifecycle to one of its systems, identifying for each phase the key decisions, the accountable role, and one control you would prioritize.

4.  Evaluate why so many organizations keep personal information indefinitely and propose a retention governance program that addresses schedules, backups and replicas, legal holds, and evidence of deletion.

5.  A Québec retailer plans to move customer data to a U.S.-hosted cloud service with global support teams. Analyze the cross-border obligations under Québec law, PIPEDA, and, where relevant, the GDPR, and recommend the architecture and contract decisions management should make.

### Multiple-choice questions

Choose the best answer. Answers are listed in the appendix "Answers to multiple-choice questions".

**1.** Under Québec's Law 25, who is the person in charge of the protection of personal information by default?

> A. The chief information security officer
>
> B. The person exercising the highest authority in the enterprise
>
> C. A data protection officer registered with the CAI
>
> D. The head of legal affairs or compliance

**2.** An analytics team replaces customer names with tokens but keeps a separate key that allows records to be re-linked. How should this dataset be treated?

> A. As anonymized information outside the scope of privacy law
>
> B. As public information once the key is encrypted
>
> C. As commercial data governed only by confidentiality rules
>
> D. As personal information still subject to privacy obligations

**3.** According to the NIST Privacy Framework, how does privacy risk differ from cybersecurity risk?

> A. Privacy risk arises from data processing that can cause problems for individuals
>
> B. Privacy risk arises only from loss of confidentiality of personal records
>
> C. Privacy risk is limited to breaches that require regulatory notification
>
> D. Privacy risk concerns legal penalties, while security risk concerns attackers

**4.** A software-as-a-service provider hosts its customers' client files and also runs its own billing and marketing. What roles does it hold?

> A. Controller for all data, because it operates the platform
>
> B. Processor for all data, because it provides a service
>
> C. Processor for customers' data and controller for its own billing and marketing data
>
> D. Joint controller with each customer for every dataset

**5.** Which Law 25 phase gave individuals a right to data portability in a structured, commonly used technological format?

> A. September 22, 2021
>
> B. September 22, 2022
>
> C. September 22, 2024
>
> D. September 22, 2023

**6.** An enterprise subject to Québec's private-sector law had worldwide turnover of \$800 million in the preceding fiscal year. What is the maximum administrative monetary penalty the CAI could impose?

> A. \$16 million
>
> B. \$10 million
>
> C. \$25 million
>
> D. \$32 million

**7.** Under PIPEDA, how long must organizations keep records of all breaches of security safeguards?

> A. Five years after becoming aware
>
> B. Until the OPC closes its review
>
> C. Twelve months after notification
>
> D. At least 24 months

**8.** As of October 2026, what is the status of Bill C-36 according to the chapter?

> A. It is in force and has replaced PIPEDA's private-sector rules
>
> B. It was introduced in June 2026 but not enacted, so PIPEDA still applies
>
> C. It died on the Order Paper when Parliament was prorogued in 2025
>
> D. It applies only to federally regulated banks and telecommunications firms

**9.** A Montréal company plans to host customer data in a data centre in Ontario. What does Québec's law require before the information is communicated?

> A. A privacy impact assessment, because outside Québec includes the rest of Canada
>
> B. No assessment, because Ontario is covered by federal privacy law
>
> C. Only a notice to customers that data will be stored in Canada
>
> D. Approval from the CAI before any interprovincial transfer

**10.** Under the GDPR, within what timeframe must a breach generally be notified to the supervisory authority?

> A. Promptly, with no fixed deadline
>
> B. As soon as feasible after harm is confirmed
>
> C. Within 72 hours, where feasible
>
> D. Within 30 days of discovery

**11.** A project charter lists the retention period for a new customer database as to be determined. What is the chapter's view?

> A. It is acceptable until the first annual records review
>
> B. It is a reliable predictor of indefinite retention and should not be approved
>
> C. It is acceptable if the database is encrypted at rest
>
> D. It is a records management issue outside the project's scope

**12.** Developers find that application logs record full request payloads, including customer details. Which minimization practice addresses this?

> A. Segregation of sensitive attributes
>
> B. Purpose-bound datasets
>
> C. Lean collection interfaces
>
> D. Minimal logging of content

**13.** Under Québec's 2024 anonymization regulation, which step must an enterprise complete before treating information as anonymized?

> A. Encrypt the dataset with customer-managed keys
>
> B. Obtain express consent from every person concerned
>
> C. Notify the CAI and publish the anonymized dataset
>
> D. Analyze re-identification risk under qualified supervision and keep a register

**14.** A cloud provider encrypts all customer data at rest by default with keys it manages. Against which threat does this mainly protect?

> A. Access by the provider's own administrators
>
> B. Theft of physical storage media
>
> C. An authorized user exporting records
>
> D. An application compromised while data is decrypted

**15.** In the chapter's layered approach to data loss prevention, what should come first?

> A. Hard blocks on all external sharing
>
> B. A cloud access security broker deployment
>
> C. Classification policies and labels for personal information
>
> D. Pattern matching on card and social insurance numbers

**16.** In a Québec privacy impact assessment, what does the proportionality analysis examine that a security assessment usually does not?

> A. Whether the processing is justified, minimally intrusive, and beneficial overall
>
> B. Whether encryption and access controls meet the classification scheme
>
> C. Whether the vendor holds ISO/IEC 27001 and SOC 2 Type II reports
>
> D. Whether incident notification can be completed within 72 hours

**17.** In the Chaudière scenario, why was the vendor's EU–U.S. Data Privacy Framework certification irrelevant to the assessment?

> A. Because the transfer was from Québec, so Québec's transfer analysis applied instead
>
> B. Because the Data Privacy Framework had been invalidated by the courts
>
> C. Because the vendor stored all primary data in Canada
>
> D. Because ISO/IEC 27018 certification replaces any transfer mechanism

**18.** In the Chaudière scenario, why did the residual risk decision go to the executive committee rather than the project team?

> A. Because the CAI must approve all transfers outside Canada
>
> B. Because the AI call-summary feature was classified as prohibited
>
> C. Because the vendor refused to negotiate a written agreement
>
> D. Because the appetite statement requires executive approval for moderate risks over 50,000 people

**19.** What changed with the 2025 edition of ISO/IEC 27701?

> A. It merged with ISO/IEC 27018 into a single cloud privacy standard
>
> B. It became mandatory for organizations subject to Law 25
>
> C. It became a stand-alone standard certifiable without ISO/IEC 27001
>
> D. It replaced ISO/IEC 29134 as the guide for impact assessments

**20.** A privacy office spends most of its time completing PIAs for project teams and maintaining inventories that owners never consult. What is the main governance problem?

> A. The privacy office should report to internal audit instead
>
> B. The privacy office is doing first-line work and lacks capacity to challenge
>
> C. Project teams should not participate in privacy impact assessments
>
> D. Inventories should be maintained by the CISO rather than owners
