<!-- Introduction to Cybersecurity Governance, 3rd ed., v2.1g (Aug 2026) - (c) Marc-André Léger. Licensed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/). Third-party standards, frameworks and quotations remain under their owners' terms. -->

# Chapter 9: Cybersecurity Control Frameworks

Earlier chapters presented cybersecurity governance as a managerial discipline. Boards, executives, CIOs, CISOs, risk owners, and business technology managers must set direction, define risk appetite and tolerance, allocate resources, assign accountability, and monitor whether cybersecurity practices achieve the intended outcomes. At some point, however, governance must become concrete. Managers must be able to answer practical questions:

- How should cybersecurity responsibilities be organized?

- Which controls should be implemented to reduce identified risks?

- How can the organization demonstrate that its practices are reasonable, effective, and aligned with regulatory, contractual, and stakeholder expectations?

Control frameworks and standards help answer these questions. They provide a shared vocabulary, structured control catalogues, implementation guidance, governance models, and assessment criteria. Some frameworks support high-level governance and performance management, while others offer detailed technical and organizational controls. Some standards define formal management systems that can be audited and, in certain cases, certified. For business technology managers, the goal is not to memorize every control but to understand how frameworks help select, adapt, justify, implement, monitor, and improve cybersecurity practices.

This chapter introduces the main categories of standards and frameworks and then examines several contemporary references commonly used in cybersecurity governance and management. Standards presented include ISO/IEC 27001, the international standard for an information security management system; ISO/IEC 27002, the supporting catalogue of information security controls; and ISO/IEC 27005, guidance for information security risk management. As for frameworks, this book looks at the NIST ecosystem, including the Risk Management Framework, SP 800-53 Rev. 5, SP 800-53B control baselines, and the Cybersecurity Framework; and COBIT 2019, ISACA’s framework for the governance and management of enterprise information and technology.

The chapter also highlights a practical reality: organizations rarely use a single standard or framework in isolation. In practice, they build hybrid governance architectures that integrate ISO, NIST, COBIT, CIS Controls, sector-specific requirements, privacy and cybersecurity laws, contractual obligations, audit expectations, and internal policies. The managerial challenge is therefore not simply to choose a framework but to understand how multiple references can be combined into a coherent, risk-based, auditable, and operationally realistic cybersecurity governance system.

## What is a standard?

A standard is a documented agreement that describes rules, guidelines or characteristics for activities or their results. In everyday language we talk about norms or best practices. In English, the word standard is generally preferred in this context.

Standards play several roles in cybersecurity governance:

- First, they **normalize vocabulary**. When a bank says we have an ISMS in place or we are aligned with ISO/IEC 27001, everyone in the industry has a reasonably precise understanding of what this means. This common language is vital when you negotiate with suppliers, regulators, insurers, or external auditors.

- Second, standards **support comparability**. They let organizations position themselves relative to peers or expectations: we are certified to ISO/IEC 27001, our controls meet the NIST SP 800-53 moderate-impact baseline, and so on. In governance terms, this helps boards and executives answer the question are we doing enough? without inventing their own benchmark from scratch.

- Third, standards **reduce design effort.** No organization has the resources to invent every policy, procedure and control from first principles. Control catalogues such as ISO/IEC 27002:2022 or NIST SP 800-53 Rev.5 provide a menu of proven safeguards which you can select, tailor and combine, guided by risk analysis.

- Finally, some standards are linked to **formal certification or regulatory obligations**. For example, ISO/IEC 27001 certification provides external assurance that your information security management system meets the standard's requirements. NIST-based standards are mandated for U.S. federal agencies and, in practice, influence many Canadian organizations that sell to the U.S. public sector or operate in regulated industries.

To understand where these documents come from, it is helpful to distinguish four broad categories of standards.

## What Is a Framework?

A framework is a structured reference model that helps an organization organize activities, responsibilities, controls, processes, and decisions around a specific objective. In cybersecurity governance, a framework helps you think about what must be governed, how components relate to one another, and how an organization can design, assess, and improve its cybersecurity practices over time.

A framework is typically broader and more flexible than a standard. A standard often sets requirements, specifications, terminology, or control expectations. A framework, by contrast, usually provides an organizing structure that can be adapted to the organization’s context. It may define functions, domains, processes, outcomes, maturity levels, control objectives, or governance practices. For example, the NIST Cybersecurity Framework organizes cybersecurity outcomes into functions such as Govern, Identify, Protect, Detect, Respond, and Recover. COBIT organizes governance and management of enterprise information and technology around objectives, components, goals, roles, and performance management.

The distinction is important, but it should not be treated as absolute. Some documents called standards contain framework-like elements, and some frameworks include detailed practices or controls. ISO/IEC 27001 is a standard because it defines requirements for an information security management system. ISO/IEC 27002 is a control guidance document because it provides recommended information security controls. The NIST Cybersecurity Framework is a framework because it organizes cybersecurity outcomes and communication within a flexible structure. COBIT is a governance framework because it helps organizations align stakeholder needs, enterprise goals, decision rights, processes, accountability, performance, and assurance.

For managers, a simple distinction is useful. A standard often helps answer the question, “What requirements or expectations should we meet?” A framework often helps answer the question, “How should we organize our thinking, responsibilities, and activities?” Standards support consistency, comparability, and assurance, while frameworks support structure, alignment, communication, and adaptation.

In practice, organizations often use multiple frameworks. A cybersecurity program may use ISO/IEC 27001 to structure an information security management system, ISO/IEC 27002 or NIST SP 800-53 to select controls, the NIST Cybersecurity Framework to communicate outcomes to executives, and COBIT to align cybersecurity with enterprise governance. The value of a framework is that it helps managers view cybersecurity as an organized system rather than a disconnected list of tools, policies, and technical tasks.

## Why Standards and Frameworks Matter to Managers

Managers do not select cybersecurity controls in isolation. In practice, they rely on standards and frameworks to structure decisions, define a common language, assign accountability, support assurance, and demonstrate that cybersecurity practices are reasonable relative to risk. Modern governance increasingly expects more than informal good intentions. It requires documented criteria, repeatable processes, evidence of control implementation, and a clear connection among risk, controls, performance, and oversight.

- **ISO/IEC 27001:2022** is the leading international standard for establishing, implementing, maintaining, and continually improving an Information Security Management System (ISMS). For managers, its importance lies in treating information security as a governed management system rather than a collection of isolated technical controls. It requires organizations to define information security risk assessment and risk treatment processes, apply defined criteria to assess likelihood and consequences, retain documented results, and ensure that risk assessments are consistent and comparable over time. For governance, ISO/IEC 27001 helps management demonstrate that information security risks are identified, treated, monitored, and accepted through intentional processes rather than informal judgment.

- **ISO/IEC 27002:2022** complements ISO/IEC 27001 by providing guidance on information security controls. It organizes 93 controls into four themes: organizational, people, physical, and technological. It also introduces control attributes that allow organizations to view and classify controls by factors such as control type, information security properties, cybersecurity concepts, operational capabilities, and security domains. This makes ISO/IEC 27002 useful not only as a control reference but also as a tool for mapping controls, explaining coverage, and supporting governance reporting. The 2022 edition also reflects contemporary risk realities through controls such as threat intelligence, information security for cloud services, and ICT readiness for business continuity.

- **ISO/IEC 27005:2022** provides guidance on information security risk management and supports alignment with ISO/IEC 27001 and broader risk management principles. It does not prescribe a single mandatory risk assessment method. Rather, it explains what a credible risk management process should include and how it can be integrated into an ISMS. For managers, its value lies in clarifying how risk criteria, risk identification, analysis, evaluation, treatment, communication, monitoring, and review should be organized. It also supports both event-based and asset-based approaches to risk identification, which is especially useful when organizations develop cybersecurity risk scenarios.

- The **NIST ecosystem** offers a complementary, widely used approach, especially in public-sector, regulated, and supplier-assurance contexts. The NIST Risk Management Framework presents a structured lifecycle for managing security and privacy risk, including steps such as preparing, categorizing systems, selecting controls, implementing controls, assessing controls, authorizing systems, and monitoring over time. NIST SP 800-53 provides a detailed catalogue of security and privacy controls, while NIST SP 800-53B provides control baselines for different impact levels. The NIST Cybersecurity Framework offers a higher-level governance and communication structure that helps organizations organize cybersecurity outcomes across functions such as Govern, Identify, Protect, Detect, Respond, and Recover.

- **COBIT 2019** occupies a distinct yet complementary position. It is not a certifiable information security management system standard, nor is it primarily a technical control catalogue. It is an enterprise governance framework for information and technology. COBIT helps boards and executives align stakeholder needs, enterprise objectives, decision rights, risk, resources, performance, and assurance. For cybersecurity managers, COBIT is especially useful when the question is not only "Which control should we implement?" but also "Who is accountable?", "Which business objective does the control support?", "How will performance be measured?", and "How will management know that the governance system is working?"

The governance relevance of these standards and frameworks is immediate. They translate abstract intent, such as "we care about cybersecurity," into structured accountability, defined risk processes, selected controls, measurable performance, documented evidence, and credible assurance. They also help managers explain cybersecurity decisions to boards, auditors, regulators, customers, suppliers, insurers, and internal stakeholders. Used well, standards and frameworks do not replace judgment; they make it more disciplined, transparent, and defensible.

## Categories of standards and frameworks

Standards and frameworks do not all come from the same source, nor do they all carry the same authority. Some are developed through formal international consensus processes. Others emerge because governments require them, because professional associations promote them, or because the market adopts them so widely that they become unavoidable. Understanding these categories helps managers interpret the status of a reference document and determine how much weight it should carry in governance, procurement, audit, and risk management.

### Consensus-Based Standards

Consensus-based standards are developed by recognized standards bodies through structured processes that involve multiple stakeholders. Technical committees typically bring together experts from industry, government, academia, professional associations, and, at times, consumer or public-interest groups. Drafts are proposed, discussed, revised, and commented on, and are eventually approved when a sufficiently broad level of agreement is reached.

In cybersecurity and information security, one of the most important international consensus environments is the work of the International Organization for Standardization (ISO) and the International Electrotechnical Commission (IEC). They carry out their work on information security, cybersecurity, and privacy protection through ISO/IEC Joint Technical Committee 1, Subcommittee 27. Key standards in this ecosystem include ISO/IEC 27001, ISO/IEC 27002, and ISO/IEC 27005.

Other consensus-oriented bodies are also relevant to information technology and cybersecurity. The International Telecommunication Union (ITU) develops telecommunications and network standards. The Institute of Electrical and Electronics Engineers (IEEE) publishes widely used technical standards, including those for Ethernet and Wi-Fi. These documents may not all be cybersecurity governance standards, but they shape the technological environment that cybersecurity governance must address.

Consensus-based standards have two characteristics that matter for governance.

- First, they are **deliberately conservative.** Because they must be acceptable to many stakeholders, they rarely represent the absolute cutting edge of research or product innovation. Instead, they codify practices that a broad community of experts and practitioners agrees are reasonable, stable, and defensible. From a governance perspective, this conservatism is often valuable. Boards, regulators, auditors, customers, and insurers are usually more comfortable with a recognized, widely accepted standard than with an experimental approach not yet validated across many organizations.

- Second, consensus-based standards **evolve in cycles**. They are reviewed, revised, and republished over time. For ISO/IEC, the typical revision cycle is 3 to 5 years. As a result, a delay often exists between emerging practice and formal standardization, and another between the publication of a new edition and widespread industry adoption. Organizations need time to update policies, procedures, control mappings, contracts, training, audit programs, and certification arrangements. Managers should therefore understand not only which standard applies but also which version is in use and whether the organization has a transition plan when standards are updated.

### Market-Driven, Ad Hoc, and De Facto Standards

Not all standards originate through formal committees. Some emerge when a particular product, protocol, platform, or practice becomes so widely adopted that it effectively defines how things are done. These are often described as ad hoc or de facto standards.

An ad hoc standard emerges informally when a technology or practice becomes widely used before a formal standard exists or before a formal standard has become dominant. A de facto standard is similar, but it emphasizes factual dominance: something becomes a standard in practice because many organizations use it, not because a recognized standards body has approved it or because users have explicitly selected it as best practice.

This distinction is useful, but it should not be overstated. In practice, both ad hoc and de facto standards influence governance by shaping what is technically feasible, commercially viable, and operationally economical. A dominant operating system, cloud platform, collaboration suite, identity provider, programming framework, encryption library, or network protocol can set expectations that managers must address. Even when a technology is not ideal from a security perspective, its market dominance may make it difficult to avoid.

From a governance perspective, treat market-driven standards as constraints. Ignoring them does not make them disappear. At the same time, blindly accepting them can introduce hidden risk if the organization has not assessed their implications for cybersecurity, privacy, resilience, and supplier dependency. Mature governance therefore asks whether widely adopted technologies are secure by default, whether they can be configured to meet organizational requirements, whether they create concentration risk, and whether compensating controls are needed.

Over time, some market-driven practices become formalized as consensus standards. Others remain dominant only until better technology or regulatory pressure replaces them. Managers should therefore treat de facto standards as operational realities, but not necessarily as evidence of good governance.

### Government-Mandated Standards and Public-Sector Requirements

Some standards, baselines, and guidance documents are created or mandated by government agencies. They may be required by law, regulation, procurement policy, public-sector contracting rules, or as a condition of doing business with the state. In the United States, the National Institute of Standards and Technology plays a central role in this area for federal agencies and many organizations that support them. NIST publications such as SP 800-53, SP 800-53B, and the Risk Management Framework are important examples of government-developed cybersecurity guidance that influence control selection, assessment, authorization, and continuous monitoring.

Although many NIST publications are written primarily for U.S. federal contexts, their influence extends well beyond that environment. They are widely used because they are publicly available, detailed, regularly maintained, and supported by a broader ecosystem of guidance. Organizations outside the United States may align with NIST references due to customer expectations, regulatory influence, supplier-assurance requirements, cyber-insurance expectations, or participation in supply chains connected to U.S. public-sector or critical-infrastructure environments.

In Canada and Québec, public-sector policies, privacy laws, cloud requirements, critical-infrastructure expectations, and sector-specific rules may also function as standards, even if they are not standards in the ISO sense. They define required or expected practices for organizations that provide services to government, process regulated information, operate in sensitive sectors, or manage critical services. For managers, the key point is that government-developed requirements can become binding through law, contract, procurement, audit, or regulatory supervision.

### Professional Governance and Control Frameworks

Not every influential reference is a formal standard. Professional associations, industry communities, and specialized organizations also publish frameworks that structure practices, objectives, controls, and responsibilities. COBIT, developed by ISACA, is a major framework for enterprise governance of information and technology. The CIS Controls provide a prioritized set of cybersecurity safeguards with a practical, operational focus. OWASP guidance is influential in application security and, increasingly, in software and AI-related security practices.

These references often serve a different purpose than formal standards. A formal standard may define requirements against which an organization can be audited or certified. A framework may instead provide an organizing model that the organization adopts, tailors, maps to other obligations, and uses to improve decision-making. Frameworks can be highly influential even when they are not certifiable standards, because they help organizations structure responsibilities, prioritize controls, communicate expectations, and assess maturity.

COBIT is especially useful in this regard because it is an integrating governance framework. It does not replace ISO, NIST, CIS, privacy laws, financial-sector rules, contractual requirements, or internal policies. Instead, it helps organizations align those requirements with enterprise objectives, decision rights, governance structures, management practices, performance measures, and assurance mechanisms. In other words, COBIT helps managers ask not only whether a control exists, but also why it exists, who owns it, how it supports business objectives, how performance is measured, and how governance knows whether it is working.

The managerial lesson is that the source of a standard or framework matters. Consensus standards provide legitimacy and comparability. Market-driven standards reflect operational reality. Government-mandated standards create legal, contractual, or procurement obligations. Professional frameworks help organize governance, controls, and maturity. A well-governed organization recognizes these differences and integrates the relevant references into a coherent control architecture rather than treating them as disconnected documents.

## ISO/IEC 27001:2022 and the Information Security Management System (ISMS)

Among the many standards in the ISO/IEC 27000 family, ISO/IEC 27001:2022 is the core certifiable standard. It sets out the requirements for establishing, implementing, maintaining, and continually improving an Information Security Management System (ISMS). An ISMS is not merely a list of technical safeguards. It is a management system that defines how an organization governs information security risks, selects and operates controls, monitors performance, reviews results, and improves over time.

This distinction is important for cybersecurity governance. ISO/IEC 27001 does not require an organization to implement every possible control. Instead, it requires the organization to understand its context, define the ISMS scope, identify information security risks, determine risk treatment options, select appropriate controls, retain documented evidence, evaluate performance, and continually improve the system. In other words, ISO/IEC 27001 turns information security from a collection of isolated practices into a disciplined governance and management process.

ISO/IEC 27001:2022 was amended in 2024 through Amendment 1, which introduced climate-action-related changes into the management system context analysis. Organizations must consider whether climate change is a relevant external issue and whether interested parties have related requirements. For cybersecurity governance, the practical lesson is that the ISMS context must remain current as the organization’s environment changes. The amendment does not create a separate climate-security control catalogue. Rather, it reinforces the broader management-system principle that information security must be understood in relation to the organization’s internal and external context.

<img src="media/image43.png" style="width:6in;height:4in" />

<span id="_Toc235166925" class="anchor"></span>Figure 43: ISO/EIC 27001:2022

### ISMS as a management system

In ISO terminology, a management system is the set of interrelated elements an organization uses to establish policies, set objectives, assign responsibilities, operate processes, measure performance, and continually improve in relation to a defined subject. It is not merely a document, a software platform, a policy manual, or a list of controls. It is the organized way the organization governs and manages an area of responsibility over time.

A management system typically includes leadership commitment, defined scope, policies, roles and responsibilities, risk assessment processes, operational controls, resources, competence, communication, documented information, performance measurement, internal audit, management review, corrective action, and continual improvement. These elements work together to help the organization move from intention to execution and from execution to evidence.

This is why ISO/IEC 27001 is best understood as a management system standard. It does not merely ask whether specific cybersecurity tools exist. Instead, it asks whether the organization has a structured, repeatable system for understanding its information security risks, selecting appropriate controls, operating those controls, monitoring their effectiveness, addressing weaknesses, and improving over time.

For managers, the concept is important because it links cybersecurity to governance. A management system makes accountability explicit. It clarifies who is responsible, what objectives must be achieved, what evidence must be retained, how performance is reviewed, and how improvements are implemented when controls fail or circumstances change. In this sense, an Information Security Management System is the mechanism through which cybersecurity intent becomes an organized, auditable, and continuously improved practice.

Like other ISO management system standards, such as ISO 9001 for quality management and ISO 14001 for environmental management, ISO/IEC 27001 follows the Plan–Do–Check–Act cycle of continuous improvement.

- **Plan**: In the planning phase, the organization understands its context, identifies interested parties and requirements, defines the ISMS scope, establishes information security objectives, identifies and assesses information security risks, and determines how to treat them.

- **Do**: During the implementation phase, the organization implements the risk treatment plan, selects and implements controls, assigns responsibilities, allocates resources, communicates expectations, and maintains the policies, processes, and documentation required for the ISMS to function.

- **Check**: During the checking phase, the organization monitors and measures performance, evaluates control effectiveness, conducts internal audits, and reviews the ISMS at the management level to determine whether it remains suitable, adequate, and effective.

- **Act**: In the improvement phase, the organization addresses nonconformities, implements corrective actions, and improves the ISMS based on audit findings, incidents, changes in risk, shifts in business objectives, and lessons learned from operations.

This continuous improvement cycle is central to cybersecurity governance. It aligns with the risk-centric logic developed earlier in the book: security is not the absolute elimination of risk. Rather, it is the ongoing management of unacceptable risk, consistent with organizational strategy, legal and contractual obligations, available resources, and risk tolerance.

### Structure of ISO/IEC 27001:2022

The ISO/IEC 27001:2022 follows the harmonized structure used across modern ISO management system standards. This structure makes the standard easier to integrate with other management systems, such as quality, environmental, business continuity, compliance, and privacy management systems. For governance purposes, the most important clauses are as follows.

- **Context of the organization (Clause 4):** The organization must understand the internal and external issues that affect its information security management system, identify interested parties and their requirements, and define the ISMS scope. The scope is a major governance decision. An organization may decide that the ISMS applies to the whole enterprise, or it may begin with a specific business unit, service, process, data centre, cloud environment, or customer-facing activity. A narrow scope may simplify implementation, but it can also create false assurance if it excludes critical dependencies. A broad scope may provide stronger governance coverage, but it requires more resources, coordination, and evidence.

- **Leadership (Clause 5):** Top management must demonstrate leadership and commitment to the ISMS, establish and communicate an information security policy, assign relevant roles and responsibilities, and ensure that information security objectives align with the organization's strategic direction. This clause formalizes a central governance principle: information security is not merely an IT responsibility. It requires leadership, authority, accountability, and visible management commitment.

- **Planning (Clause 6):** The organization must address ISMS-related risks and opportunities. This includes defining information security risk criteria, establishing risk assessment and risk treatment processes, setting information security objectives, and planning how to achieve them. Clause 6 is particularly important because it links risk appetite, risk assessment, control selection, residual-risk acceptance, and governance oversight. Annex A serves as a reference set of controls when determining which controls are necessary, but the organization must still justify control selection based on risk and context.

- **Support (Clause 7):** The organization must provide the resources, competence, awareness, communication, and documented information required to support the ISMS. This includes ensuring that people understand their responsibilities, that relevant documentation is controlled, and that evidence can be retained and produced as needed. These requirements may seem administrative, but they are essential to governance because they make expectations repeatable, auditable, and sustainable.

- **Operation (Clause 8):** The organization must plan, implement, and control the processes required to meet ISMS requirements. This includes conducting information security risk assessments, implementing the risk treatment plan, operating selected controls, and managing changes that may affect the ISMS. Clause 8 is where governance decisions become operational practice. Risk treatment is no longer merely a plan; it must be executed, maintained, and controlled.

- **Performance evaluation (Clause 9):** The organization must monitor, measure, analyze, and evaluate the ISMS’s performance and effectiveness. It must also conduct internal audits and carry out management reviews. From a governance perspective, this clause links dashboards, key risk indicators, audit findings, incident trends, control evidence, management review, and decision-making. This linkage helps leadership determine whether the ISMS remains suitable, adequate, and effective.

- **Improvement (Clause 10)**: The organization must manage nonconformities, take corrective action, and continually improve the ISMS’s suitability, adequacy, and effectiveness. This prevents the ISMS from becoming a static documentation exercise. Weaknesses identified through audits, incidents, control failures, business changes, or external developments must prompt corrective action and improvement.

- **Annex A**: Annex A of ISO/IEC 27001 provides a reference set of information security controls to support risk treatment. The 2022 edition aligns Annex A with ISO/IEC 27002:2022, which offers more detailed guidance on controls. The organization is not expected to implement every control automatically. Instead, it must determine which controls are necessary, justify inclusions and exclusions, and document those decisions in the Statement of Applicability. This makes Annex A and the Statement of Applicability central governance artifacts that link risk assessment, risk treatment, control selection, implementation status, accountability, and audit evidence.

## ISO/IEC 27002:2022: The control catalogue

While ISO/IEC 27001 defines the management system, ISO/IEC 27002:2022 provides detailed guidance on controls. It is best understood as a catalogue of information security controls that organizations can select, tailor, combine, and implement to address risks identified through the ISMS. ISO/IEC 27002 is not a certification standard. It does not define certification requirements in the same way ISO/IEC 27001 does. Instead, it provides implementation-oriented guidance that helps organizations determine which controls may be appropriate and how to apply them in practice.

The 2022 edition reorganized the control catalogue compared with the 2013 version. The controls were consolidated and reduced to 93, grouped into four high-level themes:

- Organizational controls.

- People controls.

- Physical controls.

- Technological controls.

This structure helps managers because it reinforces an important governance lesson: information security is not solely technological. Many controls address leadership, policy, responsibility, suppliers, human behaviour, physical protection, continuity, compliance, and operational discipline. A cybersecurity program that focuses only on technological safeguards will therefore overlook a significant part of the control environment.

Within each theme, ISO/IEC 27002 presents each control with a title, description, purpose, guidance, and related information. The 2022 edition also introduced control attributes, which allow organizations to classify and view controls from different perspectives. These attributes include control type, information security properties, cybersecurity concepts, operational capabilities, and security domains. This makes ISO/IEC 27002 more flexible as a governance, mapping, and reporting tool. For example, the same control can be analyzed to determine whether it is preventive, detective, or corrective; whether it primarily supports confidentiality, integrity, or availability; or whether it relates to governance, protection, detection, response, or recovery.

<img src="media/image44.png" style="width:6in;height:4.5in" />

<span id="_Toc235166926" class="anchor"></span>Figure 44: ISO/EIC 27002:2022

Without reproducing the full control catalogue, the four themes can be summarized as follows:

- **Organizational controls** address the governance and management environment in which information security operates. They include information security policies, roles and responsibilities, segregation of duties, threat intelligence, supplier relationships, information classification, legal and contractual compliance, business continuity readiness, and the management of information security in projects.

- **People controls** address the human dimension of information security. They include screening, terms and conditions of employment, awareness and training, disciplinary processes, responsibilities during employment, and procedures for termination or changes in employment. These controls recognize that people can be sources of risk but are also essential to maintaining security.

- **Physical controls** protect facilities, equipment, and physical environments. They include secure areas, physical access controls, protection against physical and environmental threats, equipment placement and protection, secure disposal or reuse of equipment, clear-desk and clear-screen practices, and related safeguards. These controls are important because physical access can enable digital compromise.

- **Technological controls** encompass the technical safeguards applied to systems, networks, applications, data, and infrastructure. They include identity and access controls, authentication, privileged access, cryptography, secure configuration, logging and monitoring, malware protection, backups, vulnerability management, network security, secure development, and technical testing.

Organizations can use ISO/IEC 27002 as a control catalogue to mine, not mechanically copy. Evaluate each control against the organization’s risks, obligations, business processes, information assets, threat exposure, maturity, resources, and operating constraints. The governance task is to justify which controls are necessary, how to implement them, who owns them, what evidence demonstrates their operation, and whether they reduce risk to an acceptable level.

This is why ISO/IEC 27002 is closely linked to the Statement of Applicability required by ISO/IEC 27001. The organization must explain which controls it includes or excludes, why those decisions are appropriate, and the implementation status. In that sense, ISO/IEC 27002 is more than a technical checklist. It bridges risk assessment, control design, accountability, assurance, and continual improvement.

### Annex A of ISO/IEC 27001 and ISO/IEC 27002

Annex A of ISO/IEC 27001:2022 provides a reference set of information security controls that organizations must consider when determining how to treat information security risks. In the 2022 edition, Annex A is aligned with ISO/IEC 27002:2022 and contains the same 93 controls, organized under four themes: organizational, people, physical, and technological controls. However, Annex A presents these controls in concise form. It supports the management system and the Statement of Applicability, not detailed implementation guidance.

ISO/IEC 27002:2022 provides supporting guidance for these controls. It explains the purpose of each control, offers implementation guidance, and provides additional information to help organizations interpret and apply the controls in practice. In simple terms, ISO/IEC 27001 establishes the management-system requirement to determine and justify controls, while ISO/IEC 27002 helps organizations understand how to implement those controls.

The connection between the two documents is most evident in the Statement of Applicability. The Statement of Applicability is a core ISMS document that identifies which controls are necessary, whether they have been implemented, and why they have been included. It must also justify the exclusion of any Annex A controls. This does not mean that every Annex A control must be implemented automatically. Instead, the organization must select controls based on its risk assessment, legal and contractual obligations, business context, and chosen risk treatment options.

The Statement of Applicability may also include controls beyond Annex A if the organization determines that additional safeguards are necessary. Annex A is therefore not a closed checklist. It is a baseline reference set that helps ensure that important control areas are not overlooked.

In practice, managers and auditors work with ISO/IEC 27001 and ISO/IEC 27002 together. ISO/IEC 27001 defines the management-system obligations: risk assessment, risk treatment, control selection, documented justification, performance evaluation, and continual improvement. ISO/IEC 27002 provides detailed guidance on controls to make those decisions practical. Together, they connect governance, risk analysis, control design, evidence, and assurance.

## ISO/IEC 27005:2022: Information security risk management

Risk management links governance decisions to control selection.

Risk management links governance decisions to control selection. ISO/IEC 27005:2022 provides guidance on information security risk management that supports ISO/IEC 27001 and aligns with the general risk management principles of ISO 31000. Its value lies in helping organizations structure risk-related decisions before selecting, implementing, monitoring, or accepting controls.

### Scope and Role of ISO/IEC 27005

It is important to understand what ISO/IEC 27005 is and what it is not. It is not a complete, prescriptive risk assessment methodology with mandatory formulas, scoring rules, or predefined risk matrices. It does not tell every organization exactly how to calculate likelihood, impact, or residual risk. Instead, it provides a structured vocabulary and process guidance for designing, evaluating, and improving an information security risk management process.

This makes ISO/IEC 27005 particularly useful for governance. A university, bank, hospital, retailer, municipality, or cloud service provider may each use a different risk assessment method, but each method should still explain how risks are identified, analyzed, evaluated, treated, accepted, communicated, monitored, and reviewed. ISO/IEC 27005 provides a reference model for assessing whether the organization’s method is complete, coherent, repeatable, and aligned with its information security management system.

The standard highlights several core activities:

- **Establishing the context** involves understanding the organization’s objectives, internal and external issues, interested parties, assets, business processes, legal and contractual obligations, and risk criteria. This step clarifies what is in scope, what matters, who is affected, and what level of risk may be acceptable. Without a clear context, risk analysis becomes abstract and may not reflect business reality.

- **Risk assessment** includes risk identification, risk analysis, and risk evaluation. Risk identification determines what could happen, which assets or processes could be affected, which threats or events may be involved, which vulnerabilities or control weaknesses may exist, and what consequences could follow. Risk analysis estimates the likelihood and consequences of identified risks. Risk evaluation compares the analyzed risks against defined criteria so that management can decide which risks require treatment and which should receive priority.

- **Risk treatment** involves selecting options to modify risk. These options may include avoiding, accepting, sharing or transferring, or mitigating risk through controls. This is where control catalogues such as ISO/IEC 27002 or NIST SP 800-53 become practically relevant. They help the organization identify controls that may reduce likelihood, reduce impact, improve resilience, or strengthen monitoring and response.

- **Risk acceptance** is a governance decision. After selecting or implementing a treatment, residual risk remains. The appropriate risk owner must decide whether that residual risk is acceptable considering the organization’s risk criteria, obligations, and objectives. Document this decision and review it periodically. Many serious incidents do not arise from completely unknown risks; they arise from risks implicitly tolerated without clear ownership, rationale, or review.

- **Risk communication and consultation** ensure that relevant stakeholders understand the assumptions, limitations, consequences, and responsibilities associated with risk decisions. Information security risk management is not only a technical exercise. It involves business owners, system owners, executives, legal and compliance functions, privacy officers, IT operations, suppliers, and sometimes customers or regulators.

- **Monitoring and review** ensure that risk decisions remain valid over time. Risks change when the organization introduces new systems, adopts cloud services, changes suppliers, launches new products, experiences incidents, faces new legal obligations, or faces new threat activity. A risk assessment that was reasonable last year may become outdated if the business environment or the threat landscape changes.

The Identification–Prioritization–Mobilization model (IPM), introduced earlier in the book, follows the same managerial logic. Identification corresponds to understanding what can go wrong. Prioritization corresponds to analyzing and evaluating which risks matter most. Mobilization corresponds to selecting treatments, assigning ownership, allocating resources, and monitoring whether actions are completed. ISO/IEC 27005 provides a more detailed, standards-based vocabulary for this risk management cycle.

<img src="media/image45.png" style="width:6in;height:4.5in" />

<span id="_Toc235166927" class="anchor"></span>Figure 45: ISO/EIC 27005:2002

### Qualitative and Quantitative Approaches

In practice, many organizations use qualitative or semi-quantitative approaches to information security risk assessment. They may rate likelihood and impact using categories such as low, medium, high, and critical, or assign numerical values to these categories within a risk matrix. These methods are relatively easy to explain and apply, but they are inherently judgment-based. Different analysts may rate the same scenario differently unless the organization defines clear criteria and applies them consistently.

More sophisticated approaches aim to quantify risk in monetary terms or to develop risk indices and key risk indicators that aggregate multiple factors, including threat presence, exploitability, vulnerability severity, asset criticality, expected damage, maximum damage, resilience, and control effectiveness. These methods can support more rigorous prioritization and cost-benefit analysis, but they require more data, stronger assumptions, and careful interpretation. A precise-looking number is not inherently more reliable than a well-documented qualitative judgment.

ISO/IEC 27005 does not prescribe a single level of mathematical sophistication. Instead, it emphasizes that the chosen approach should align with the organization’s context, decision-making needs, available information, risk culture, and governance maturity. A smaller organization may need a simple yet consistent method, whereas a large, regulated organization may require a more formal, documented, and auditable approach that integrates with enterprise risk management, compliance reporting, internal audit, and board oversight.

For business technology managers, the key competence is not advanced mathematics. It is ensuring that the method used is documented, repeatable, transparent, evidence-informed, and aligned with organizational objectives. Managers should be able to explain how risks were identified, how ratings were assigned, what assumptions were made, which controls were selected, who accepted residual risk, and how the organization will know whether the risk picture has changed.

## The NIST Cybersecurity ecosystem

The NIST ecosystem comprises complementary publications that support governance, risk management, control selection, assessment, authorization, and continuous monitoring. These documents are especially important in the United States federal context, but their influence extends far beyond US federal agencies. Many private-sector, Canadian, and international organizations use NIST publications because they are detailed, publicly available, regularly maintained, and widely recognized by auditors, regulators, suppliers, and security professionals.

Several NIST references are particularly important for cybersecurity governance. NIST SP 800-37 Rev. 2 defines the Risk Management Framework. NIST SP 800-53 Rev. 5 provides a detailed catalogue of security and privacy controls. NIST SP 800-53B provides control baselines for systems at different impact levels. The NIST Cybersecurity Framework provides a higher-level structure for organizing cybersecurity outcomes and for communicating with executives.

Together, these documents form an ecosystem. The Risk Management Framework explains how to integrate risk management into the system lifecycle. SP 800-53 identifies controls that may be selected and tailored. SP 800-53B provides baseline starting points. The Cybersecurity Framework helps organizations communicate cybersecurity outcomes in a way that is understandable to management, boards, and business stakeholders.

### NIST SP 800-53 and SP 800-53B

NIST Special Publication 800-53 Rev. 5 is a comprehensive catalogue of security and privacy controls for information systems and organizations. The controls are grouped into families, including Access Control, Audit and Accountability, Configuration Management, Contingency Planning, Identification and Authentication, Incident Response, Risk Assessment, System and Communications Protection, System and Information Integrity, and more.

Compared with ISO/IEC 27002, NIST SP 800-53 is generally more granular and more explicitly structured for system-level control selection, assessment, authorization, and continuous monitoring. It is especially useful when an organization needs detailed control language, specific control enhancements, and a structured basis for assessing whether controls have been implemented and are operating as intended.

NIST SP 800-53B complements SP 800-53 by providing control baselines. A baseline is a starting set of controls considered appropriate for systems at a particular impact level, typically low, moderate, or high. These baselines are not the end of the decision process. They must be tailored to the organization’s mission, system characteristics, threat environment, legal obligations, and risk tolerance. In governance terms, SP 800-53 identifies possible controls, while SP 800-53B helps determine a minimum starting point for systems of different criticality.

This baseline logic is useful for managers because it prevents control selection from becoming arbitrary. Instead of asking each project team to invent its own security requirements, the organization can start with a recognized baseline and then tailor it. Controls may be added, removed, refined, or supplemented based on risk analysis. The result is a more consistent and defensible approach to system security.

NIST controls are also frequently mapped to other frameworks and standards, including ISO/IEC 27001, ISO/IEC 27002, CIS Controls, privacy requirements, contractual obligations, and sector-specific rules. This mapping is important for organizations that must satisfy multiple assurance expectations simultaneously. A control may be implemented once, yet it may provide evidence for several frameworks or obligations.

### NIST Risk Management Framework

The NIST Risk Management Framework, defined in SP 800-37 Rev. 2, provides a structured process for integrating security and privacy risk management throughout the system development lifecycle. It is commonly summarized in seven steps: Prepare, Categorize, Select, Implement, Assess, Authorize, and Monitor.

The Prepare step establishes organizational and system-level readiness for risk management. The Categorize step determines the system's impact level and the information it processes. The Select step identifies appropriate controls, often using SP 800-53 and SP 800-53B. The Implement step puts those controls into operation. The Assess step evaluates whether the controls are implemented correctly, operating as intended, and producing the desired outcome. The Authorize step involves a formal, risk-based decision by an accountable official. The Monitor step ensures that the security and privacy posture is maintained over time as systems, threats, vulnerabilities, and organizational conditions change.

For business technology managers, the RMF is important because it operationalizes risk governance. It integrates system classification, control selection, implementation, assessment, authorization, and continuous monitoring into a disciplined lifecycle. Even when an organization is not legally required to follow the RMF, its logic can be used to strengthen internal security governance, especially for high-risk systems, cloud platforms, regulated data, critical applications, and major technology projects.

The authorization step is particularly significant from a governance perspective. It makes clear that an accountable authority must decide whether a system may operate with its residual risk. This aligns closely with the concept of residual-risk acceptance discussed earlier in the book. Risk should not be accepted passively simply because a system is already in production. It should be accepted deliberately, with evidence, documentation, and accountability.

### NIST Cybersecurity Framework

The NIST Cybersecurity Framework is a high-level framework that helps organizations understand, organize, communicate, and improve cybersecurity outcomes. It was originally developed for critical infrastructure, but it is now used across many sectors and countries because it is flexible, accessible, and suitable for executive communication.

The current version, CSF 2.0, groups cybersecurity outcomes into six core functions:

- **Govern**: establishing, communicating, and monitoring the organization’s cybersecurity risk management strategy, expectations, policies, roles, responsibilities, and oversight.

- **Identify**: understanding assets, business context, systems, data, suppliers, dependencies, and related cybersecurity risks.

- **Protect**: implementing safeguards to manage cybersecurity risks and reduce the likelihood or impact of adverse events.

- **Detect**: developing and implementing activities to identify possible cybersecurity events.

- **Respond**: acting regarding detected cybersecurity incidents.

- **Recover**: maintaining and restoring capabilities, services, and resilience after incidents.

Within these functions, categories and subcategories describe the outcomes organizations may seek to achieve. The CSF deliberately avoids prescribing a specific technology stack or a single control catalogue. Instead, it provides an organizing structure that can be mapped to other references, including ISO/IEC 27001, ISO/IEC 27002, NIST SP 800-53, CIS Controls, sector-specific rules, and internal policies.

Two CSF concepts are particularly useful for governance:

- **Profiles** allow an organization to describe its current cybersecurity outcomes and its desired target outcomes. The gap between the current and target profiles supports planning, prioritization, investment decisions, and communication with executives.

- **Implementation tiers** help the organization describe the maturity and integration of its cybersecurity risk management practices. The tiers are not a simple certification score. Rather, they help management assess whether cybersecurity risk management is partial, risk-informed, repeatable, or adaptive.

The CSF is often useful for boards and executives because it communicates cybersecurity in outcome-oriented language. Senior leaders may not need the full detail of SP 800-53 controls or ISO/IEC 27002 guidance, but they do need to understand whether the organization has governance, asset visibility, protection, detection, response, and recovery capabilities. In that sense, the CSF serves as a bridge between technical control frameworks and executive governance.

The managerial lesson is that the NIST ecosystem can be applied at multiple levels. SP 800-53 and SP 800-53B support detailed control selection and baselining. The RMF supports disciplined system authorizations and lifecycle risk management. The CSF supports strategic communication, governance, and cybersecurity improvement planning. Together, these references help organizations move from general cybersecurity intent to structured control decisions, accountable risk acceptance, and continuous monitoring.

<img src="media/image46.png" style="width:6in;height:4.5in" />

<span id="_Toc235166928" class="anchor"></span>Figure 46: NIST cybersecurity ecosystem

## COBIT 2019: Enterprise governance of information and technology

COBIT is a framework developed by ISACA for the governance and management of enterprise information and technology. Its scope is enterprise-wide. It addresses information and technology wherever they support organizational objectives, not only within the formal IT department. This is important because cybersecurity risks frequently arise from business processes, suppliers, cloud services, data practices, applications, operational technologies, and strategic technology decisions that extend beyond the traditional boundaries of IT.

COBIT complements the security standards and frameworks discussed earlier in this chapter. ISO/IEC 27001 sets requirements for an information security management system. ISO/IEC 27002 and NIST SP 800-53 provide detailed control guidance. ISO/IEC 27005 and the NIST Risk Management Framework structure risk management activities. The NIST Cybersecurity Framework presents cybersecurity outcomes in a concise, executive-friendly way. COBIT adds a broader governance architecture by linking information and technology to enterprise goals, decision rights, accountability, resource allocation, performance measurement, and assurance.

### Purpose and Value

COBIT's central purpose is to help the enterprise create value from information and technology. Value is understood as a balance among three outcomes: benefits realization, risk optimization, and resource optimization.

Benefits realization means ensuring that technology-enabled investments, services, data, and capabilities deliver the intended business value. Risk optimization means ensuring that I&T-related risks, including cybersecurity, privacy, resilience, third-party, cloud, and data risks, remain within the organization’s risk appetite and tolerance. Resource optimization means ensuring that people, information, applications, infrastructure, services, suppliers, and financial resources are used responsibly.

This framing is important for cybersecurity governance. A control is not valuable merely because it appears in a catalogue. It is valuable when it supports an enterprise objective, mitigates a material risk, meets a legal or contractual obligation, strengthens resilience, or enables trustworthy operations at a reasonable cost. COBIT helps managers keep cybersecurity aligned with enterprise value rather than treating it as a separate technical checklist.

<img src="media/image47.png" style="width:6in;height:4in" />

<span id="_Toc235166929" class="anchor"></span>Figure 47: COBIT 2019

### Governance and Management Are Different

COBIT explicitly separates governance from management. Governance evaluates stakeholder needs, conditions, and options; sets priorities and decision principles for the organization; and monitors performance, risk, and conformance. These responsibilities primarily belong to the governing body and senior executives.

Management plans, builds, runs, and monitors activities in accordance with governance direction. These responsibilities fall to executive management, business managers, technology leaders, security leaders, and operational teams.

This distinction is central to cybersecurity. Governance determines what must be achieved, what risk is acceptable, who has authority, how priorities are set, and how performance will be monitored. Management translates that direction into policies, programs, controls, services, projects, monitoring, reporting, and improvement activities.

### The COBIT Core Model

The COBIT Core Model groups 40 governance and management objectives into five domains.

- **EDM**: Evaluate, Direct and Monitor contains governance objectives carried out under the authority of the governing body. This domain focuses on evaluating stakeholder needs, directing management, and monitoring outcomes.

- **APO**: Align, Plan and Organize covers strategy, architecture, innovation, budgets, people, suppliers, risk, security, data, and other planning and organizational capabilities.

- **BAI**: Build, Acquire and Implement addresses programs, projects, solutions, change, availability, assets, configuration, and organizational change enablement.

- **DSS**: Deliver, Service and Support covers operations, service delivery, incidents, problems, continuity, security services, and business process controls.

- **MEA**: Monitoring, Evaluation, and Assessment addresses performance, internal control, compliance, and assurance.

These domains should not be treated as isolated silos. They form a governance chain. For example, a board may use EDM03 Ensured Risk Optimization to set risk-related expectations. Management may rely on APO12 Managed Risk and APO13 Managed Security to organize risk and security programs. Operations may apply DSS04 Managed Continuity and DSS05 Managed Security Services to support resilience and daily security operations. Assurance functions may use MEA02 Managed System of Internal Control, MEA03 Managed Compliance with External Requirements, and MEA04 Managed Assurance to evaluate whether the governance and control system is working as intended.

### The COBIT Goals Cascade

One of COBIT’s most useful managerial concepts is the goals cascade. It keeps cybersecurity connected to business priorities. It creates a traceable path from stakeholder needs to enterprise goals, from enterprise goals to alignment goals for information and technology, and from those alignment goals to governance and management objectives.

For example, a board may identify customer trust, operational resilience, regulatory compliance, and digital transformation as strategic priorities. These priorities can be translated into enterprise and alignment goals covering managed business risk, continuity of critical services, reliable information, secure technology-enabled innovation, and compliance with external requirements. COBIT then helps identify which governance and management objectives warrant attention.

The value of the goals cascade is that it helps managers justify cybersecurity investments in business terms. Rather than saying a control is needed because a framework mentions it, management can explain that the control supports a specific enterprise objective, reduces a defined risk, strengthens accountability, or fulfils an obligation.

### Seven Components of a Governance System

COBIT emphasizes that an effective governance system is more than a set of processes. Seven interacting components must be considered:

- Processes.

- Organizational structures.

- Principles, policies, and procedures.

- Information flows and items.

- Culture, ethics, and behaviour.

- People, skills, and competencies.

- Services, infrastructure, and applications.

This holistic view is especially relevant to cybersecurity. A technically correct control may still fail if responsibilities are unclear, staff lack competence, reporting information is unreliable, incentives reward unsafe behaviour, or supporting technology is poorly maintained. COBIT therefore encourages managers and auditors to assess the entire governance system rather than checking process documents alone.

For example, a vulnerability management process may exist on paper, but it will not be effective if asset ownership is unclear, business units do not prioritize remediation, reporting is incomplete, patching capacity is insufficient, exceptions are not reviewed, and the board never sees unresolved exposure in critical assets. COBIT helps reveal these governance dependencies.

### Design Factors and Tailoring

COBIT is not intended to be implemented mechanically or in full. Its purpose is to help organizations design a governance system that fits their context. COBIT’s design factors help determine which governance and management objectives should be prioritized and what level of capability may be appropriate.

Important design factors include enterprise strategy, goals, risk profile, I&T-related issues, the threat landscape, compliance requirements, the role of IT, the I&T sourcing model, IT implementation methods, technology adoption strategy, and enterprise size.

These factors matter because organizations have different governance needs. A highly regulated financial institution with extensive outsourcing, a severe threat landscape, and critical digital services will require a different governance design than a small local organization with limited digital dependence. A cloud-native software company will have different priorities than a manufacturing organization with operational technology and safety constraints.

COBIT also supports the concept of focus areas. A focus area allows governance guidance to be adapted to a specific topic, domain, or problem. ISACA has published a COBIT Focus Area for Information Security, which helps organizations identify security-relevant governance and management practices, activities, and metrics within the broader framework.

### Performance Management and Capability

COBIT performance management supports the assessment of capability on a scale from 0 to 5. At level 0, a process is incomplete. At level 1, it is performed. At level 2, it is managed. At level 3, it is established. At level 4, it is predictable. At level 5, it is optimized.

Managers should not assume that every governance or management objective must reach the highest capability level. Target capability should be justified by enterprise goals, design factors, risk exposure, compliance obligations, cost, and operational importance. A well-governed organization may intentionally set higher capability targets for risk management, security, continuity, compliance, identity, and assurance, while setting lower targets for less critical areas.

The important governance question is not whether every process is perfect. It is whether the target capability is appropriate, whether the current capability is known, whether the gap is understood, and whether management has an improvement plan for material weaknesses.

### Using COBIT to Govern Cybersecurity

A practical cybersecurity application of COBIT can follow this sequence:

- Identify the business drivers, pain points, risks, regulatory obligations, and stakeholders that require cybersecurity attention.

- Use the goals cascade and design factors to prioritize relevant governance and management objectives.

- Map existing roles, committees, policies, processes, controls, information flows, services, skills, and metrics to the COBIT governance system components.

- Assess current capability and define realistic target capability levels.

- Map detailed controls from ISO/IEC 27002, NIST SP 800-53, CIS Controls, OWASP guidance, or sector requirements to the selected COBIT objectives.

- Develop a prioritized improvement roadmap with accountable owners, resources, milestones, and performance measures.

- Monitor performance, control effectiveness, risk, conformance, and assurance results, then repeat the improvement cycle.

COBIT therefore helps answer managerial questions that a control catalogue alone cannot fully address. Who has decision authority? Who is accountable for risk acceptance? Which committee monitors cybersecurity performance? What information reaches the board? How do technology investments support enterprise goals? How are risk, compliance, internal control, and assurance coordinated?

### COBIT and the Other Frameworks

The main references in this chapter can be viewed as complementary layers.

COBIT provides enterprise governance, accountability, goals, decision rights, performance management, resource optimization, and assurance. ISO/IEC 27001 sets auditable requirements for an information security management system. ISO/IEC 27005 and the NIST Risk Management Framework structure risk management activities. ISO/IEC 27002 and NIST SP 800-53 provide control guidance at varying levels of detail. NIST CSF 2.0 offers an outcome-based structure for communicating cybersecurity posture and priorities. CIS Controls and OWASP guidance provide prioritized or technology-specific implementation support.

COBIT should not be presented as a substitute for these references. Its value lies in helping integrate them into a coherent system of enterprise governance. It helps managers connect cybersecurity controls to enterprise objectives, risk decisions, accountability, funding, metrics, assurance, and continual improvement.

### Certification and Assurance

COBIT is not an organizational certification standard comparable to ISO/IEC 27001. An organization does not become “COBIT certified” through a certification audit of its governance system. ISACA offers COBIT-related credentials for individuals, while organizations use COBIT to design governance systems, assess capabilities, define audit criteria, structure assurance work, and guide improvement programs.

COBIT’s MEA domain is particularly relevant to internal audit, compliance, and assurance because it links performance monitoring, internal control, external requirements, and independent assurance. For cybersecurity governance, this is important because controls must not only exist; they must be monitored, evidenced, reviewed, and improved. COBIT helps ensure that cybersecurity assurance is integrated with the broader enterprise governance system rather than treated as a standalone technical review.

## Combining frameworks in practice

Real organizations rarely adopt a single standard or framework in isolation. Instead, they build a composite governance architecture that reflects their regulatory obligations, sector expectations, customer requirements, legacy practices, technology environment, risk profile, and strategic choices. The goal is not to accumulate frameworks for their own sake but to combine them coherently so that governance, risk management, control selection, implementation, monitoring, assurance, and reporting reinforce one another.

For example, a Canadian financial institution might combine several references as follows:

- At the board and executive levels, COBIT can be used to clarify governance responsibilities, align information and technology objectives with enterprise goals, guide risk and resource decisions, and structure performance and assurance reporting.

- NIST Cybersecurity Framework 2.0 can be used to structure cybersecurity outcomes and executive reporting through the functions of Govern, Identify, Protect, Detect, Respond, and Recover.

- Internally, the organization may implement an ISO/IEC 27001:2022-conformant Information Security Management System (ISMS) to govern information security across defined business units, services, data environments, or enterprise processes. The ISMS defines the scope, policy, roles and responsibilities, risk assessment processes, objectives, documented information, internal audit, management review, and continual improvement.

- For control selection and implementation guidance, the organization may rely on ISO/IEC 27002:2022 and the Annex A controls of ISO/IEC 27001. These controls can be tailored to internal control baselines for different types of systems, data classifications, business processes, cloud services, suppliers, or operational environments.

- For systems that handle highly sensitive information, support critical services, or operate under U.S. public-sector or supplier-assurance expectations, the organization may map its ISO-based controls to NIST SP 800-53 Rev. 5 and use the applicable SP 800-53B baselines as a more detailed control reference.

- The organization’s risk process may be designed to align with ISO/IEC 27005:2022, the NIST Risk Management Framework, and the enterprise-wide risk management framework. COBIT objectives such as EDM03 Ensured Risk Optimization, APO12 Managed Risk, and APO13 Managed Security can help link those risk and security processes to executive oversight and governance decision-making.

- Monitoring and assurance may be coordinated through ISO management review and internal audit requirements, NIST control assessment practices, COBIT’s MEA objectives, and the organization’s internal audit and compliance functions. This approach helps the organization avoid duplicating assurance activities while still meeting the needs of multiple stakeholders.

<img src="media/image48.png" style="width:6in;height:4.5in" />

<span id="_Toc235166930" class="anchor"></span>Figure 48: Combining frameworks in practice

In a smaller organization, the combination may be simpler. NIST CSF 2.0 can provide an outcome-based structure for cybersecurity planning and communication. CIS Controls v8.1 can provide a prioritized baseline of safeguards. A proportionate risk assessment can be informed by ISO/IEC 27005 and the NIST RMF without adopting their full complexity. A limited subset of COBIT objectives can be used to clarify accountability, reporting, supplier governance, resource decisions, and performance measures.

Implementation teams may also complement governance frameworks with more technical references. CIS Controls v8.1 provides a prioritized set of safeguards mapped to multiple legal, regulatory, and policy frameworks. The OWASP Top 10:2025 provides an awareness baseline for major web application security risks. The OWASP Top 10 for LLMs and Gen AI Apps 2025 provides a focused reference for risks associated with generative AI and large language model applications. These references are not substitutes for governance frameworks, but they help translate governance expectations into practical implementation concerns.

The key governance point is that these references are not mutually exclusive competitors. They answer different questions. COBIT helps explain why governance decisions are needed, who is accountable, how decisions are structured, and how performance and assurance are monitored. ISO/IEC 27001 establishes a management system for information security. ISO/IEC 27002 and NIST SP 800-53 support control selection and implementation. ISO/IEC 27005 and the NIST RMF structure risk management. NIST CSF 2.0 supports outcome-based communication and improvement planning. CIS and OWASP provide prioritized or technology-specific implementation guidance.

What matters is coherence. A well-governed organization should be able to explain how its chosen combination supports strategy, meets obligations, manages risk, allocates responsibility, avoids duplication, and provides reasonable assurance to stakeholders. The danger is not using multiple frameworks; the danger is treating them as disconnected checklists. Mature governance integrates them into a traceable architecture in which business objectives lead to risk decisions, risk decisions lead to controls, controls lead to evidence, and evidence supports oversight, assurance, and continual improvement.

## Certification, audits and maturity

Certification and audits are often the most visible ways standards appear in practice. They translate standards from abstract reference documents into evidence, findings, management reviews, corrective actions, and assurance statements that can be presented to customers, regulators, partners, insurers, boards, and internal stakeholders.

ISO/IEC 27001:2022 is certifiable. An accredited certification body can audit an organization’s Information Security Management System against the standard’s requirements and, if those requirements are met, issue a certificate for a defined scope. The scope is important. Certification does not necessarily mean the entire organization is certified, nor does it mean that every system, business unit, supplier, or process is included. It means the ISMS, as defined by the certification scope, has been assessed against ISO/IEC 27001 requirements.

An ISO/IEC 27001 certificate does not guarantee that no cybersecurity incident will occur. It does not prove that every control is perfect or that every risk has been eliminated. Rather, it indicates that the organization has implemented a recognized management-system discipline for identifying information security risks, selecting and operating controls, monitoring performance, conducting reviews, addressing nonconformities, and improving over time. For governance purposes, the certificate is part of the organization’s assurance story, but it should not be treated as a substitute for ongoing risk management.

Internal audits are also central to ISO/IEC 27001. Clause 9 requires the organization to evaluate ISMS performance, conduct internal audits, and carry out management reviews. Internal audits assess whether the ISMS conforms to the organization’s requirements, ISO/IEC 27001 requirements, and relevant legal, contractual, or regulatory expectations. Findings from internal audits should inform corrective actions, management review, risk treatment decisions, and continual improvement. A mature organization therefore treats internal audit not as a paperwork exercise but as a mechanism for learning whether the ISMS is operating as intended.

External certification audits provide independent assurance, while internal audits provide ongoing management insight. Both are useful, but they serve different purposes. External certification can support trust among external stakeholders. Internal audit supports governance by identifying weaknesses before they become certification issues, regulatory concerns, or incidents. Together, they help management understand whether documented controls, processes, and responsibilities are actually being followed.

NIST-based environments use different terminology, but the underlying governance logic is similar. Rather than certification, the NIST Risk Management Framework emphasizes control assessment, authorization, and continuous monitoring. NIST SP 800-53A provides assessment procedures to determine whether controls in SP 800-53 are implemented correctly, operating as intended, and producing the desired outcome. The result is not an ISO-style certificate. Rather, it supports an authorization decision: an accountable official determines whether the system may operate with the remaining risk.

This authorization logic is highly relevant beyond the U.S. federal environment. It reinforces an important governance principle: systems should not operate with significant residual risk simply because they are already in production. Someone with appropriate authority should review the evidence, assess the risk, and decide whether operation is acceptable. That decision should be documented and revisited as systems, threats, vulnerabilities, controls, and business conditions change.

COBIT offers a distinct performance-management perspective. It is not an organizational certification standard comparable to ISO/IEC 27001. Instead, COBIT supports the assessment of governance and management capability. Process capability can be assessed on a scale from level 0 to level 5, ranging from incomplete to optimized. COBIT also allows maturity to be considered across focus areas that combine several governance and management objectives, components, practices, information flows, roles, and measures.

Capability and maturity assessments complement formal compliance. Compliance asks whether requirements are met. Maturity asks how well practices are institutionalized, measured, managed, and improved. A process may technically exist yet remain immature if it relies on informal knowledge, inconsistent execution, weak evidence, unclear ownership, or reactive decision-making. Conversely, a mature process is repeatable, owned, measured, reviewed, improved, and aligned with business risk.

Managers should not assume that every process must reach the highest maturity or capability level. Higher maturity often entails additional cost, expertise, tooling, documentation, and management attention. The appropriate target should be based on risk exposure, regulatory obligations, business criticality, stakeholder expectations, and resource constraints. A highly regulated organization may require strong maturity in identity management, incident response, continuity, supplier governance, and assurance, whereas a smaller organization may adopt a more proportionate target.

<img src="media/image49.png" style="width:6in;height:4.5in" />

<span id="_Toc235166931" class="anchor"></span>Figure 49: Certifications, Audits and Maturity

For business technology managers, the key role is translation. Audit findings, certification results, control assessments, and maturity ratings must be converted into language executives can use. For example: incident response may be repeatable but not yet measured; supplier security may be partially implemented; access reviews may meet policy but lack strong evidence; and vulnerability management may be technically operational but fail to meet business-priority remediation targets. The managerial task is to explain what these findings mean, the risk they pose, the improvement required, the cost, and how progress will be monitored.

Certification, audits, authorizations, and maturity assessments should not be treated as separate activities. Together, they form an assurance system. They help the organization determine whether controls exist, whether they operate, whether risks are consciously accepted, whether weaknesses are corrected, and whether cybersecurity governance is improving over time.

## Implications for business technology managers

Control frameworks and standards should not appear as an arcane concern reserved for auditors or compliance specialists. They are essential tools for governance. They help managers translate cybersecurity from broad intent into structured accountability, risk-based control selection, measurable performance, evidence, and assurance.

For business technology managers, several implications stand out:

- First, managers must recognize the main standards and frameworks and understand their distinct roles. When someone says that the organization wants to obtain ISO/IEC 27001 certification, that a client requires alignment with NIST SP 800-53 controls, that the board wants reporting based on NIST CSF 2.0, or that internal audit is using COBIT to assess governance capability, the manager should understand what kind of reference is involved. The requirement may concern a management system, a control catalogue, a risk management process, an enterprise governance model, a maturity assessment, or a communication structure for executives and stakeholders.

- Second, managers must be comfortable with scoping and tailoring. Very few organizations can apply every control from every framework to every system, process, business unit, supplier, and data set at the same time. Governance requires explicit decisions about what is in scope, which risks matter most, which controls are necessary, which objectives deserve priority, which evidence to retain, and which residual risks to accept. These decisions should not be hidden inside technical implementation. They should be documented, justified, approved by the appropriate risk owners, and reviewed as conditions change.

- Third, managers must understand the link between risk management and control selection. ISO/IEC 27005 and the NIST Risk Management Framework both reinforce the same principle: controls are not ends in themselves. They are selected to treat specific risks, satisfy obligations, support business objectives, and maintain acceptable levels of residual risk. In practice, managers will often be asked why a control is necessary, why a control was not implemented, or why a control can be relaxed. Connecting these discussions to risk analysis, business impact, legal obligations, and risk appetite is central to responsible decision-making.

- Fourth, managers should treat audits, certification, assessments, and maturity reviews as normal components of governance rather than as hostile inspections. Internal audits, external audits, control assessments, certification audits, and maturity reviews help the organization determine whether its policies, processes, and controls are working. Findings should not be viewed only as problems to defend against. They provide evidence to support corrective action, resource allocation, management review, and improvement roadmaps.

- Fifth, managers must learn to navigate between levels of abstraction. On one day, they may discuss NIST SP 800-53 control requirements, ISO/IEC 27002 safeguards, CIS Controls, or OWASP guidance with technical teams. On another day, they may use the NIST Cybersecurity Framework to explain cybersecurity outcomes to executives, COBIT’s goals cascade to connect cybersecurity to enterprise objectives, or ISO/IEC 27001 audit results to explain the status of an ISMS. Business technology managers must therefore be translators: they connect technical controls to risk, risk to governance, governance to performance, and performance to assurance.

- Finally, managers must avoid treating frameworks as disconnected checklists. The value of standards and frameworks comes from coherence. A mature organization can explain how its chosen references fit together: how business objectives lead to risk decisions, how risk decisions lead to control selection, how controls generate evidence, how evidence supports assurance, and how assurance drives improvement. This ability to connect strategy, accountability, risk, controls, performance, and assurance is one of the central competencies of cybersecurity governance.

## Conclusion

Cybersecurity control frameworks and standards are more than reference documents. They are practical governance instruments that help organizations translate broad cybersecurity intentions into structured management systems, risk processes, control decisions, evidence, assurance, and continual improvement. Without them, cybersecurity governance risks becoming reliant on personal judgment, fragmented practices, inconsistent terminology, and undocumented assumptions.

This chapter has shown that different references serve distinct purposes. ISO/IEC 27001 provides a certifiable management-system structure for information security. ISO/IEC 27002 and NIST SP 800-53 offer control guidance at varying levels of detail. ISO/IEC 27005 and the NIST Risk Management Framework help structure risk management and risk-based control selection. NIST CSF 2.0 provides a concise, outcome-based language for communicating cybersecurity posture and improvement priorities. COBIT connects information and technology to enterprise governance, decision rights, performance, resources, assurance, and accountability. CIS and OWASP guidance can then support more focused implementation decisions in specific technical domains.

The most important managerial lesson is that no framework should be applied mechanically. Frameworks must be interpreted, scoped, tailored, mapped, and justified against organizational objectives, risk appetite, legal and contractual obligations, available resources, technology architecture, supplier dependencies, and stakeholder expectations. A control is valuable not because it appears in a catalogue, but because it addresses a meaningful risk, supports a business objective, satisfies an obligation, strengthens resilience, or provides credible assurance.

For business technology managers, the ability to work with these frameworks is a core professional skill. Managers must be able to move between the languages of executives, auditors, regulators, technical teams, suppliers, and business owners. They must understand how a board-level concern becomes a risk criterion, how a risk criterion becomes a control requirement, how a control requirement becomes an implementation project, how implementation generates evidence, and how that evidence supports assurance and improvement.

Ultimately, standards and frameworks make cybersecurity governable. They do not eliminate uncertainty, guarantee security, or replace managerial judgment. Instead, they make judgment more disciplined, transparent, comparable, and defensible. A mature organization uses them not as disconnected checklists but as an integrated architecture that aligns cybersecurity with strategy, risk, compliance, resilience, and trust.
