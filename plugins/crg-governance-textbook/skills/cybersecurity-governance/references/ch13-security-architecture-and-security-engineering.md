<!-- Léger, M.-A. (2026). Introduction to Cybersecurity Governance for Business Technology Management (3rd ed., v2.51). Chapter 13: Security Architecture and Security Engineering. CC BY-NC 4.0 -->

# Chapter 13: Security Architecture and Security Engineering

## Learning outcomes

After completing this chapter, you should be able to:

1.  Distinguish security architecture from security engineering and explain how risk owners and governance complete a single accountability chain.

2.  Explain why architectural controls offer greater risk-reduction leverage than detective or procedural controls, using the scenario logic of Chapters 4 to 6.

3.  Compare SABSA, TOGAF, and Open Security Architecture and recommend a defensible way to combine them in an organization.

4.  Analyze the seven architectural pillars as an interacting system and identify the governing decision and the evidence expected for each.

5.  Evaluate a zero trust program against NIST SP 800-207, the CISA Zero Trust Maturity Model Version 2.0, and the governance pitfalls described in the chapter.

6.  Design the governance mechanisms that make architecture governable: threat modelling gates, reference patterns, a design authority, architecture decision records, and exceptions with expiry.

7.  Justify treating security debt as cybersecurity risk and recommend coverage-based metrics and key risk indicators for reporting architecture to leadership.

Cybersecurity governance is often taught through principles, policies, and standards. Cybersecurity operations are often taught through tools, detection, response, and teams. Both are necessary. Yet managers frequently struggle in the space between them: translating risk priorities into a coherent set of design decisions that shape how technology is built, connected, operated, and recovered. That space is the domain of security architecture and security engineering. It is where a board's statement of risk appetite either becomes a structural property of the organization's systems or remains a sentence in a policy.

**Security architecture** is the disciplined design of protection capabilities across the organization's systems, data flows, and processes. **Security engineering** is the practical implementation of those designs in real environments—where constraints, legacy dependencies, budgets, suppliers, and human behaviour determine what actually works. For managers, neither is a technical topic in the narrow sense. Both are managerial control domains, because they determine the organization's structural exposure to cyber risk, the cost of security over the life of each system, the operational friction imposed on business units, and the credibility of every assurance claim made to a board, regulator, auditor, or customer.

Standards and control catalogues, examined in Chapter 9, describe what should exist. Architecture provides the rationale and structure for what the organization will actually build, how the parts fit together, what must be enforced by design rather than by vigilance, and how the result will be operated and recovered. Executives rarely approve control catalogues; they approve roadmaps, platform choices, operating models, dependencies, and trade-offs. Architecture is the language of those approvals, and the place where a poor approval is most expensive to reverse.

This chapter builds on the risk logic of Chapters 4 to 6. Scenario-based risk assessment tells managers which combinations of threat, vulnerability, and impact matter most; architecture is how the organization reduces the likelihood and impact of those scenarios durably, across many systems at once. It also connects to neighbouring chapters: identity and access management (Chapter 10), baselines (Chapter 14), cloud (Chapter 15), third parties (Chapter 19), cryptography (Chapter 20), and investment (Chapter 17) all depend on architectural choices made, or avoided, earlier.

The chapter first distinguishes architecture from engineering and introduces security by design and secure by default as the organizing philosophy of modern programs. It then examines the enterprise architecture frameworks that give security architecture its method—SABSA, TOGAF, and Open Security Architecture—and the seven architectural pillars that most programs govern and invest in, treated at the level of architectural decisions while the detailed practices are left to other chapters. Zero trust architecture is presented as a governance concept rather than a product category. The final sections address how architecture is governed in practice: threat modelling, reference architectures and patterns, review boards and design authority, decision records and exceptions, technical debt as risk, and the metrics that make the whole visible to leadership. An illustrative scenario and questions for managers close the chapter.

## Architecture and engineering: what managers must govern

The words *architecture* and *engineering* are used loosely in practice, and the looseness causes real governance problems. When everyone calls themselves an architect, no one owns the target state. When engineering is understood only as tool deployment, no one owns whether the tools add up to a coherent defence. Managers do not need to master the technical content of either discipline, but they do need a clear model of what each is for, who decides what, and what evidence each should produce.

### Two disciplines, one accountability chain

A simple managerial distinction is helpful. Security architecture answers the question: *what must be true about the organization's technology environment for risk to remain within appetite?* It defines target states, standard patterns, reference architectures, mandatory design constraints, and the principles against which proposed designs are judged. Its purpose is to structure systems so that security outcomes do not depend on heroics, memory, or the goodwill of individual administrators.

Security engineering answers a different question: *how will we build and operate those target states in our real environment?* It selects implementation methods and tooling, writes the configuration code, integrates controls into delivery pipelines, defines how controls will be tested, and ensures that what is designed can actually be run by the operations teams described in Chapter 7. Engineering turns design into working capability and, critically, produces the evidence that the capability exists and functions.

The two disciplines form a single accountability chain. Architecture without engineering produces elegant diagrams that never reach production. Engineering without architecture produces a patchwork in which controls exist but do not align, monitoring lacks coverage, privileged access is unmanaged, and recovery has never been tested. Mature organizations treat both as competencies with named owners, decision rights, and measurable outcomes, connected to the risk register so that each architectural decision can be traced to the scenarios it addresses.

- **Architecture owns the "what" and the "why":** target states, principles, patterns, and the design constraints that projects must respect.

- **Engineering owns the "how" and the proof:** implementation, automation, testing, and the operational evidence that a design works as intended.

- **Risk owners own the residual risk:** when a design falls short of the target state, the business owner of the affected service, not the architect, decides whether to accept the gap, within the limits of risk appetite.

- **Governance owns the system as a whole:** it sets the principles, funds the roadmap, resolves conflicts between delivery speed and risk, and monitors whether architecture is actually reducing exposure.

### Why architecture is a risk management instrument

Risk management, as developed in Chapters 4 to 6, is scenario-based: threat actors exploit vulnerabilities to produce impacts on things the organization values. Controls can act at many points in that chain, but they do not all act with the same leverage. A detective control catches one instance of a problem. A procedural control relies on people remembering to follow it. An architectural control changes the structure of the environment so that whole classes of scenarios become less likely or less harmful, for every system built on that structure, without anyone having to remember anything.

Consider a few structural mechanisms. Segmentation limits the blast radius of a compromised workstation. Phishing-resistant authentication removes an entire family of credential-theft scenarios. Standard baselines enforced as code prevent configuration drift across thousands of assets. Immutable, isolated backups convert a catastrophic ransomware scenario into a survivable, if painful, disruption. Each of these is a design decision taken once and inherited many times. That is why a single well-chosen architectural pattern can reduce risk across dozens or hundreds of systems, often at a lower total cost than compensating through detection, manual review, or after-the-fact remediation.

This is also why architecture belongs directly under governance. When a board asks whether the organization is managing its cyber risk responsibly, the operational proof is frequently found in architecture: whether privileged access is constrained, whether critical systems are segmented, whether logging supports detection and investigation, whether encryption keys are controlled, and whether recovery is realistic. These are questions about structure, and structure is what architecture governs. **Architectural decisions are among the highest-leverage risk decisions an organization makes, and they deserve the same governance attention as major investment decisions.**

| Policies tell people what to do; architecture determines what the environment allows. **Where the two disagree, the environment usually wins.** |
|-------------------------------------------------------------------------------------------------------------------------------------------------|

### Where architecture appears in standards and regulation

Security architecture is not an optional refinement that sits outside the frameworks discussed in Chapter 9. ISO/IEC 27002:2022 includes a specific control, 8.27 *Secure system architecture and engineering principles*, which expects organizations to establish, document, maintain, and apply such principles to any information system development activity, alongside related controls on the secure development life cycle (8.25), secure coding (8.28), and configuration management (8.9). An ISMS certified to ISO/IEC 27001:2022 should therefore be able to show not only that principles exist, but that projects actually apply them.

NIST CSF 2.0 places architectural concerns in several categories of the Protect function, notably Platform Security (PR.PS) and Technology Infrastructure Resilience (PR.IR), while the Govern function expects cybersecurity strategy, roles, and policy to shape how such outcomes are pursued. In COBIT 2019 terms, security architecture lives within the management objective for managed enterprise architecture (APO03) and the build objectives that follow it, which helps connect security design to enterprise goals rather than treating it as an isolated technical concern.

In Canada, federally regulated financial institutions face explicit expectations. OSFI Guideline B-13, *Technology and Cyber Risk Management*, published in July 2022 and effective since January 1, 2024, states as Principle 4 that institutions should implement a technology architecture framework with supporting processes, expects systems to be designed for security (secure-by-design) and resilience (resilience-by-design) using a risk-based approach, and asks institutions to adopt secure-by-design practices to safeguard technology assets. For organizations outside that perimeter, the same logic increasingly appears in privacy law: Québec's Law 25 requires a privacy impact assessment for any project to acquire, develop, or overhaul an information system involving personal information and requires that technological products offered to the public provide the highest level of confidentiality by default; the EU GDPR's Article 25 requires data protection by design and by default. Architecture is where these legal duties become technically real.

## Security by design and secure by default

The most consequential shift in security architecture over the past decade is not a technology. It is a change in where security work is expected to happen. For many years, security was something added to a system after it was designed, often by a different team, frequently at the end of a project when the cost of change was highest. **Security by design** reverses that sequence: security requirements, threat analysis, and architectural constraints are introduced at the beginning and carried through the life of the system. **Secure by default** adds a second expectation: that systems and products should be safe in their out-of-the-box configuration, without requiring the customer or the operator to discover and enable protective settings.

### From retrofitting to design-time decisions

The economic argument for security by design is straightforward. Decisions made early—which identity provider a system trusts, whether a component is exposed to the Internet, how data is partitioned, which protocols are allowed—constrain everything that follows. Changing them after deployment typically means redesign, data migration, retesting, and service disruption. Adding compensating controls instead creates permanent operating cost. The organization pays either way; the question is whether it pays once, at design time, or repeatedly, for the life of the system.

NIST SP 800-160 Vol. 1 Rev. 1, *Engineering Trustworthy Secure Systems*, published in November 2022, provides the most comprehensive public treatment of this discipline. It frames systems security engineering as an integral part of systems engineering rather than a parallel activity, organizes the work around a problem context (what must be protected and why), a solution context (how protection is achieved), and a trustworthiness context (how stakeholders gain justified confidence that the solution works), and stresses that the approach applies to agile and DevOps delivery as much as to traditional life cycles. Its companion, NIST SP 800-160 Vol. 2 Rev. 1, focuses on cyber resiliency engineering—designing systems that can anticipate, withstand, recover from, and adapt to adverse conditions, including successful attacks.

For managers, the essential lesson of SP 800-160 is its emphasis on justified confidence. A secure design is not one that the architect believes to be secure; it is one for which there is an argument, supported by evidence, that the security requirements are complete, consistent, and actually met. That framing fits naturally with the book's emphasis on defensible decisions and on evidence that can withstand audit and incident review.

### CISA Secure by Design and the customer's role

The United States Cybersecurity and Infrastructure Security Agency (CISA) and partner agencies have pushed the same idea from the side of technology producers. Their joint guidance *Shifting the Balance of Cybersecurity Risk* was first released in April 2023 (subtitled *Principles and Approaches for Security-by-Design and -Default*) and reissued in an expanded edition in October 2023 under the subtitle *Principles and Approaches for Secure by Design Software* with a broader group of international partners, including the Canadian Centre for Cyber Security (CCCS). It articulates three principles for software manufacturers: take ownership of customer security outcomes, embrace radical transparency and accountability, and lead from the top. The underlying argument is that the burden of security has fallen disproportionately on customers, who are least able to carry it, and that manufacturers should eliminate classes of vulnerability—for example, through memory-safe languages, default multi-factor authentication, and the removal of default passwords—rather than shipping products that are secure only if configured correctly.

Most organizations reading this book are customers rather than manufacturers, so the managerial relevance lies in procurement and in internal development. CISA's subsequent Secure by Demand guidance (2024) helps customers ask suppliers the right questions: how they authenticate users by default, whether they publish vulnerability disclosure policies, whether security logs are provided to customers at no additional charge, and whether they can provide a software bill of materials. These questions belong in the third-party assessment processes described in Chapter 19, and Chapter 14 applies the secure-by-default principle to configuration baselines. Regulation is moving in the same direction: the EU Cyber Resilience Act, Regulation (EU) 2024/2847, imposes essential cybersecurity requirements on products with digital elements sold in the EU, with manufacturers' vulnerability and incident reporting obligations applying from September 11, 2026 and most other obligations from December 11, 2027 (see Chapters 11 and 19). Canadian firms selling software or connected devices into Europe should treat secure by default as a market-access condition, not merely a good practice.

### Secure software development as a governed process

When the organization itself builds software—whether a customer portal, an internal integration, or a commercial product—security by design must be embedded in the development process. NIST SP 800-218, the *Secure Software Development Framework* (SSDF) Version 1.1, published in February 2022, is the reference most widely used for this purpose. It groups practices into four families: prepare the organization, protect the software, produce well-secured software, and respond to vulnerabilities. SP 800-218A (July 2024) extends the SSDF with a community profile for generative AI and dual-use foundation models, which links this chapter to Chapter 18. At the time of writing, a draft Revision 1 (SSDF Version 1.2) released in December 2025 was still at the public-draft stage; managers should confirm its status before citing it contractually.

The governance value of the SSDF is that it is outcome-oriented and life-cycle neutral: it prescribes no methodology, but it gives buyers and suppliers a common vocabulary and gives internal audit a basis for asking whether secure development is actually practised. Application weaknesses themselves are catalogued in references such as the OWASP Top 10:2025, and testing is discussed in Chapter 8. The architectural point is narrower: the organization should decide which development practices are mandatory for which classes of system, and require evidence of them before a system enters production.

- **Security by design is a sequencing decision:** it moves security analysis to the point where change is cheapest and design options are widest.

- **Secure by default is a burden-allocation decision:** it places responsibility for safe configuration on the party best able to carry it, whether a supplier or an internal platform team.

- **Both require evidence:** threat models, design reviews, test results, and configuration records that show the principles were applied, not merely endorsed.

## Enterprise architecture frameworks and security architecture

Security architecture does not exist in isolation. Most medium and large organizations practise some form of enterprise architecture: a structured description of business capabilities, information, applications, and technology, together with a method for moving from the current state to a desired future state. Security architecture is most effective when it is woven into that method rather than run as a parallel exercise. Three references are particularly useful for business technology managers: SABSA, TOGAF, and Open Security Architecture.

### SABSA: business-driven security architecture

SABSA (originally the Sherwood Applied Business Security Architecture) is a methodology for developing security architectures that are driven by business requirements rather than by technology. Its central idea is that security should be justified by traceable links to business attributes—qualities such as *available*, *confidential*, *auditable*, *compliant*, or *reputation-protected* that the business needs from a service. Through a technique called business attributes profiling, stakeholders define what each attribute means for a given service and how it will be measured, and every architectural element is traced back to the attributes it supports.

SABSA organizes the work into layers that mirror the questions asked by different stakeholders: a contextual layer (the business view), a conceptual layer (the architect's view of what is needed), a logical layer (the designer's view), a physical layer (the builder's view), a component layer (the tradesperson's view of specific products and standards), and a cross-cutting service management layer concerned with operating the result. The managerial strength of this structure is two-way traceability: a board can ask why a control exists and be led back to a business attribute, and an engineer can ask which business outcome would suffer if a component failed. The SABSA Institute maintains the methodology and its certifications.

### TOGAF and security in the enterprise architecture cycle

The TOGAF Standard, maintained by The Open Group, is among the most widely used enterprise architecture frameworks. Its current edition, the TOGAF Standard, 10th Edition, was launched in April 2022 with a modular structure that separates a stable body of fundamental content from series guides addressing specific topics. Its core is the Architecture Development Method (ADM), an iterative cycle that moves from architecture vision through business, data, application, and technology architectures to opportunities and solutions, migration planning, implementation governance, and architecture change management, all anchored by requirements management.

TOGAF's significance for security is that it provides the organizational machinery—architecture principles, an architecture repository, architecture contracts, compliance reviews, and an architecture board—within which security requirements can be expressed and enforced. The Open Group has published guidance on integrating security and risk into the ADM, and as early as 2011 it issued, with The SABSA Institute, a white paper on integrating TOGAF and SABSA so that the business-attribute traceability of SABSA could run through the TOGAF cycle. In practice, organizations that already use TOGAF should not create a separate security architecture method; they should ensure that security requirements, threat analysis, and risk acceptance appear at each relevant phase of the ADM and that the architecture board's terms of reference explicitly include security.

### Open Security Architecture and pattern libraries

Open Security Architecture (OSA) is a community resource, active since 2008 and recently relaunched on a modernized platform, that publishes security architecture patterns and a control catalogue under an open Creative Commons licence. Its patterns describe common situations—cloud computing, identity management, remote access, industrial control systems, payment environments—and the controls and threat mitigations each requires, with its control catalogue rooted in NIST SP 800-53 and mapped to other frameworks such as ISO/IEC 27001:2022, the CIS Controls, and NIST CSF 2.0.

For a resource-constrained organization, such a library offers a reviewed starting point already mapped to the frameworks auditors ask about. Cloud providers' reference architectures and well-architected frameworks, and CCCS architecture guidance, serve a similar purpose for particular platforms (see Chapter 15). The governance caution is that external patterns must be adapted to the organization's risk appetite, threat profile, and regulatory context. A pattern adopted without tailoring becomes a form of compliance theatre: the diagram looks authoritative, but no one has asked whether it addresses the organization's actual scenarios.

### Choosing and combining frameworks

As with the control frameworks in Chapter 9, organizations rarely adopt one architecture framework in pure form. A common and defensible combination uses TOGAF, or a lighter internal equivalent, as the enterprise architecture method; SABSA's business-attribute logic to justify and trace security requirements; and curated patterns from OSA, cloud providers, and CCCS as the starting point for reference architectures. The more important managerial decision is not which framework to name, but whether the organization has an architecture method at all, whether security is integrated into it, and whether the method produces decisions that are recorded, reviewed, and traceable to risk.

## The seven architectural pillars

Whatever framework an organization uses, its security architecture ultimately resolves into a limited number of structural capabilities that must be designed, funded, and governed. This book groups them into seven pillars. The pillars are not a control catalogue; they are the major architectural decision areas in which the organization's risk appetite becomes concrete. Each pillar has detailed practices that belong to other chapters, and the sections that follow deliberately stay at the level of architecture: what must be decided, who should decide it, and what evidence should show that the decision was implemented.

The figure below presents the seven pillars. Six of them are protective and detective structures. The seventh, resilience architecture, is shown as the foundation on which the others rest, because it is the pillar that assumes the others will sometimes fail.

<span id="_Toc243635559" class="anchor"></span>Figure 60: The seven architectural pillars of security architecture

### Reading the pillars as a system

The pillars interact, and their value lies in the interaction. Identity determines who can act; segmentation determines what a compromised identity or device can reach; baselines determine how much an attacker can exploit once there; encryption and key management determine what data is worth stealing; logging determines whether anyone notices; and resilience determines whether the organization can recover when all of the above prove insufficient. A weakness in one pillar can sometimes be compensated by strength in another, which is exactly why architecture must be governed as a whole. A decision to defer segmentation, for example, raises the importance of identity controls and detection coverage, and should be recorded as such.

The pillars also map to frameworks managers already use: identity and privileged access to CIS Controls 5 and 6 and to the identity management, authentication, and access control category (PR.AA) of NIST CSF 2.0; baselines to CIS Control 4 and PR.PS; logging to CIS Control 8 and the Detect function; resilience to CIS Control 11, PR.IR, and the Recover function. The organization can therefore report on architecture in its chosen framework's language while governing it as a set of structural programs.

### Pillar 1: identity and privileged access architecture

Identity is the control plane of modern cybersecurity. In cloud, SaaS, remote work, and hybrid environments, the most important security boundary is no longer the network perimeter; it is the identity and authorization model. Many serious intrusions either begin with an identity compromise or quickly become a privileged access problem, which is why organizations must govern identity as critical infrastructure rather than as a directory service.

The architectural decisions in this pillar are few but consequential. The organization must decide whether it will operate a single authoritative identity provider or tolerate several, and how external partners and customers will be federated. It must set its authentication standard, including where phishing-resistant authentication is mandatory, in line with the assurance levels of NIST SP 800-63-4. It must choose an authorization model—role-based, attribute-based, or a combination—that can actually be enforced across its systems. It must decide how privileged access will be separated from ordinary access, whether privilege will be standing or granted just in time, how administrative pathways will be isolated, and how non-human identities such as service accounts, workloads, and AI agents will be governed. Finally, it must decide how identity itself will be recovered if the identity provider is compromised or unavailable.

The detailed practices—joiner–mover–leaver processes, access reviews, privileged access management tooling, session recording, credential vaulting, and break-glass procedures—are treated in Chapter 10. The architectural control point is captured by a single question: *can we demonstrate that access is minimal, auditable, and difficult to abuse, especially for privileged actions?* If identity design is weak, no amount of endpoint tooling or policy will fully compensate.

### Pillar 2: endpoint and workload security architecture

Endpoints (laptops, workstations, mobile devices) and workloads (servers, virtual machines, containers, serverless functions) remain the places where code executes and where attackers establish footholds. Endpoint and workload architecture must therefore prevent compromise where feasible, detect suspicious activity reliably, support rapid containment, and preserve the information needed for investigation.

This pillar is primarily about standardization and enforceability, because a secure posture is very hard to sustain in a heterogeneous, partly unmanaged estate. The architectural decisions concern which device and workload classes the organization will support, what management and telemetry every asset must carry before it may connect to corporate resources, how much variation will be tolerated, and how workloads will be built—for example, from minimal, hardened, centrally maintained images rather than ad hoc builds. For cloud-native workloads, the decision extends to whether workloads are treated as immutable, replaced rather than patched, which changes both the risk profile and the operating model (see Chapter 15).

Three questions keep this pillar honest. What proportion of endpoints and workloads are truly managed to a standard baseline, as opposed to merely inventoried? How quickly can a compromised endpoint or workload be isolated without paralyzing the business process it supports? Can the organization produce evidence of control execution—patch status, encryption status, endpoint detection and response coverage, and containment actions—on demand? Detection engineering and response capabilities built on this telemetry are discussed in Chapter 8.

### Pillar 3: segmentation and connectivity architecture

Segmentation limits blast radius. Where identity controls determine who may do what, segmentation determines what a compromised system or identity can reach. Both are needed: when one fails, the other should prevent a contained incident from becoming an enterprise-wide failure. In environments dominated by ransomware and hands-on-keyboard intrusion, segmentation is often the decisive difference between those two outcomes.

Segmentation is frequently misunderstood as a set of firewall rules. Architecturally, it is a strategy that defines zones and trust boundaries, the flows permitted between them, and the monitoring required at each boundary. Its purpose is to prevent lateral movement and to protect the organization's most valuable and most dangerous assets: identity infrastructure, finance and payment systems, sensitive data stores, backup systems, cloud management planes, and operational technology (see Chapter 11). In modern and cloud designs it increasingly operates at the level of accounts, workloads, and identities rather than network ranges, which is where it converges with zero trust.

Segmentation is a governance topic because it involves visible trade-offs. Strong boundaries create operational friction, and business units may resist changes that break long-standing integrations. Managers should therefore treat segmentation as a risk-based policy in which critical assets justify stronger boundaries, and they should reframe the debate. The relevant question is not whether segmentation is inconvenient, but what lateral movement would cost if a compromise occurred—a question that the scenario-based assessment methods of Chapter 6 are designed to answer.

## Pillars that sustain trust: baselines, keys, logs, and recovery

The first three pillars shape who can act and what they can reach. The remaining four determine whether the environment stays in the state the architecture intended, whether the most sensitive data remains protected even when infrastructure is compromised, whether the organization can see and prove what happened, and whether it can recover. They are less visible than identity or segmentation, and for that reason they are more often underfunded.

### Pillar 4: secure configuration baselines and hardening

Misconfiguration remains one of the most persistent sources of exposure in both on-premises and cloud environments. It is rarely a technology problem; it is a discipline problem. **Secure configuration baselines** provide that discipline by defining a minimum acceptable configuration for each class of asset—endpoints, servers, network devices, cloud accounts and services, containers, and key applications—and by making deviations visible.

The architectural decisions in this pillar are about how baselines are expressed and enforced, not about the individual settings. The organization must decide which external references will anchor its baselines (for example, CIS Benchmarks or vendor security baselines), who owns each baseline, how baselines will be enforced—increasingly through infrastructure-as-code and policy-as-code, so that environments are deployed securely by default rather than retrofitted—and how drift will be detected and corrected. Chapter 14 treats baseline governance in depth, including tailoring, exception management, and evidence.

Managers should avoid confusing having a baseline document with having baseline control. **The governance requirement is evidence: demonstrable coverage, measurable adherence, and a controlled exception process.** A baseline that applies to only part of the estate, or that cannot be measured, is a statement of intent rather than a control.

### Pillar 5: encryption and key management architecture

Encryption is a foundational mechanism for confidentiality and, through signatures and message authentication, for integrity. Yet encryption without disciplined key management offers only superficial comfort. Key management is where governance most often fails, because it sits at the intersection of operations, cloud provider services, third parties, and legal obligations.

The architectural decisions begin with data classification and threat analysis: which data must be protected, against which scenarios, and with what consequences if protection fails. From there, the organization must decide where encryption is mandatory (at rest, in transit, and increasingly in use for the most sensitive workloads), who controls the keys for each class of data (provider-managed, customer-managed, or customer-held), where keys are stored and with what hardware protection, how key administration is separated from data administration, and how keys are backed up and recovered. A newer architectural decision concerns **crypto-agility**: whether systems are designed so that algorithms can be replaced without redesign, which matters for the transition to post-quantum cryptography. Chapter 20 develops these topics in managerial terms.

Managers should insist on clarity about key ownership, separation of duties for key administrators, logging of key use, and tested recovery procedures. A common anti-pattern is to mandate encryption broadly without managing the key life cycle. The result is hidden fragility: during an incident, an organization may discover that it cannot recover its own data because the keys, or the people who understood them, are no longer available.

### Pillar 6: logging architecture, monitoring, and evidence

If governance is accountability, logging is its technical foundation. Without high-quality logs and an architecture that collects, protects, and analyses them, organizations cannot reliably detect intrusions, investigate incidents, meet notification obligations under laws such as Québec's Law 25 or PIPEDA, or demonstrate control operation to auditors. Logs are also the raw material for the credible metrics that senior leadership needs.

Logging architecture is not "turn on the logs". It is a design that specifies what must be logged and at what level of detail, how logs are centralized, how they are protected against tampering and deletion, how long they are retained, how they feed detection, and who may access them. The highest-value sources are usually identity and authentication events, privileged actions, administrative changes, cloud control-plane activity, endpoint and workload telemetry, boundary traffic, and the logs of security services themselves. The architectural decisions concern coverage priorities, the central platform, retention aligned with legal and investigative needs, and the separation of log storage from the systems being logged, so that an attacker who compromises a system cannot erase the record of having done so.

Logging is expensive at scale and must be prioritized by risk. The managerial question is whether coverage exists for the assets and control planes whose compromise would be catastrophic, and whether auditable evidence can be produced when asked. The operations and detection engineering that consume these logs are covered in Chapters 7 and 8.

### Pillar 7: resilience architecture

Many organizations still treat resilience as IT continuity and security as prevention. That separation is a governance error. Against ransomware, destructive attacks, supply-chain compromise, and persistent intrusion, prevention will eventually fail somewhere. What distinguishes mature organizations is the ability to recover critical services within tolerances the business has defined in advance.

Resilience architecture includes backups that are isolated and immutable, restoration mechanisms that work at the required scale, redundancy for critical systems, and the dependency mapping that reveals which services cannot be restored until others are. Its architectural decisions concern recovery time and recovery point objectives for each critical service, the degree of isolation between production and backup environments, the design of a recovery environment that can be trusted after a compromise, and the recovery of identity itself, without which nothing else can be restored. Cloud dependencies require explicit decisions as well: what happens if a region fails, a provider service degrades, or a configuration error propagates across environments.

The core deliverable of this pillar is not a plan document but **demonstrable recoverability under realistic conditions**. Managers should insist on scenario-based validation, including technical restoration tests and exercises, because resilience that has not been tested is frequently fictional. Chapter 12 addresses incident management, business continuity, and resilience in full; NIST SP 800-160 Vol. 2 Rev. 1 provides the engineering view of designing for cyber resiliency.

| Each pillar has a single governing question: **can we show, with evidence, that the architectural decision was implemented across the assets that matter most?** |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## Zero trust architecture as a governance concept

Few terms in cybersecurity have been stretched as far as *zero trust*. It is used to describe products, programs, network designs, and aspirations, often in the same presentation. For managers, the useful approach is to return to the reference definitions, to treat zero trust as a set of architectural principles and a multi-year direction of travel, and to govern it accordingly. Zero trust is not something an organization buys; it is something an organization progressively becomes.

### What NIST SP 800-207 actually says

NIST SP 800-207, *Zero Trust Architecture*, published in August 2020, remains the foundational reference. It defines **zero trust** as "a collection of concepts and ideas designed to minimize uncertainty in enforcing accurate, least privilege per-request access decisions in information systems and services in the face of a network viewed as compromised." A **zero trust architecture**, in NIST's words, is "an enterprise's cybersecurity plan that utilizes zero trust concepts and encompasses component relationships, workflow planning, and access policies." The central shift is the abandonment of implicit trust based on network location: being "inside" the corporate network no longer confers access.

SP 800-207 sets out a small number of tenets that are useful for governance precisely because they are technology-neutral. Every data source and computing service counts as a resource. Communication must be protected whether it originates inside or outside the corporate network. Access to individual resources is granted per session, on the basis of dynamic policy that considers the identity of the requester, the state of the device, the sensitivity of the resource, and other behavioural and environmental attributes. The enterprise monitors the integrity and security posture of all assets it owns, enforces authentication and authorization dynamically before access, and collects as much information as it can about the current state of its assets and communications to improve its security posture. Architecturally, these tenets are realized through a policy decision point—a policy engine and policy administrator that evaluate each request—and policy enforcement points that allow or deny the connection.

NIST has since added implementation guidance. SP 800-207A (September 2023) addresses access control for cloud-native applications in multi-location, multi-cloud environments, and SP 1800-35, *Implementing a Zero Trust Architecture*, published by the National Cybersecurity Center of Excellence in June 2025, documents 19 example implementations built with commercial technologies from two dozen vendors. The latter is particularly instructive for managers because it shows that every practical zero trust implementation combines multiple products and that integration, not acquisition, is the hard part.

### The CISA Zero Trust Maturity Model as a roadmap

If SP 800-207 explains what zero trust is, the CISA Zero Trust Maturity Model (ZTMM) Version 2.0, released in April 2023, helps organizations plan how to get there. It organizes zero trust into five pillars—Identity, Devices, Networks, Applications and Workloads, and Data—supported by three cross-cutting capabilities: Visibility and Analytics, Automation and Orchestration, and Governance. Each pillar is described across four maturity stages: Traditional, Initial, Advanced, and Optimal. Version 2.0 added the Initial stage to recognize that most organizations begin with incremental, partly manual steps rather than a leap to sophisticated automation.

The figure below presents this maturity path in managerial terms. Its value for governance is that it turns an abstract principle into a sequence of observable states that can be assessed, funded, and reported. A board does not need to understand policy enforcement points to understand that the identity pillar has moved from Initial to Advanced, or that the data pillar remains Traditional and is therefore the priority for the next budget cycle.

<span id="_Toc243635560" class="anchor"></span>Figure 61: A zero trust maturity path based on the CISA model

Two features of the ZTMM deserve emphasis. First, Governance is one of its three cross-cutting capabilities: CISA expects policies, roles, and enforcement to mature alongside the technology, not after it. Second, the pillars need not advance in lockstep. An organization may reasonably reach Advanced maturity in identity while remaining at Initial in data, provided the choice is deliberate, tied to risk scenarios, and recorded. The mistake is to declare an organization "zero trust" because one pillar has advanced.

### Governance pitfalls

Zero trust programs fail in recognizable ways, and most of the failures are governance failures rather than technical ones.

- **Zero trust as a product:** treating the purchase of a single platform, often an access broker or network product, as the delivery of zero trust. CCCS guidance explicitly cautions against vendors who claim that one product can deliver full zero trust.

- **Zero trust as a slogan:** announcing a zero trust strategy without a current-state assessment, a target maturity per pillar, a funded roadmap, or metrics, which converts an architectural direction into compliance theatre.

- **Zero trust without identity hygiene:** investing in dynamic access policy while privileged access remains standing, service accounts are unmanaged, and multi-factor authentication is incomplete; dynamic policy is only as good as the identities it evaluates.

- **Zero trust without an asset and data inventory:** per-request decisions that consider resource sensitivity are impossible if the organization does not know what its resources are or how sensitive they are.

- **Zero trust without an exception model:** legacy systems, operational technology, and partner integrations that cannot meet zero trust requirements are left undocumented rather than isolated and recorded as exceptions with expiry dates.

### Zero trust in the Canadian context

For Canadian organizations, the CCCS publication *A zero trust approach to security architecture* (ITSM.10.008), published in March 2023, is a useful companion to the NIST and CISA references. It identifies preventing lateral movement as the primary goal of zero trust, recommends incremental adoption in a hybrid model alongside existing perimeter defences, pairs zero trust with risk-based security management, and is candid about the obstacles: incompatible legacy technology, user friction, cost, scarce skills, and transitions measured in years. It also notes that the Government of Canada has been developing its own zero trust framework aligned with NIST and CISA. In the United States, federal agencies have been directed toward zero trust since OMB Memorandum M-22-09 (January 2022), so Canadian suppliers to U.S. public-sector clients will meet these expectations in contracts.

The managerial conclusion is consistent across these sources. **Zero trust is best governed as a multi-year architectural program with a maturity target per pillar, funded in stages, measured with observable criteria, and reported to the board like any other strategic transformation.**

## Threat modelling as a managerial control

Architecture decisions are only as good as the analysis that informs them. **Threat modelling** is the structured analysis of a system's design to identify what could go wrong, how, and what should be done about it, before the system is built or changed. It is often presented as a technical practice for security specialists, and the techniques are indeed technical. Its governance significance, however, is that it is the moment at which a specific design is confronted with specific threats, and the moment at which design-level risk decisions are made explicit.

### Four questions and a few methods

The most accessible framing is the four-question frame associated with Adam Shostack and reflected in the community *Threat Modeling Manifesto* (2020): *What are we working on? What can go wrong? What are we going to do about it? Did we do a good enough job?* Each has a managerial counterpart: the first requires an accurate model of the system and its data flows, which many projects lack; the second is the threat analysis itself; the third produces design changes, compensating controls, or explicit risk acceptance; and the fourth asks whether the analysis was adequate for the system's criticality.

Several methods help with the second question. **STRIDE**, developed at Microsoft, prompts analysts to consider spoofing, tampering, repudiation, information disclosure, denial of service, and elevation of privilege for each element of a design. LINDDUN applies a similar approach to privacy threats, which makes it useful alongside the privacy impact assessments required by Law 25 (see Chapter 16). PASTA offers a more risk-centric, multi-stage process; attack trees decompose an attacker's goal into alternative paths; and MITRE ATT&CK provides a catalogue of observed adversary techniques that grounds the analysis in real behaviour. Managers do not need to choose among these in detail. They need to ensure that a method is chosen, applied consistently, and scaled to the importance of the system.

### Making threat modelling governable

Threat modelling becomes a managerial control when it is required, proportionate, and evidenced. It should be required at defined points: when a new system or significant change is proposed, when a system crosses a trust boundary or begins processing more sensitive data, when a new supplier or integration is introduced, and after a significant incident reveals that earlier assumptions were wrong. It should be proportionate: a lightweight, one-hour exercise may suffice for a low-impact internal tool, whereas a payment platform or a system controlling physical processes warrants a deeper, documented analysis. And it should be evidenced: the output is a short record of the system model, the threats considered, the decisions taken, and the residual risks accepted and by whom.

There is a natural link between threat modelling and the scenario-based risk assessments of Chapters 5 and 6. Enterprise risk scenarios describe what the organization fears at a strategic level, such as ransomware disrupting clinical operations or a data breach at a key supplier. Threat models describe how a particular system could contribute to those scenarios. When the two are connected, each system-level decision can be traced to an enterprise risk, and each enterprise risk can be traced to the design decisions that address it. Tools such as CyberRiskGuardian can help maintain that traceability by linking each scenario's causal chain to its treatment measures and to the key risk indicators that monitor them.

- **Required:** threat modelling is a mandatory gate for defined classes of change, not a discretionary activity.

- **Proportionate:** depth scales with system criticality, data sensitivity, and exposure.

- **Evidenced:** each threat model produces a short, retained record that an auditor or incident investigator can read.

- **Connected:** significant threats and accepted residual risks flow into the risk register and to the accountable risk owner.

## Reference architectures and patterns

If every project designed its security from first principles, architecture would be slow, inconsistent, and expensive. **Reference architectures** and **security patterns** solve this problem by capturing approved designs for recurring situations—a customer-facing web application, a data analytics platform, a remote administration pathway, a partner integration, a SaaS onboarding—so that projects can reuse them rather than reinvent them.

### Patterns as the paved road

A reference architecture describes the components, trust boundaries, data flows, and mandatory controls for a class of system. A pattern is a smaller, reusable solution to a specific design problem, such as how to expose an internal service to a partner, how to store secrets, or how to authenticate a workload to a cloud service. In organizations with mature platform engineering, these are increasingly implemented as templates, modules, and pipelines—sometimes called a **paved road**—so that the secure way to build is also the easiest way to build.

The governance benefits are substantial. Patterns reduce design effort and delivery time, removing much of the friction that makes security seem like an obstacle. They make assurance cheaper, because a conforming design can be reviewed quickly with attention on deviations. They make metrics possible, because the organization can measure how many systems are built on approved patterns. And they embody secure by default internally: the platform team gets security right once, rather than each project carrying the burden repeatedly.

### Governing the pattern life cycle

Patterns are assets, and like all assets they must be owned and maintained. A pattern that was secure three years ago may now rely on deprecated protocols, unsupported components, or assumptions that no longer hold. Each pattern should therefore have a named owner, a version, a review date, a mapping to the controls and frameworks it satisfies, and a record of the systems that use it, so that when a pattern changes the organization knows which systems are affected.

The organization must also decide how patterns become mandatory. A sensible approach distinguishes between patterns that are required (for example, the approved pathway for privileged administration), patterns that are preferred but may be departed from with justification, and patterns that are offered as guidance. Departures from required patterns are exceptions and should be governed as such, and external patterns should be adopted only after tailoring to the organization's risk appetite, obligations, and estate.

## Architecture governance: review boards, design authority, and decisions

To be governable, security architecture needs an operating model: a defined set of roles, forums, decision rights, artifacts, and feedback loops. Without one, architecture becomes advice that projects may ignore; with an overly heavy one, it becomes a bottleneck that projects learn to route around. The figure below summarizes an operating model that balances authority with flow.

<span id="_Toc243635561" class="anchor"></span>Figure 62: An operating model for architecture governance

### Design authority and architecture review boards

A **design authority** is the role or body empowered to set architectural standards, approve designs against them, and decide—or escalate—when a design departs from them. In smaller organizations it may be a single senior architect with a clear mandate from the CIO and CISO; in larger ones it is typically an **architecture review board** (ARB) that brings together enterprise, security, infrastructure, data, and application architects, often with representation from risk management and privacy. Security architecture should either sit on the enterprise ARB with explicit authority over security matters or operate as a security design authority whose decisions the ARB is bound to respect.

The board's terms of reference matter more than its membership. They should specify which changes require review, the criteria against which designs are assessed, the board's authority to approve, approve with conditions, or reject, the escalation path when a business sponsor disagrees, and the service levels the board commits to so that delivery teams can plan. They should also make clear that the board does not accept residual risk on behalf of the business. When a design cannot meet a requirement, the board's role is to describe the resulting risk clearly; the decision to accept it belongs to the accountable risk owner, within the limits of risk appetite and the delegation of authority described in Chapter 4.

### Integrating architecture with delivery

Architecture review that occurs only at the end of a project is almost always too late; by then, the choice is between accepting a flawed design and delaying a launch. Effective organizations integrate architecture at several points: at intake, where a short triage determines whether a change is significant enough for review and which patterns apply; at design, where the threat model and solution architecture are reviewed; before production, where engineering evidence confirms that the approved design was built; and after go-live, where monitoring and periodic review detect drift from the approved design.

In agile and DevOps environments, these checkpoints must be lightweight and largely automated. Patterns implemented as code, policy-as-code checks in deployment pipelines, and automated evidence collection allow most changes to proceed without convening a board at all. The board's attention is then reserved for novel designs, significant deviations, and high-impact systems. This is the essential trade: **the more the organization invests in paved roads and automated guardrails, the less it needs to rely on manual review, and the more its review capacity can be focused where risk is highest.**

### Architecture decision records

Architectural decisions are often made in meetings and remembered only by the people who attended. When those people move on, the organization loses the reasoning behind its structure and repeats old debates or, worse, reverses good decisions without understanding why they were made. An **architecture decision record** (ADR) is a short, structured document that captures a single significant decision: the context, the options considered, the decision taken, the rationale, the consequences (including the security and risk consequences), and who made it.

ADRs are a lightweight but powerful governance artifact. They provide the audit trail showing that design decisions were deliberate and risk-informed, which is what regulators and auditors look for under OSFI B-13 or ISO/IEC 27001. After an incident, they distinguish a known risk that was consciously accepted from one that was never considered—a distinction that matters greatly for accountability and for the defensibility of management's conduct. A good ADR is one or two pages; the discipline lies in writing it when the decision is made, not reconstructing it later.

### Exceptions with expiry

Exceptions are inevitable. Legacy systems cannot always meet current standards, suppliers impose constraints, and business urgency sometimes justifies a temporary departure. What destroys architecture is not exceptions but unmanaged exceptions: departures that are granted informally, never recorded, never revisited, and eventually forgotten until an incident exposes them.

A credible exception process is tied directly to risk appetite. Each exception should record the requirement being departed from, the reason, the resulting risk described in scenario terms, the compensating controls, the accountable risk owner who accepts the residual risk, the approval authority (which should rise with the size of the risk), and an expiry date with a remediation plan. Exceptions that would create residual risk beyond tolerance must not be approved at any level below the one authorized to accept such risk, and some should not be approved at all. The exception register should be reported to senior management as a portfolio: how many exceptions exist, how many have expired without remediation, which systems carry the most, and whether the trend is improving. Chapter 14 applies the same discipline to configuration baselines.

| An exception without an owner, a compensating control, and an expiry date is not an exception; **it is an undocumented change to the organization's risk appetite.** |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## Technical debt as cybersecurity risk

Every organization carries technical debt: the accumulated cost of past design shortcuts, deferred upgrades, unsupported platforms, undocumented integrations, and expedient workarounds that became permanent. **Security debt** is the portion of that debt that increases cyber exposure. Unlike financial debt, it rarely appears on a balance sheet, and its interest is paid in forms that are hard to see until something goes wrong: emergency patching, compensating controls, slower incident response, failed audits, and, ultimately, incidents whose impact is magnified by the weaknesses the debt created.

### How security debt accumulates

Security debt accumulates in predictable ways: systems reach end of support while still running critical processes; exceptions are granted and never closed; acquisitions bring in environments never integrated into the organization's identity, logging, or baseline architecture; projects ship with known weaknesses on the promise of a fix that is never funded; and cloud adoption outpaces its guardrails. Each decision may have been reasonable at the time. The problem is that the debt compounds, and its aggregate effect on the risk profile is rarely assessed.

Architecture is both a cause of and a cure for security debt. Poor or absent architecture generates debt, because each project solves the same problems differently. Good architecture prevents new debt by steering projects toward approved patterns and provides the target state against which existing debt is measured; without one, an organization cannot even say how much debt it carries.

### Making debt visible and governable

The first governance step is to treat significant security debt as risk and record it in the risk register, not merely in an engineering backlog. Each item should be described in scenario terms—what could happen, how likely it is, and what it would cost—so that it can be compared with other risks and prioritized accordingly. Unsupported operating systems on a production network, a legacy application that cannot use modern authentication, or an acquired subsidiary that still shares administrative credentials are not technical curiosities; they are open risk scenarios with owners.

The second step is to measure the debt and its interest. Useful indicators include the number of assets running unsupported software, the number of open and expired architectural exceptions, the proportion of critical systems not yet on approved patterns, the cost of compensating controls maintained solely because of legacy constraints, and the time required to patch critical vulnerabilities on legacy systems compared with modern ones. Trended over time, these become key risk indicators that tell leadership whether the organization is paying down its debt or adding to it.

### Paying down debt: investment logic

Managers often experience cybersecurity budgeting as ambiguous because security seems to have no natural limit. Architecture reduces that ambiguity by turning investment into a portfolio of structural capabilities that each reduce broad classes of risk. The aim is not to buy everything, but to build durable protective capacity where it matters most and to retire the debt that most increases exposure.

A practical sequence for many organizations begins with identity and privileged access, because compromise there reverberates everywhere. It then addresses baseline hardening and vulnerability remediation discipline, because misconfiguration and exploitation of known weaknesses are persistent failure modes. It strengthens logging and monitoring to improve detection and evidentiary capability, enhances segmentation to reduce blast radius, and validates resilience and recovery, because recoverability is the final safeguard against catastrophic outcomes. Retirement or isolation of unsupported systems runs alongside this sequence rather than after it. The order is not universal and must be adapted to sector, dependencies, and threat profile—an operational technology environment or a hospital may need to prioritize segmentation and resilience earlier—but the governing principle holds. **Architectural investment is how an organization converts abstract risk appetite into defensible choices, and debt reduction should be funded as risk reduction, not treated as discretionary maintenance.** Chapter 17 develops the economic tools for making that case.

## Measuring architecture and aligning with standards

What is not measured is rarely governed. Architecture has traditionally been difficult to measure because its outputs—designs, principles, patterns—are not obviously quantifiable. The solution is to measure architecture through its effects on the environment: coverage, conformance, and outcomes.

### Metrics and key risk indicators

Architecture metrics should be oriented toward coverage and outcomes rather than activity. The number of design reviews conducted says little; the proportion of critical systems built on approved patterns says a great deal. Useful measures can be organized by pillar and by the governance processes described earlier in this chapter.

| **Area**                       | **Example indicator**                                                          | **What it tells leadership**                       |
|--------------------------------|--------------------------------------------------------------------------------|----------------------------------------------------|
| Identity and privileged access | Share of privileged accounts under just-in-time or vaulted control             | Whether the most dangerous access is constrained   |
| Baselines and workloads        | Share of critical assets enforced and monitored against a baseline             | Whether intended configuration actually holds      |
| Segmentation                   | Share of crown-jewel systems isolated in defined zones                         | Whether blast radius is limited where it matters   |
| Logging                        | Control-plane and identity log coverage for critical services                  | Whether incidents can be detected and investigated |
| Resilience                     | Successful restoration tests within RTO for critical services                  | Whether recovery is real or assumed                |
| Architecture governance        | Open, expired, and high-risk exceptions; share of systems on approved patterns | Whether the architecture is holding or eroding     |
| Zero trust                     | Maturity stage per ZTMM pillar against target                                  | Whether the transformation is on track             |

These indicators should be set with thresholds that reflect risk tolerance, so that a decline triggers attention before it becomes an incident. They should also be reported in a form that connects to the risk scenarios the board already knows. A hypothetical report that segmentation coverage for clinical systems has risen from 40% to 85% means far more when presented alongside the ransomware scenario it addresses and the estimated reduction in that scenario's impact.

### Standards provide the language, architecture provides the structure

Standards such as ISO/IEC 27001 and ISO/IEC 27002, frameworks such as NIST CSF 2.0 and NIST SP 800-53 Rev. 5, and prioritized safeguards such as the CIS Controls v8.1 are essential for governance credibility. The managerial trap is to treat them as lists of controls to implement one by one. Architecture is the mechanism that makes standards implementable at scale.

A mature organization uses standards to define what must be covered and uses architecture to define how coverage will be achieved coherently and measurably. Access control expectations in ISO/IEC 27002 are met through the identity and privileged access pillar. Logging and monitoring expectations are met through logging architecture and detection engineering. Expectations for ICT readiness for business continuity are met through resilience architecture and tested recovery. Control 8.27 on secure architecture and engineering principles is met through the design authority, reference architectures, threat modelling, and decision records described in this chapter. **Standards provide the assurance language; architecture provides the execution structure.** Managers should therefore expect their organizations to translate standards into a limited number of architectural programs, each with clear ownership, a roadmap, and measurable outcomes, rather than into hundreds of disconnected control tasks.

The three lines model, introduced earlier in the book, applies here as well. Architecture and engineering teams, together with the business owners of systems, form the first line, designing and operating controls. The second line—risk management and the security governance function—sets architectural policy and risk criteria, challenges significant decisions, and monitors exceptions and debt. Internal audit, as the third line, provides independent assurance that the architecture governance process works: that reviews happen, decisions are recorded, exceptions expire, and metrics are accurate.

## Architecture in practice

The concepts in this chapter become clearer when applied to a single organization facing realistic constraints. The scenario that follows is fictional, but its pressures are typical of growing Canadian technology firms.

### Illustrative scenario: a Montréal health-software scale-up

Clairvue Santé is a Montréal-based software company of about 350 employees that provides a scheduling and patient-communication platform to clinics and hospitals in Québec and Ontario, with a growing customer base in the northeastern United States and early interest from a European distributor. The platform processes personal health information, which brings it within the scope of Québec's Law 25, Ontario's health privacy legislation through its clients, and contractual security requirements imposed by hospital customers. The company grew quickly through two small acquisitions, and its platform runs on a public cloud provider, with some legacy components in a colocation facility inherited from one of the acquired firms.

A major hospital network's procurement team sends Clairvue a security questionnaire that asks, among other things, for evidence of a secure development life cycle, phishing-resistant authentication for administrators, tenant isolation, customer-accessible audit logs, tested recovery objectives, and a roadmap toward zero trust. The CEO asks the newly appointed head of security architecture whether the company can answer credibly. The honest answer is "partly". Engineering practices are good in the core product team but inconsistent in the acquired teams; administrators use multi-factor authentication, but not a phishing-resistant form; tenant isolation is enforced in the application but not in the data layer for one legacy reporting module; logs are collected but not exposed to customers; and recovery has been tested for the main database but not for the identity provider or the legacy components.

Rather than answering the questionnaire item by item, the head of security architecture proposes treating it as a prompt for an architectural program. Working with the CTO and the risk committee, she frames three enterprise risk scenarios: compromise of an administrator account leading to cross-tenant data exposure; ransomware spreading from the legacy colocation environment into the cloud platform; and a prolonged outage caused by the loss of the identity provider. Each is assessed using the company's scenario-based method, producing likelihood and impact estimates in dollars, regulatory consequences, and reputational terms. The assessment shows that the second and third scenarios exceed the risk tolerance the board approved the previous year.

The resulting roadmap is expressed in the language of the seven pillars and the CISA maturity model. In the first six months, the company moves all administrators to phishing-resistant authentication and just-in-time privilege, isolates the legacy colocation environment behind a tightly controlled connection with no standing administrative access, and conducts a full restoration test of the identity provider and platform from isolated backups. In the following six months, it adopts the SSDF as its secure development standard across all teams, introduces lightweight threat modelling as a gate for significant changes, publishes three reference architectures as reusable templates, and begins exposing tenant audit logs to customers. The legacy reporting module is recorded as an architectural exception with an owner (the vice-president of product), compensating controls (additional monitoring and restricted access), and an eighteen-month expiry tied to a funded replacement project.

Governance changes accompany the technical work. A small architecture review board, chaired by the CTO with the head of security architecture holding a veto on security matters subject to escalation to the CEO, meets every two weeks and reviews only novel or high-risk designs; everything else proceeds through automated pipeline checks. Significant decisions are recorded as ADRs. The board's risk committee receives a quarterly dashboard showing maturity per zero trust pillar against target, privileged access coverage, exception counts and ages, and restoration test results. Within a year, Clairvue can answer the hospital network's questionnaire with evidence rather than assertions, and—more importantly—the two scenarios that exceeded tolerance have been brought back within it. The European distributor's questions about the Cyber Resilience Act, which would have been alarming a year earlier, now map onto practices the company already follows.

The lessons transfer readily: the customer's demands became an input to risk-based architecture rather than a compliance exercise, investments were sequenced by risk rather than by the order of the questionnaire, exceptions were owned and time-bound, and governance stayed light enough that delivery teams experienced architecture as an accelerator.

### Questions managers and boards should ask

The following questions help managers and boards test whether security architecture is genuinely governed or merely described.

- **Target state:** Do we have a documented target security architecture, approved by management and linked to our risk appetite and top risk scenarios?

- **Decision rights:** Who has the authority to approve, condition, or reject a design on security grounds, and how are disagreements with business sponsors escalated?

- **Security by design:** At what point in our delivery process are security requirements and threat models produced, and what evidence shows that this happens for significant changes?

- **Suppliers:** Do our procurement processes ask suppliers secure-by-design and secure-by-default questions, and do we act on the answers?

- **Pillars:** For each of the seven pillars, can we show coverage across the assets that would cause the greatest harm if compromised?

- **Privileged access:** How much privileged access is standing rather than just-in-time, and how quickly is that number falling?

- **Zero trust:** What maturity stage have we reached in each pillar of the CISA model, what is our target, and is the roadmap funded?

- **Patterns:** What proportion of our critical systems are built on approved reference architectures, and who maintains those patterns?

- **Decisions:** Are significant architectural decisions recorded with their rationale and risk consequences, so that we could explain them to a regulator after an incident?

- **Exceptions:** How many architectural exceptions are open, how many have expired without remediation, and who has accepted the residual risk for each?

- **Technical debt:** Is our security debt recorded and quantified as risk, and are we paying it down faster than we are adding to it?

- **Recovery:** When did we last restore our identity provider and a critical business service from isolated backups, and did we meet our recovery objectives?

## Bringing it together: architecture as the practical face of governance

Cybersecurity governance is the managerial system that sets objectives, makes trade-offs, allocates resources, and enforces accountability. Security architecture and security engineering are where that system becomes physical. They determine whether security is achieved by design or hoped for through effort, whether risk appetite is a structural property of the environment or an aspiration in a policy, and whether the organization can produce evidence of its claims when a regulator, customer, or court asks for it.

This chapter has argued that managers should govern architecture as a portfolio of structural decisions rather than as a collection of tools. Security by design and secure by default move security to the point where it is cheapest and most effective. Frameworks such as SABSA, TOGAF, and Open Security Architecture give the work a method and connect it to business value. The seven pillars define the structural capabilities that most programs must build, each with detailed practices owned by other chapters. Zero trust, read through NIST SP 800-207 and the CISA Zero Trust Maturity Model Version 2.0, provides a long-term direction of travel that can be planned, funded, and measured in stages.

The governance machinery matters as much as the designs. Threat modelling makes design-level risk explicit; patterns make the secure path the easy path; design authorities hold decision rights and focus review where risk is highest; decision records preserve the reasoning that makes decisions defensible; exceptions with owners and expiry dates keep departures within risk appetite; and treating technical debt as risk makes the cost of past shortcuts visible and fundable. Coverage-based metrics, reported in the language of the organization's risk scenarios, let leadership see whether the architecture is holding.

Once architecture is governed in this way, the neighbouring chapters become easier to apply. Identity, cyber-physical systems, resilience, baselines, cloud, data protection, cryptography, supply-chain governance, and investment decisions all become elaborations of a coherent structure rather than separate initiatives competing for attention. Risk scenarios can be translated into architectural requirements, standards into programs, and budgets into investments in durable capability. Most importantly, the organization moves from fragmented security activity to a disciplined, evidenced, and defensible state of acceptable risk—which is, in the end, what cybersecurity governance is for.

## Chapter summary

Security architecture and security engineering are where a board's risk appetite either becomes a structural property of the organization's systems or remains a sentence in a policy. Architecture defines what must be true for risk to stay within appetite; engineering builds it and proves that it works. Because a single architectural decision is inherited by many systems, architecture is one of the highest-leverage risk instruments managers govern. The chapter presents security by design and secure by default, the SABSA, TOGAF, and Open Security Architecture frameworks, seven architectural pillars, and zero trust as a multi-year governance program. It closes with the machinery that makes architecture governable: threat modelling, patterns, design authority, decision records, exceptions with expiry, security debt, and coverage-based metrics.

### Key points to retain

- Architecture owns the what and the why, engineering owns the how and the proof, risk owners own residual risk, and governance owns the system as a whole.

- Architectural controls change the structure of the environment, so one decision reduces whole classes of risk scenarios across every system built on it.

- Architecture is expected by standards and regulation, including ISO/IEC 27002:2022 control 8.27, NIST CSF 2.0 (PR.PS, PR.IR), COBIT 2019 APO03, OSFI B-13, Law 25, and GDPR Article 25.

- Security by design is a sequencing decision and secure by default is a burden-allocation decision; both require evidence that the principles were applied.

- SABSA traces security to business attributes, TOGAF provides the enterprise architecture machinery, and Open Security Architecture offers reusable patterns that must be tailored.

- The seven pillars (identity, endpoints and workloads, segmentation, baselines, encryption and keys, logging, and resilience) interact, and each is governed by one question: can we show, with evidence, that the decision was implemented where it matters most?

- Reference architectures and patterns form a paved road that makes the secure way the easiest way, provided each pattern has an owner, a version, and a review date.

- Zero trust abandons implicit trust based on network location; it is something an organization progressively becomes, not something it buys.

- The CISA Zero Trust Maturity Model Version 2.0 turns zero trust into observable maturity stages per pillar that can be assessed, funded, and reported.

- Threat modelling becomes a managerial control when it is required, proportionate, evidenced, and connected to the risk register.

- A design authority holds decision rights and focuses review where risk is highest, but never accepts residual risk; its significant decisions are captured in architecture decision records.

- Exceptions need an owner, compensating controls, and an expiry date, and significant security debt belongs in the risk register, measured and funded as risk reduction; architecture itself is measured by coverage and outcomes reported in the language of risk scenarios.

### Key terms

- **Security architecture** — The disciplined design of protection capabilities across systems, data flows, and processes, defining what must be true for risk to remain within appetite.

- **Security engineering** — The practical implementation of security designs in real environments, including automation, testing, and the evidence that controls exist and function.

- **Security by design** — Introducing security requirements, threat analysis, and architectural constraints at the beginning of a system's life and carrying them through it.

- **Secure by default** — The expectation that systems and products are safe in their out-of-the-box configuration without the customer having to discover and enable protective settings.

- **Secure Software Development Framework (SSDF)** — NIST SP 800-218, an outcome-oriented set of practices grouped into preparing the organization, protecting the software, producing well-secured software, and responding to vulnerabilities.

- **SABSA** — A business-driven security architecture methodology that traces every architectural element back to business attributes through layered stakeholder views.

- **TOGAF Architecture Development Method (ADM)** — The iterative enterprise architecture cycle of the TOGAF Standard, within which security requirements, threat analysis, and risk acceptance should be embedded.

- **Architectural pillars** — The seven major architectural decision areas in which the organization's risk appetite becomes concrete, from identity to resilience.

- **Zero trust** — Per NIST SP 800-207, concepts that minimize uncertainty in enforcing accurate, least-privilege, per-request access decisions in a network viewed as compromised.

- **Policy decision and enforcement points** — The zero trust components that evaluate each access request against dynamic policy and then allow or deny the connection.

- **Threat modelling** — The structured analysis of a system's design to identify what could go wrong, how, and what should be done about it before the system is built or changed.

- **Reference architecture and pattern** — Approved designs for a class of system or a specific design problem, increasingly implemented as templates and pipelines forming a paved road.

- **Design authority** — The role or body, often an architecture review board, empowered to set architectural standards, approve designs, and decide or escalate departures.

- **Architecture decision record (ADR)** — A short, structured record of one significant decision: context, options, decision, rationale, consequences including risk, and who decided.

- **Security debt** — The portion of technical debt, such as unsupported platforms and unclosed exceptions, that increases the organization's cyber exposure.

## Review questions

### Essay questions

1.  The chapter argues that architectural decisions are among the highest-leverage risk decisions an organization makes. Using at least three structural mechanisms from the chapter, analyze why an architectural control can outperform detective and procedural controls, and discuss what this implies for how boards should govern architecture.

2.  A mid-sized Canadian organization already uses TOGAF for enterprise architecture but has no security architecture method. Evaluate how it should integrate SABSA and pattern libraries such as Open Security Architecture, and explain the governance risks of adopting external patterns without tailoring.

3.  Choose an organization you know (or use Clairvue Santé from the chapter). Assess its likely position on the seven architectural pillars, identify the two pillars that most need investment, and justify a sequence for the next 12 months using the chapter's investment logic.

4.  A vendor tells your executive committee that its platform will make the organization 'zero trust in six months'. Using NIST SP 800-207, the CISA Zero Trust Maturity Model Version 2.0, and CCCS guidance, critically evaluate this claim and propose how zero trust should instead be governed.

5.  Design an architecture governance operating model for a growing organization that delivers software through agile teams. Explain how a design authority, threat modelling gates, architecture decision records, and exceptions with expiry would work together without becoming a bottleneck.

### Multiple-choice questions

Choose the best answer. Answers are listed in the appendix "Answers to multiple-choice questions".

**1.** According to the chapter, which question does security architecture answer?

> A. How will we build and operate target states in our real environment?
>
> B. What must be true about our environment for risk to stay within appetite?
>
> C. Which tools should the security operations centre deploy to detect intrusions?
>
> D. Which business owner accepts the residual risk when a design falls short?

**2.** When a proposed design falls short of the target state, who decides whether to accept the resulting gap?

> A. The security architect who reviewed and documented the design
>
> B. The architecture review board, by a majority vote of its members
>
> C. The engineering team responsible for implementing the design
>
> D. The business owner of the affected service, within risk appetite

**3.** Why does the chapter describe architectural controls as having more leverage than detective or procedural controls?

> A. They change the environment's structure so whole classes of scenarios become less likely or harmful
>
> B. They cost less to purchase than detection tools and carry no ongoing operating cost
>
> C. They rely on staff following written procedures, which auditors find easier to test
>
> D. They remove the need for risk owners to accept any residual risk at all

**4.** Which ISO/IEC 27002:2022 control specifically addresses secure system architecture and engineering principles?

> A. Control 8.9, configuration management
>
> B. Control 8.25, secure development life cycle
>
> C. Control 8.27, secure system architecture and engineering principles
>
> D. Control 8.28, secure coding

**5.** After an incident, what important distinction does a well-kept architecture decision record allow the organization to make?

> A. Whether the vendor or the customer introduced the vulnerability
>
> B. Whether the incident must be reported under Québec's Law 25
>
> C. Whether a risk was consciously accepted or never considered
>
> D. Whether the review board met its committed service levels

**6.** The chapter calls security by design a sequencing decision. What does that mean?

> A. It moves security analysis to the point where change is cheapest and options are widest
>
> B. It places responsibility for safe configuration on the party best able to carry it
>
> C. It requires security testing to be completed only after deployment to production
>
> D. It orders controls by cost so that the cheapest ones are implemented first

**7.** Which of the following is one of the three principles in CISA's Secure by Design guidance for software manufacturers?

> A. Shift security testing entirely to the customer's operations team
>
> B. Certify every product against ISO/IEC 27001 before its release
>
> C. Require customers to enable multi-factor authentication themselves
>
> D. Take ownership of customer security outcomes

**8.** A Canadian firm already uses TOGAF for enterprise architecture. According to the chapter, how should it handle security architecture?

> A. Create a separate security architecture method running in parallel with TOGAF
>
> B. Embed security requirements, threat analysis, and risk acceptance in relevant ADM phases
>
> C. Replace TOGAF entirely with SABSA because SABSA is specific to security
>
> D. Adopt Open Security Architecture patterns untailored to save design time

**9.** What is the main managerial benefit of SABSA's business attributes profiling?

> A. Two-way traceability between business attributes and architectural elements
>
> B. A maturity scale from Traditional to Optimal for each security domain
>
> C. A catalogue of observed adversary techniques grounded in real behaviour
>
> D. A machine-readable format for checking configuration compliance at scale

**10.** Why does the chapter warn against adopting an external security pattern without tailoring it?

> A. External patterns are rarely mapped to any recognized control framework
>
> B. Creative Commons licences prohibit commercial organizations from using them
>
> C. It becomes compliance theatre if no one checks it addresses actual scenarios
>
> D. Auditors refuse to accept patterns that were not developed internally

**11.** Why is resilience architecture shown as the foundation of the seven architectural pillars?

> A. It is the cheapest pillar and should therefore always be completed first
>
> B. It is the pillar that assumes the other pillars will sometimes fail
>
> C. It is the only pillar that maps directly to NIST CSF 2.0 outcomes
>
> D. It replaces the need for segmentation in cloud-native environments

**12.** An organization decides to defer its segmentation program by one year. What does the chapter say it should do?

> A. Nothing, because segmentation is an optional refinement in modern designs
>
> B. Suspend identity improvements until segmentation can be funded with them
>
> C. Treat the deferral as an operational matter for the network team alone
>
> D. Record the decision and raise the importance of identity and detection coverage

**13.** A firm mandates encryption across all systems but does not manage the key life cycle. What is the chapter's main concern?

> A. Encryption will slow application performance below acceptable thresholds
>
> B. Regulators prohibit encryption unless the customer holds all keys
>
> C. Post-quantum algorithms will immediately make the encryption useless
>
> D. Hidden fragility: it may be unable to recover its own data in an incident

**14.** According to the chapter, what is the first governance step for significant security debt?

> A. Add it to the engineering backlog for the next delivery sprint
>
> B. Record it in the risk register, described in scenario terms
>
> C. Defer it until the affected systems reach their end of support
>
> D. Report it to the external auditor as a control deficiency

**15.** What is the central shift introduced by NIST SP 800-207, Zero Trust Architecture?

> A. Replacing perimeter firewalls with a single access broker product
>
> B. Requiring all employees to work only from the corporate network
>
> C. Abandoning implicit trust based on network location
>
> D. Encrypting all data at rest with customer-held keys

**16.** An organization reaches Advanced maturity in the CISA ZTMM identity pillar while its data pillar remains Initial. How does the chapter view this?

> A. Acceptable if the choice is deliberate, tied to risk scenarios, and recorded
>
> B. Unacceptable, because all pillars must advance in lockstep each year
>
> C. Sufficient to declare the organization a zero trust organization
>
> D. Evidence that the Governance capability can safely be matured later

**17.** Which threat modelling method applies a structured approach specifically to privacy threats, making it useful alongside Law 25 privacy impact assessments?

> A. LINDDUN
>
> B. STRIDE
>
> C. PASTA
>
> D. MITRE ATT&CK

**18.** A team performs a one-hour threat model for a low-impact internal tool but a deep, documented analysis for its payment platform. Which property of governable threat modelling does this illustrate?

> A. Required
>
> B. Evidenced
>
> C. Connected
>
> D. Proportionate

**19.** What is the role of the architecture review board when a design cannot meet a security requirement?

> A. Accept the residual risk on behalf of the business sponsor
>
> B. Reject the project outright and cancel its remaining funding
>
> C. Describe the resulting risk clearly for the accountable risk owner
>
> D. Approve the design automatically when delivery is urgent

**20.** A hypothetical report shows segmentation coverage for clinical systems rising from 40% to 85%. What would make this most meaningful to a board?

> A. Expressing it as the number of firewall rules changed this quarter
>
> B. Presenting it with the ransomware scenario it addresses and the estimated impact reduction
>
> C. Comparing it with the number of design reviews conducted this year
>
> D. Reporting it only to internal audit in its third-line role
