<!-- Léger, M.-A. (2026). Introduction to Cybersecurity Governance for Business Technology Management (3rd ed., v2.51). Chapter 14: Security Baselines and Configuration Governance. CC BY-NC 4.0 -->

# Chapter 14: Security Baselines and Configuration Governance

## Learning outcomes

After completing this chapter, you should be able to:

1.  Distinguish a security baseline from a baseline configuration, an external benchmark, and hardening, and explain how drift undermines assurance.

2.  Explain why baselines are a governance mechanism rather than technical hygiene, with reference to measurability, scale, and the business-technical interface.

3.  Compare the main configuration references (CIS Benchmarks, DISA STIGs, vendor baselines, NIST SP 800-70 Rev. 5, SCAP 1.4, and NIST SP 800-128) and recommend how to combine them.

4.  Apply the blast-radius principle to decide which domains should have a baseline and in what priority.

5.  Design a tiered, testable baseline with explicit ownership, decision rights, versioning, and documented deviations from external benchmarks.

6.  Evaluate secure-by-default mechanisms, change management, and drift detection and remediation as a closed loop.

7.  Design an exception process with compensating controls, approval levels scaled to residual risk, an exception register, and expiry.

8.  Recommend coverage, compliance, drift, and exception metrics and KRIs that provide audit-ready evidence while avoiding compliance theatre.

Many of the most damaging cybersecurity failures are not caused by novel exploits or sophisticated adversaries. They are caused by ordinary misconfigurations and uncontrolled change: cloud storage exposed to the public Internet, identity policies weakened for convenience, logging disabled to save money, firewall rules opened to the whole Internet "for a few minutes", default passwords left unchanged, and temporary exceptions that quietly become permanent. Post-incident reviews return again and again to the same finding: the control existed on paper, and the system was not configured to match it.

These failures are so common because configuration is where real systems meet real operations. Even excellent policies and well-chosen control frameworks collapse if the organization cannot consistently deploy systems in secure configurations and keep them that way over time. In cloud and modern enterprise environments, configuration is also the fastest-moving part of the security surface. Deployments occur daily, infrastructure is ephemeral, software-as-a-service (SaaS) vendors change their defaults without asking, and changes arrive through many pathways at once—console clicks, scripts, CI/CD pipelines, vendor-managed updates, and the occasional well-intentioned administrator working late at night.

For managers, this makes security baselines and configuration governance one of the highest-leverage control domains available. Unlike many advanced controls, baseline governance can prevent broad classes of incidents by enforcing a secure minimum standard at scale. It also produces concrete evidence of due diligence: which standards exist, where they apply, how compliance is measured, how exceptions are approved, and how drift is detected and corrected. Few areas of cybersecurity allow a board to ask "are we doing what we said we would do?" and receive an answer that is both precise and verifiable.

This chapter builds directly on earlier material. Chapter 9 introduced the control frameworks—ISO/IEC 27001 and 27002, the NIST ecosystem, COBIT 2019, and the CIS Controls—that tell an organization *which* controls to implement; this chapter explains how the configuration-related controls in those frameworks become enforceable reality. Chapters 5 and 6 showed how scenario-based risk assessment identifies the exposures that matter, and many of those scenarios begin with a misconfigured system. Chapter 13 described security architecture as the set of patterns that make systems defensible by design; baselines are how those patterns are expressed, deployed, and kept intact. Chapter 15 will examine cloud security in depth, and this chapter deliberately limits itself to the configuration governance questions that span cloud and on-premises environments alike.

The chapter treats hardening standards, drift detection, and secure-by-default deployment as governance mechanisms, not as mere technical best practice. It begins with core concepts and the reasons baselines are a governance matter, surveys the reference ecosystem (CIS Benchmarks, DISA STIGs, vendor baselines, the NIST National Checklist Program, SCAP, and NIST SP 800-128), and then works through scope, design, secure-by-default deployment, change management, drift, exceptions, cloud and enterprise specifics, and metrics. An illustrative scenario follows a Canadian credit union through the aftermath of a misconfiguration incident. The objective throughout is to show how organizations turn a desired security posture into enforceable, measurable, and continuously maintained reality.

## Core concepts: baselines, hardening, drift, and secure-by-default

Configuration governance has its own vocabulary, and loose use of it is a source of risk. When one team says "baseline" and means a vendor's recommended settings while another means the organization's approved minimum, nobody can say with confidence whether a system is compliant.

### What a security baseline is (and is not)

A **security baseline** is a documented, approved minimum secure configuration for a defined class of systems. It expresses what "good enough by default" means for endpoints, servers, identity platforms, cloud accounts, network devices, SaaS tenants, databases, containers, and similar assets. A baseline is not aspirational guidance; it is an enforceable minimum against which actual systems can be compared. A useful test is whether an independent reviewer, given the baseline and access to a system, could determine without interpretation whether the system meets it.

NIST SP 800-128, *Guide for Security-Focused Configuration Management of Information Systems*, uses a closely related term, the **baseline configuration**, which it defines as "a documented set of specifications for a system, or a configuration item within a system, that has been formally reviewed and agreed on at a given point in time, and which can be changed only through change control procedures." The two ideas fit together. The security baseline defines the minimum secure settings for a class of assets; the baseline configuration records the approved state of a particular system, which should incorporate the security baseline plus any approved, documented deviations.

A baseline is also not the same as a benchmark. External benchmarks—CIS Benchmarks, DISA Security Technical Implementation Guides, vendor security baselines—are reference inputs. The organization's baseline is the decision it makes after considering those inputs in light of its own risk appetite, operational constraints, and obligations. Treating an external benchmark as if it were automatically the organization's baseline is a common shortcut, and it usually produces either excessive friction or unexamined gaps.

### Hardening and attack surface reduction

**Hardening** is the practice of reducing attack surface by removing or disabling unnecessary features, services, ports, accounts, permissions, and defaults, and by configuring what remains securely. Hardening often includes patching, but it is broader. It covers identity settings, network rules, logging configurations, management interface restrictions, cryptographic settings, and the secure configuration of applications and platforms. The principle behind it is what NIST SP 800-53 Rev. 5 calls least functionality: a system should provide only the capabilities it needs for its mission.

For managers, the important insight is that hardening is a trade-off, not an absolute. Every disabled feature is a feature someone might have wanted; every restricted protocol may break a legacy integration. Hardening decisions therefore carry business consequences, and the more aggressive the hardening, the more those consequences need to be understood and owned. This is why hardening belongs inside a governance process rather than being left to the preferences of individual engineers.

### Configuration drift

**Configuration drift** is the gradual divergence between the intended baseline and the actual deployed state. Drift happens for predictable reasons: emergency changes, manual console edits, temporary exceptions that are never reversed, vendor updates that alter defaults, imperfect automation, and changes introduced by different teams using different tools. In SaaS platforms, drift can occur even when nobody in the organization touches anything, because the vendor introduces a new feature that is enabled by default.

Drift is a governance problem, not only a technical one, because it creates untracked risk and destroys the organization's ability to claim that controls are in place. An organization that reports "all production servers meet the hardening standard" on the basis of how they were built, rather than how they are configured today, is making a statement it cannot support. Drift turns accurate assurance into inaccurate assurance without anyone deciding to change it.

### Secure-by-default and secure-by-design

**Secure-by-default deployment** means that systems are deployed in a secure configuration automatically, without requiring each team to reinvent hardening. In modern environments, this is achieved through golden images, approved infrastructure-as-code modules, standardized platform configurations, landing zones, and CI/CD policy gates. The secure configuration becomes the path of least resistance, and insecure configurations require deliberate, visible effort.

The idea also has an external dimension. Since 2023, the U.S. Cybersecurity and Infrastructure Security Agency (CISA), together with international partners including the Canadian Centre for Cyber Security (CCCS), has promoted *secure by design* and *secure by default* principles for technology manufacturers. The core argument is that products should ship with secure settings enabled and that customers should not have to discover and apply dozens of hardening steps to reach a reasonable security posture. For a customer organization, these principles become procurement criteria: products that ship insecure by default transfer configuration risk and cost to the buyer.

### How the concepts form a lifecycle

These concepts are tightly linked. Baselines define the desired secure state, secure-by-default mechanisms deploy that state, monitoring detects drift away from it, remediation or approved exceptions resolve each deviation, and the results feed back into the next version of the baseline. NIST SP 800-128 describes a comparable cycle in four phases—planning, identifying and implementing configurations, controlling configuration changes, and monitoring—and the CIS Controls v8.1 expect the secure configuration process itself to be reviewed and updated at least annually or when significant enterprise changes occur. The figure below summarizes the lifecycle used in this chapter.

<span id="_Toc243635562" class="anchor"></span>Figure 63: The security baseline governance lifecycle

The managerial point of the lifecycle is that no single stage delivers security on its own. A well-written baseline that is not deployed by default produces paperwork. Excellent automation without drift detection produces confidence that erodes silently. Drift detection without remediation authority produces dashboards. Exceptions without expiry produce permanent, unpriced risk. **Configuration governance is valuable only as a closed loop.**

## Why baselines are governance, not technical hygiene

It is tempting to treat configuration as operational detail well below the attention of executives and boards. Yet configuration has several properties that make it unusually important for governance, and organizations that leave it entirely to operational teams tend to discover this only after an incident.

### Measurability and evidence

First, baselines are measurable. Unlike many policy statements—"the organization shall protect sensitive information"—baseline compliance can be evaluated objectively: a configuration either meets the requirement or it does not. Multi-factor authentication is either enforced for administrative access or it is not. Public access to a storage account is either blocked or it is not. This makes baselines a foundation for evidence-based governance and one of the most effective antidotes to compliance theatre.

Measurability also changes the nature of assurance. Internal audit, external auditors, regulators, and cyber insurers increasingly ask for evidence of control *operation*, not just control *design*. A policy demonstrates design. A baseline with continuous compliance measurement demonstrates operation, across the population of assets, at a point in time and over time. For an ISO/IEC 27001 certification audit, a supervisory review, or an insurer's underwriting questionnaire, that difference is material.

### Scale and blast radius

Second, baselines scale. A single baseline applied consistently across thousands of assets can reduce risk more effectively than many bespoke controls. The reverse is also true: a single weak default, replicated through a template or image, can create the same vulnerability on every system built from it. Configuration therefore has a multiplier effect in both directions.

This multiplier is most pronounced in cloud and identity domains, where one setting—a conditional access exemption, an anonymous-access storage policy, a broad role assignment—affects everything within its scope. In the language of the risk triangle introduced in Chapter 1 and developed in Chapter 4, configuration decisions directly shape both vulnerability and the potential impact of a successful threat.

### The interface between business trade-offs and technical requirements

Third, baselines bridge business and technical accountability. A baseline converts risk appetite into minimum technical requirements, and the exception process converts business trade-offs into documented risk acceptance. This is the governance interface. When a team wants to deviate from the baseline, it must justify the deviation, propose compensating controls, and obtain approval from the appropriate risk owner. A configuration choice that would otherwise be invisible becomes an explicit, attributable decision.

This interface is what distinguishes configuration governance from configuration management in the narrow IT service management sense. Configuration management records and controls what is deployed. Configuration *governance* asks who decided what the minimum should be, on what basis, who may approve departures from it, and how the organization knows that its decisions are being honoured.

### Where baselines sit in the control frameworks

The major frameworks discussed in Chapter 9 all treat secure configuration as a core control, which gives managers a ready vocabulary for mapping and reporting:

- **ISO/IEC 27002:2022 control 8.9, Configuration management,** introduced as a new control in the 2022 edition, calls for configurations, including security configurations, of hardware, software, services, and networks to be established, documented, implemented, monitored, and reviewed. It also appears in Annex A of ISO/IEC 27001:2022, so certified organizations must address it in their Statement of Applicability.

- **NIST CSF 2.0** addresses configuration within the Protect function's Platform Security category, where outcome PR.PS-01 states that configuration management practices are established and applied. The Govern function supplies the policy, roles, and risk appetite that give those practices direction.

- **CIS Controls v8.1, Control 4, Secure Configuration of Enterprise Assets and Software,** requires the organization to establish and maintain the secure configuration of end-user devices, network devices, non-computing and IoT devices, servers, and software. Its twelve safeguards range from establishing a documented secure configuration process (4.1) and a separate process for network infrastructure (4.2), through managing default accounts (4.7) and disabling unnecessary services (4.8), to protections for portable and mobile devices.

- **NIST SP 800-53 Rev. 5** contains a full Configuration Management (CM) family, including baseline configuration (CM-2), configuration change control (CM-3), configuration settings (CM-6), and least functionality (CM-7).

- **Sector requirements** reinforce the same expectations. For federally regulated financial institutions in Canada, OSFI Guideline B-13, *Technology and Cyber Risk Management*, effective January 1, 2024, sets expectations for technology operations, change management, and cyber security that presuppose controlled, secure configurations. In the European Union, DORA and NIS2 impose comparable ICT risk management obligations on financial entities and on essential and important entities respectively.

| Configuration is where policy becomes fact. **A baseline is the point at which risk appetite stops being a statement and becomes a setting that can be checked.** |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|

Baseline governance is therefore not merely hardening. It is the operating system that makes secure configuration consistent, enforceable, auditable, and sustainable.

## The reference ecosystem: benchmarks, technical guides, and checklists

Very few organizations should write their baselines from scratch. A mature ecosystem of public references exists, much of it machine-readable. The managerial task is not to master these references but to understand what each offers, how authoritative it is in a given context, and how the organization will decide which settings to adopt.

### CIS Benchmarks and CIS Control 4

The **CIS Benchmarks**, published by the Center for Internet Security, are consensus-developed configuration guides for operating systems, cloud platforms, network devices, databases, browsers, office suites, container platforms, and many other technologies. They are developed by communities of practitioners and vendors and are widely used in both the public and private sectors. Most benchmarks define profiles: a Level 1 profile intended to be practical and broadly applicable with limited operational impact, and a Level 2 profile intended for higher-security environments where some loss of functionality is acceptable.

CIS also publishes hardened virtual machine images, pre-configured to its benchmarks, through major cloud marketplaces. Together with CIS Control 4, the benchmarks give an organization a coherent path: the control says *that* secure configurations must be established and maintained, and the benchmark supplies a defensible starting point for *what* those configurations should be. For many small and medium organizations, adopting CIS Level 1 profiles with a documented set of local deviations is a reasonable and defensible baseline strategy.

### DISA Security Technical Implementation Guides

The U.S. Defense Information Systems Agency (DISA) publishes **Security Technical Implementation Guides (STIGs)**, detailed configuration standards for products used in U.S. Department of Defense environments, along with higher-level Security Requirements Guides (SRGs) from which product-specific STIGs are derived. STIGs are mandatory for DoD systems and are frequently required in defence supply chains, including for Canadian firms that supply U.S. defence programs.

STIGs are typically more prescriptive and restrictive than CIS Level 1 profiles, and each finding is assigned a severity category. Outside defence contexts, they are useful as a reference for high-assurance systems and as a check on whether a less restrictive baseline has omitted something important. Organizations that are not subject to them should generally not adopt them wholesale; their strictness reflects a specific threat model and operating environment.

### Vendor security baselines

Major technology vendors publish their own recommended security configurations. Microsoft, for example, publishes security baselines for current Windows client and server releases, Microsoft Edge, and Microsoft 365 Apps for Enterprise through its Security Compliance Toolkit, and offers security baselines that can be applied to managed devices through Microsoft Intune. Cloud providers publish security reference architectures and built-in policy sets, and many SaaS vendors publish hardening guides for their tenants.

Vendor baselines have two advantages: they are written by the people who know the product's settings best, and they are often updated quickly when the product changes. Their limitation is perspective. A vendor's baseline reflects what the vendor believes is appropriate for a typical customer and what its product can support without generating support calls. It is a strong input, not a substitute for the organization's own risk-based decision.

Governments sometimes go further: CISA's Secure Cloud Business Applications (SCuBA) project publishes secure configuration baselines for Microsoft 365, and Binding Operational Directive 25-01 requires U.S. federal civilian agencies to implement the mandatory policies in those baselines, assess their tenants with automated tools, and report the results to CISA. Deviations from mandatory policies must be accepted by an authorizing official and explained in the reported results—an instructive model of baselines, measurement, and documented risk acceptance working together, even for organizations that are not bound by it.

### NIST SP 800-70 Rev. 5 and the National Checklist Program

NIST operates the **National Checklist Program (NCP)**, a public repository of security configuration checklists—also called benchmarks, lockdown guides, or hardening guides—for IT products. The program is defined in NIST SP 800-70, *National Checklist Program for IT Products: Guidelines for Checklist Users and Developers*. Revision 5, published in May 2026, supersedes Revision 4 (2018). According to NIST's release announcement, the revision links each checklist setting to CSF 2.0 outcomes, SP 800-53 controls, and Common Configuration Enumeration (CCE) identifiers. It also reaches beyond traditional servers and desktops to cloud platforms, IoT, and AI systems, accommodates automated checklist formats, and explains how checklists should be tailored to the environment in which a system runs, from a stand-alone machine to a managed enterprise, a specialized high-security system, or a legacy platform.

For managers, the NCP is a curated index of checklists from CIS, DISA, vendors, and others. More importantly, the traceability emphasis in Revision 5 aligns with the governance question that matters: when a setting is enforced, which control objective and which risk does it serve? An organization that can answer that question for its high-impact settings can defend its baseline choices to auditors and to its own engineers.

### SCAP and machine-readable configuration content

The **Security Content Automation Protocol (SCAP)** is a suite of NIST specifications for expressing configuration checks, vulnerability definitions, and assessment results in standardized, machine-readable formats, so that different tools can evaluate the same baseline and produce comparable results. Its components include formats for checklists (XCCDF) and checks (OVAL), and standardized identifiers for configuration items and platforms. SCAP 1.4, specified in NIST SP 800-126 Rev. 4 and finalized in 2026, is the current version; relative to SCAP 1.3, it simplifies the specification by no longer requiring tools to support older versions and by updating its digital signature and reference rules.

Managers do not need to understand SCAP's internal structure. What matters is the principle it embodies: **a baseline that can be expressed as machine-readable content can be checked automatically, continuously, and consistently across tools.** When evaluating configuration assessment products, it is reasonable to ask whether they consume standard content such as SCAP data streams or other open policy formats, or whether every baseline must be re-encoded in a proprietary language, creating lock-in and a risk of translation errors.

### NIST SP 800-128: configuration management as a discipline

Where the references above describe *what* to configure, NIST SP 800-128 describes *how to manage* configurations over time. Originally published in 2011 and updated in 2019, it remains the principal NIST guide to security-focused configuration management. It frames configuration management as a set of activities—establishing an organizational program and policy, developing secure baseline configurations, controlling changes through formal processes that include security impact analysis, and monitoring configurations to detect unauthorized change—and it assigns these activities to roles such as a configuration control board.

Its enduring value is that it treats configuration as a managed lifecycle rather than a one-time hardening event, linking it directly to change management.

### Choosing and combining references

Most organizations combine references rather than selecting one. The table below summarizes how the main references are typically used in governance.

| **Reference**                | **Publisher**                           | **Typical governance use**                                        |
|------------------------------|-----------------------------------------|-------------------------------------------------------------------|
| CIS Controls v8.1, Control 4 | Center for Internet Security            | Program-level requirement for a secure configuration process      |
| CIS Benchmarks               | Center for Internet Security            | Consensus starting point for platform baselines                   |
| DISA STIGs                   | U.S. Defense Information Systems Agency | Mandatory in defence contexts; high-assurance reference elsewhere |
| Vendor baselines             | Microsoft, cloud and SaaS vendors       | Product-specific defaults, updated as products change             |
| NIST SP 800-70 Rev. 5        | NIST                                    | Checklist index; traceability to controls and CSF outcomes        |
| SCAP 1.4 (SP 800-126 Rev. 4) | NIST                                    | Machine-readable content for automated assessment                 |
| NIST SP 800-128              | NIST                                    | Configuration management process and roles                        |
| ISO/IEC 27002:2022, 8.9      | ISO/IEC                                 | Control requirement within an ISMS and certification scope        |

A practical approach is to choose one primary benchmark family per technology—often CIS for breadth, or the vendor baseline for platforms that evolve quickly—record the rationale, and use other references as cross-checks for high-impact settings. Whatever the choice, the organization should record the source and version of each baseline, because benchmarks change and auditors will ask which version was adopted and when it was last reviewed.

## Defining baseline scope: what should have a baseline

A common failure is creating baselines only for servers and workstations—the assets that traditional hardening guides were written for—while leaving higher-leverage domains loosely governed. A mature program defines baselines according to risk and blast radius, not according to which assets were historically managed by the infrastructure team.

### Scope by blast radius, not by asset count

The guiding principle is to place baseline governance where misconfiguration yields disproportionate harm. A single identity provider tenant has more security significance than a thousand desktop computers, because its configuration determines who can access everything else. A cloud organization's root policies determine what every account beneath them is allowed to do. Scope decisions should therefore begin with the question, "If this configuration were wrong, how much would be exposed, and how quickly?"

Scope should also follow the asset inventory. CIS Controls 1 and 2, covering inventories of enterprise assets and software, precede Control 4 for good reason: an organization cannot claim baseline coverage for assets it does not know it has. Coverage gaps—systems that no tool is assessing—are often more dangerous than known non-compliance, because they are invisible in every report.

The figure below presents baseline scope as a set of layers, from the identity and control-plane settings that govern everything else to the endpoints and application delivery patterns through which most day-to-day configurations are created.

<span id="_Toc243635563" class="anchor"></span>Figure 64: Layers of baseline scope, ordered by blast radius

### Identity and privileged access

Identity systems and privileged access settings should be governed by baseline, because identity misconfiguration is among the most common enablers of major compromise. An identity baseline typically covers multi-factor authentication requirements and permitted methods, conditional access rules, legacy authentication protocols, privileged role assignment and activation, session lifetimes, guest and external user policies, and secure defaults for service accounts and application registrations. Chapter 10 discusses identity governance in depth; the point here is that identity *configuration* deserves the same baseline discipline as server configuration, and usually more.

### Cloud control planes and network connectivity

Cloud tenant and account configurations should be governed by baseline, because many cloud failures arise at the control-plane level rather than in individual workloads. A control-plane baseline covers account and subscription structure, organization-wide guardrail policies, audit logging, default network segmentation, storage access defaults, encryption and key management settings, and restrictions on public exposure. Network security baselines cover firewall rules, security groups, inbound and outbound constraints, remote access, and management-plane access. Default "allow from anywhere" patterns should be prohibited by baseline except where explicitly justified and approved.

### Endpoints, servers, and administrative workstations

Endpoints and servers still matter, particularly those used by administrators and engineers, whose compromise yields privileged access. Baselines here cover operating system hardening, patch and update configuration, endpoint detection and response coverage, disk encryption, local administrator rights, application control, scripting restrictions, removable media, and secure remote access. Privileged access workstations—dedicated, highly restricted devices used only for administration—warrant their own, stricter tier.

### SaaS platforms and collaboration tools

SaaS platforms often represent large data and identity surfaces that are configured by business administrators rather than by IT. Baselines for SaaS should define secure sharing defaults, external collaboration constraints, third-party application consent, audit logging and retention, administrative role governance, and data loss prevention settings where appropriate. Because SaaS vendors release features continuously, SaaS baselines need a defined process for evaluating new settings as they appear, rather than discovering months later that a new external-sharing feature was enabled for everyone.

### Application delivery: containers, pipelines, and code

Finally, application deployment patterns should be baseline-governed, especially for containerized and cloud-native workloads. Baselines may cover container image provenance and hardening, orchestration platform settings, secret management, minimum logging, dependency and software supply chain practices, and the policy gates that every deployment must pass in CI/CD. The delivery pipeline is itself a high-value configuration domain: whoever can change the pipeline can change what is deployed everywhere, which is why pipeline configuration belongs near the top of the scope ladder alongside identity. Chapter 13 discusses the architectural patterns, and Chapter 19 addresses software supply chain governance.

| Scope baselines by blast radius. **The configuration of one identity tenant or one deployment pipeline can matter more than the configuration of every desktop in the organization.** |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## Baseline design: turning risk appetite into minimum secure configurations

A baseline is effective only when it is clear, implementable, and aligned with real operational needs. Baselines that are too weak provide false assurance; baselines that are too strict generate exception volume, workarounds, and resentment, and eventually lose credibility. Baseline design is therefore an exercise in calibrated judgment, and it should follow several managerial rules.

### Tiering baselines to risk

A single baseline for all systems is a false economy. Systems handling highly sensitive personal information, payment data, or critical operations require stricter baselines than low-impact systems, and imposing the strictest requirements everywhere wastes effort and invites resistance. Tiering also reduces exception volume, because teams can select the baseline tier appropriate to their system's classification rather than requesting individual exemptions from requirements that were never designed for them.

A common pattern defines three tiers. A *foundation* tier applies to every asset of a given type and contains the non-negotiable settings: no default credentials, logging enabled, supported software versions, encryption where available. An *enhanced* tier applies to systems processing confidential data or supporting important business services. A *high-assurance* tier applies to crown-jewel systems, privileged administration infrastructure, and systems subject to specific regulatory obligations. The tier assigned to a system should follow from its data classification and business impact assessment, linking baseline selection directly to the risk assessments described in Chapter 6.

### Explicit, testable requirements

Baselines must be explicit and testable. Statements such as "systems shall be hardened" or "appropriate logging shall be configured" are useless for governance, because compliance with them cannot be measured. Baselines should specify concrete settings or concrete outcomes: administrative access requires phishing-resistant multi-factor authentication; public access to storage is blocked at the account level; security logs are forwarded to the central log platform and retained for a defined period; default credentials are prohibited; encryption at rest is enabled with organization-managed keys for the high-assurance tier.

Each requirement should ideally carry four attributes: the setting or outcome required, the rationale or risk it addresses, the method of verification (automated check, manual review, or attestation), and the severity of non-compliance. The severity attribute matters because it later drives remediation timelines and exception approval levels. A requirement without a verification method is a hope, not a control.

### Adapting external benchmarks

Baselines should be aligned with recognized frameworks and benchmarks but adapted to the organization. Blindly copying external benchmarks creates burden: a full CIS Level 2 profile or a DISA STIG may include hundreds of settings, some of which break business functions or address threats irrelevant to the organization's context. Governance requires judgment—enforce high-impact settings rigorously, avoid trivial settings that create friction without meaningful risk reduction, and adapt where operational constraints exist.

The adaptation itself must be documented. When the organization adopts a benchmark but deviates from specific recommendations, it should record each deviation and its rationale at the baseline level, so that it is not re-litigated as an individual exception on every system. This *baseline-level deviation record* is a powerful governance artifact: it shows auditors that departures from a recognized benchmark were deliberate, reviewed, and approved, rather than accidental.

### Ownership and decision rights

Baselines must have owners. In practice, ownership is usually split, and the split should be explicit:

- **The CISO or security architecture function** typically owns baseline *content*: which requirements exist, at which tier, and why. It also proposes changes when threats or technologies evolve.

- **Platform and infrastructure teams** own baseline *implementation*: the images, templates, modules, and policy code that deploy and enforce the baseline, and the operational feasibility of each requirement.

- **System and business owners** own *compliance* for their systems and are accountable for remediating drift or requesting exceptions.

- **Risk owners** at the appropriate management level own *risk acceptance* for exceptions, in line with the organization's risk appetite and delegation of authority.

- **Internal audit and the second line** provide independent assurance that the baselines exist, are enforced, and are measured honestly, consistent with the three lines model.

Where a configuration control board exists, as NIST SP 800-128 recommends, it is the natural forum for approving baseline changes that affect many systems. The key governance principle is that the people who write the baseline should not be the only people who decide whether it is workable, and the people who must comply should not be able to waive it unilaterally.

### Versioning and change control

Baselines must be versioned and change-controlled. They evolve as vendors release new versions, as benchmarks are updated, and as the threat landscape shifts. When a baseline changes, the organization must know which systems are on which baseline version, how and by when migration will occur, and how exceptions approved under the old version will be treated. A change to a baseline is itself a change with security and operational impact, and it should pass through the same analysis as any significant change.

A baseline without maintenance governance becomes obsolete rapidly, particularly in cloud and SaaS environments where vendor defaults and services evolve continuously. A sensible minimum is a scheduled review cycle for each baseline—at least annually, as CIS Safeguard 4.1 expects—supplemented by event-driven reviews when a major product version is released, a significant vulnerability class emerges, or an incident reveals a gap.

### Decision point: how strict is strict enough?

Managers are frequently asked to arbitrate between security teams that want stricter baselines and delivery teams that want fewer constraints. The arbitration should not be settled by seniority or volume. It should be settled by reference to risk appetite and evidence. Useful questions include: Which threat scenario does this requirement address, and how plausible is it in our context? What is the operational cost of the requirement, and has it been measured or only asserted? Does a less disruptive configuration achieve most of the risk reduction? What does our exception data say about the requirement—is it routinely waived, which suggests it is unrealistic, or routinely met, which suggests it is cheap? **A defensible baseline is one whose strictness can be explained in terms of the risks it reduces and the costs it imposes.**

## Secure-by-default deployment: making compliance automatic

The most effective baseline is the one that teams do not have to think about. Secure-by-default deployment aims to make the default path the secure path, so that compliance is an outcome of how systems are built rather than a separate activity performed afterwards. This shifts configuration governance from inspection to design—a recurring theme in this book.

### Golden images and hardened templates

In traditional environments, secure-by-default often means **golden images**: pre-hardened, approved, versioned operating system and application images from which servers, virtual machines, and workstations are built. A golden image embeds the baseline, the required security agents, logging configuration, and approved software, so that every system starts in a known-good state. Network templates and standardized configuration management scripts play a similar role for network devices and servers.

Golden images carry their own governance requirements. They must have an owner, a build process that is itself controlled, a regular rebuild cadence so that they incorporate current patches, and a retirement process so that outdated images are not used to build new systems. An image that was hardened two years ago and never updated is not a golden image; it is a template for deploying known vulnerabilities. Managers should ask how old the oldest image in active use is and how many systems were built from images that are no longer approved.

### Infrastructure as code and approved modules

In cloud and modern delivery environments, secure-by-default is increasingly implemented through **infrastructure as code (IaC)**—the practice of defining infrastructure in version-controlled, machine-readable files rather than through manual configuration—and through platform engineering. Organizations create approved modules, templates, and landing zones that embed baseline controls: secure network patterns, logging defaults, encryption settings, restricted identity roles, and preconfigured monitoring. Teams deploy using these patterns rather than building environments manually.

This approach shifts security earlier in the delivery lifecycle without increasing friction, and it reduces the number of unique configurations, which reduces both risk and operational complexity. When every team builds differently, drift and misconfiguration are inevitable. When teams build through shared secure patterns, risk becomes governable. IaC also produces a natural evidence trail: the intended configuration is recorded in a repository, every change is attributable to a person and a review, and the history of the configuration can be reconstructed for any point in time.

### Policy as code and pipeline gates

**Policy as code** expresses baseline requirements as executable rules that are evaluated automatically—against infrastructure code before deployment, against deployment requests at the moment of deployment, and against running resources afterwards. A pipeline gate might reject an infrastructure change that creates a publicly accessible storage bucket, a database without encryption, or a security group open to the Internet on an administrative port. Cloud providers offer native policy services that can deny non-compliant resource creation at the organization or account level, providing a guardrail that applies regardless of which tool or person initiates the change.

From a governance perspective, policy as code has three important properties. It makes baseline requirements unambiguous, because a rule either passes or fails. It enforces requirements consistently, without depending on reviewer attention. And it makes the policy itself subject to version control, review, and testing, which means changes to the rules are visible and attributable. Its risk is that policy code becomes a hidden layer of governance that only engineers can read. Mature organizations therefore maintain traceability from each policy rule back to the baseline requirement and risk it implements, and they route changes to high-impact rules through the same approval process as changes to the baseline document. Each rule should also be deliberately set to *deny* (blocking deployment) or *audit* (allowing but flagging), in proportion to the severity of the requirement it enforces.

### Procurement as a secure-by-default lever

Secure-by-default also extends to procurement and third-party governance. Where possible, organizations should require that new platforms and services ship with secure defaults, support baseline enforcement through configuration APIs and policy, expose configuration state for automated assessment, and provide audit logging of administrative changes. Products whose security depends on dozens of manual hardening steps, or whose configuration cannot be read programmatically, impose a lasting governance cost and may be unacceptable for high-risk use cases. The secure-by-design principles promoted by CISA and its international partners, including CCCS, give buyers a useful vocabulary for these requirements, and Chapter 19 discusses how to embed them in supplier due diligence and contracts.

## Change management and configuration control

Baselines describe a desired state; change management determines whether the organization can move from one approved state to another without losing control along the way. The two are inseparable. A baseline without change control decays; change control without a baseline has nothing to protect.

### Linking baselines to change management

Every significant change to an in-scope system should answer one configuration question before it is approved: will the system still conform to its baseline after the change, and if not, is an exception being requested? NIST SP 800-128 frames this as a **security impact analysis**, performed before a change is implemented and followed by verification afterwards that the change was made as approved and that the system still meets its baseline configuration. In organizations using IaC, much of this analysis can be automated through policy checks on the proposed change; in others, it is a question on the change request form that a reviewer must actually answer.

The governance aim is to make baseline conformance a normal criterion of change approval, not a separate security review that delivery teams experience as an obstacle. When the change advisory board or its automated equivalent routinely asks "what is the baseline impact?", configuration governance becomes part of how the organization works rather than something that happens to it.

### Emergency change and break-glass access

Emergencies are the most common origin of serious drift. During an outage, an engineer opens a firewall rule, disables a security agent, or grants a broad role to restore service. These actions may be entirely justified. The governance failure is not the emergency change but the absence of reconciliation afterwards. A sound emergency change process permits rapid action using controlled break-glass accounts, logs every action, requires retrospective review within a short, defined period, and creates a tracked item to either revert the change or convert it into a formal exception. Monitoring should flag break-glass account use automatically, and a pattern of frequent emergency changes in one area should be treated as a signal that the normal change process or the baseline itself needs attention.

### Configuration items, inventory, and the CMDB problem

Configuration governance relies on knowing what exists and how it is configured. Many organizations maintain a configuration management database (CMDB) for this purpose, and many of those databases are incomplete or out of date. In dynamic environments, a manually maintained CMDB cannot keep pace with resources that are created and destroyed many times a day. Mature organizations therefore treat authoritative, automatically discovered inventories—cloud provider APIs, endpoint management platforms, identity directories, and IaC state—as the primary sources of truth, and use the CMDB as an integration and ownership layer rather than as the sole record. For managers, the practical test is simple: ask for the number of assets of a given type from two different sources and see whether the numbers agree. When they do not, the difference is the organization's blind spot.

## Drift detection and remediation: keeping reality aligned with intent

Even with secure-by-default deployment, drift occurs. Automation has gaps, emergencies happen, vendors change defaults, and people find ways around controls they consider inconvenient. Drift governance therefore requires four capabilities: detection, prioritization, remediation enforcement, and learning.

### Detection: continuous, not annual

Detection means the organization can measure actual configuration state and compare it with the baseline. This is implemented through configuration compliance scanners, endpoint and mobile device management platforms, cloud security posture management, SaaS security posture management, identity posture tools, and continuous policy evaluation. The specific tools matter less than the governance principle: **compliance must be continuously measurable, not manually audited once per year.** An annual configuration review describes a system as it was on the day of the review; continuous measurement describes it as it is.

Detection must also cover its own blind spots. Every assessment tool covers some asset types and not others, and assets can fall outside assessment because an agent was never installed, a cloud account was never connected, or a SaaS tenant was never registered. Coverage—the proportion of in-scope assets actually being assessed—is therefore the first drift metric managers should request, before any compliance percentage.

### Prioritization by risk

Prioritization means the organization focuses on drift that matters. Not all deviations carry equal risk, and treating them as equal produces long lists that nobody can work through. Governance should classify drift by severity and impact, using the severity attribute assigned to each baseline requirement and the tier and exposure of the affected system. Public exposure of a storage bucket containing personal information is critical; a deviation in screen-lock timeout on a kiosk is minor. Exposure context matters as well: the same misconfiguration is more urgent on an Internet-facing system than on an isolated test machine.

By prioritizing real risk, the organization reduces alert fatigue and directs scarce remediation capacity to where it changes the risk picture. This is the configuration equivalent of the risk-based vulnerability prioritization discussed in Chapter 8, and in practice the two should be managed together, since attackers chain misconfigurations and unpatched vulnerabilities.

### Remediation enforcement and automated correction

Remediation enforcement means drift is corrected, not merely reported. Many organizations can produce posture dashboards but fail to close issues. Governance maturity includes defined remediation ownership, timelines by severity, and escalation pathways that reach accountable managers when timelines are missed. For critical baseline violations, remediation should be automatic where feasible—a configuration management tool that reapplies the approved setting, or a cloud policy that reverts public access—or should trigger rapid escalation to system owners with clear deadlines.

Automated remediation is powerful but must itself be governed. Automatically reverting a change can break a production service if the change was an approved emergency fix not yet recorded. Organizations should therefore decide, rule by rule, whether drift is auto-corrected, auto-corrected with notification, or reported for human action, and they should be able to explain that decision. A reasonable pattern is to auto-correct the small set of settings whose violation is never acceptable—public exposure of sensitive data stores, disabled audit logging, unrestricted administrative ports—and route the rest through owned remediation workflows.

### Drift as a signal about the operating model

An effective drift governance program treats drift as information, not only as non-compliance. High drift rates in one domain often indicate that baselines are unrealistic, automation is incomplete, or manual change pathways dominate. Recurring drift on the same setting suggests that a team needs that setting to function and that the baseline or the platform should accommodate the need safely. Drift concentrated in one business unit may reveal a staffing, skills, or tooling problem rather than a discipline problem.

Root-cause analysis of recurring drift, reviewed periodically by the baseline owner and platform teams, is one of the most productive improvement loops in a security program. The question to ask is not only "who drifted?" but "why does our system make drifting easier than complying?"

| A dashboard full of red is not evidence of governance. **Drift governance is proven by closure: findings that are owned, corrected within defined timelines, and analyzed so that they do not recur.** |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## Exceptions and compensating controls: making trade-offs explicit

In every mature organization, some systems cannot meet baseline requirements because of operational constraints, legacy dependencies, vendor limitations, or economic realities. A medical device may run an operating system that cannot be updated without affecting its regulatory approval. A core banking platform may require a legacy protocol for an interface that cannot be replaced until a multi-year modernization completes. The governance failure is not having exceptions; it is having undocumented, uncontrolled exceptions whose risk nobody has accepted.

### Anatomy of a defensible exception

Chapter 13 introduced exceptions with expiry as a discipline for architecture standards. The same logic applies to baselines, where exceptions are far more numerous and therefore need a more systematic process. Each exception request should state, at minimum:

- **The requirement:** which baseline requirement, at which tier and version, cannot be met, and on which systems.

- **The reason:** why it cannot be met, with enough evidence that a reviewer can judge whether the constraint is genuine or merely inconvenient.

- **The risk:** which threat scenarios the deviation enables, what data or services are exposed, and the resulting residual risk rating using the organization's standard criteria.

- **The compensating controls:** what alternative measures will reduce the risk, how they will be verified, and who operates them.

- **The duration and exit plan:** when the exception expires and what must happen—a project, a replacement, a vendor fix—for the system to return to compliance.

- **The approval:** the risk owner whose authority matches the residual risk level, not only a technical team or the requester's own manager.

The approval level should scale with residual risk. Minor deviations may be approved by the baseline owner; deviations that leave a high residual risk should require a senior business risk owner; and those that would exceed risk tolerance should be escalated to the executive or committee level that sets tolerance. This aligns exceptions with the delegation of authority for risk acceptance described in Chapter 4.

### Compensating controls that actually compensate

**Compensating controls** are alternative measures that reduce risk to an acceptable level when the baseline requirement cannot be met. If a system cannot be patched or hardened, compensating controls may include network isolation, restricting access to a small set of managed hosts, application allow-listing, enhanced monitoring with specific detection use cases, removal of unnecessary functionality, or additional backup and recovery measures. The objective is to keep risk within tolerance even when the baseline cannot be met.

A compensating control must address the same risk as the requirement it replaces. Enhanced monitoring does not compensate for an exposed administrative interface if nobody responds to the alerts; network isolation does not compensate if the isolation is a firewall rule with a broad exception of its own. Compensating controls should therefore be verified when the exception is approved and at each renewal, and their failure should automatically reopen the exception for review.

### The exception register and expiry

All exceptions should be recorded in a single **exception register**, not scattered across email threads, ticketing queues, and spreadsheets owned by different teams. The register records each exception's attributes, links it to the affected assets and to the risk register, and drives reminders before expiry. Exceptions must be time-bound. Otherwise, temporary deviations become permanent and the organization accumulates risk that was accepted once, by someone who may no longer work there, under conditions that may no longer apply.

Expiry should have consequences. When an exception expires without renewal, the system is treated as non-compliant and enters the normal remediation and escalation process. Renewal should require fresh justification, evidence that compensating controls are still operating, and progress against the exit plan. A common and effective rule limits the number of renewals, or the total duration, before escalation to a higher approval level, which prevents the indefinite rollover of "temporary" arrangements. Where configuration tools support it, the exception should also be recorded in the tool itself—suppressing the finding only for the named assets and only until the expiry date—so that the technical view and the governance view stay consistent.

The figure below summarizes the exception governance steps, from request to closure.

<span id="_Toc243635564" class="anchor"></span>Figure 65: Governing baseline exceptions from request to closure

### Exceptions as a portfolio

Exceptions should also be managed as a portfolio. How many exist, which domains and business units are most affected, how many are past expiry, what residual risk accumulates across them, and what investment would be required to eliminate the highest-risk ones? Portfolio views often reveal that a large share of exceptions trace back to a small number of root causes—a legacy platform, a vendor that does not support a required setting, an unrealistic requirement—and that addressing those causes would retire many exceptions at once. This turns exception data into an input for investment decisions, a topic Chapter 17 develops.

This process makes the politics of security visible: trade-offs become accountable decisions rather than hidden configurations.

## Baselines in the cloud: configuration governance under shared responsibility

Cloud environments amplify both the risk of misconfiguration and the ability to enforce baselines at scale. Everything in a public cloud is configured through an API, which means everything can be inspected and enforced programmatically—and everything can also be changed in seconds by anyone with sufficient permissions. Governance must therefore be explicit about what the organization controls, what the provider controls, and how the organization's portion is kept within baseline. Chapter 15 treats cloud security comprehensively; this section focuses on the configuration governance mechanisms.

### Control-plane baselines and landing zones

In the shared responsibility model, which Chapter 15 examines as a governance instrument, cloud providers secure the underlying infrastructure, while customers are responsible for configuring identities, networks, storage permissions, and workloads. Many high-profile cloud incidents arise from errors in those customer-controlled layers rather than from failures of the provider. A cloud baseline should therefore focus heavily on control-plane governance: account structure and separation of environments, privileged access, organization-wide logging, network segmentation defaults, restrictions on public exposure, encryption and key management settings, secure service configurations, and continuous posture monitoring.

The most effective vehicle for these controls is a **landing zone**: a pre-configured, governed cloud foundation into which new accounts or subscriptions are provisioned with baseline guardrails already in place. A landing zone typically enforces centralized logging that workload teams cannot disable, denies the creation of resources in unapproved regions, blocks public access to storage by default, and assigns standard security roles. Its governance value is that it moves baseline enforcement from each workload team to the platform, so that the baseline applies even to teams that have never read it.

### CSPM and SSPM

**Cloud security posture management (CSPM)** tools continuously evaluate cloud configurations against baselines, benchmarks, and regulatory control sets, identify misconfigurations and risky combinations of settings, and often support automated remediation. Cloud providers include native posture services, and independent products cover multiple clouds. **SaaS security posture management (SSPM)** performs the equivalent function for SaaS tenants such as collaboration suites, CRM platforms, and file-sharing services, where configuration is often controlled by business administrators and changed by vendor releases.

For managers, these tools raise three governance questions. First, which baseline are they measuring against? Out-of-the-box rule sets mapped to CIS Benchmarks or regulatory frameworks are useful, but the organization's own tiered baseline and approved deviations should be reflected, or the tool will report noise and miss what the organization actually cares about. Second, what is their coverage? A posture tool connected to some accounts but not others, or to the main collaboration suite but not to departmental SaaS, provides partial assurance presented as complete. Third, who acts on findings? A posture tool without remediation ownership becomes an expensive generator of unread reports.

### Console changes, ClickOps, and reconciliation

Cloud introduces a central governance tension: ease of change versus control. The web console makes it easy to bypass automation, a practice engineers sometimes call "ClickOps". Changes made directly in the console are not reviewed, not recorded in the IaC repository, and may be overwritten—or may silently persist—depending on how automation is configured. Governance should therefore restrict manual changes to critical configurations in production, monitor and alert on such changes, and require that any manual change be reconciled back into code or reverted.

The objective is not to prevent emergency action but to ensure emergency action is logged, reviewed, and reconciled with the baseline. IaC drift detection, which compares deployed resources with the state recorded in code, is a practical control here: it identifies resources that were changed outside the pipeline and resources that exist with no code at all.

### Regulatory and residency constraints

Cloud baselines should also encode regulatory and contractual constraints. Data residency requirements, encryption obligations, and key ownership models may require baseline enforcement at the tenant or organization level, not only at the application level. Québec Law 25, for example, requires a privacy impact assessment before personal information is communicated outside Québec, and organizations that commit to keeping certain data in Canadian regions can make that commitment enforceable by denying resource creation elsewhere. Baselines that encode such constraints convert legal commitments into technical guardrails and provide evidence that the commitments are honoured. Chapter 16 discusses the privacy obligations themselves.

## Configuration governance in the enterprise: endpoints, identity, and forgotten defaults

Baselines are equally critical in on-premises and hybrid enterprise environments, which carry long histories of accumulated configuration.

### Identity as the highest-leverage baseline

Of all enterprise configuration domains, identity usually has the widest reach, and the identity baseline settings described earlier in this chapter are only part of the picture. On-premises directory services add their own accumulated risks: stale accounts, excessive delegation, weak legacy protocols, and trust relationships established years ago for reasons nobody remembers. Governance should insist that identity baselines are enforced and monitored with the same seriousness as network baselines, and that identity configuration changes—especially to conditional access, privileged roles, and federation settings—are treated as high-impact changes requiring review. Chapter 10 develops identity governance; the configuration lesson is that identity settings should be in the first tier of continuously monitored baselines.

### Administrator and engineer endpoints

Endpoints used by administrators and engineers deserve particular baseline attention because they are often the initial foothold for attacks that escalate to domain-wide or tenant-wide compromise. Baselines for these devices should include full-disk encryption, endpoint detection and response coverage, prompt patching, restrictions on local administrative rights, application and script control, secure configuration of remote access, and separation between administrative and everyday activity. Microsoft's security baselines and the corresponding CIS Benchmarks provide a sound starting point for Windows fleets, with comparable references available for macOS and Linux.

### Forgotten defaults outside IT

The forgotten defaults in enterprise environments sit in devices and platforms deployed or administered outside central IT: printers and multifunction devices, cameras, building management systems, conference room equipment, laboratory instruments, and departmental SaaS applications. These often ship with default credentials, unnecessary services, and management interfaces reachable from the corporate network. CIS Control 4 explicitly includes non-computing and IoT devices for this reason, and Safeguard 4.7 requires default accounts to be managed.

Governing baselines for these environments is harder because ownership is fragmented and the people who buy the devices are rarely security specialists. This is why baseline governance must intersect with procurement and accountability models: a purchasing rule that any network-connected device must be registered, assigned an owner, and configured to a minimum baseline before connection does more than any scanner. Chapter 11 discusses operational technology and cyber-physical systems, where configuration changes can have safety consequences and must be governed with particular care.

## Metrics and evidence: how managers know baseline governance works

Baseline governance is one of the most measurable security domains, which is both an opportunity and a temptation. The opportunity is precise, objective reporting. The temptation is to report large compliance percentages that look reassuring but hide coverage gaps, low-severity padding, and long-lived exceptions. Managers should insist on metrics that reflect control effectiveness rather than activity.

### Coverage, compliance, and drift metrics

Useful indicators fall into a small number of families. **Coverage** measures the proportion of in-scope assets that are assessed against an assigned baseline; it should always be reported alongside compliance, because high compliance on low coverage means little. **Compliance** measures the proportion of assessed assets, or of high-severity requirements, that meet the baseline, reported by tier and domain. **Drift rate** measures how often compliant assets become non-compliant over a period, which reveals whether secure-by-default mechanisms are holding. **Remediation performance** measures the time taken to correct critical and high-severity deviations against defined targets. **Exception health** measures the number, age, and residual risk of exceptions, and the proportion past expiry. **Recurrence** measures how often the same misconfiguration type reappears after correction.

### Configuration drift KRIs and thresholds

Some of these measures should be elevated to key risk indicators (KRIs), with thresholds linked to risk appetite, so that breaches trigger escalation rather than simply appearing in a monthly report. The table below gives illustrative examples; actual thresholds must be set by each organization in line with its own appetite and tolerance.

| **KRI**                                     | **What it signals**                       | **Illustrative threshold**                   |
|---------------------------------------------|-------------------------------------------|----------------------------------------------|
| Baseline assessment coverage, tier 1 assets | Blind spots in assurance                  | Below 98% triggers escalation                |
| Critical deviations open beyond target      | Remediation capacity or ownership failure | Any open beyond 72 hours                     |
| Exceptions past expiry                      | Erosion of risk acceptance discipline     | Any high-risk exception past expiry          |
| Unreconciled manual production changes      | Bypass of secure-by-default pathways      | More than 5 per month in one platform        |
| Recurring misconfiguration types            | Baseline or platform design problem       | Same finding recurs three times in a quarter |

The KRIs connect to the scenario-based risk assessments of Chapters 5 and 6. If a top-ranked scenario involves exposure of member or customer data through cloud misconfiguration, then public exposure findings and unreconciled console changes in that environment are leading indicators of that scenario's likelihood. In the CyberRiskGuardian method, such configuration KRIs do not change the calculation by themselves; they are recorded as evidence that the analyst uses, at each review, to revisit the scenario's probability of exploitation Pb(ψ,A) and its resilience θ, so that the assessment rests on measured conditions rather than on periodic expert judgment alone.

### Evidence for auditors, regulators, and insurers

Configuration governance produces several types of evidence that auditors, regulators, and insurers recognize: approved and versioned baseline documents with owners and review dates; baseline-level deviation records; mappings from baseline requirements to frameworks such as ISO/IEC 27002:2022 control 8.9, NIST CSF 2.0 PR.PS-01, and CIS Control 4; continuous compliance reports with coverage; change records showing security impact analysis; and the exception register with approvals, compensating controls, and expiry dates. Evidence is strongest when it is generated by the operating process itself rather than assembled for the audit. An organization that can export its posture history for the past twelve months demonstrates operation over time; one that produces a screenshot taken the week before the audit demonstrates only that it knew the audit was coming.

### Avoiding compliance theatre in configuration metrics

Several patterns turn configuration metrics into compliance theatre. Reporting compliance across all requirements equally lets thousands of trivial settings mask a handful of critical failures. Excluding assets that are "out of scope" without a documented scope decision quietly removes the hardest systems from the denominator. Suppressing findings in tools without corresponding exception records makes the dashboard green while the risk remains. Counting remediation tickets closed rather than deviations actually corrected rewards paperwork.

The most important governance metric is not any single number; it is whether the organization can show a closed-loop process. Standards exist, they are implemented by default, compliance is continuously measured with known coverage, drift is detected and corrected within defined timelines, exceptions are accountable and time-bound, and baselines evolve with technology and risk. This is what transforms baseline governance into audit-ready evidence of due diligence.

## Configuration governance in practice

The principles in this chapter are easiest to appreciate when they are absent, as the following scenario shows.

### Illustrative scenario: a credit union after a misconfiguration incident

Ridgeview Credit Union (a fictional organization) is a provincially regulated Ontario credit union with about 120,000 members, a growing digital banking channel, and a small security team of five reporting to the chief risk officer. Two years ago it moved its loan origination analytics to a public cloud platform, using an external integrator to build the environment. Its board receives a quarterly cybersecurity report showing "95% configuration compliance".

On a Friday afternoon, a security researcher contacts the credit union to report that a cloud storage container holding exported loan application files—names, addresses, income details, and scanned identity documents—is publicly readable. The investigation finds that an integrator engineer had enabled public access eleven weeks earlier to troubleshoot a data transfer with a credit bureau and never reverted it. Access logging on the container had been disabled during the same session to "reduce noise". The posture tool had flagged the public access, but the alert went to a shared mailbox owned by the integrator. The 95% compliance figure was accurate: it measured on-premises servers and workstations against CIS Level 1 benchmarks and did not include the cloud environment at all.

The immediate response follows the incident processes described in Chapter 12: access is revoked, the scope of exposure is assessed as far as the limited logs permit, and the credit union's privacy officer evaluates the real risk of significant harm under PIPEDA, notifies the Office of the Privacy Commissioner of Canada and affected members, and informs the provincial regulator. Because logging had been disabled, the credit union cannot prove that the files were not downloaded, and it must assume the worst. The cost of that single disabled setting is therefore measured not in technical terms but in notification scope, member trust, and regulatory attention.

The post-incident review, chaired by the chief risk officer, deliberately focuses on governance failures rather than on the integrator's engineer. It identifies six:

- **Scope:** the cloud environment had no assigned baseline and was outside the compliance metric reported to the board, so coverage was misrepresented as compliance.

- **Secure-by-default:** the environment was built manually, without a landing zone that would have blocked public storage access and protected logging from being disabled.

- **Change management:** the integrator's change was never raised as a change request, and no security impact analysis occurred.

- **Drift response:** the posture finding was routed to a mailbox with no accountable owner and no escalation timeline.

- **Exceptions:** the temporary public access was, in effect, an unapproved exception with no expiry, no compensating control, and no risk owner.

- **Third-party governance:** the integrator contract did not require adherence to the credit union's baselines or define responsibilities for configuration monitoring.

The management response, approved by the board's risk committee, is structured around the lifecycle in this chapter. The credit union adopts a tiered cloud baseline based on the CIS Benchmark for its cloud platform, with documented local deviations; classifies the loan analytics environment as high-assurance; and rebuilds it inside a landing zone whose guardrails deny public storage access and prevent audit logging from being disabled. Infrastructure changes must now pass through code review with policy-as-code checks, and console changes in production trigger alerts to the security team. Critical posture findings are routed to named system owners with a 24-hour remediation target and automatic escalation to the chief risk officer. An exception register is introduced, with a 90-day maximum duration and approval by the business risk owner for any high-risk deviation. The integrator contract is amended to require baseline adherence, change logging, and notification of configuration changes.

The board report changes as well. Instead of a single compliance percentage, the risk committee now sees coverage and compliance by tier, the count of critical deviations open beyond target, unreconciled manual changes, and the number and age of exceptions. Six months later, the figures are less flattering than "95%" but far more useful: cloud coverage is complete, two critical findings breached their target and were escalated, and three exceptions were closed when the credit bureau integration was redesigned. **The credit union did not become more secure by buying a new tool; it already had one. It became more secure by assigning scope, ownership, timelines, and expiry to the configuration decisions it was already making.**

### Questions managers and boards should ask

The scenario suggests a set of questions that managers, executives, and directors can ask about any organization's configuration governance:

- **Scope:** Which asset classes have an approved baseline, and which high-impact domains—identity, cloud control planes, SaaS tenants, delivery pipelines—do not?

- **Coverage:** What proportion of in-scope assets is actually being assessed, and how do we know the inventory is complete?

- **Source:** Which external benchmarks and versions are our baselines based on, and where are our documented deviations from them?

- **Ownership:** Who owns the content, implementation, compliance, and risk acceptance for each baseline, and are those roles written down?

- **Default path:** What proportion of new systems are deployed through golden images, approved IaC modules, or landing zones rather than built manually?

- **Change linkage:** Does every significant change include a security impact analysis against the baseline, including changes made by suppliers and integrators?

- **Drift response:** What are the remediation targets for critical deviations, how often are they missed, and who is notified when they are?

- **Exceptions:** How many exceptions are open, how many are past expiry, what residual risk do they represent, and who approved the highest-risk ones?

- **Recurrence:** Which misconfigurations keep recurring, and what has been changed in the baseline or platform to stop them?

- **Evidence:** Could we demonstrate to an auditor or regulator, from data generated by our normal processes, that our baselines were enforced over the past year?

- **Third parties:** Do our contracts require vendors, integrators, and managed service providers to adhere to our baselines and to report configuration changes?

| Ask for coverage before compliance. **A compliance percentage measured on part of the estate is a statement about the part, not about the organization.** |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------|

## Bringing it together: making secure configuration the default and drift the exception

Security baselines and configuration governance are among the most decisive elements of cybersecurity governance because they address the most common root causes of failure: misconfiguration, uncontrolled change, and fragile defaults. They sit at the precise point where the policies, frameworks, and risk appetite discussed throughout this book must become concrete settings on concrete systems, and where the gap between what an organization claims and what it actually does is most easily measured.

Baselines are a governance mechanism, not merely technical hygiene: they are measurable, they scale in both directions, and they turn deviations into explicit, owned decisions. Public references mean that no organization needs to invent its baselines alone, while ISO/IEC 27002:2022 control 8.9 and NIST CSF 2.0 place secure configuration firmly within the management system.

For managers, the governance objective is not to master configuration detail. It is to ensure that the organization has a repeatable system that converts risk appetite into tiered, testable, owned minimum configurations; deploys those configurations by default through golden images, infrastructure as code, landing zones, and policy as code; links every significant change to a baseline impact analysis; detects and corrects drift continuously through tools such as CSPM and SSPM, with known coverage and defined timelines; and manages exceptions as explicit, time-bound risk decisions recorded in a single register. Each stage depends on the others, and the value lies in the closed loop rather than in any single stage.

When baseline governance is mature, the organization becomes harder to compromise not because of any particular product, but because its systems are consistently configured to be defensible and it knows when they are not. This is one of the rare areas in cybersecurity where governance discipline reliably translates into measurable reduction of risk at scale. The next chapter extends these ideas to cloud cybersecurity more broadly, where shared responsibility, provider dependencies, and the speed of change make configuration governance not just valuable but indispensable.

## Chapter summary

Many damaging incidents stem from ordinary misconfiguration and uncontrolled change, not novel exploits: the control existed on paper, but the system was not configured to match it. This chapter argues that security baselines and configuration governance are among the highest-leverage control domains managers can govern, because baselines convert risk appetite into minimum settings that can be checked. It surveys the reference ecosystem, scopes baselines by blast radius, and shows how to design tiered, testable, owned baselines. It then treats secure-by-default deployment, change control, drift detection, exceptions, cloud and enterprise specifics, and metrics as one closed loop that produces evidence of due diligence, illustrated by a credit union recovering from a misconfiguration incident.

### Key points to retain

- A security baseline is an approved, enforceable minimum configuration for a class of systems; external benchmarks are inputs to it, not substitutes for it.

- Configuration drift silently turns accurate assurance into inaccurate assurance, so compliance must reflect current state, not how systems were built.

- Baselines are governance because they are measurable, they scale risk in both directions, and exceptions turn hidden trade-offs into attributable decisions.

- ISO/IEC 27002:2022 control 8.9, NIST CSF 2.0 PR.PS-01, CIS Control 4, and the NIST SP 800-53 CM family all treat secure configuration as a core control.

- Organizations should choose one primary benchmark family per technology, use others as cross-checks, and record the source and version of each baseline.

- Baseline scope should follow blast radius and the asset inventory, putting identity, cloud control planes, and delivery pipelines near the top.

- Effective baselines are tiered to risk, explicit and testable, owned through split decision rights, versioned, and reviewed at least annually.

- Secure-by-default deployment through golden images, infrastructure as code, policy as code, landing zones, and procurement makes compliance an outcome of how systems are built.

- Every significant change needs a baseline impact analysis, emergency changes must be reconciled, and drift governance requires continuous detection with known coverage, risk-based prioritization, enforced remediation, and root-cause learning.

- In the cloud and the enterprise, landing zones, CSPM and SSPM with known coverage, ClickOps reconciliation, residency guardrails, identity baselines, and owned devices outside IT keep configuration governed.

- Defensible exceptions state the requirement, reason, risk, compensating controls, expiry, and approval by a risk owner whose authority matches the residual risk.

- Managers should ask for coverage before compliance and expect evidence generated by the operating process itself, not assembled for an audit.

### Key terms

- **Security baseline** — A documented, approved minimum secure configuration for a defined class of systems against which actual systems can be compared without interpretation.

- **Baseline configuration** — In NIST SP 800-128, a formally reviewed and agreed set of specifications for a system that can be changed only through change control.

- **Benchmark** — An external configuration reference, such as a CIS Benchmark, DISA STIG, or vendor baseline, used as an input to the organization's own baseline decision.

- **Hardening** — Reducing attack surface by removing or disabling unnecessary features, services, ports, accounts, and defaults, and configuring what remains securely.

- **Configuration drift** — The gradual divergence between the intended baseline and the actual deployed state of a system.

- **Secure-by-default deployment** — Deploying systems in a secure configuration automatically, so that the secure path is the path of least resistance.

- **Golden image** — A pre-hardened, approved, versioned operating system or application image from which systems are built in a known-good state.

- **Infrastructure as code (IaC)** — Defining infrastructure in version-controlled, machine-readable files rather than through manual configuration.

- **Policy as code** — Baseline requirements expressed as executable rules evaluated automatically before deployment, at deployment, and against running resources.

- **Landing zone** — A pre-configured, governed cloud foundation into which new accounts or subscriptions are provisioned with baseline guardrails already in place.

- **Cloud security posture management (CSPM)** — Tools that continuously evaluate cloud configurations against baselines, benchmarks, and control sets and often support automated remediation.

- **SaaS security posture management (SSPM)** — The equivalent of CSPM for SaaS tenants, where configuration is often controlled by business administrators and changed by vendor releases.

- **SCAP** — The Security Content Automation Protocol, NIST specifications for machine-readable configuration checks and results so different tools evaluate the same baseline.

- **Compensating control** — An alternative measure that addresses the same risk as an unmet baseline requirement and keeps risk within tolerance.

- **Exception register** — The single record of all baseline exceptions, linked to assets and the risk register, that drives reminders and consequences at expiry.

## Review questions

### Essay questions

1.  The chapter claims that baselines are 'governance, not technical hygiene'. Critically evaluate this claim using the three properties the chapter identifies, and explain how a board should use baseline evidence differently from policy evidence.

2.  Using the Ridgeview Credit Union scenario, analyze how each of the six governance failures contributed to the incident and its cost, and evaluate whether the management response closes the baseline governance loop.

3.  Choose an organization you know. Apply the blast-radius principle to propose its baseline scope and tiering, and justify which three domains should receive baselines first and which reference sources each should use.

4.  A delivery team argues that the security team's baselines are too strict and slow releases. As the manager asked to arbitrate, explain how you would decide 'how strict is strict enough', using the chapter's decision questions, exception data, and secure-by-default mechanisms.

5.  Design a set of configuration metrics and KRIs for a board report. Explain how your design avoids the compliance theatre patterns described in the chapter and how the KRIs connect to scenario-based risk assessment in CyberRiskGuardian.

### Multiple-choice questions

Choose the best answer. Answers are listed in the appendix "Answers to multiple-choice questions".

**1.** Which statement best defines a security baseline as the chapter uses the term?

> A. A vendor's recommended settings for a typical customer's environment
>
> B. An aspirational guide describing good configuration practices
>
> C. A documented, approved minimum secure configuration for a class of systems
>
> D. The recorded, approved state of one particular system under change control

**2.** What is configuration drift?

> A. The gradual divergence between the intended baseline and the actual deployed state
>
> B. The planned migration of systems from one baseline version to the next
>
> C. The difference in strictness between CIS Level 1 and Level 2 profiles
>
> D. The process of adapting an external benchmark to local constraints

**3.** A manager reports that 'all production servers meet the hardening standard' because they were built from approved images. Why can this statement not be supported?

> A. It overstates the number of servers in the production environment
>
> B. It relies on the wrong external benchmark for server hardening
>
> C. It is acceptable as long as every image used was a golden image
>
> D. It describes how systems were built, not how they are configured today

**4.** Which control, new in the 2022 edition, calls for configurations to be established, documented, implemented, monitored, and reviewed?

> A. CIS Controls v8.1, Control 4
>
> B. ISO/IEC 27002:2022, control 8.9
>
> C. NIST CSF 2.0, outcome PR.PS-01
>
> D. NIST SP 800-53 Rev. 5, CM-7

**5.** Tier 1 baseline assessment coverage is 96%, and 99% of assessed assets are compliant. Using the chapter's illustrative KRIs, what should happen?

> A. Escalate, because coverage is below the illustrative 98% threshold
>
> B. Take no action, because compliance is above 98% of assessed assets
>
> C. Escalate only if a critical deviation stays open beyond 72 hours
>
> D. Report the 99% compliance figure to the board as the headline

**6.** An organization that is not subject to DISA STIGs wants to use them. What does the chapter recommend?

> A. Adopt them wholesale to achieve the highest possible security posture
>
> B. Ignore them, because they apply only to U.S. Department of Defense systems
>
> C. Replace CIS Benchmarks with them across the whole estate
>
> D. Use them as a reference for high-assurance systems and as a cross-check

**7.** What is the main managerial principle embodied by SCAP and machine-readable configuration content?

> A. Every baseline should be re-encoded in each tool's own proprietary language
>
> B. A machine-readable baseline can be checked automatically, continuously, and consistently across tools
>
> C. Configuration checks should remain manual to avoid translation errors between tools
>
> D. Only NIST may publish configuration checklists for IT products

**8.** Applying the chapter's blast-radius principle, which configuration domains belong near the top of the baseline scope ladder?

> A. Office desktops, because they are the most numerous asset class
>
> B. Conference room equipment, because it is often forgotten by IT
>
> C. The identity provider tenant and the deployment pipeline
>
> D. Test servers, because their configuration changes most frequently

**9.** Why does the chapter consider coverage gaps more dangerous than known non-compliance?

> A. They always involve systems that store personal information
>
> B. Auditors penalize them more heavily than any other finding
>
> C. They are always more expensive to remediate than known deviations
>
> D. They are invisible in every report the organization produces

**10.** What is a key benefit of tiering baselines into foundation, enhanced, and high-assurance levels?

> A. It ensures the strictest settings apply to every system everywhere
>
> B. It reduces exception volume because teams select a tier matching their system's classification
>
> C. It removes the need for risk owners to approve any deviations
>
> D. It lets each engineer choose the settings they prefer for their system

**11.** A baseline states that 'appropriate logging shall be configured'. What is the main problem with this requirement?

> A. Compliance with it cannot be measured, so it is useless for governance
>
> B. It is too strict for most systems and will generate many exceptions
>
> C. It conflicts with the safeguards defined in CIS Control 4
>
> D. It can be assessed only with SCAP-compatible scanning tools

**12.** Each baseline requirement should carry four attributes. Which attribute later drives remediation timelines and exception approval levels?

> A. The rationale or risk it addresses
>
> B. The method of verification
>
> C. The severity of non-compliance
>
> D. The setting or outcome required

**13.** In the CyberRiskGuardian method, how are configuration KRIs such as unreconciled console changes used?

> A. They automatically recalculate estimated risk at the end of each month
>
> B. They are recorded as evidence to revisit Pb(ψ,A) and resilience θ at each review
>
> C. They replace scenario assessment with measured compliance percentages
>
> D. They set the organization's risk appetite and tolerance for the year

**14.** Under the chapter's split ownership model, who owns risk acceptance for baseline exceptions?

> A. Risk owners at the appropriate management level, within delegated authority
>
> B. Platform and infrastructure teams that implement the baseline
>
> C. The CISO or security architecture function that writes the content
>
> D. Internal audit, as the independent third line of assurance

**15.** A golden image was hardened two years ago and has never been rebuilt. How does the chapter characterize it?

> A. Still a golden image, provided it was formally approved
>
> B. A candidate for immediate conversion to infrastructure as code
>
> C. A template for deploying known vulnerabilities
>
> D. Acceptable if it was based on a CIS hardened image

**16.** What governance risk of policy as code does the chapter highlight?

> A. Policy rules cannot be placed under version control or review
>
> B. Enforcement depends on how attentive individual reviewers are
>
> C. Rules are ambiguous because they can partly pass or fail
>
> D. Policy code can become a hidden governance layer only engineers can read

**17.** During an outage, an engineer opens a firewall rule using a break-glass account to restore service. According to the chapter, what would be the governance failure?

> A. Making an emergency change during a production outage
>
> B. Using a controlled break-glass account to act quickly
>
> C. The absence of reconciliation and retrospective review afterwards
>
> D. Logging every action taken with the break-glass account

**18.** A manager asks for the number of servers from the CMDB and from the cloud provider's API, and the numbers differ. What does the chapter say this difference represents?

> A. An error in the cloud API, because the CMDB is always the authoritative record
>
> B. A rounding issue that should be resolved by reporting the higher of the two numbers
>
> C. Evidence that the posture management tool is faulty and should be replaced
>
> D. The organization's blind spot: assets that one source of truth does not know about

**19.** Which drift is most appropriate for automatic correction, according to the chapter?

> A. Disabled audit logging on a sensitive data store
>
> B. A non-compliant screen-lock timeout on a kiosk
>
> C. An approved emergency fix not yet recorded
>
> D. Every deviation, regardless of production impact

**20.** Ridgeview Credit Union's board saw '95% configuration compliance' before the incident. What was the fundamental flaw in that figure?

> A. CIS Level 1 benchmarks are too weak to support any compliance claim
>
> B. It covered only on-premises servers and workstations, so coverage was presented as compliance
>
> C. The figure was calculated incorrectly by the posture management tool
>
> D. The posture tool failed to detect the public storage container
