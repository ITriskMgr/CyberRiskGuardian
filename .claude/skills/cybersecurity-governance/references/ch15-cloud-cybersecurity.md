<!-- Léger, M.-A. (2026). Introduction to Cybersecurity Governance for Business Technology Management (3rd ed., v2.51). Chapter 15: Cloud Cybersecurity. CC BY-NC 4.0 -->

# Chapter 15: Cloud Cybersecurity

## Learning outcomes

After completing this chapter, you should be able to:

1.  Explain how virtualization, containers, orchestration, and serverless computing change the control boundary and concentrate risk in the management layer.

2.  Distinguish the cloud service and deployment models and analyze the governance consequences of each NIST SP 800-145 essential characteristic.

3.  Apply the shared responsibility model as a governance instrument, distinguishing responsibility from accountability and mapping complementary user entity controls to internal owners.

4.  Design a cloud governance operating model with a cloud centre of excellence, documented decision rights, landing zones, account structure, guardrails, the three lines, and FinOps alignment.

5.  Evaluate controls for identity as the cloud control plane, covering human privileged access, workload identities and secrets, and federation of SaaS identity.

6.  Recommend governance responses to misconfiguration, SaaS sprawl, data residency and sovereignty, logging and incident response, concentration risk, and exit planning.

7.  Compare the standards and assurance references for cloud security (ISO/IEC 27001, 27002, 27017, 27018, NIST CSF 2.0, CSA CCM and STAR, SOC 2, CIS Foundations Benchmarks, and Canadian references) and justify a control backbone.

8.  Apply the seven-step method to build an automated, measured cloud security controls baseline for an organization.

Cloud computing has become the default operating model for most organizations. The reason is not that it is inherently more secure. It is that cloud services offer a degree of speed, scalability, resilience, and cost flexibility that on-premises environments struggle to match. A start-up can launch a global service without buying a single server, a university can provision a research cluster in an afternoon, and a manufacturer can replace a decade-old enterprise system with a subscription. From a governance perspective, this shift is decisive. When critical business processes, sensitive data, and operational dependencies move onto platforms the organization does not own, cybersecurity becomes inseparable from vendor management, service design, contractual accountability, and continuous control assurance.

The governance problem is that cloud adoption redistributes control without redistributing accountability. The provider operates the data centres, the hypervisors, and much of the software stack. The customer still answers to its board, its regulators, its customers, and the courts for the protection of its data and the continuity of its services. Between those two facts lies a zone of shared responsibility that is often poorly understood, rarely documented with enough precision, and frequently the place where incidents begin. Most cloud security failures are not failures of the provider's infrastructure. They are failures of the customer to configure, monitor, and govern what it was always responsible for.

Cloud also changes the tempo of risk. In a traditional data centre, building a new environment took weeks, and every change passed through people who could be asked questions. In the cloud, an engineer with the right permissions can create, expose, or delete an entire environment in minutes, often through code rather than through a console. That speed is the business value of the cloud, and it is also why governance must move from periodic review to built-in guardrails, from documents to automated policy, and from annual audits to continuous evidence.

This chapter builds on several earlier ones. Chapter 2 framed governance as direction, decision rights, and accountability; Chapters 5 and 6 developed scenario-based risk assessment; Chapter 9 introduced the control frameworks that make decisions auditable; Chapter 10 treated identity and access management; and Chapter 14 developed the logic of security baselines and configuration governance. Cloud is where all of these converge, because it is an environment in which identity is the perimeter, configuration is the control, and evidence is generated (or lost) by the platform itself.

The chapter proceeds in four movements. It first explains the technical foundations managers need: virtualization, containers, and the service and deployment models that define the control boundary. It then examines what actually changes in the cloud risk profile and treats the shared responsibility model as a governance instrument rather than a marketing diagram. The central sections describe how to govern cloud in practice: the operating model, identity as the control plane, misconfiguration and posture management, SaaS sprawl, data residency, logging and detection, resilience and exit, and the interplay between cost management and security. The final sections anchor the program to current standards, present a repeatable method for building a cloud controls baseline, work through an illustrative scenario, and close with the questions managers and boards should ask.

## Virtualization and containers: the foundation and the concentration of risk

Virtualization is the technical idea that made cloud computing possible. Managers do not need to configure hypervisors or container clusters, but they do need a working understanding of what changes when computing becomes logical, portable, and dynamically allocated. Those changes reshape the organization's threat surface, its control strategies, and its assurance responsibilities.

### What virtualization changes

**Virtualization** creates a logical computing environment decoupled from a specific physical machine. Instead of tying an operating system and its applications to a single server, virtualization allows multiple isolated environments, called virtual machines (VMs), to run on a shared physical host. A layer of software called the **hypervisor** allocates processor, memory, storage, and network resources to each VM and enforces the isolation between them. The same principle extends to networks (software-defined networking) and storage (software-defined storage), so that an entire data centre can be described, created, and changed through software.

The advantages are strategically significant. Portability improves resilience: a VM image can be moved from one host to another quickly, supporting continuity when hardware fails or capacity must be rebalanced. Virtualization supports rapid development and operational stability through versioning and rollback, so that organizations can deploy quickly and revert quickly when a change introduces instability. Snapshots and sandboxes provide safe, isolated environments for testing patches, prototypes, and security controls without exposing production systems. Elastic scaling allows capacity to expand or contract far faster than buying and installing physical servers.

### Containers, orchestration, and serverless

Many organizations now go one step further with **containerization**. A container packages an application together with its libraries and runtime dependencies into a lightweight unit that shares the host operating system's kernel rather than carrying a full operating system of its own. Containers start in seconds and run consistently across a developer's laptop, a test environment, and production. At scale, they are managed by an **orchestration** platform, most commonly Kubernetes, which schedules containers across clusters of machines, restarts failed ones, and manages networking among them. NIST SP 800-190, the *Application Container Security Guide*, remains a useful reference for the specific risks of images, registries, orchestrators, and container runtimes.

Beyond containers lies **serverless** computing, in which the organization supplies only code (a "function") and the provider runs it on demand, scaling automatically and billing by execution. Serverless removes the need to manage servers altogether, but it does not remove the need to manage identity, permissions, data flows, dependencies, and logging. In each step from VMs to containers to functions, the customer hands more of the stack to the provider and keeps less to configure, yet the parts that remain—who can invoke what, with what permissions, touching which data—become proportionally more important.

### Concentrated risk: the management layer as crown jewel

Virtualization also concentrates risk. The hypervisor, the container runtime, and the orchestration layer become high-value targets because a compromise there can break isolation across many workloads at once. In practice, however, the main virtualization-related risks in organizations are governance failures rather than exotic hypervisor exploits. They include excessive administrative privileges in management consoles, weak authentication for virtualization and cloud administrators, incomplete separation between development and production, unmanaged VM and container sprawl, container images built from unvetted public sources, and inconsistent configuration discipline.

**Virtualization amplifies the consequences of unclear decision rights and weak operational control.** The same management console that lets an administrator rebuild a hundred servers in an hour lets an attacker who has stolen that administrator's credentials destroy them in the same hour. This is why modern security governance relies on **DevSecOps** logic: security requirements and verification are built into the development and deployment pipeline rather than inspected afterward. The aim is to ensure that speed does not come at the cost of unmanaged risk, and that repeatability—the ability to build the same secure environment every time—becomes a security asset rather than a way to replicate mistakes.

## Cloud computing: service models and deployment models

Cloud computing is the industrialization of computing capacity. Instead of purchasing, operating, and maintaining all infrastructure internally, organizations consume computing as a service, scale usage up and down, and pay according to consumption. For managers, the choice of service model and deployment model is not merely a technical or financial decision. It defines where the control boundary lies, and therefore which controls the organization must design, operate, and evidence itself.

### The NIST definition and why it still matters

The most widely cited definition comes from NIST SP 800-145, *The NIST Definition of Cloud Computing* (2011), which describes cloud computing as "a model for enabling ubiquitous, convenient, on-demand network access to a shared pool of configurable computing resources (e.g., networks, servers, storage, applications, and services) that can be rapidly provisioned and released with minimal management effort or service provider interaction." The definition identifies five essential characteristics: on-demand self-service, broad network access, resource pooling, rapid elasticity, and measured service. Its companion, NIST SP 800-146, *Cloud Computing Synopsis and Recommendations* (2012), discusses the benefits, open issues, and risks of each model in terms that remain recognizable more than a decade later.

The definition is old, but it is still useful because each characteristic has a governance consequence. On-demand self-service means that anyone with permissions can create resources without a procurement step, which is why cloud sprawl and shadow IT appear. Broad network access means that management interfaces are reachable from the Internet unless deliberately restricted. Resource pooling means that the customer shares physical infrastructure with others and must rely on the provider's isolation. Rapid elasticity means that costs and exposure can grow quickly. Measured service means that the platform produces detailed usage records, which can serve as evidence—if the organization collects and keeps them.

### Service models: IaaS, PaaS, SaaS, and the blurring middle

The three classic service models remain the starting point. In **Infrastructure as a Service (IaaS)**, the provider delivers foundational computing resources—virtual machines, storage, networks—while the customer remains responsible for operating systems, middleware, applications, and most security configuration. In **Platform as a Service (PaaS)**, the provider delivers a managed platform such as a database service, an application runtime, or a container platform, and the customer deploys and configures applications on it. In **Software as a Service (SaaS)**, the provider delivers a complete application, and the customer configures it, manages who uses it, and governs the data placed in it.

In practice the boundaries blur. Managed Kubernetes services sit between IaaS and PaaS. Serverless functions are a form of PaaS. Many SaaS products expose extensive configuration, integration, and development capabilities that make them behave like platforms. A large enterprise may consume several hundred distinct cloud services across all three models from dozens of providers. The governance implication is that the organization cannot have a single cloud policy written for one model; it needs a framework that classifies each service by model and by criticality and applies the corresponding responsibilities and controls.

### Deployment models: public, private, hybrid, multicloud, and sovereign

Deployment models describe who shares the infrastructure. **Public cloud** services are offered to the general public on provider-owned infrastructure. **Private cloud** refers to infrastructure operated for a single organization, whether on its premises or hosted by a third party. **Community cloud** serves a defined group with shared concerns, such as government departments or a consortium of hospitals. **Hybrid cloud** combines two or more of these, often because certain data or workloads face regulatory, privacy, latency, or business constraints that make full public deployment undesirable.

Two further patterns have become common. **Multicloud** refers to the deliberate (or accidental) use of more than one public provider, either to use each provider's strengths or to reduce dependency on any one of them. **Sovereign cloud** offerings promise that data, operations, and sometimes personnel remain within a specific jurisdiction and under its laws; these are discussed later in the chapter. Many architectures also include **edge computing**, placing computing or caching capacity close to users, factories, or devices to improve performance and reliability. Each pattern widens the set of environments that must be governed consistently. Multicloud, in particular, multiplies the number of identity systems, logging formats, configuration languages, and skill sets the organization must master, so it should be a deliberate choice with an owner and a rationale, not an accumulation of departmental preferences.

## What actually changes in the cloud: the risk profile

Cloud adoption changes the risk landscape in predictable ways. Viewed through the risk triangle of Chapters 1 and 4 (threat, vulnerability, and impact), the cloud does not invent entirely new categories of risk. It shifts where vulnerabilities concentrate, how quickly threats can exploit them, and how far the impact can spread. Understanding these shifts is what allows managers to write realistic risk scenarios and to set priorities that reflect how cloud incidents actually happen.

### The management plane as the dominant attack surface

Every cloud platform has two layers: the **data plane**, where workloads run and data is processed, and the **management plane** (or control plane), the set of consoles and application programming interfaces (APIs) through which resources are created, configured, and deleted. In a traditional data centre, the equivalent of the management plane was a locked room and a separate administrative network. In the cloud, it is a website and an API reachable from anywhere, protected chiefly by identity.

This is why a large share of real cloud incidents begin with stolen credentials, overly permissive roles, exposed access keys, misconfigured storage, or weak API security, rather than with any defeat of the provider's encryption or isolation. An attacker who obtains a highly privileged cloud identity does not need to break anything; the platform will faithfully execute whatever that identity is permitted to do. **In the cloud, the management plane is the crown jewel, and identity is the lock on it.**

### Speed, scale, and the economics of mistakes

Cloud changes the economics of error. A misconfiguration that once affected one server can now be replicated across hundreds of environments by a single template. A storage bucket accidentally made public can be found by automated scanners within hours. A deleted backup policy can leave an entire business unit without recoverable data. Conversely, the same automation allows a correct configuration to be applied everywhere at once, and a misconfiguration to be detected and corrected across the estate in minutes.

The managerial lesson is that cloud rewards organizations that invest in standardization and automation and punishes those that rely on individual diligence. A team that builds environments by hand, each slightly different, accumulates configuration debt that becomes very difficult to audit. A team that builds from approved templates and enforces policy automatically can demonstrate its posture continuously. This is the central reason the rest of this chapter emphasizes guardrails, baselines, and posture management.

### Dependency, jurisdiction, and evidence

Three further shifts complete the picture. First, cloud increases dependency on connectivity and on provider availability. If Internet connectivity fails, or if a provider suffers a regional outage, access to services may fail; resilience planning must account for this explicitly. Second, cloud raises confidentiality and privacy questions when sensitive data is processed outside the organization's traditional perimeter and possibly across borders. Where data is stored, which jurisdictions' laws apply, and which contractual mechanisms govern access and disclosure become security questions, not legal footnotes.

Third, logging and monitoring become both more powerful and more fragile. Cloud platforms can generate rich audit data about every administrative action, but many logs are not enabled by default, are retained for short periods, or are stored where an attacker with sufficient privilege could delete them. An organization that has not deliberately configured its logging may discover, in the middle of an incident, that it lacks the evidence needed to determine what happened, to notify regulators accurately, or to support legal proceedings. **Cloud security is not primarily about buying more tools; it is about governing identity, configuration, data handling, and evidence in a distributed environment.**

## The shared responsibility model as a governance instrument

The **shared responsibility model** is the managerial key to cloud security. It states that the provider and the customer each have defined security responsibilities and that the division between them depends on the service model. Every major provider publishes a version of the model. Too often, organizations treat it as a reassuring diagram. Used properly, it is a governance instrument: a way of assigning responsibilities, identifying gaps, writing contracts, and directing assurance work.

### How responsibility shifts across service models

The general pattern is consistent. The provider is always responsible for the physical security of data centres, the hardware, and the virtualization layer—the "security of the cloud." The customer is always responsible for its data, its identities and access decisions, and the way it configures the services it uses—the "security in the cloud." Between those fixed points, responsibility moves with the service model.

In IaaS, the customer's responsibilities are broad: it must govern operating systems, patching, vulnerability management, network segmentation, firewall rules, encryption and keys, workload identities, logging, and backup. In PaaS, the provider manages the operating system and platform software, and many security mechanisms are built in; the customer governs application security, the configuration of platform services, identities and permissions, and data protection. In SaaS, the provider controls most of the technology stack, and the customer's governance focus narrows to identity, access, data classification and handling, the security settings the service exposes, integrations with other systems, and monitoring of usage. The figure below contrasts the three models.

<span id="_Toc243635565" class="anchor"></span>Figure 66: Shared responsibility across IaaS, PaaS, and SaaS

A common error is to assume that the narrowing of customer responsibility in SaaS means the narrowing of customer risk. It does not. The data in a SaaS application is often the organization's most sensitive information, the identities that access it are often the organization's entire workforce, and the configuration settings that determine whether a file is shared publicly or a guest account can download everything belong to the customer.

### Responsibility is not accountability

The shared responsibility model divides operational responsibilities. It does not divide accountability. Regulators, courts, and customers hold the organization accountable for the protection of personal information and the continuity of critical services, whatever its contract says about who operates which layer. Québec's Law 25, for example, keeps the enterprise accountable for personal information it entrusts to a service provider, and OSFI Guideline B-10, *Third-Party Risk Management* (effective May 1, 2024), states plainly that a federally regulated financial institution remains accountable for activities it outsources.

This distinction has practical consequences. It means that the customer must obtain evidence that the provider is fulfilling its share, through independent assurance reports, certifications, and contractual rights, rather than assuming it. It also means that the customer must be able to show that it fulfils its own share, through configuration evidence, logs, and tested processes. **The provider's assurance report proves only half of the shared responsibility model; the other half must be proven by the customer.**

### Making shared responsibility contractual and testable

Managers can turn the model into a working instrument in three steps. The first is to produce a **responsibility matrix** for each critical cloud service, listing control areas (identity, encryption, logging, backup, vulnerability management, incident notification, and so on) and recording, for each, whether the provider, the customer, or both are responsible, and what evidence demonstrates performance. The CSA Cloud Controls Matrix, discussed later, is structured to support exactly this kind of analysis, and ISO/IEC 27017 distinguishes guidance for cloud service customers from guidance for cloud service providers.

The second step is to read provider assurance reports properly. A SOC 2 Type II report, prepared under the AICPA Trust Services Criteria, describes the provider's controls and an auditor's tests of their operating effectiveness over a period. It almost always lists **complementary user entity controls**—the controls the customer must operate for the provider's controls to achieve their objectives. These are, in effect, the provider's written statement of the customer's half of the model, and they should be mapped explicitly to internal control owners. The third step is to align contracts and internal policies with the matrix, so that notification duties, audit rights, data location, and exit assistance are defined before they are needed. Chapter 19 develops the contractual dimension of third-party governance in depth.

## The cloud governance operating model

Cloud security cannot be governed through policy documents alone. The rate of change is too high and the number of people who can make consequential changes is too large. Effective organizations build an operating model in which decision rights are clear, secure environments are provided as a default, and guardrails prevent the most dangerous configurations automatically. The figure below presents this operating model as a set of layers, from strategic direction at the top to evidence at the bottom.

<span id="_Toc243635566" class="anchor"></span>Figure 67: The cloud governance operating model

### The cloud centre of excellence and decision rights

Many organizations establish a **cloud centre of excellence (CCoE)**, a small cross-functional team that sets cloud standards, builds shared platforms, and helps business and engineering teams adopt cloud consistently. A well-designed CCoE includes cloud architecture, security, network, identity, finance, and operations expertise, and it works closely with procurement, legal, privacy, and risk management. Its purpose is not to approve every change; that would recreate the bottlenecks the cloud was meant to remove. Its purpose is to make the secure way the easy way.

The governance questions are about decision rights. Who decides which cloud providers and services are approved? Who decides which data classifications may be placed in which services and regions? Who may create a new account or subscription? Who may grant administrative privileges, approve exceptions to guardrails, and accept residual risk? Who owns each production workload as a business risk owner? These should be documented, ideally in a RACI-style matrix, and connected to the organization's existing governance bodies (see Chapter 2) rather than creating a parallel structure. The CCoE recommends and builds; risk owners and executives decide and accept.

### Landing zones and account structure

Chapter 14 introduced the **landing zone** as the main vehicle for enforcing cloud baselines: a pre-configured, secure, multi-account (or multi-subscription) cloud environment into which workloads are deployed. In the operating model, it provides baseline identity integration, network design, logging, security tooling, and policy enforcement from the first day, so that each new project starts inside a governed environment rather than building its own. All major providers offer reference landing-zone designs, and the Government of Canada's cloud guardrails, discussed later, are essentially a minimum specification for what a landing zone must include within a department's first 30 business days of cloud access.

The **account or subscription structure** is one of the most consequential and least visible governance decisions in the cloud. Cloud accounts (AWS), subscriptions (Azure), and projects (Google Cloud) are the primary units of isolation, billing, and policy. A sound structure separates production from non-production, isolates highly sensitive workloads, places logging and security tooling in dedicated accounts that application teams cannot alter, and groups accounts into a hierarchy (organizational units, management groups, or folders) to which policies can be applied centrally. A poor structure—a single account shared by many teams, or hundreds of accounts created without hierarchy—makes it almost impossible to enforce least privilege, contain an incident, or attribute costs.

### Guardrails: preventive, detective, and the exception path

**Guardrails** are policies enforced by the platform that keep teams within acceptable boundaries without requiring manual review of each action. *Preventive* guardrails block prohibited actions outright: creating resources in unapproved regions, disabling audit logging, making storage publicly accessible, or launching resources without encryption. *Detective* guardrails continuously evaluate configurations and raise findings when something drifts from the baseline. *Corrective* guardrails go further and automatically remediate certain findings, for example by reverting a public storage setting.

The managerial challenge is to calibrate guardrails to risk appetite. Too few, and the organization depends on every engineer's judgment. Too many, and teams route around them, creating the shadow environments the guardrails were meant to prevent. A credible guardrail program therefore includes a defined **exception path**: a documented request, a risk assessment proportionate to the exception, an approving risk owner, compensating controls where needed, and an expiry date after which the exception must be renewed or removed. Chapter 14 develops this logic for baselines generally; in the cloud it is enforced by policy engines rather than by memo.

### The three lines in the cloud

The three lines model applies to cloud as it does elsewhere. In the first line, product and platform teams build and operate cloud services and own their risks; the CCoE and cloud platform team provide the paved road. In the second line, the cybersecurity and risk functions set standards, define guardrails, monitor posture, and challenge exceptions. In the third line, internal audit provides independent assurance that the cloud governance system works, including tests of guardrail effectiveness and of the completeness of the cloud inventory. A frequent weakness is that the second line lacks cloud skills and read access to cloud environments, leaving it unable to challenge effectively; boards should ask whether oversight functions can actually see what the first line has built.

### FinOps and security: one inventory, two disciplines

**FinOps**, the practice of managing cloud spending through shared accountability among finance, engineering, and business teams, has become a standard part of the cloud operating model. Its interplay with security is closer than it first appears. Both disciplines depend on a complete and accurate inventory of cloud resources, on consistent **tagging** that records owner, application, environment, and data classification, and on clear accountability for each account. An untagged resource is both an unallocated cost and an unowned risk.

The two disciplines also reinforce each other operationally. Unexpected cost spikes are often the first visible sign of compromise, for example when attackers use stolen credentials to run cryptocurrency mining or exfiltrate large volumes of data. Idle and orphaned resources that FinOps teams flag for deletion are frequently unpatched and unmonitored. At the same time, tensions must be managed explicitly. Logging, backup copies in a second region, extended log retention, and security tooling all cost money, and a cost-reduction drive that quietly shortens log retention or removes cross-region backups changes the organization's risk position. **Cost optimization decisions that reduce security controls are risk acceptance decisions and should be made by the risk owner, not hidden in a billing report.** Chapter 17 discusses the broader economics of security investment.

## Identity as the cloud control plane

Chapter 10 treats identity and access management in full. This section highlights what is distinctive about identity in the cloud, where it functions not only as an access control but as the control plane itself. Every API call to create, modify, or delete a cloud resource is authorized by an identity and a set of permissions. **Governing cloud identity is therefore governing the cloud.**

### Human privileged access

The first priority is human administrative access. Cloud administrators, platform engineers, and identity administrators hold permissions that can affect the entire estate, and their accounts are prime targets for phishing, session hijacking, and social engineering of help desks. Strong practice includes phishing-resistant multi-factor authentication for all privileged roles (consistent with NIST SP 800-63-4), federation of cloud access through the organization's central identity provider rather than local cloud accounts, just-in-time elevation so that standing administrative privilege is rare, separation of duties between those who manage identities and those who manage workloads, and dedicated, protected "break-glass" accounts for emergencies, with alerting on every use.

The root or global administrator accounts that every cloud tenancy creates deserve special attention. They should be secured with the strongest available authentication, used almost never, and monitored continuously. Managers should expect to see a short, named list of individuals with the highest privileges in each cloud environment, and to see that list reviewed on a defined schedule.

### Workload identities, keys, and secrets

The second priority is the identities used by software rather than people: service accounts, roles assumed by applications, API keys, access tokens, and the credentials embedded in automation pipelines. In most cloud environments these non-human identities far outnumber human ones, and they are frequently over-privileged because it is easier to grant broad permissions than to determine the minimum required. Long-lived access keys left in source code repositories, configuration files, or container images remain one of the most common starting points for cloud compromise.

Governance expectations should include an inventory of workload identities with named owners, a preference for short-lived credentials issued automatically by the platform over static keys, secrets stored in a managed vault rather than in code, automated scanning of code repositories for exposed secrets, and periodic analysis to remove permissions that are granted but never used. Cloud providers offer tools that identify unused permissions and externally shared resources; whether they are actually used, and whether findings are acted upon, is a governance question.

### Federation and SaaS identity

The third priority is the identity fabric that connects the organization to its SaaS estate. Single sign-on through a central identity provider allows the organization to enforce consistent authentication policies, conditional access, and prompt removal of access when people leave. Each SaaS application that is not connected to single sign-on is a separate identity island with its own passwords, its own administrators, and its own joiner–mover–leaver gaps. Because the central identity provider becomes a single point of control for hundreds of services, it also becomes a single point of failure and a primary target; its own administrative roles, recovery processes, and logs require the highest level of protection.

## Misconfiguration and cloud posture management

If identity is the most common way into a cloud environment, misconfiguration is the most common reason the door was left open. Publicly exposed storage, databases reachable from the Internet, disabled logging, overly permissive network rules, unencrypted volumes, and default settings that were never changed are recurring causes of cloud data exposure. These are rarely the result of ignorance; they are the result of speed, complexity, and the absence of automated checks.

### Why misconfiguration dominates

Several features of the cloud make misconfiguration likely. The number of configurable settings across hundreds of services is enormous, and providers add new services and options continuously. Defaults vary between services and change over time. Responsibility for configuration is distributed across many teams, each focused on delivering its own product. And the effect of a configuration is not always obvious: a permission that seems narrow may, combined with another, allow access to far more than intended. Chapter 14 describes configuration drift as the gap between intended and actual state; in the cloud, drift can occur many times a day.

### CSPM, CNAPP, and what managers should demand

A family of tools has emerged to address this problem. **Cloud Security Posture Management (CSPM)** tools, whose governance questions Chapter 14 discussed, continuously assess cloud configurations against benchmarks and policies, such as the CIS Benchmarks discussed later, and report misconfigurations, compliance gaps, and risky combinations. **Cloud Workload Protection Platforms (CWPP)** protect the workloads themselves—VMs, containers, and functions—through vulnerability scanning and runtime protection. **Cloud Infrastructure Entitlement Management (CIEM)** focuses on identities and permissions. **Cloud-Native Application Protection Platforms (CNAPP)** combine these capabilities, often adding scanning of infrastructure code and container images before deployment, to give a unified view of risk from development to runtime.

The managerial point is not the product category. It is that posture must be assessed continuously, findings must be prioritized by actual risk rather than raw count, and remediation must be owned and measured. A CSPM dashboard showing thousands of findings is not a security control; it becomes one when findings are routed to accountable owners with service-level targets, when critical findings (such as publicly exposed sensitive data or disabled logging) trigger immediate response, and when trends are reported to management. Useful KRIs include the number of critical misconfigurations open beyond target, the percentage of accounts covered by posture monitoring, the median time to remediate high-severity findings, and the number of active guardrail exceptions past their expiry date.

### Infrastructure as code, policy as code, and DevSecOps

Because cloud infrastructure is programmable, the most effective way to prevent misconfiguration is to stop configuring by hand. Chapter 14 described how **infrastructure as code (IaC)** and **policy as code** turn baselines into reviewed templates and automated pipeline gates; in the cloud, these are not refinements but the primary means of control, because a template that would create an unencrypted database or a publicly reachable management port can be rejected before it ever exists.

This is the managerial bridge between DevOps and DevSecOps. Governance sets the standards; engineering pipelines enforce them by default; posture management detects what slips through; and exceptions are handled through a documented, time-limited path. The approach also produces evidence as a by-product: the version history of templates, the record of policy checks passed and failed, and the log of approved exceptions together form an audit trail far richer than a periodic configuration review.

| **In the cloud, configuration is the control.** An organization that cannot describe its intended configuration as code, test it automatically, and detect drift continuously is relying on hope rather than governance. |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## SaaS security posture and SaaS sprawl

Much of the discussion of cloud security focuses on IaaS and PaaS, where engineering teams build systems. Yet for many organizations, the larger and less governed exposure lies in SaaS. Collaboration suites, customer relationship management platforms, human resources systems, file-sharing services, e-signature tools, and hundreds of smaller applications hold sensitive data and are configured, often by business users, with little security oversight.

### Shadow SaaS and the inventory problem

**SaaS sprawl** describes the uncontrolled growth in the number of SaaS applications in use, many adopted by individual departments or employees with a corporate credit card or a free trial. Each one may hold customer data, intellectual property, or personal information, and each one creates an identity, a contract (often click-through), a data location, and an integration surface. Because SaaS adoption does not require infrastructure, it frequently bypasses procurement, security review, and privacy assessment entirely.

The first governance response is visibility. Organizations can discover SaaS use through single sign-on logs, expense and procurement records, network and browser telemetry, and the OAuth grants recorded in their identity provider. The second response is a proportionate intake process: a short, fast review for low-risk tools and a fuller assessment for applications that will hold sensitive data or integrate with core systems. If the approved path is slow and burdensome, people will use the unapproved one.

### SaaS configuration and integration risk

Even approved SaaS applications carry configuration risk. External sharing settings, guest access, administrative roles, data retention, audit logging, and session policies are customer responsibilities, and their defaults are not always conservative. **SaaS Security Posture Management (SSPM)** tools apply the same logic as CSPM to SaaS, comparing tenant configurations with recommended baselines. Public references such as the CIS Microsoft 365 Foundations Benchmark and CISA's SCuBA baselines (see Chapter 14) are useful for any organization using those platforms.

A distinct and growing risk lies in **SaaS-to-SaaS integrations**. When a user connects one application to another, typically by granting an OAuth token, the third-party application may receive broad and persistent access to email, files, or customer records. These tokens often survive password changes, are rarely reviewed, and can be abused if the connected application is compromised. Governance should require periodic review of third-party application grants, restrictions on which users can authorize integrations, and prompt revocation of unused or high-risk grants. Chapter 19 treats the broader issue of fourth-party and supply-chain exposure.

### Assigning SaaS ownership

The underlying problem with SaaS is often ownership. Infrastructure has an IT owner by default; a SaaS application adopted by the marketing department may have no one accountable for its security configuration, its administrator accounts, or its data. A sound model assigns, for each significant SaaS application, a business owner accountable for its use and data, a technical administrator responsible for its configuration, and a security contact responsible for posture review. It also records the data classification, the contract and assurance evidence, the integration points, and the exit approach. These records are the SaaS equivalent of the landing zone: the minimum structure within which an application may operate.

## Data protection, residency, and sovereignty in the cloud

Chapter 16 addresses data protection and privacy in depth, including cross-border transfers and privacy impact assessments, and Chapter 20 addresses cryptography and key management. This section focuses on what is specific to cloud decisions.

### Encryption and key control as governance choices

Cloud providers encrypt most data at rest and in transit by default, which is valuable but not decisive. The governance question is who controls the keys and under what conditions the provider can access plaintext. Options range from provider-managed keys, through customer-managed keys held in the provider's key management service, to keys held in a customer-controlled hardware security module or external key manager; Chapter 20 examines this spectrum in detail. Each step increases customer control and also customer responsibility, since a lost or revoked key can make data permanently unrecoverable and an external key service becomes a dependency for availability. The choice should therefore follow data classification and risk appetite rather than a blanket rule: customer-managed keys with strong separation of duties for the most sensitive data, and provider-managed encryption where it is entirely reasonable for routine data.

### Data residency, cross-border access, and Law 25

**Data residency** refers to the physical or geographic location where data is stored and processed. **Data sovereignty** is the broader principle that data is subject to the laws of the jurisdiction where it is located and, in some cases, of the jurisdiction where the provider is headquartered. These are not identical: data stored in a Canadian region of a provider headquartered elsewhere may still be subject to lawful-access demands under the provider's home-country laws, depending on the legal arrangements involved.

For Québec organizations, Law 25 requires a privacy impact assessment before personal information is communicated outside Québec, including to a cloud provider, and requires that the information receive adequate protection in light of generally recognized principles. Other regimes, including the EU GDPR's transfer rules and sector-specific requirements, impose their own conditions. The cloud governance implication is that region selection, backup locations, support access from other countries, and the provider's subprocessors must all be known and documented. A preventive guardrail restricting resource creation to approved regions is one of the simplest and most effective controls an organization can implement, and the Government of Canada guardrails include data location as an explicit requirement.

### Sovereign cloud as a risk treatment

Providers increasingly offer **sovereign cloud** arrangements: regions or partner-operated environments in which data, operations, support personnel, and sometimes encryption keys remain within a jurisdiction. These can be legitimate risk treatments for data whose exposure to foreign legal processes would exceed risk appetite. They also tend to cost more, to lag the main platform in available features, and to introduce their own dependencies. Managers should treat sovereignty claims as assertions to be verified—which controls, contractual terms, and independent assurances support the claim—rather than as labels. The decision should be expressed in the language of risk: which scenarios the arrangement mitigates, what residual risk remains, and who accepts it.

## Logging, detection, and response in the cloud

Chapter 8 covers detection, orchestration, and response, and Chapter 12 covers incident and crisis management. In the cloud, these capabilities depend on a set of deliberate configuration and governance decisions that are easy to overlook until an incident reveals their absence.

### Which logs matter

Cloud logging operates at several levels. **Management-plane or audit logs** record administrative actions: who created, modified, or deleted which resource, from where, and when. **Identity logs** record sign-ins, token issuance, privilege elevation, and changes to roles. **Data-access logs** record reads and writes to storage and databases, though many are disabled by default because of their volume and cost. **Network flow logs** record traffic patterns. **Workload and application logs** record what happens inside VMs, containers, and functions. SaaS applications have their own audit logs, whose availability and retention often depend on the licence tier purchased.

The governance decision is which of these must be enabled, for which environments, and for how long. That decision should be driven by risk scenarios: if the organization's top scenarios involve stolen administrator credentials and data exfiltration from storage, then audit logs, identity logs, and data-access logs for sensitive stores are non-negotiable. The decision should also consider legal and regulatory needs, such as the ability to determine what personal information was accessed for breach notification under Law 25, PIPEDA, or the GDPR.

### Protecting and centralizing evidence

Logs are only useful if they survive the incident. A capable attacker with administrative access will often attempt to disable logging or delete logs to conceal activity. Good practice is to send logs to a dedicated, centrally controlled logging account to which workload teams have no write or delete permissions, to use immutable or write-once storage for critical audit records, to alert immediately whenever logging is disabled or modified, and to retain logs long enough to cover realistic detection delays. Logs must also be correlated: cloud audit records, identity provider sign-ins, SaaS audit logs, and endpoint telemetry need to reach the security operations function in a form that allows an analyst to reconstruct a sequence of events across them.

### Cloud incident response

Cloud incidents unfold differently from on-premises incidents. Containment may mean revoking credentials and sessions, applying restrictive policies to an account, or isolating a workload by changing its network rules, rather than unplugging a server. Forensic evidence may be collected by snapshotting volumes and preserving logs rather than seizing disks. The provider may need to be engaged, and its role, contacts, and timelines should be understood in advance. NIST SP 800-61 Rev. 3, which aligns incident response with CSF 2.0, provides the overall frame; cloud-specific playbooks should address scenarios such as compromised administrator identity, exposed storage, leaked access keys, ransomware against cloud backups, and compromise of a SaaS tenant. **A cloud incident response plan that has never been exercised against a cloud scenario is an assumption, not a capability.**

## Resilience, concentration risk, and exit strategy

Cloud can improve resilience dramatically. Multiple availability zones, automated failover, and geographically separated backups are available to organizations that could never have built them alone. But cloud also creates new dependencies, and those dependencies concentrate across the economy. Chapter 12 addresses business continuity and resilience, and Chapter 19 addresses concentration risk and exit planning across all third parties; this section highlights the cloud-specific aspects.

### Designing for failure

Resilience in the cloud is an architectural decision made by the customer. Providers offer the building blocks—zones, regions, replication, backup services—but a workload deployed in a single zone with no tested recovery path is no more resilient than a server in a closet. Managers should insist that each critical service has defined recovery time and recovery point objectives, an architecture that meets them, and recovery tests that demonstrate it. Backup design must address cloud-specific scenarios: a compromised administrator who deletes both production data and backups in the same account, ransomware that encrypts cloud storage, accidental deletion propagated by replication, and a SaaS provider that offers only limited native restoration. Logically isolated backups in a separate account, with separate credentials and deletion protection, are the cloud counterpart of the offline backup.

### Concentration risk

**Concentration risk** arises when many critical services depend on the same provider, region, or underlying component. At the level of a single organization, it appears when core business systems, identity, collaboration, and backups all reside with one provider, so that one provider-level outage or account compromise affects everything. At the level of the economy, it appears when large numbers of banks, hospitals, and governments depend on a few hyperscale providers, so that a single failure can have systemic effects.

Regulators have responded. OSFI Guideline B-10 expects federally regulated financial institutions to take reasonable steps to manage concentration risks in their own third-party arrangements and to assess systemic concentration risk as far as possible. In the European Union, the Digital Operational Resilience Act (DORA), applicable since January 17, 2025, created an oversight regime for critical ICT third-party providers, and the European Supervisory Authorities published their first list of designated critical providers, including hyperscale cloud providers, in November 2025. For managers, the practical response is not necessarily multicloud for every workload, which can increase complexity and risk. It is to understand dependencies explicitly, to identify which business services would be affected by the failure of each provider or region, and to decide deliberately which services warrant diversification, which warrant multi-region design, and which can tolerate a provider outage with manual workarounds.

### Exit strategy: the control everyone forgets

ISO/IEC 27002:2022 control 5.23, *Information security for use of cloud services*, requires organizations to establish processes for the acquisition, use, management, and **exit** from cloud services. Exit is the part most organizations neglect. An **exit strategy** documents how the organization would move a service to another provider or back in-house, whether because of provider failure, a commercial dispute, a change in law, a security concern, or a strategic decision. It identifies the data and configuration to be extracted, the formats and tools required, the time and cost involved, the dependencies on proprietary services, and the contractual assistance the provider must give.

The legal environment is evolving in the customer's favour. The EU Data Act's switching provisions, which apply from September 12, 2025, require providers of data processing services in the EU to remove obstacles to switching, to support customer exit, and to phase out switching charges entirely from January 12, 2027. Canadian organizations may not benefit directly, but the provisions are shaping contract terms globally. Even so, contractual rights do not make an exit feasible if the organization has built deeply on proprietary services without understanding the cost of leaving. **An exit strategy that has never been tested, even as a tabletop exercise, provides little real assurance.** For the most critical services, managers should expect a documented exit plan, an estimate of exit time and cost, and periodic verification that data can actually be extracted in a usable form.

| Resilience in the cloud is bought, not inherited. **The provider offers the building blocks; the customer decides whether to use them, and must prove that recovery and exit actually work.** |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## Anchoring cloud security to standards and assurance

To govern cloud security credibly, organizations should anchor their program to recognized standards and make those standards operational through policies, baselines, and assurance mechanisms. Chapter 9 explained how standards and frameworks combine; this section identifies the references most relevant to cloud and the question each answers.

### The ISO/IEC family

ISO/IEC 27001:2022 (with Amendment 1:2024) provides the management-system backbone. Its most cloud-relevant mechanisms are disciplined risk assessment and reassessment, a risk treatment plan, and the Statement of Applicability (SoA), which documents which controls apply, why, and how they are implemented. For cloud programs, the SoA compels executives and control owners to justify the security posture as an auditable set of decisions rather than an informal collection of good intentions. ISO/IEC 27002:2022 supplies the control guidance, including control 5.23 on cloud services, and ISO/IEC 27005:2022 supports the modelling of cross-border exposure, provider dependency, and impact amplification.

Two cloud-specific standards build on this foundation. ISO/IEC 27017, the code of practice for information security controls for cloud services, was first published in 2015; its second edition, ISO/IEC 27017:2026, published in July 2026, builds on ISO/IEC 27002 with cloud-specific implementation guidance and additional cloud controls and supersedes the 2015 edition. Organizations whose contracts, certifications, or policies cite ISO/IEC 27017:2015 should plan their transition. ISO/IEC 27018:2025, the third edition of the guidelines for protecting personally identifiable information in public clouds acting as PII processors, aligns with ISO/IEC 27002:2022 and is particularly relevant when assessing providers that process personal information on the organization's behalf. Certification claims against these documents are typically made as extensions of an ISO/IEC 27001 certification, so managers should check the scope statement to confirm which services and locations are covered.

### NIST and the CSA ecosystem

On the NIST side, CSF 2.0 provides an outcome-based umbrella for organizing cloud security, and its Govern function—including the supply-chain risk management category—speaks directly to provider governance. NIST SP 800-53 Rev. 5 provides detailed controls, including the external system services controls, and NIST SP 800-145 and SP 800-146 provide the conceptual definitions used throughout this chapter. For organizations that sell to the U.S. federal government, FedRAMP applies NIST controls to cloud authorization.

The Cloud Security Alliance (CSA) provides the most widely used cloud-specific control framework. The **Cloud Controls Matrix (CCM)** v4.1, released in January 2026, organizes cloud security controls into domains such as identity and access management, logging and monitoring, and supply-chain management, and it indicates how responsibility for each control is typically shared between provider and customer. The accompanying Consensus Assessments Initiative Questionnaire (CAIQ) v4.1 provides a standardized set of questions that providers answer. The CSA **STAR** (Security, Trust, Assurance and Risk) program publishes these answers in a public registry, at Level 1 as provider self-assessments and at Level 2 as third-party audited certifications or attestations. CSA has announced a transition in which v4.0-based submissions remain accepted alongside v4.1 until December 2027, after which only v4.1 will be accepted. For managers, the practical value is efficiency: a STAR entry or an audited CCM mapping can replace a bespoke security questionnaire for many providers, and the CCM's mappings help reconcile cloud controls with ISO, NIST, and other frameworks.

### Assurance reports and configuration benchmarks

**SOC 2** reports, issued under the AICPA's Trust Services Criteria (2017, with revised points of focus in 2022), are the most common form of independent provider assurance in North America. A Type I report evaluates control design at a point in time; a Type II report also tests operating effectiveness over a period, typically six to twelve months, and is the more useful for governance. Readers should check the scope (which services and locations are included), the period covered, any exceptions found by the auditor, any subservice organizations excluded through the "carve-out" method, and the complementary user entity controls described earlier. A provider's SOC 2 report says nothing about how well the customer has configured the service.

**CIS Benchmarks** translate good practice into specific, testable configuration recommendations. The CIS Foundations Benchmarks for Amazon Web Services, Microsoft Azure, and Google Cloud Platform cover identity, logging, monitoring, networking, and storage settings at the account level, and they are built into most CSPM tools. They are revised frequently as services evolve; at the time of writing, the AWS Foundations Benchmark had reached version 7.1.0, only months after version 7.0.0. The baseline should therefore cite the specific version adopted and include a process for evaluating new releases, rather than referring to "the CIS Benchmark" in general.

### Canadian public-sector and regulatory references

Canadian organizations have several additional references. The **Government of Canada Cloud Guardrails**, issued under the Treasury Board's *Directive on Service and Digital* and most recently updated on October 8, 2024, define thirteen minimum controls that departments must implement within the first 30 business days of obtaining a cloud account: protecting user accounts and identities, managing access, securing endpoints, enterprise monitoring accounts, data location, protection of data at rest, protection of data in transit, segmentation and separation, network security services, cyber defence services, logging and monitoring, configuration of cloud marketplaces, and continuity planning. Although mandatory only for federal departments, they are an excellent, concise checklist for any organization's landing zone.

The **CCCS Medium Cloud Control Profile** (Annex B of ITSP.50.103), published by the Canadian Centre for Cyber Security and derived from its ITSG-33 risk management guidance, specifies the controls expected of cloud services that handle information categorized as Protected B, with medium integrity and medium availability; it replaced the control list formerly appended to the Treasury Board's security control profile for cloud-based services, and CCCS assesses major providers against it. For federally regulated financial institutions, **OSFI Guideline B-10** sets expectations for third-party risk management, including cloud arrangements, while Guideline B-13 addresses technology and cyber risk more broadly. The table below summarizes how these references serve different governance questions.

| **Reference**                          | **Governance question it answers**                              | **Typical managerial use**                         |
|----------------------------------------|-----------------------------------------------------------------|----------------------------------------------------|
| ISO/IEC 27001, 27002, 27017, 27018     | Is cloud security managed as a system, with justified controls? | ISMS scope, SoA, provider certification review     |
| NIST CSF 2.0, SP 800-53 Rev. 5         | Which outcomes and controls are expected?                       | Program structure, detailed control selection      |
| CSA CCM v4.1, CAIQ, STAR               | Who does what, and what has the provider declared or proven?    | Responsibility matrix, provider assessment         |
| SOC 2 Type II                          | Did the provider's controls operate effectively?                | Assurance review, mapping of customer controls     |
| CIS Foundations Benchmarks             | How exactly should accounts be configured?                      | Technical baseline, CSPM rules                     |
| GC Guardrails, CCCS profile, OSFI B-10 | What do Canadian authorities expect?                            | Landing-zone minimums, regulated-sector compliance |

**Standards are not substitutes for security; they are instruments for making security decisions explicit, testable, and defensible.** The organization that cites five frameworks but cannot show the configuration of its production accounts has achieved compliance theatre, not assurance.

## Building a cloud security controls baseline: a practical, repeatable method

To convert principles into operational reality, organizations should formalize a **cloud security baseline**: the minimum set of controls that must be present for any cloud workload or service to be approved, operated, and maintained. Chapter 14 explains baseline governance in general terms; this section presents a cloud-specific method in seven steps, illustrated in the figure below.

<span id="_Toc243635567" class="anchor"></span>Figure 68: A seven-step method for building a cloud controls baseline

### The seven steps

1.  **Establish the current state.** Inventory the cloud providers, accounts, subscriptions, SaaS applications, and data stores in use; identify owners; and record how identities are managed, what logging exists, where misconfiguration and privilege drift occur, and which third-party integrations are embedded. Discovery tools help, but the inventory must be reconciled with procurement and finance records to catch what tools miss.

2.  **Map obligations and risk appetite.** Identify the legal, regulatory, and contractual requirements that apply (privacy law, sector regulation, customer commitments) and translate the organization's risk appetite into cloud terms, such as which data classifications may reside in which service models and regions.

3.  **Analyze priority risk scenarios.** Using the scenario-based approach of Chapters 5 and 6, develop and assess the cloud scenarios that matter most: compromised administrator identity, exposed sensitive storage, leaked workload credentials, provider regional outage, SaaS tenant takeover, ransomware against cloud backups, and loss of access to a critical provider.

4.  **Select the control backbone.** Combine a governance framework (ISO/IEC 27001 or NIST CSF 2.0), a control catalogue (ISO/IEC 27002 with ISO/IEC 27017, or NIST SP 800-53 Rev. 5), a responsibility and assurance reference (CSA CCM v4.1), and configuration benchmarks (the relevant CIS Foundations Benchmarks), adding sector-specific requirements such as the GC Guardrails or OSFI B-10 where applicable.

5.  **Define tiered baseline requirements.** Specify controls across the domains that reflect real cloud failure patterns—identity and privileged access, encryption and key management, network exposure and segmentation, secure configuration and change through IaC, centralized and protected logging, vulnerability management for images and dependencies, API security, backup and recovery, SaaS configuration, and exit planning—with stronger requirements for higher-criticality tiers.

6.  **Automate enforcement.** Implement the baseline in landing-zone templates, preventive guardrails, policy-as-code checks in deployment pipelines, and posture monitoring, so that the baseline is the default rather than a document. Define the exception path, with owners and expiry dates.

7.  **Assure and improve.** Measure baseline coverage and compliance through KRIs, report to management, test controls through internal audit and exercises, review the baseline when providers release new services or benchmarks change, and feed lessons from incidents back into the requirements.

The most common failure in baseline work is stopping at step 5: producing an excellent document that no pipeline enforces and no dashboard measures. **A cloud baseline that is not automated will drift, and a baseline that is not measured cannot be defended.**

### Illustrative scenario: a Canadian insurer moves its claims platform to the cloud

Laurentide Mutual is a fictional mid-sized Canadian property-and-casualty insurer, federally regulated and therefore subject to OSFI Guidelines B-10 and B-13. It has 1,400 employees, offices in Montréal, Québec City, and Toronto, and a strategic plan to replace its ageing on-premises claims system with a cloud-native platform on a public IaaS and PaaS provider. It already uses a SaaS collaboration suite and a SaaS customer relationship management platform, and an internal discovery exercise has found roughly 180 other SaaS applications in use, about a third of them unknown to IT.

The board's risk committee asks management to demonstrate, before migration, that the cloud environment will be governed within the insurer's risk appetite. Management responds in several moves. First, it establishes a CCoE reporting to the CIO, with a dotted line to the CISO and representation from privacy, risk management, procurement, and finance. Decision rights are documented: the CCoE recommends providers and regions, the CISO owns guardrails and the cloud baseline, the vice-president of claims is the risk owner for the claims platform, and the chief risk officer must approve any residual risk above appetite. Second, the CCoE builds a landing zone with separate accounts for production, non-production, shared services, security tooling, and centralized logging; preventive guardrails restrict resources to Canadian regions, block public storage, require encryption with customer-managed keys for claims data, and prevent anyone outside the security team from altering logging.

Third, the risk team runs a scenario workshop. The highest-rated scenario is not a sophisticated attack on the provider; it is the compromise of a cloud administrator through phishing of a help-desk password reset, followed by deletion of production data and backups. The treatment portfolio includes phishing-resistant authentication for all privileged roles, just-in-time elevation, help-desk identity verification procedures, and immutable backups in a separate account with independent credentials. A second high-rated scenario concerns the collaboration suite: guest sharing is enabled broadly, and dozens of OAuth integrations have access to mailboxes. The treatment includes an SSPM review against the CIS Microsoft 365 benchmark, removal of unused integrations, and a rule that new integrations require approval.

Fourth, for B-10 purposes, procurement and risk management assess the provider's SOC 2 Type II report, its CSA STAR Level 2 entry, and its ISO/IEC 27001 certificate scope; they map the complementary user entity controls to named internal owners. The contract is reviewed for audit rights, incident notification timelines, data location commitments, and exit assistance. A documented exit plan estimates that moving the claims platform to another provider would take about nine months because of reliance on two proprietary services; the board accepts this, provided the architecture team limits further proprietary dependencies and the exit plan is reviewed annually.

Finally, management agrees on a quarterly cloud governance report to the risk committee: percentage of accounts within the landing zone, critical misconfigurations open beyond target, privileged accounts without phishing-resistant authentication, guardrail exceptions past expiry, results of the latest recovery test, and the status of SaaS discovery and ownership assignment. Six months after go-live, the report shows that a FinOps review flagged a sudden spike in compute spending in a development account; investigation found a leaked access key used for cryptocurrency mining. The incident was contained in hours because logging was centralized and protected, and the post-incident review led to a new guardrail prohibiting long-lived access keys in non-production accounts. The board's conclusion is that the governance system did what it was designed to do: it did not prevent every error, but it limited impact, produced evidence, and improved itself.

### Questions managers and boards should ask

- **Inventory:** Do we have a complete, reconciled inventory of our cloud accounts, subscriptions, and SaaS applications, each with a named business owner and data classification?

- **Decision rights:** Who approves new cloud providers, new regions, guardrail exceptions, and residual cloud risk, and is that documented and followed?

- **Shared responsibility:** For each critical cloud service, do we have a responsibility matrix, and have we mapped the provider's complementary user entity controls to internal owners?

- **Privileged identity:** How many people hold the highest privileges in each cloud environment, are they all using phishing-resistant authentication, and how much of that privilege is standing rather than just-in-time?

- **Workload identities:** Do we know how many non-human identities and long-lived keys exist, who owns them, and when unused permissions were last removed?

- **Posture:** What percentage of our cloud estate is covered by continuous posture monitoring, how many critical findings are open beyond target, and what is the trend?

- **Guardrails:** Which dangerous configurations are prevented automatically, and how many exceptions are active or past their expiry date?

- **Evidence:** If an administrator account were compromised today, would we have protected logs sufficient to determine what was accessed and to meet notification obligations?

- **Data location:** Can we demonstrate where our sensitive data, backups, and support access are located, and have we completed the assessments required by Law 25 or other applicable laws?

- **Resilience:** When did we last restore a critical cloud service from isolated backups, and did we meet our recovery objectives?

- **Concentration and exit:** Which business services would be disrupted by the failure of our main provider, and do we have a tested exit plan with realistic time and cost estimates for the most critical ones?

- **Cost and risk:** When cost-reduction initiatives change logging, retention, backup, or security tooling, who approves the resulting change in risk?

| **A cloud governance report that the board can read in five minutes—coverage, critical findings, privileged access, exceptions, recovery tests—is worth more than a hundred-page policy no one can verify.** |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## Concluding integration: the managerial bottom line

Cloud computing and virtualization shift cybersecurity from a perimeter-centric model to a governance-centric model. The organization gains agility, scalability, and access to resilience capabilities it could not build alone, but it also inherits dependency, complexity, and shared control boundaries that must be governed deliberately. The provider secures the cloud; the customer secures what it puts in the cloud and how it configures it; and the customer remains accountable for both outcomes to its regulators, customers, and board.

The chapter's argument converges on a small number of managerial disciplines. Organizations succeed in the cloud when they govern identity as the control plane, treating privileged and workload identities as their most valuable assets. They succeed when they govern configuration as a continuous discipline, expressing intended state as code, enforcing guardrails automatically, and monitoring posture across IaaS, PaaS, and SaaS. They succeed when they govern data through deliberate decisions about classification, key control, and location; govern visibility through protected, centralized logging tied to response authority; and govern resilience through tested recovery, explicit treatment of concentration risk, and an exit strategy that has been checked against reality.

These disciplines need an operating model to hold them together: a cloud centre of excellence that makes the secure path the easy one, a landing zone and account structure that provide isolation and policy by default, clear decision rights for providers, regions, exceptions, and residual risk, and FinOps and security working from the same inventory and tags. They also need standards that make decisions auditable—the ISO/IEC 27000 family, including the newly revised ISO/IEC 27017:2026, NIST CSF 2.0, the CSA Cloud Controls Matrix and STAR, SOC 2 assurance, CIS Foundations Benchmarks, and the Canadian references summarized earlier—combined into a baseline that is automated and measured rather than merely written.

Cloud security, therefore, is not a cloud team problem. It is a governance problem. The organization must define who decides, who is accountable, which minimum controls are non-negotiable, how compliance is continuously verified, how evidence is preserved, and how it will recover or exit when disruption inevitably occurs. Managers who can answer those questions with evidence, rather than with assurances from a provider's marketing material, are governing the cloud. Those who cannot are simply renting someone else's infrastructure and hoping for the best.

## Chapter summary

Cloud has become the default operating model not because it is inherently more secure, but because of its speed, scale, resilience, and cost flexibility. Its central governance problem is that it redistributes control without redistributing accountability: the provider operates much of the stack, yet the customer still answers to its board, regulators, and customers. Most cloud failures are customer failures to configure, monitor, and govern what they always owned. This chapter explains the technical foundations, the changed risk profile, and shared responsibility as a governance instrument, then shows how to govern cloud through an operating model, identity, posture management, SaaS ownership, data location, protected logging, tested recovery and exit, recognized standards, and an automated, measured controls baseline.

### Key points to retain

- Virtualization, containers, and serverless hand more of the stack to the provider, but the parts the customer keeps, such as identity, permissions, and data flows, become proportionally more important.

- Service and deployment models define the control boundary, and each NIST essential characteristic, such as on-demand self-service, has a direct governance consequence.

- The management plane is the dominant cloud attack surface: it is reachable from anywhere and protected chiefly by identity, so a privileged identity needs no exploit to cause harm.

- Cloud changes the economics of mistakes, rewarding standardization and automation and punishing reliance on individual diligence.

- Shared responsibility divides operational duties but not accountability; the provider's assurance report proves only its half, and the customer must prove its own.

- A cloud centre of excellence recommends and builds the paved road, while documented decision rights leave acceptance of residual risk with risk owners and executives.

- Landing zones, a sound account hierarchy, and preventive, detective, and corrective guardrails with a time-limited exception path keep teams within appetite by default.

- FinOps and security share one inventory and tagging scheme, and cost cuts that weaken logging, retention, or backups are risk acceptance decisions.

- Identity is the cloud control plane: privileged humans need phishing-resistant, just-in-time access, and workload identities need owners, short-lived credentials, and vaulted secrets.

- Misconfiguration and SaaS sprawl are governed through continuous posture assessment, owned remediation, proportionate SaaS intake, OAuth grant review, and a named owner for each significant application.

- Key control, data residency, sovereignty, protected central logging, cloud-specific response playbooks, tested recovery, concentration analysis, and a verified exit plan are deliberate customer decisions.

- Standards make decisions auditable, and a cloud baseline must be automated and measured through all seven steps; stopping at the document is the most common failure.

### Key terms

- **Hypervisor** — The software layer that allocates physical resources to virtual machines and enforces the isolation between them.

- **Management plane** — The consoles and APIs through which cloud resources are created, configured, and deleted, distinct from the data plane where workloads run.

- **Shared responsibility model** — The division of security responsibilities between provider and customer, which shifts with the service model while customer accountability remains.

- **Complementary user entity controls** — Controls listed in a SOC 2 report that the customer must operate for the provider's controls to achieve their objectives.

- **Cloud centre of excellence (CCoE)** — A small cross-functional team that sets cloud standards, builds shared platforms, and makes the secure way the easy way.

- **Landing zone** — A pre-configured, secure, multi-account cloud environment that provides identity integration, network design, logging, tooling, and policy from day one.

- **Guardrail** — A platform-enforced policy, preventive, detective, or corrective, that keeps teams within acceptable boundaries without manual review of each action.

- **FinOps** — The practice of managing cloud spending through shared accountability among finance, engineering, and business teams.

- **Workload identity** — A non-human identity, such as a service account, application role, API key, or pipeline credential, used by software to call cloud services.

- **Cloud-Native Application Protection Platform (CNAPP)** — A platform combining posture management, workload protection, entitlement management, and pre-deployment scanning for a unified view of cloud risk.

- **SaaS sprawl** — The uncontrolled growth in SaaS applications adopted by departments or individuals, often bypassing procurement, security review, and privacy assessment.

- **Data residency** — The physical or geographic location where data is stored and processed.

- **Data sovereignty** — The principle that data is subject to the laws of the jurisdiction where it is located and, in some cases, of the provider's home jurisdiction.

- **Concentration risk** — The risk that many critical services depend on the same provider, region, or component, so that one failure has broad or systemic effects.

- **Exit strategy** — A documented plan for moving a cloud service to another provider or in-house, covering data, formats, time, cost, dependencies, and provider assistance.

## Review questions

### Essay questions

1.  The chapter states that cloud 'redistributes control without redistributing accountability'. Analyze this statement using the shared responsibility model, OSFI Guideline B-10, and Québec's Law 25, and evaluate what evidence a customer must produce to demonstrate that it is fulfilling its half of the model.

2.  A university wants to move student information systems and research computing to a public cloud. Design a cloud governance operating model for it, covering the CCoE, decision rights, landing zone and account structure, guardrails with an exception path, the three lines, and FinOps alignment.

3.  Choose an organization you know (or use Laurentide Mutual from the chapter). Identify its three highest-priority cloud risk scenarios, justify the ranking, and recommend a treatment portfolio that addresses identity, configuration, and evidence for each.

4.  SaaS is often the larger and less governed cloud exposure. Evaluate the risks of SaaS sprawl and SaaS-to-SaaS integrations, and propose a governance model for discovery, intake, configuration, and ownership that will not drive users to unapproved tools.

5.  A board member argues that multicloud is the obvious answer to concentration risk and that contractual rights make exit easy. Critically evaluate both claims using the chapter's discussion of resilience, concentration risk, DORA, the EU Data Act, and ISO/IEC 27002:2022 control 5.23.

### Multiple-choice questions

Choose the best answer. Answers are listed in the appendix "Answers to multiple-choice questions".

**1.** What does the chapter identify as the central governance problem of cloud adoption?

> A. Provider infrastructure is generally less secure than on-premises data centres
>
> B. Cloud removes the need for vendor management and contractual oversight
>
> C. Cloud costs grow unpredictably and cannot be governed by finance
>
> D. Cloud redistributes control without redistributing accountability

**2.** Which NIST SP 800-145 essential characteristic most directly explains the appearance of cloud sprawl and shadow IT?

> A. Resource pooling
>
> B. On-demand self-service
>
> C. Measured service
>
> D. Rapid elasticity

**3.** As an organization moves from virtual machines to containers to serverless functions, what happens to the customer's security responsibilities?

> A. They disappear, because the provider runs the code on demand
>
> B. Identity and permissions become far less important than before
>
> C. The customer configures less, but what remains becomes proportionally more important
>
> D. The provider becomes accountable for the customer's data and flows

**4.** Why does the chapter describe the management plane as the dominant cloud attack surface?

> A. It is reachable from anywhere through consoles and APIs and protected chiefly by identity
>
> B. Provider encryption and isolation are routinely defeated by attackers
>
> C. Hypervisor exploits are the most common cause of cloud incidents
>
> D. The data plane produces no logs that defenders could use

**5.** Regardless of the service model, which security responsibility always remains with the customer?

> A. Physical security of the provider's data centres
>
> B. Its data, identities and access decisions, and how it configures services
>
> C. Security of the virtualization layer and underlying hardware
>
> D. Operating system patching for every service it consumes

**6.** An executive argues that moving to SaaS means the organization's cyber risk has narrowed along with its responsibilities. How does the chapter respond?

> A. It agrees, because the provider controls most of the technology stack
>
> B. It agrees, provided the provider holds an ISO/IEC 27001 certificate
>
> C. It disagrees: the data, workforce identities, and sharing settings remain the customer's
>
> D. It disagrees, because SaaS providers are not covered by assurance reports

**7.** Under OSFI Guideline B-10, what is the position of a federally regulated financial institution that outsources a service to a cloud provider?

> A. It remains accountable for the activities it outsources
>
> B. Accountability transfers to the provider under the contract
>
> C. Accountability is split equally between the institution and provider
>
> D. Accountability applies only to IaaS, not to SaaS arrangements

**8.** In a provider's SOC 2 Type II report, what are complementary user entity controls, and what should the customer do with them?

> A. Controls the auditor tested at the provider, to be filed as evidence
>
> B. Subservice organizations excluded from scope through the carve-out method
>
> C. Exceptions the auditor found, to be raised with the provider's management
>
> D. Controls the customer must operate, to be mapped to named internal owners

**9.** What is the purpose of a cloud centre of excellence, according to the chapter?

> A. To approve every cloud change before it reaches production
>
> B. To accept residual risk on behalf of each workload's business owner
>
> C. To make the secure way the easy way by setting standards and building platforms
>
> D. To replace the organization's existing governance bodies for cloud matters

**10.** At Laurentide Mutual, the top scenario was administrator compromise via a help-desk reset, followed by deletion of production data and backups. Which treatment most directly limits the backup deletion?

> A. Immutable backups in a separate account with independent credentials
>
> B. An SSPM review of the collaboration suite against the CIS benchmark
>
> C. A guardrail restricting resource creation to Canadian regions
>
> D. A quarterly cloud governance report to the risk committee

**11.** A cloud policy automatically reverts any storage bucket that becomes publicly accessible. Which type of guardrail is this?

> A. Preventive
>
> B. Detective
>
> C. Exception
>
> D. Corrective

**12.** A cost-reduction initiative quietly shortens log retention and removes cross-region backup copies. How should this decision be treated?

> A. As a routine FinOps optimization recorded in the billing report
>
> B. As a risk acceptance decision to be made by the risk owner
>
> C. As automatically acceptable whenever the savings exceed tooling cost
>
> D. As a technical choice for the cloud centre of excellence alone

**13.** Which practice reflects the chapter's governance expectations for workload identities and secrets?

> A. Prefer short-lived, platform-issued credentials and store secrets in a managed vault
>
> B. Keep long-lived access keys in configuration files so automation stays reliable
>
> C. Grant broad permissions by default to avoid outages caused by missing rights
>
> D. Review non-human identities only when a human owner leaves the organization

**14.** A CSPM dashboard shows thousands of findings. According to the chapter, when does it become a security control?

> A. When the tool detects more findings than in the previous quarter
>
> B. When the organization upgrades from CSPM to a full CNAPP platform
>
> C. When findings go to owners with targets and critical ones trigger immediate response
>
> D. When raw finding counts are reported to the board each month

**15.** Why do SaaS-to-SaaS OAuth integrations create a distinct and growing risk?

> A. Their tokens expire whenever the user changes the account password
>
> B. They can grant access only to basic profile information
>
> C. Each SaaS vendor reviews every token granted to third parties
>
> D. They can give broad, persistent access that survives password changes and is rarely reviewed

**16.** Data is stored in a Canadian region of a provider headquartered in another country. What does the chapter say about this arrangement?

> A. Only Canadian law can apply because residency determines jurisdiction
>
> B. The data may still face lawful-access demands under the provider's home-country laws
>
> C. The arrangement is exempt from Law 25's privacy impact assessment rules
>
> D. The data cannot be encrypted with customer-managed keys in that region

**17.** What does Québec's Law 25 require before personal information is communicated outside Québec, including to a cloud provider?

> A. A SOC 2 Type II report from the receiving cloud provider
>
> B. A sovereign cloud arrangement located within the province
>
> C. Encryption with keys held only by the enterprise itself
>
> D. A privacy impact assessment and adequate protection of the information

**18.** Which logging design best ensures that evidence survives an attacker who gains administrative access to a workload account?

> A. Central logging account without workload write or delete rights, immutable storage, alerts on changes
>
> B. Logs kept in each workload account so the application team can manage them
>
> C. Data-access logging enabled everywhere regardless of volume and cost
>
> D. Provider default settings and retention, which are designed for investigations

**19.** What does ISO/IEC 27002:2022 control 5.23, Information security for use of cloud services, require organizations to establish?

> A. Encryption of all cloud data using customer-held keys
>
> B. Processes for the acquisition, use, management, and exit from cloud services
>
> C. A multicloud architecture for every critical business service
>
> D. An annual SOC 2 Type II report for each cloud provider used

**20.** According to the chapter, what is the most common failure when organizations build a cloud security controls baseline using the seven-step method?

> A. Skipping the current-state inventory and starting with framework selection
>
> B. Combining too many frameworks into a single control backbone
>
> C. Stopping at step 5 with a document no pipeline enforces or dashboard measures
>
> D. Defining more baseline tiers than the organization can manage
