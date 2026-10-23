<!-- Léger, M.-A. (2026). Introduction to Cybersecurity Governance for Business Technology Management (3rd ed., v2.51). Chapter 19: Third-Party and Supply-Chain Cybersecurity Governance. CC BY-NC 4.0 -->

# Chapter 19: Third-Party and Supply-Chain Cybersecurity Governance

## Learning outcomes

After completing this chapter, you should be able to:

1.  Explain why third-party risk must be defined by consequence rather than by contract type, and distinguish third-party, fourth-party, and supply-chain exposure.

2.  Describe the seven stages of the third-party governance lifecycle and justify the use of gates with defined decision rights at each stage.

3.  Apply the four drivers of inherent risk to classify a supplier into a tier and link that tier to minimum evidence, approval authority, and monitoring cadence.

4.  Evaluate the assurance value of questionnaires, SOC reports, and ISO/IEC 27001 certificates, including scope, period, exceptions, carve-outs, and complementary user entity controls.

5.  Recommend security requirements, disqualifiers, and contract clauses that make supplier obligations enforceable and aligned with the organization's own regulatory deadlines.

6.  Analyze concentration and fourth-party risk, using the July 2024 outage and software supply-chain incidents to identify governance responses.

7.  Design incident coordination and exit arrangements for a critical supplier, including notification triggers, disconnection authority, data portability, and evidence at termination.

8.  Assign roles and decision rights for third-party governance across business owners, procurement, legal, privacy, the CISO function, and the three lines.

Cybersecurity governance increasingly succeeds or fails at organizational boundaries. Many of the most disruptive incidents of the past decade were not caused by an organization's internal control weaknesses alone, but by weaknesses in its ecosystem: cloud and SaaS dependencies, managed service providers, software components, payment processors, customer support platforms, and specialized vendors with privileged access. A ransomware group that compromises a remote management tool used by a managed service provider reaches hundreds of that provider's clients at once. A flaw in a widely used file-transfer product exposes the data of thousands of organizations that never had a contract with the product's maker. A faulty update from a trusted security vendor halts airlines, hospitals, and retailers on the same morning. In each case, the affected organizations discovered that their cybersecurity posture was only as strong as the external parties they depended on.

In this environment, third-party risk is not a procurement formality. It is the governance discipline that determines whether the organization's cybersecurity expectations become enforceable obligations or remain aspirational policies. Internal controls can be designed, implemented, and tested by the organization itself. External controls cannot. They must be specified in advance, negotiated, contracted, evidenced, monitored, and, when necessary, enforced or abandoned. Each of those verbs describes a managerial act, not a technical one.

For business technology managers, third-party and supply-chain security is one of the most tangible areas of cybersecurity governance. It is where risk appetite is translated into minimum requirements, where auditability becomes contractual, and where accountability becomes operational: who must do what, by when, with what evidence, and what happens if they do not. It is also an area where managers outside the security function—procurement officers, business owners who sponsor a SaaS purchase, lawyers who negotiate terms, finance leaders who approve renewals—make decisions that shape cybersecurity exposure for years, often without realizing it.

This chapter builds on earlier material: risk appetite and tolerance (Chapter 4) provide the basis for tiering, scenario-based risk assessment (Chapters 5 and 6) supplies the method for asking what could go wrong at a supplier, the frameworks of Chapter 9 all contain supplier requirements, incident management and continuity (Chapter 12) are what supplier failures most often test, and the cloud shared responsibility model (Chapter 15) is a specific case of the broader problem treated here.

The chapter presents third-party and supply-chain cybersecurity as a full lifecycle: planning and risk tiering, pre-contract due diligence, security requirements in procurement, contracting and enforceability, onboarding and continuous monitoring, concentration and fourth-party risk, software supply-chain assurance, incident coordination, and exit planning. It then describes the operating model that makes the program work and the questions managers and boards should ask. The goal is not simply to assess vendors, but to operate a defensible assurance program that reduces real risk and produces evidence of due diligence.

## Concepts and scope: what third-party and supply chain mean in practice

Organizations often use the term third-party risk narrowly to mean vendors who have network access. That definition was never adequate, and it is now dangerously incomplete. A modern organization depends on external parties for identity, communication, storage, computing, payments, analytics, customer engagement, and even its own security monitoring. Some of these parties never touch the corporate network, yet their failure would stop the business. Governance must therefore begin by defining the scope of the problem correctly, because everything that follows—tiering, due diligence, contracts, monitoring—applies only to what is inside that scope.

### Defining the third party by consequence, not by contract

A **third party**, for cybersecurity governance purposes, is any external entity whose services, products, or operations can materially affect the confidentiality, integrity, or availability of the organization's information, the privacy of the people whose data it holds, or the continuity of its critical services. A free browser extension adopted by a marketing team, a payment gateway, an open-source library embedded in a customer-facing application, and an affiliate company that provides shared IT services all qualify if their failure or compromise would harm the organization.

This consequence-based definition matches the direction taken by regulators. OSFI's Guideline B-10, *Third-Party Risk Management*, published in April 2023 and effective May 1, 2024, replaced the older outsourcing-focused guideline with a definition of third-party arrangement that extends well beyond outsourcing, and it explicitly includes arrangements with affiliates. The EU Digital Operational Resilience Act (DORA), Regulation (EU) 2022/2554, which has applied since January 17, 2025, similarly treats ICT services broadly, from cloud infrastructure to software and data services. The managerial lesson is that an organization that inventories only its contracted IT outsourcers is not managing its third-party risk; it is managing a subset of it.

A consequence-based definition also clarifies a common source of confusion: **third-party risk** is not the same as vendor insecurity. A supplier can be well secured and still be a source of serious risk if the organization depends on it without alternatives, if its contract gives the organization no rights, or if the organization's own integration with it is poorly controlled. Governance must consider the relationship, not only the supplier.

### Six categories of external dependency

A practical scope definition usually distinguishes at least six categories of external dependency.

- **Technology and cloud providers** include infrastructure, platform, and software-as-a-service providers, identity providers, email and collaboration platforms, and managed databases. These are often the most critical dependencies and the most difficult to replace.

- **Managed service and security providers** include MSPs, managed security service providers (MSSPs), managed detection and response (MDR) providers, and incident response retainers. They typically hold privileged access to many clients simultaneously, which makes them attractive targets.

- **Software suppliers and integrators** include commercial software vendors, open-source components, contractors who develop internal systems, and system integrators who deploy platforms. Their risk lies in the code and configurations they deliver.

- **Business process outsourcing providers** include payroll services, customer contact centres, collection agencies, marketing agencies, and data analytics firms. They often process large volumes of personal information outside the organization's direct oversight.

- **Data partners and intermediaries** receive, enrich, broker, or process data on the organization's behalf. Their risk is primarily one of confidentiality, privacy, and secondary use.

- **Operational dependencies** such as payment processors, telecommunications carriers, logistics platforms, and operational technology vendors may feel non-technical, but they are cyber-relevant because their compromise or outage interrupts critical services (see Chapter 11 for OT vendors specifically).

### Supply chain, fourth parties, and transitive exposure

**Supply-chain cybersecurity** overlaps with third-party risk but emphasizes upstream dependencies and transitive exposure: the components, subcontractors, libraries, build tools, and service providers on which the organization—and its direct vendors—depend. NIST describes this discipline as cybersecurity supply chain risk management, or C-SCRM. Supply-chain thinking forces two questions that a vendor-by-vendor view tends to miss. What is inside the products and services we buy? And on whom do our suppliers themselves depend?

The second question introduces **fourth-party risk**: the risk that arises from the vendor's vendors. A SaaS provider runs on a hyperscale cloud, uses a third-party email delivery service, embeds a payment processor, and relies on an outsourced support desk. Each of these is a fourth party to the SaaS customer. The customer usually has no contract with them and no direct visibility into their practices, yet their failure can propagate to the customer's critical services. Fourth-party exposure is also what makes concentration risk systemic: when many vendors rely on the same few upstream providers, an apparently diversified supplier base may conceal a single point of failure.

Third-party risk management is therefore not a single process but an **ecosystem governance program**: risk-based, scope-driven, and integrated with procurement, legal, privacy, business continuity, and cybersecurity operations.

### The reference landscape

Several standards and frameworks now address this domain directly, and managers should know what each contributes. The ISO/IEC 27036 series, *Cybersecurity — Supplier relationships*, provides the most specific international guidance: Part 1 (2021) gives an overview and concepts, Part 2 (2022) specifies requirements for defining, implementing, and managing information security in supplier relationships, Part 3 (2023) provides guidelines for hardware, software, and services supply-chain security, and Part 4 (2016) addresses the security of cloud services. Within the ISMS itself, ISO/IEC 27001:2022 and ISO/IEC 27002:2022 contain a cluster of organizational controls (5.19 to 5.23) covering information security in supplier relationships, security within supplier agreements, the ICT supply chain, monitoring and change management of supplier services, and the use of cloud services.

NIST SP 800-161 Rev. 1, *Cybersecurity Supply Chain Risk Management Practices for Systems and Organizations*, published in May 2022 and updated through an errata release (Update 1) in November 2024 that clarified guidance on SBOMs and vulnerability advisories, is the most detailed public treatment of C-SCRM, with enterprise, mission, and system-level practices. NIST CSF 2.0 elevated supply-chain risk management into the Govern function as its own category, GV.SC, with ten outcomes ranging from establishing a C-SCRM program and assigning roles to prioritizing suppliers by criticality, embedding requirements in contracts, including suppliers in incident planning, and managing post-relationship activities. CIS Controls v8.1 addresses the topic operationally through Control 15, Service Provider Management. For federally regulated financial institutions in Canada, OSFI B-10 sets supervisory expectations, and for EU financial entities DORA's ICT third-party provisions are binding law. These references are complementary: ISO/IEC 27036 and NIST SP 800-161 explain how to do the work, CSF 2.0 states the outcomes governance should achieve, and OSFI B-10 and DORA define what supervisors will examine.

| Third-party risk is defined by consequence, not by contract type: if an external party's failure would harm critical services or sensitive information, it belongs inside the governance program. |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## A lifecycle model for third-party cybersecurity governance

A credible governance program follows a lifecycle that mirrors how relationships actually evolve. The lifecycle matters because risk is not constant across a relationship: it is shaped before the contract, spikes during integration, drifts during operation, and resurfaces at exit. A program that concentrates all its effort at one point—typically a security questionnaire during procurement—will be strong at that point and blind everywhere else. ISO/IEC 27036-2 and NIST CSF 2.0 (GV.SC-09 and GV.SC-10 in particular) both frame supplier security as something to be integrated and monitored throughout the relationship, including after it ends.

### Seven stages of the relationship

The first stage is **planning and classification**. The organization defines what types of third parties exist, what minimum requirements apply to each, and how relationships are classified by risk. Classification has practical consequences: it determines required controls, required evidence, approval levels, and the intensity of ongoing monitoring. This stage also includes maintaining the third-party register discussed later in this chapter.

The second stage is **pre-contract due diligence**. Before committing, the organization assesses whether the prospective supplier can meet its requirements. This includes reviewing security posture and assurance evidence, privacy obligations and data residency, financial viability, operational resilience, and the supplier's own dependencies. Due diligence is proportionate: a critical cloud platform warrants weeks of review, while a low-risk office supplier warrants minutes.

The third stage is **contracting and enforceability**. This is where policy becomes legally binding obligation: security requirements, audit and evidence rights, incident notification duties, subcontractor controls, data handling and deletion commitments, and remedies for non-compliance. What is not in the contract is, in practice, largely unenforceable.

The fourth stage is **onboarding and integration**. Risk often spikes here: accounts are created, credentials and API keys are exchanged, network connections are opened, and data begins to flow. Onboarding should therefore include technical gating controls—no production access until required configurations are verified—and evidence capture for later audit.

The fifth stage is **ongoing assurance and monitoring**. Suppliers change ownership, architecture, personnel, and subcontractors. The organization must continuously maintain confidence rather than relying on a snapshot taken at contract signature.

The sixth stage is **incident coordination**. When a supplier suffers an incident, or when the organization is breached through a supplier, coordination and evidence become decisive. Governance must pre-define communication channels, notification timelines, cooperation expectations, and decision rights before the incident occurs.

The seventh stage is **renewal, exit, and transition**. Renewal is a decision point at which the relationship's risk should be reassessed, not a calendar formality. Termination requires secure transition: data return or destruction, credential revocation, service continuity, and risk-managed migration to an alternative. The figure below summarizes the lifecycle.

<span id="_Toc243635577" class="anchor"></span>Figure 78: The third-party cybersecurity governance lifecycle

### The accept-and-forget failure

The most common failure mode in third-party governance is what can be called **accept and forget**: risk is assessed and accepted at onboarding, and the supplier then effectively disappears from governance view until something goes wrong. Procurement controls naturally cluster around the purchase event, business sponsors move on once the service is live, security teams have little capacity left for reassessment, and auto-renewing contracts remove the natural decision point.

The consequences are predictable: a low-risk supplier gradually receives payment data, a vendor is acquired by a company with weaker practices, or a pilot integration becomes a permanent, privileged connection, and none of these changes triggers reassessment. The lifecycle model counters this failure by making each stage explicit, assigning an owner to it, and defining the events that move a relationship back into review.

### Lifecycle gates and decision rights

A lifecycle becomes governance only when its stages have **gates**: points at which a defined person must make a recorded decision based on defined evidence. Typical gates include approval to proceed with a supplier into contract negotiation, approval of contract terms that deviate from the standard security schedule, approval to grant production access after onboarding checks, renewal approval, and approval to terminate with a verified exit plan. Each gate should specify who decides, what evidence is required, and how exceptions are handled.

Decision rights at these gates should scale with risk tier: a low-risk supplier might pass with the business owner's approval and an automated check, while a critical supplier might require the CISO's assessment, the privacy officer's sign-off, legal approval of non-standard terms, and explicit acceptance of residual risk by an executive risk owner. When requirements cannot be met, the gate should produce a documented exception with an owner, compensating controls, and an expiry date—the same discipline applied to other exceptions in this book—rather than a silent waiver. Gates of this kind are what turn a lifecycle diagram into a defensible decision record that internal audit, regulators, and boards can examine.

## Risk tiering: turning risk appetite into tiered requirements

Organizations often claim to have third-party risk management but apply the same process to every vendor. That approach creates two failures at the same time: critical suppliers receive insufficient scrutiny, and low-risk suppliers create bureaucratic overload that slows the business and erodes support for the program. Both investment governance and risk governance require **tiering**: classifying third-party relationships by inherent risk so that effort, evidence, and oversight are proportionate. OSFI B-10 expresses this as a risk-based approach in which the intensity of management is commensurate with the risk and criticality of each arrangement, and CSF 2.0 (GV.SC-04) calls for suppliers to be known and prioritized by criticality.

### Four drivers of inherent risk

A practical tiering model typically considers four drivers. Each can be captured through a short intake questionnaire completed by the business owner before any security assessment begins, which keeps tiering fast and places accountability for an accurate description of the relationship where it belongs.

- **Data sensitivity** asks what information the supplier will store, process, or access. Personal information, financial data, health data, payment card data, intellectual property, authentication secrets, and confidential business information raise the tier, as does volume. The jurisdictions involved matter as well: under Québec's Law 25, for example, communicating personal information outside Québec requires a privacy impact assessment, and any transfer to a service provider must be supported by a written contract specifying protective measures.

- **Service criticality** asks what happens if the supplier fails. How quickly does harm escalate? Is the supplier on the critical path for revenue, customer service, safety, regulatory compliance, or financial reporting? OSFI B-10 is explicit that criticality focuses on the impact of a disruption, irrespective of how likely it is, which is a useful discipline for any organization.

- **Access and integration** asks how deeply the supplier is connected. Privileged or administrative access, persistent network connectivity, API access to core systems, or the ability to push code or updates into the environment all raise risk sharply, because they turn a supplier compromise into an organizational compromise.

- **Substitutability and concentration** asks whether the organization could realistically replace the supplier within an acceptable time, and whether the supplier is a single point of failure because of product uniqueness, market concentration, proprietary data formats, or internal integration complexity.

### From drivers to tiers

Using these drivers, organizations define a small number of tiers—usually three or four. More tiers add precision but also argument about borderline cases; fewer tiers make the program easier to explain and operate. The highest tier typically includes cloud identity providers, core SaaS and infrastructure platforms, managed service providers with privileged access, and suppliers processing regulated personal information at scale. The lowest tier includes commodity suppliers with no data access and no operational criticality. The table below shows a sample model; each organization must calibrate its own thresholds to its risk appetite.

| **Tier**         | **Typical examples**                                                 | **Minimum evidence**                                               | **Reassessment**            |
|------------------|----------------------------------------------------------------------|--------------------------------------------------------------------|-----------------------------|
| **1 — Critical** | Identity provider, core ERP or cloud platform, MSP with admin access | Independent report with matching scope, targeted review, exit plan | Annual, plus event triggers |
| **2 — High**     | SaaS holding personal information, payroll, contact centre           | Independent report or detailed evidence pack                       | Annual or at renewal        |
| **3 — Moderate** | Analytics tool with limited data, niche software                     | Standard questionnaire with key evidence                           | Every two to three years    |
| **4 — Low**      | No data, no access, easily replaced                                  | Intake form only                                                   | At renewal                  |

The tier, once assigned, drives everything downstream: the RFP template used, the depth of due diligence, the contract schedule attached, the onboarding gates applied, the monitoring cadence, the approval authority for exceptions, and whether an exit plan is required. This is what makes the model enforceable. Without such linkages, tiering is a labelling exercise that changes nothing.

### Inherent risk, residual risk, and calibration

Tiering is based on **inherent risk**—the exposure created by the nature of the relationship before considering the supplier's controls. Due diligence then informs **residual risk**: the exposure that remains given what the organization knows about the supplier's controls and its own compensating measures. Keeping the two distinct prevents a reputable supplier from being placed in a low tier and never examined closely. A critical identity provider remains Tier 1 even if its controls are excellent; what changes with good evidence is the residual risk rating and the confidence with which the business owner can accept it.

Calibration should be tested against risk appetite. If the board has stated a low appetite for the loss of customer personal information and a moderate appetite for short service disruptions in non-core functions, the tiering thresholds should reflect those statements: a supplier holding customer data at scale should be Tier 1 or 2 regardless of its operational role, while a supplier whose outage would inconvenience an internal team for a day might be Tier 3. Scenario-based risk assessment, as described in Chapters 5 and 6, is the natural tool for checking calibration. Building two or three concrete scenarios for a sample of suppliers in each tier—a supplier breach exposing data, a supplier outage, a compromised update pushed through a supplier's channel—quickly reveals whether the tiers are aligned with the losses the organization actually cares about.

### The third-party register as the foundation

Tiering presupposes that the organization knows who its third parties are. In practice, this is often the hardest step. Purchases made with corporate cards, free SaaS tools adopted without procurement, integrations created by developers, and legacy contracts that predate the current program all escape the formal inventory. A **third-party register** that records each relationship, its business owner, its tier, the data and systems involved, its key contract terms, its material subcontractors, and its assessment status is the foundation of the program. OSFI B-10 expects federally regulated financial institutions to maintain a third-party inventory, and DORA requires EU financial entities to maintain a register of information on all contractual arrangements for ICT services, which supervisors use, among other things, to identify critical providers across the sector.

Building the register usually requires combining several sources: accounts payable data, contract management systems, single sign-on and identity logs showing which external applications employees use, network and cloud access-broker telemetry, and interviews with business units. The register should be treated as a living control rather than an annual compilation. Its quality is itself a key risk indicator: the proportion of active suppliers with an assigned tier and owner tells the board more about the program's real coverage than any number of completed questionnaires. The figure below shows how risk appetite flows through tiering to concrete requirements.

<span id="_Toc243635578" class="anchor"></span>Figure 79: From risk appetite to tiered third-party requirements

| **Tiering is what makes third-party governance enforceable**: each tier must carry its own minimum controls, minimum evidence, approval authority, and monitoring cadence, or the tiers change nothing. |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## Due diligence and assurance: questionnaires versus evidence-based confidence

One of the most important governance decisions in this domain is how the organization obtains assurance about a supplier's security. Many programs rely heavily on questionnaires because they are cheap, scalable, and familiar. Questionnaires have a place, but they often create false confidence because they measure statements rather than capabilities. A supplier that answers yes to "Do you enforce multifactor authentication for all administrative access?" has made a claim; it has not demonstrated a control. The governance question is how much confidence the organization needs for a given relationship, and what kind of evidence can deliver it.

### What questionnaires can and cannot do

Questionnaires are best understood as a **screening and scoping tool**. They identify suppliers with no security program at all, reveal the basic shape of a supplier's environment, confirm the existence of policies and certifications, provide security contacts, and surface disclosed incidents. Standardized questionnaires reduce friction for both sides. The Cloud Security Alliance's Consensus Assessments Initiative Questionnaire (CAIQ), now at version 4.1 alongside the Cloud Controls Matrix v4.1 released in January 2026, is widely used for cloud providers, and many providers publish completed CAIQs in the CSA STAR registry, allowing customers to review them without sending yet another custom questionnaire.

The limitations are equally clear. Questionnaires are self-attested, often completed by sales or compliance staff rather than the people who operate controls, and answered in the most favourable reasonable interpretation. They measure design, rarely operating effectiveness. And a completed questionnaire filed in a procurement system can give a program the appearance of diligence without any analysis of the answers—a classic form of compliance theatre. The test of a questionnaire process is not how many are completed, but whether answers are reviewed by someone competent, whether material gaps lead to decisions, and whether claims that matter are corroborated.

### Reading independent assurance reports

At a higher level, **evidence-based assurance** relies on independent artifacts. The most common are service organization control reports and certifications. A **SOC 2** report, prepared by an independent auditor under AICPA attestation standards against the Trust Services Criteria (security, availability, processing integrity, confidentiality, and privacy), describes a service organization's system and controls. A Type I report addresses whether controls were suitably designed at a point in time; a **Type II** report also tests whether they operated effectively over a period, typically six to twelve months, and is therefore far more informative. SOC 1 reports, issued in the United States under SSAE 18, internationally under ISAE 3402, and in Canada under CSAE 3416, address controls relevant to customers' financial reporting rather than security as such, and they matter to finance and external audit teams. ISO/IEC 27001 certification provides assurance that a management system meets the standard's requirements within a defined scope, but it does not by itself attest to the operating effectiveness of specific controls in the way a Type II report does.

The governance value of these artifacts depends entirely on how they are read. Managers should ensure that reviewers check at least the following elements:

- **Scope and boundaries** determine whether the report or certificate actually covers the service, the locations, and the systems the organization uses. An ISO/IEC 27001 certificate for a supplier's head office does not cover its data centre operations unless the statement of scope says so.

- **The period and its currency** matter because a Type II report covering a period that ended many months ago says little about today. Suppliers commonly provide a **bridge letter** asserting that nothing material has changed since the period end; it is useful, but it is a management assertion, not audit evidence.

- **Exceptions and qualified opinions** are where the real information lies. A clean summary page may hide control exceptions in the testing section that are directly relevant to the organization's risk.

- **Carved-out subservice organizations** reveal fourth parties. If the report carves out the supplier's cloud hosting provider, the controls of that fourth party are not covered and must be assured another way.

- **Complementary user entity controls** are controls the supplier assumes its customers will perform—for example, managing their own user accounts or reviewing access logs. If the organization does not perform them, the assurance in the report does not hold for that organization.

The last point deserves emphasis because it is frequently overlooked. A SOC 2 report is not only evidence about the supplier; it is also a list of obligations for the customer. This mirrors the shared responsibility model discussed in Chapter 15, and it means that reviewing a report should produce actions for internal teams, not only a pass or fail for the supplier.

### Deeper validation for critical suppliers

For Tier 1 relationships, governance may require validation beyond standard reports. Options include targeted assessments against specific risk scenarios; architecture reviews focused on tenant isolation, data segregation, key management, and administrative access paths; review of the supplier's incident response plan and evidence of recent exercises; independent penetration test summaries with remediation status; and, where contractual rights allow, on-site or remote audits. For cloud services, deeper validation often focuses on configuration and integration rather than the provider's internal controls, because the most likely failure is in the customer's own use of the service (see Chapter 15).

### Proportionality and the assurance budget

The principle that ties these layers together is simple: **the more material the risk, the more evidence-driven the assurance must be**. Proportionality is a design principle, not a concession: it spends scarce assessment capacity where it reduces the most risk. An organization with three hundred suppliers and two analysts cannot review everyone deeply, but it can ensure that its twenty Tier 1 suppliers receive genuine scrutiny, that Tier 2 suppliers provide independent evidence, and that lower tiers are handled through efficient, largely automated processes.

Managers should also recognize that assurance has a cost, and that this cost belongs in investment decisions (see Chapter 17). The time spent assessing, negotiating, and monitoring a critical supplier is part of the total cost of the relationship. When a business unit proposes a new Tier 1 supplier, the business case should include the assurance effort required, just as it includes licence fees and integration costs. Doing so makes security review a planned cost of a high-risk decision rather than an unbudgeted obstacle.

## Security requirements in RFPs and procurement: building security in rather than negotiating it later

The most effective point of control is often the request for proposal (RFP) stage. If security is introduced only after a vendor has been selected, the organization's leverage drops sharply. Timelines are set, and the supplier knows the customer is unlikely to walk away over a security clause. Governance maturity is demonstrated by integrating minimum security requirements into procurement criteria before a supplier is chosen, so that security becomes a factor in selection rather than an obstacle to signature.

### Requirements before selection

In practice, this means that procurement and business owners define security and privacy requirements alongside functional requirements, with the security function providing standard content rather than bespoke review of every RFP. Typical requirements include single sign-on and multifactor authentication, customer-accessible audit logs, encryption with customer-managed key options for sensitive workloads, data residency choices, defined patching timelines, a vulnerability disclosure policy, incident notification commitments, independent assurance reports, and transparency about material subcontractors. For software products, requirements increasingly include evidence of secure development practices and the ability to supply a software bill of materials, discussed later in this chapter.

Requirements should be written as verifiable statements, and RFP responses should be evaluated by people competent to judge them. "The solution supports single sign-on through SAML 2.0 or OpenID Connect and enforces MFA for all administrative roles" can be tested during a proof of concept; "The vendor takes security seriously" cannot.

### Disqualifiers as an expression of risk appetite

Procurement requirements are also an effective way to operationalize risk appetite. If the organization's appetite does not permit certain exposures—for example, a critical supplier without any independent assurance report, an identity-bearing application that cannot integrate with the corporate identity provider, or a platform that cannot keep regulated data within approved jurisdictions—those conditions should become **disqualifying criteria** stated in the RFP rather than post-selection negotiation points. Stating disqualifiers up front is fairer to suppliers, who can decline to bid, and it removes a recurring source of conflict between business sponsors and the security function.

Disqualifiers must be used sparingly, because each one narrows the market and may create the very concentration risk the organization wishes to avoid. The appropriate pattern is a short list of non-negotiable requirements derived from board-level appetite statements, a longer list of weighted requirements that influence scoring, and a documented exception path for cases where business value justifies accepting a gap with compensating controls.

### Tier-based templates and evaluation weighting

A useful governance pattern is to provide **tier-based RFP templates**. Low-risk purchases receive a lightweight set of requirements that a business owner can include without assistance. Higher-risk purchases receive structured requirements, evidence requests, and a defined weighting of security in the overall evaluation. The weighting itself is a governance decision: if security accounts for five percent of the score on a critical platform purchase, the organization has signalled that it is willing to trade most security concerns for price or features, and that signal should be deliberate rather than accidental.

Tier-based templates reduce friction while increasing consistency, and consistency is what makes decisions defensible. When a regulator, auditor, or board asks why a particular supplier was chosen despite known weaknesses, the organization can show the requirements that applied, how the supplier scored, which gaps were accepted, and who accepted them. That record is the practical meaning of a defensible decision in procurement.

## Contracting and enforceability: where governance becomes legally real

A policy is not enforceable against a supplier until it becomes a binding obligation. Contracting is therefore central to third-party cybersecurity governance. It is the point at which the organization translates its security expectations into clear clauses, measurable commitments, and remedies. It is also the point at which many programs lose ground: standard supplier terms are accepted because negotiation would delay the project, security schedules are attached but never read by the supplier's operational staff, and clauses are drafted in language so general that no breach could ever be established. ISO/IEC 27002:2022 control 5.20 and CSF 2.0 outcome GV.SC-05 both call for security requirements to be established in supplier agreements; the governance challenge is to make those requirements specific enough to matter.

### Security obligations and the division of responsibility

Contracts should specify baseline security obligations appropriate to the tier: access control and privileged access management, encryption, secure development where relevant, vulnerability management with defined remediation timelines, logging and retention, security monitoring, personnel security, and business continuity and recovery capabilities with stated objectives. The most effective approach is a standard **security schedule** maintained by the security and legal functions, with tier-specific variants, so that each contract does not have to be drafted from scratch and deviations are visible.

Where shared responsibility applies, as it does for nearly every cloud and managed service, the division of responsibilities must be explicit rather than assumed. A responsibility matrix attached to the contract—who manages identities, who configures logging, who applies patches to which layer, who responds to which alerts—removes ambiguity that otherwise surfaces only during an incident. Managed service arrangements deserve particular care, because the provider often operates with administrative privileges inside the customer's environment. The contract should state how the provider's own access is secured (for example, phishing-resistant MFA, dedicated administrative accounts, just-in-time access, and logging visible to the customer), because the compromise of a provider's remote management tooling is one of the most efficient attack paths available to criminals.

### Audit rights and evidence obligations

Contracts should define what assurance artifacts the supplier must provide, how often, and under what conditions the customer may request additional evidence or conduct an audit. For high-risk suppliers, the organization may require annual SOC 2 Type II reports, current ISO/IEC 27001 certificates with appropriate scope, penetration test summaries, and notification of material changes to the control environment. Audit rights should include reasonable limitations that respect the supplier's confidentiality and the interests of its other customers, but they must exist in a form strong enough to support governance and regulatory scrutiny. For federally regulated financial institutions, OSFI B-10 expects contracts for high-risk and critical arrangements to provide audit and access rights for both the institution and OSFI itself, and DORA similarly requires that contracts give financial entities and their competent authorities rights of access, inspection, and audit.

Large providers frequently offer their standard assurance reports instead of individual audit rights. That is often a reasonable compromise if the reports cover the services used and are contractually due on a schedule; the governance error is accepting a contract with no evidence rights at all.

### Incident notification and cooperation

Contracts must define incident notification timelines, the information to be shared, and cooperation obligations for investigation, containment, legal requirements, and customer communications. Notification clauses should recognize that an incident is broader than a confirmed breach, because early warning of a suspected compromise is often what allows the customer to limit harm—by rotating credentials, suspending an integration, or preparing regulatory notifications.

Notification timelines should also be aligned with the customer's own regulatory obligations. A Canadian organization subject to PIPEDA's breach reporting requirements, Québec's Law 25 confidentiality incident provisions, OSFI's technology and cyber incident reporting expectations, or, for EU operations, NIS2 or DORA incident reporting, cannot meet its deadlines if its suppliers are allowed longer to inform it than the regulator allows it to inform the authorities. Cooperation clauses should cover preservation of logs and forensic evidence, access to supplier personnel during an investigation, participation in joint exercises, and coordination of public statements. CSF 2.0 (GV.SC-08) explicitly calls for relevant suppliers to be included in incident planning, response, and recovery activities.

### Subcontracting, data handling, and remedies

Suppliers should be required to impose equivalent security obligations on their subcontractors, to disclose material subcontractors—especially those handling sensitive data or supporting critical functions—and to notify the customer, ideally with a right to object, before material subcontracting changes. This is the contractual basis for managing fourth-party risk, discussed later in the chapter.

Contracts must also specify data ownership, permitted uses (including any use of customer data for analytics or for training AI models, a question that has become routine since generative AI features began appearing in mainstream SaaS products; see Chapter 18), retention limits, data location, data return in usable formats, and secure destruction obligations with evidence. These are not only privacy controls; they are **breach-impact controls**, because data that has been deleted cannot be stolen. In Québec, Law 25 requires that a contract with a service provider receiving personal information be in writing and specify the measures the provider must take to protect it, to use it only for the mandate, and to not retain it after the mandate ends.

Finally, governance requires clarity about what happens if obligations are not met: rights to terminate for security breach, service credits for availability failures, indemnities and liability caps appropriate to the risk, commitments to remediate identified gaps within defined timelines, and step-in or transition assistance rights for critical services. Liability caps deserve executive attention. A standard cap equal to twelve months of fees may be reasonable for a low-risk tool but wholly inadequate for a supplier whose breach could expose millions of customer records. Without consequences, requirements are optional in practice.

### Regulatory minimums and standard terms

For regulated organizations, contract content is partly prescribed. OSFI B-10 sets out in an annex the provisions that contracts for high-risk and critical arrangements should address, including audit rights, incident reporting, subcontracting, data location, and termination. DORA's Article 30 lists key contractual provisions for all ICT services contracts and additional provisions for those supporting critical or important functions, including service level descriptions, notice periods, participation in the financial entity's security awareness and resilience testing, and exit strategies. Organizations outside these sectors can still use these lists as well-tested checklists.

Managers should also expect that some critical suppliers will offer only standard, non-negotiable terms. OSFI B-10 acknowledges this reality: it recognizes that standardized contracts leave limited room to negotiate and that not every arrangement will have a tailored contract, and it expects the resulting risks to be managed through other means, such as monitoring, business continuity, and contingency planning. The same discipline applies elsewhere: when the organization cannot negotiate, it should document the gap between its requirements and the supplier's terms, identify compensating controls (for example, stronger internal monitoring, encryption under its own keys, or a more developed exit plan), and have the residual risk accepted by the accountable executive. Documented acceptance is what turns an unenforceable gap into an owned decision.

## Onboarding, continuous monitoring, and ongoing assurance: recognizing that suppliers change

Third-party risk is dynamic. Suppliers add features, change architectures, restructure teams, replace subcontractors, migrate to new hosting platforms, suffer incidents, and are acquired. The organization's own use of a supplier also changes: more data, more users, more integrations. A one-time due diligence snapshot cannot provide ongoing confidence. CSF 2.0 (GV.SC-07) calls for supplier risks to be understood, recorded, prioritized, assessed, responded to, and monitored over the course of the relationship, and ISO/IEC 27002:2022 control 5.22 addresses monitoring, review, and change management of supplier services.

### Onboarding as a control point

Onboarding is where contractual promises meet technical reality, and it is a point of elevated risk. Accounts are created, credentials and API keys are exchanged—sometimes by email—network paths are opened, and data migrations begin. Governance should require that onboarding for Tier 1 and Tier 2 suppliers include verification of the controls the contract depends on before production access is granted: single sign-on and MFA configured, privileged accounts assigned to named individuals, API keys scoped to least privilege and stored in a secrets manager, logging enabled and forwarded to the organization's monitoring, and data flows documented. This is the integration side of the supplier relationship, and it belongs to the organization, not the supplier (see Chapter 10 for the identity controls involved).

### Governance monitoring and technical monitoring

Continuous monitoring can be understood in two layers. The first is **governance monitoring**: periodic refresh of key evidence (updated assurance reports, certificate renewals, policy changes), review of service performance and incident history, confirmation of subcontractor lists, tracking of remediation commitments, and renewal assessments that revisit the tier. Its cadence follows the tier, as shown in the sample tiering table earlier in the chapter.

The second is **technical monitoring**. This may include external attack surface monitoring and security ratings, threat intelligence on supplier compromises, monitoring of integration points (API behaviour, authentication patterns, privileged sessions), and alerting on abnormal data transfers. Security ratings, which score organizations on externally observable indicators, should be treated cautiously: they detect obvious problems such as exposed services and scale well, but they see only the external surface, may misattribute assets, and say nothing about internal controls. They are a useful alarm, not a substitute for evidence-based assurance.

Often the most valuable monitoring is the organization's own telemetry: provider accounts authenticating from new locations, an integration downloading far more records than usual, or a support account used outside agreed maintenance windows.

### Triggers, indicators, and escalation

Monitoring becomes governance when it is connected to decisions. Programs should define **reassessment triggers**: events that move a supplier back into review regardless of the calendar. Typical triggers include a disclosed incident at the supplier, a change of ownership or headquarters, a material change in the services or data involved, the loss or qualification of an assurance report, a significant deterioration in external monitoring results, public exploitation of a vulnerability in the supplier's product, and repeated failure to meet remediation commitments.

At the portfolio level, a small set of key risk indicators allows executives and boards to see whether the program is working. Useful indicators include the percentage of active suppliers with an assigned tier and owner; the percentage of Tier 1 and Tier 2 suppliers with current assurance evidence; the number of open exceptions and how many are past their expiry; the median age of open remediation items at critical suppliers; the number of critical suppliers without a tested exit plan; and the number of supplier-originated incidents in the period. Each indicator should have a tolerance threshold linked to risk appetite, so that a breach of tolerance prompts a decision rather than a footnote in a report.

A mature program integrates these outputs into broader risk reporting. If a critical supplier's risk posture degrades, executives should receive a clear recommendation among a defined set of options: accept with a remediation plan and deadline, apply compensating controls, enforce contractual remedies, restrict the scope of the relationship, or begin transition planning. The value of the program lies in making that choice explicit, timely, and owned.

## Concentration risk: single-provider dependency as a modern governance problem

One of the most under-governed aspects of third-party cybersecurity is **concentration risk**. Even highly secure suppliers can create fragile dependency. If a major cloud provider, identity provider, endpoint security vendor, or managed service provider experiences an outage or compromise, the organization may lose critical capabilities regardless of the quality of its internal controls. Concentration risk is therefore a question of systemic dependency, not vendor insecurity: reliance on a single provider for a critical function, at a scale where switching is slow or impractical. Chapter 15 discusses the cloud-specific aspects, such as multi-region design and logically isolated backups; this section treats concentration across all external dependencies, cloud and otherwise.

### Recognizing concentration at three levels

Concentration appears at three levels, and governance should examine each. **Organizational concentration** arises when one supplier supports many of the organization's critical services, so that its failure produces simultaneous disruption across the business—for instance, when identity, email, collaboration, document storage, and endpoint management all depend on a single platform. **Fourth-party concentration** arises when apparently independent suppliers rely on the same upstream provider, so that diversification at the direct-supplier level is illusory. **Sector or systemic concentration** arises when many organizations in an industry or economy depend on the same few providers, so that a single failure produces correlated harm across peers, customers, and counterparties at the same moment.

Systemic concentration is the reason regulators have intervened. DORA established an EU-level oversight framework for **critical ICT third-party providers** (CTPPs). In November 2025, the European Supervisory Authorities published the first list of designated CTPPs—nineteen providers, mainly large cloud, platform, and technology firms—based in part on the registers of information submitted by financial entities. Each designated provider is assigned a lead overseer and becomes subject to direct supervisory examination, with oversight activity beginning in 2026. OSFI B-10 likewise asks federally regulated financial institutions to assess concentration risk across several dimensions, including geography, supplier, and subcontractor, and to consider both their own overreliance on a provider and systemic risk across institutions.

### The July 2024 lesson

The global IT outage of July 19, 2024 illustrated concentration risk without any malicious actor. A defective content update to CrowdStrike's Falcon sensor caused Windows systems running it to crash; Microsoft estimated that about 8.5 million Windows devices were affected. Airlines, hospitals, broadcasters, banks, and retailers in many countries were disrupted on the same day, and recovery for many organizations required hands-on remediation of individual machines. The supplier was a well-regarded security vendor, chosen precisely to reduce risk, and its product required deep, privileged integration with the operating system to do its job.

The governance lessons are not about that particular vendor. They concern questions that apply to any supplier with privileged, automatic update channels into critical systems. Does the organization know which suppliers can push changes into its production environment without its own approval? Does it understand the supplier's update testing and staged rollout practices, and does the contract give it any control over update timing? Do its business continuity plans include a scenario in which a large share of endpoints fail simultaneously, and does recovery depend on tools or people that would themselves be affected? Chapter 12 discusses resilience planning in depth; the point here is that third-party governance must feed the business impact analysis with realistic dependency information.

### Measuring and treating concentration

Concentration cannot be managed without being made visible. The third-party register, enriched with information about which critical services each supplier supports and which material fourth parties each supplier relies on, allows the organization to produce a simple **dependency map**: for each critical service, the chain of external providers on which it depends. Even a rough map typically reveals surprises, such as several SaaS suppliers hosted in the same cloud region.

Governance responses vary by criticality and feasibility. In some cases, multi-region design within a single provider and robust continuity planning are sufficient. In others, multi-provider strategies or architectural decoupling are warranted, particularly for identity, critical communications, and backup storage, where a single compromise or outage would otherwise disable recovery itself. Keeping immutable backups and a minimal out-of-band communication capability outside the primary provider's ecosystem is a common and proportionate measure. In still other cases, the organization may decide to accept concentration because the benefits of a single integrated platform outweigh the risk. That is a legitimate choice, but it should be made explicitly, documented as a **strategic dependency** with board-level awareness, and supported by contingency plans, rather than discovered during an outage.

| Concentration risk is often the reason recovery plans fail: when the bottleneck is an external dependency, the organization cannot restore what it does not control. |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## Fourth-party risk and the software supply chain: managing transitive exposure

Fourth-party risk and software supply-chain risk are the most practical expressions of supply-chain cybersecurity. They concern the external services and components that suppliers use to deliver what the organization buys: subcontractors, hosting providers, software libraries, open-source packages, build systems, and specialized service partners. The organization cannot audit every fourth party, and it cannot inspect every line of code in every product it uses. Governance must therefore rely on scalable mechanisms that bound, monitor, and contractually manage transitive exposure rather than attempting to eliminate it.

### Managing fourth parties through contract, architecture, and evidence

Three mechanisms make fourth-party risk manageable. The first is **contractual**: requiring suppliers to impose equivalent obligations on subcontractors, to maintain and share a list of material subcontractors, to notify the organization of changes, and to remain accountable for subcontracted services. The second is **architectural**: limiting what any supplier can expose, through encryption under keys the organization controls, data minimization, segmentation, and least-privilege integration, so that even if a fourth party is compromised the blast radius is limited. The third is **evidence-driven**: examining how suppliers govern their own suppliers, including how assurance reports treat carved-out subservice organizations and whether the supplier has a supplier risk program of its own.

### Why software supply chains became a board-level issue

Software supply-chain attacks target the process by which software is built and distributed, so that a single compromise reaches every downstream user. The compromise of the SolarWinds Orion build process, disclosed in December 2020, inserted malicious code into signed updates distributed to thousands of customers, including government agencies. The Kaseya VSA incident of July 2021 used a remote management product relied on by managed service providers to deliver ransomware to those providers' own clients. In March 2024, a backdoor deliberately introduced into the open-source XZ Utils compression library, by a contributor who had spent a long period gaining the maintainers' trust, was discovered shortly before it could reach widely used Linux distributions. Each case exploited trust: customers accepted updates and components because they came from legitimate, trusted sources.

For managers, the lesson is that the security of purchased and open-source software depends on the supplier's development and build practices as much as on the product's features. Modern applications are assembled largely from third-party and open-source components, which means every organization that builds or buys software is a participant in a software supply chain whether or not it thinks of itself that way. Governance cannot evaluate code directly, but it can demand evidence of disciplined practice and transparency about components (Chapter 13 addresses the engineering side of secure development).

### What to demand: SSDF, SBOM, and SLSA

Three references give managers a vocabulary for those demands. The **Secure Software Development Framework (SSDF)**, NIST SP 800-218 version 1.1 (February 2022), describes high-level secure development practices grouped into preparing the organization, protecting the software, producing well-secured software, and responding to vulnerabilities. It underpins the U.S. federal secure software development attestation that software producers selling to U.S. agencies must provide. NIST released an initial public draft of SP 800-218 Rev. 1, SSDF version 1.2, in December 2025; at the time of writing it had not been finalized, so organizations should reference version 1.1 in contracts while monitoring the revision.

A **software bill of materials (SBOM)** is a formal, machine-readable inventory of the components in a piece of software and their relationships. Its governance value is speed: when a vulnerability is announced in a widely used component, an organization holding SBOMs for its critical software can determine exposure in hours rather than weeks. The minimum content expected of an SBOM was first defined by the U.S. NTIA in 2021. In 2026, CISA, together with U.S. and international partner agencies, published updated *Minimum Elements for a Software Bill of Materials*, following a public comment draft in August 2025. Among other changes, the update expects an SBOM to record component hashes and licences and to identify both the tool that produced it and the circumstances of its production; some of the 2021 field names were also changed. Managers need not master the formats; they should ensure that critical software contracts require SBOMs meeting current minimum elements, that the organization has the capability to ingest and query them, and that SBOMs are paired with vulnerability exploitability information so that teams are not overwhelmed by components that are present but not exploitable.

**Supply-chain Levels for Software Artifacts (SLSA)**, an OpenSSF specification, defines incremental levels of assurance about how software artifacts are built and how their provenance can be verified. Version 1.0 (2023) established the Build track; version 1.2, released in November 2025, added a Source track concerned with how source code is written, reviewed, and controlled before it is built. For a buyer, SLSA provides a way to ask a supplier how confident it can be that the software it ships was built from the intended source, by the intended process, without tampering.

The figure below positions these elements along the software supply chain and alongside the fourth-party services on which suppliers depend.

<span id="_Toc243635579" class="anchor"></span>Figure 80: Governing the software supply chain and fourth parties

### The regulatory shift: the EU Cyber Resilience Act

Regulation is moving responsibility upstream to manufacturers. The EU **Cyber Resilience Act (CRA)**, Regulation (EU) 2024/2847, entered into force on December 10, 2024. It requires manufacturers of products with digital elements placed on the EU market to meet essential cybersecurity requirements across the product lifecycle, including secure-by-design development, vulnerability handling, security updates for a defined support period, and technical documentation that includes an SBOM. Its reporting obligations have applied since September 11, 2026: manufacturers must notify actively exploited vulnerabilities and severe incidents to the designated national CSIRT and ENISA, with an early warning within 24 hours and a fuller notification within 72 hours. Most other obligations apply from December 11, 2027.

For Canadian organizations, the CRA matters in two ways. Those that manufacture software or connected products sold in the EU are directly subject to it. Those that only buy will benefit indirectly, because suppliers serving the EU market will increasingly be able to supply SBOMs, vulnerability handling commitments, and support-period statements as a matter of course. Procurement teams should ask for these artifacts even from suppliers not legally required to provide them; the CRA is rapidly turning them into market norms. The Canadian Centre for Cyber Security's guidance on assessing cyber supply-chain risk (ITSAP.10.070) offers a complementary, Canada-specific starting point for organizations building their programs.

## Incident coordination: when the breach happens somewhere else

When an incident originates at a supplier, the organization faces a peculiar governance problem: it is accountable for the consequences but controls neither the investigation nor much of the information. Preparation therefore matters more than usual, because the levers available during the event are mostly those established before it. Chapter 12 covers incident management and continuity in general; this section addresses what is distinctive about supplier-originated events.

### Preparing for supplier-originated incidents

Several preparations distinguish organizations that handle supplier incidents well. They maintain current security contacts for every Tier 1 and Tier 2 supplier, tested at least annually. They have incident playbooks for the main supplier scenarios—supplier data breach, supplier ransomware causing outage, compromised supplier credentials or update channel, and mass exploitation of a supplier product—with predefined decisions such as when to disconnect an integration. They know, from the register, which data each supplier holds, so that legal and privacy teams can assess notification obligations quickly. And they include critical suppliers in tabletop exercises, as CSF 2.0 GV.SC-08 recommends, so that both sides understand roles before they are tested.

The decision to suspend or disconnect a supplier integration deserves advance thought. Disconnecting may contain the threat but also stop a business process. The authority to make that call, the criteria for doing so, and the business continuity measures that take over should be agreed in advance with the business owner, not debated at two o'clock in the morning.

### Illustrative scenario: a file-transfer compromise at a Canadian retailer

Consider a fictional mid-sized Canadian retailer, Nordet Outfitters, headquartered in Montréal, with stores in four provinces. Among its many suppliers is a marketing analytics firm. The firm receives a weekly extract of loyalty members' names, email addresses, postal codes, and purchase histories through a managed file-transfer product that the firm hosts itself. Nordet assessed the firm three years ago as Tier 3: a questionnaire was completed, no independent report was requested, and the contract's security schedule was the supplier's standard terms. Nobody recorded which file-transfer product the firm used.

On a Friday in late spring, a criminal group begins mass exploitation of a previously unknown vulnerability in that file-transfer product, stealing data from every internet-facing installation it can find before a patch is available—the pattern seen in the MOVEit Transfer campaign of 2023, which CISA and the FBI attributed to the CL0P ransomware group. The marketing firm learns of the vulnerability from the software maker's advisory on the Tuesday, applies the patch, and begins an investigation. It notifies Nordet twelve days later, after confirming that files were taken, because its standard terms require notification only of confirmed breaches "without undue delay." By then, the group has published Nordet's name on its leak site, and a journalist has called the retailer's communications team.

Nordet's response exposes each gap in turn. The incident team cannot immediately say how many loyalty members are in the extracts, because the register does not record data volumes for Tier 3 suppliers. Legal counsel must determine whether the incident presents a risk of serious injury under Law 25 and a real risk of significant harm under PIPEDA, and must notify the Commission d'accès à l'information, the Office of the Privacy Commissioner of Canada, and affected individuals as required, but the supplier's investigation report will not arrive for weeks. The contract gives Nordet no right to the forensic findings and caps the supplier's liability at a year of fees, a small fraction of the expected cost of notification, call centre support, and credit monitoring. Meanwhile, Nordet's CISO asks the obvious question—which other suppliers use the same product?—and discovers that nobody knows. Questionnaires go out to sixty suppliers; responses trickle in over three weeks.

When the board reviews the incident, the lesson it draws is not that the marketing firm was negligent; the vulnerability was unknown and exploited before any patch existed. The lesson is that Nordet's governance treated the relationship as low risk because the firm was small and its service non-critical, while ignoring the data sensitivity driver entirely: the firm held personal information on several hundred thousand customers. The board approves four changes. Every supplier holding customer personal information above a volume threshold is moved to at least Tier 2, with independent evidence, contractual notification of suspected incidents within a defined number of hours, rights to investigation findings, and liability terms reviewed by the risk committee. The register will record material fourth parties and key software products for Tier 1 and Tier 2 suppliers, so that the next mass-exploitation event can be scoped in hours. Data minimization is applied to the weekly extract, which is reduced to pseudonymized identifiers and purchase categories, shrinking the blast radius of any future compromise. And a KRI tracking the percentage of personal-information suppliers with current evidence is added to quarterly reporting with a tolerance threshold.

### Lessons from mass exploitation events

In the MOVEit Transfer case, a single SQL injection vulnerability (CVE-2023-34362) in a widely used product led to data theft affecting a very large number of organizations, many of which were harmed not because they ran the software themselves, but because a payroll provider, pension administrator, or other service provider did. Three governance lessons generalize. First, **data sensitivity can make a supplier critical even when its service is not**, and tiering models that weight only operational criticality will misclassify such suppliers. Second, notification clauses tied to confirmed breaches are too slow for mass-exploitation events; suspected compromise is the right trigger. Third, knowing which products and fourth parties sit behind the supplier relationship is what allows an organization to scope its exposure quickly when the next widely used product is exploited.

## Exit and transition planning: preserving the ability to leave

Exit planning is a cybersecurity control. If the organization cannot transition away from a supplier without severe disruption, it has created a strategic lock-in that increases risk and reduces its ability to enforce every other requirement in this chapter. In governance terms, the absence of an exit option is a loss of autonomy, and it should be recognized as such.

### Why exit is a governance requirement

For critical suppliers, governance should require at least a documented exit strategy even if it is unlikely to be executed soon. This is not pessimism but ordinary resilience planning. Suppliers can fail financially, be sanctioned, be acquired by a competitor or by an owner in a jurisdiction the organization cannot accept, change pricing drastically, withdraw a product, or suffer prolonged outages or compromises. If the organization has no transition plan, it is forced into decisions made under pressure, with weak bargaining power and incomplete information. Regulators have made the same point: OSFI B-10 expects exit planning for third-party arrangements, and DORA requires financial entities to have exit strategies for ICT services supporting critical or important functions. CSF 2.0 (GV.SC-10) frames post-relationship activities as part of the C-SCRM outcomes. For cloud services specifically, Chapter 15 describes the exit strategy expected under ISO/IEC 27002:2022 control 5.23 and the switching rights created by the EU Data Act.

The level of detail should match the tier: a Tier 1 plan identifies alternatives, migration time and cost, and interim arrangements, while lower tiers usually rely on standard contract clauses for data return and deletion.

### Elements of a credible exit plan

A credible exit plan addresses several concrete elements:

- **Data portability** specifies what data will be returned, in what format, how often exports can be tested during the relationship, and whether the format is usable without the supplier's software.

- **Transition assistance** defines what help the supplier must provide during migration, for how long, and at what cost, as well as continuity obligations during the notice period.

- **Access revocation** lists every account, credential, API key, certificate, network path, and federation trust that must be removed, and assigns responsibility for confirming removal.

- **Continuity arrangements** describe how critical services will be maintained during the transition and what interim manual or alternative processes exist.

- **Residual data and records** cover backups, logs, and archived copies held by the supplier and its subcontractors, retention requirements for incident and audit records, and legal holds.

Testing is what distinguishes a plan from a hope. Periodically exporting a sample of data and confirming that it can be restored or loaded elsewhere is inexpensive and frequently reveals problems—proprietary formats, missing metadata, or volumes that would take weeks to transfer—that are far better discovered in advance.

### Evidence at termination

Exit planning also intersects with privacy and data protection obligations (see Chapter 16). A termination is often the moment when data return and destruction must be demonstrable, and when residual access is most likely to be forgotten. Termination checklists should therefore include evidence requirements: certificates or attestations of destruction where appropriate, confirmation of deletion from backups as backup cycles permit, written confirmation that subcontractors have also deleted data, and records showing that all access paths have been revoked. Accounts and integrations left behind after a relationship ends are a recurring source of breaches, because nobody monitors access that is supposed to no longer exist. The third-party register should keep terminated relationships visible until the exit evidence is complete.

## Operating model: who owns third-party cybersecurity governance

Because third-party cybersecurity cuts across functions, organizations often fail by assuming it is someone else's job. Procurement believes security reviews suppliers; security believes procurement controls the process; legal believes the business has accepted the terms; the business believes all three have taken care of it. Mature governance defines a clear operating model with explicit decision rights, as CSF 2.0 (GV.SC-02) recommends for C-SCRM roles and responsibilities, and as COBIT 2019 does through its management objective for managed vendors (APO10).

### Roles and decision rights

The operating model allocates responsibilities along the lines below. The labels will differ across organizations, but each responsibility must be owned by someone.

- **Business owners own the relationship and the risk acceptance.** They select suppliers aligned with business needs, provide accurate intake information for tiering, and explicitly accept residual risk when requirements cannot be met. Accountability for third-party risk sits with the business owner, not with the security team.

- **Procurement owns process discipline.** It ensures that security requirements appear early, that review gates are respected, that tier-appropriate templates are used, and that purchases outside the process are detected and brought into it.

- **Legal owns enforceability.** It translates requirements into contract language, manages audit rights, notification, liability, and termination clauses, and maintains the standard security schedules.

- **Privacy leadership owns personal information governance.** It ensures that data handling, residency, retention, and cross-border transfer requirements are met and that privacy impact assessments are carried out when required, as Law 25 does for communications outside Québec.

- **The CISO function owns security requirements and assurance.** It defines control baselines by tier, evaluates supplier evidence, designs integration controls, monitors critical suppliers, and advises business owners on residual risk.

- **Business continuity and resilience functions own dependency planning.** They incorporate supplier criticality, substitutability, and concentration into business impact analyses and recovery plans.

### Three lines and independent challenge

The three lines model clarifies how oversight works. Business owners, procurement, and operational security teams form the first line: they perform the lifecycle activities and own the risks. Risk management and compliance functions in the second line set the framework, define tiering and appetite linkages, monitor indicators, and challenge exceptions. Internal audit, as the third line, provides independent assurance that the program operates as designed—for instance, by sampling suppliers to check that their tier is correct, that required evidence was obtained and reviewed, that exceptions were approved and have not expired, and that exit evidence exists for terminated relationships. Board risk or audit committees receive reporting on the program and on critical supplier risks, and they should periodically examine the dependency map and the list of strategic dependencies accepted on the organization's behalf. Chapter 21 returns to the broader question of how accountability mechanisms are designed across the organization.

### Questions managers and boards should ask

The following questions help executives and directors test whether third-party cybersecurity governance is real or merely documented:

- **Inventory:** Do we have a complete register of third parties, and what percentage of active suppliers have an assigned tier, owner, and assessment status?

- **Critical dependencies:** Which suppliers, if they failed or were compromised tomorrow, would stop our critical services or expose our most sensitive data, and are they all in our highest tier?

- **Tiering logic:** Does our tiering model weight data sensitivity as heavily as operational criticality, so that small suppliers holding large volumes of personal information are not underestimated?

- **Evidence quality:** For our critical suppliers, are we relying on independent evidence whose scope covers the services we use, and are we performing the complementary controls those reports assume?

- **Contract strength:** Do our critical contracts require notification of suspected incidents within timelines compatible with our own regulatory deadlines, and do they give us rights to the information we would need?

- **Accepted gaps:** Which material gaps in supplier security or contract terms have we accepted, who accepted them, what compensating controls apply, and when do those acceptances expire?

- **Privileged channels:** Which suppliers can access our environment with administrative privileges or push updates into production without our approval, and how is that access secured and monitored?

- **Concentration:** Where are we dependent on a single provider, directly or through our suppliers' suppliers, and has that dependency been accepted as a strategic decision with contingency plans?

- **Software supply chain:** For the software that runs our critical services, can we determine within a day whether we are exposed to a newly announced vulnerability in a widely used component?

- **Exit:** For each critical supplier, do we have a documented exit strategy, and have we tested that our data can be exported and used elsewhere?

- **Incident readiness:** Have our critical suppliers participated in an incident exercise with us in the past year, and do we know who would decide to disconnect a compromised integration?

- **Assurance over the program:** When did internal audit last test the third-party program, and what did it find?

| Accountability for third-party risk belongs to the business owner of the relationship; security, procurement, and legal enable the decision, but they cannot accept the risk on the business's behalf. |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## Bringing it together: making third-party cybersecurity governance enforceable and auditable

Third-party and supply-chain governance converts cybersecurity intent into obligations that can be enforced against someone else. Over the course of this chapter, risk appetite became tiered minimum requirements, requirements became RFP criteria and contract clauses, assurance moved from questionnaires toward independent evidence, and resilience came to depend on knowing the dependency chain and keeping a credible way out.

The reference landscape supports this model. ISO/IEC 27036 and NIST SP 800-161 Rev. 1 describe how to build the program; NIST CSF 2.0's GV.SC category states the outcomes governance should achieve; ISO/IEC 27001:2022 integrates supplier controls into the ISMS; the SSDF, SBOM minimum elements, and SLSA provide a vocabulary for software assurance; and OSFI B-10, DORA, and the EU Cyber Resilience Act show where supervisors and legislators are taking the field. None of these references replaces judgment. Each helps make judgment structured, consistent, and defensible.

The managerial objective is not to eliminate third-party risk, which is impossible in modern ecosystems. The objective is to make it **governable**: to know which relationships matter most, to impose proportionate requirements, to obtain credible assurance beyond self-attestation, to monitor continuously and act on what monitoring reveals, to manage concentration and fourth-party risk consciously, to prepare for incidents that originate elsewhere, and to retain the ability to leave when dependency becomes unacceptable. When these practices are in place, the organization's cybersecurity posture becomes less fragile at its boundaries and more defensible to regulators, customers, and stakeholders. Most importantly, it becomes more resilient in a world where critical systems increasingly sit outside the organization's direct control. The next chapter turns to cryptography and key management, one of the few technical levers that allows an organization to retain control over its data even when that data resides with third parties.

## Chapter summary

Cybersecurity governance increasingly succeeds or fails at organizational boundaries, because cloud platforms, managed service providers, software components, and specialized vendors can affect critical services and sensitive information as much as internal systems can. This chapter argues that third-party risk is not a procurement formality but the discipline that turns cybersecurity expectations into enforceable obligations. It presents a full lifecycle: planning and tiering, due diligence, procurement requirements, contracting, onboarding and monitoring, concentration and fourth-party risk, software supply-chain assurance, incident coordination, and exit. The goal is not to eliminate third-party risk, which is impossible, but to make it governable, evidenced, and defensible, with accountability resting on the business owner of each relationship.

### Key points to retain

- A third party is any external entity whose failure or compromise could materially affect the organization's information, privacy obligations, or critical services, regardless of contract type.

- Fourth-party risk arises from suppliers' own suppliers and can conceal concentration behind an apparently diversified supplier base.

- The relationship lifecycle has seven stages, and each needs gates where a defined person makes a recorded decision on defined evidence, avoiding the accept-and-forget failure.

- Tiering by data sensitivity, service criticality, access and integration, and substitutability makes effort proportionate, and each tier must carry its own controls, evidence, approvals, and monitoring cadence.

- Questionnaires are screening tools that measure statements; independent evidence such as SOC 2 Type II reports must be read for scope, period, exceptions, carve-outs, and complementary user entity controls.

- Security requirements and a short list of disqualifiers derived from risk appetite should be set in the RFP, before selection, when the organization still has leverage.

- Contracts make governance legally real through security schedules, audit rights, suspected-incident notification aligned with regulatory deadlines, subcontractor controls, data handling terms, and meaningful remedies.

- Onboarding gates, governance and technical monitoring, reassessment triggers, and portfolio KRIs with tolerance thresholds keep assurance current as suppliers change.

- Concentration risk is systemic dependency rather than vendor insecurity, as the July 2024 outage showed, and accepted concentration must be an explicit, documented strategic decision.

- Software supply-chain assurance relies on demanding SSDF practices, SBOMs meeting current minimum elements, and SLSA provenance, while the EU Cyber Resilience Act moves obligations upstream to manufacturers.

- Supplier-originated incidents and exits must be prepared in advance through contacts, playbooks, pre-agreed disconnection authority, tested data exports, and evidence of access revocation and data destruction.

- Business owners own and accept third-party risk; procurement, legal, privacy, the CISO function, and continuity enable the decision, while the second and third lines challenge and assure it.

### Key terms

- **Third party** — Any external entity whose services, products, or operations can materially affect the confidentiality, integrity, or availability of the organization's information, its privacy obligations, or the continuity of its critical services.

- **Fourth-party risk** — Risk arising from the vendors on which the organization's suppliers depend, usually without any direct contract or visibility.

- **C-SCRM** — Cybersecurity supply chain risk management, NIST's term for managing upstream dependencies and transitive exposure in products and services.

- **Accept and forget** — The failure mode in which a supplier's risk is assessed at onboarding and then disappears from governance view until something goes wrong.

- **Lifecycle gate** — A point in the supplier relationship at which a defined person must make a recorded decision based on defined evidence.

- **Risk tiering** — Classifying third-party relationships by inherent risk so that due diligence, evidence, approvals, and monitoring are proportionate.

- **Third-party register** — A living inventory recording each supplier relationship, its owner, tier, data and systems involved, key terms, material subcontractors, and assessment status.

- **SOC 2 Type II report** — An independent auditor's report against the Trust Services Criteria that tests whether controls operated effectively over a period, typically six to twelve months.

- **Complementary user entity controls** — Controls a supplier assumes its customers will perform, without which the assurance in the supplier's report does not hold for the customer.

- **Bridge letter** — A supplier's management assertion that nothing material has changed since an assurance report's period ended; useful but not audit evidence.

- **Concentration risk** — Fragile dependency on a single provider for critical functions at organizational, fourth-party, or sector level, where switching is slow or impractical.

- **Software bill of materials (SBOM)** — A formal, machine-readable inventory of the components in a piece of software and their relationships, used to determine exposure to component vulnerabilities quickly.

- **SLSA** — Supply-chain Levels for Software Artifacts, an OpenSSF specification defining levels of assurance about how software is built and how its provenance can be verified.

- **Exit plan** — A documented, tested strategy for leaving a supplier that covers data portability, transition assistance, access revocation, continuity, and residual data.

## Review questions

### Essay questions

1.  Nordet Outfitters classified its marketing analytics supplier as Tier 3. Using the four drivers of inherent risk, analyze why this classification failed, and evaluate whether the four changes approved by the board would have materially reduced the impact of the file-transfer compromise.

2.  Many organizations rely mainly on security questionnaires for supplier assurance. Critically evaluate this practice and propose a proportionate assurance approach for an organization with three hundred suppliers and two analysts, explaining how independent reports should be read.

3.  Choose an organization you know (or a fictional one) and identify one critical supplier. Design the key contractual provisions, onboarding gates, monitoring triggers, and exit plan elements you would require for that relationship, justifying each by reference to the chapter.

4.  Concentration risk is described as a question of systemic dependency rather than vendor insecurity. Using the July 2024 outage and the DORA oversight framework, analyze the three levels of concentration and recommend how a board should decide whether to accept or reduce a concentrated dependency.

5.  Evaluate the claim that accountability for third-party risk belongs to the business owner rather than to the security team. In your answer, explain the roles of procurement, legal, privacy, the CISO function, and the three lines, and identify what goes wrong when ownership is unclear.

### Multiple-choice questions

Choose the best answer. Answers are listed in the appendix "Answers to multiple-choice questions".

**1.** According to the chapter, what determines whether an external entity belongs inside the third-party cybersecurity governance program?

> A. Whether the entity has a signed outsourcing contract with the organization
>
> B. Whether its failure or compromise could materially harm critical services or sensitive information
>
> C. Whether the entity has persistent access to the corporate network
>
> D. Whether the entity's annual contract value exceeds a defined procurement threshold

**2.** A SaaS provider used by a retailer runs on a hyperscale cloud and uses an outsourced support desk. From the retailer's perspective, what are the cloud and the support desk?

> A. Second-line oversight functions
>
> B. Complementary user entity controls
>
> C. Critical ICT third-party providers
>
> D. Fourth parties

**3.** Which NIST CSF 2.0 category places cybersecurity supply chain risk management within the Govern function?

> A. GV.SC, with outcomes from program establishment to post-relationship activities
>
> B. PR.AA, which addresses identity management and access control
>
> C. DE.CM, which addresses continuous monitoring of assets
>
> D. RS.MA, which addresses incident management activities

**4.** A Tier 1 supplier's controls cannot meet the organization's standard. Who, according to the chapter, should accept the resulting residual risk?

> A. The CISO, who evaluated the supplier's security evidence
>
> B. The procurement officer who managed the selection process
>
> C. The business owner of the relationship, as an accountable risk owner
>
> D. Legal counsel, who negotiated the contract's security schedule

**5.** What turns a third-party lifecycle diagram into governance, according to the chapter?

> A. Gates where a defined person makes a recorded decision based on defined evidence
>
> B. A single detailed security questionnaire completed during procurement
>
> C. Automatic renewal clauses that keep relationships stable over time
>
> D. An annual compilation of all suppliers sent to the audit committee

**6.** A small marketing firm with no system access receives weekly extracts of several hundred thousand customers' personal information. Which tiering driver should most clearly raise its tier?

> A. Service criticality, because marketing supports revenue
>
> B. Access and integration, because files are exchanged
>
> C. Data sensitivity, because of the volume of personal information
>
> D. Substitutability, because analytics firms are rare

**7.** A cloud identity provider presents excellent independent evidence of strong controls. How should this affect its classification?

> A. It should move to Tier 2, because controls reduce inherent risk
>
> B. It stays Tier 1; the evidence improves the residual risk rating
>
> C. It should move to Tier 3, with reassessment only at renewal
>
> D. It should be removed from the register as a trusted provider

**8.** Which indicator does the chapter suggest tells a board most about the real coverage of a third-party program?

> A. The total number of security questionnaires completed this year
>
> B. The number of suppliers holding an ISO/IEC 27001 certificate
>
> C. The average security rating score across all active suppliers
>
> D. The percentage of active suppliers with an assigned tier and owner

**9.** Why is a SOC 2 Type II report generally more informative than a Type I report?

> A. It is issued under ISAE 3402 rather than AICPA attestation standards
>
> B. It covers financial reporting controls as well as security controls
>
> C. It tests whether controls operated effectively over a period of time
>
> D. It replaces the need to review carved-out subservice organizations

**10.** A supplier's SOC 2 report assumes customers will review their own access logs. The organization does not do so. What follows?

> A. The report's assurance does not fully hold for this organization
>
> B. The supplier's auditor must reissue the report with a qualification
>
> C. The report's scope automatically excludes the organization's users
>
> D. The supplier becomes responsible for reviewing the customer's logs

**11.** Under the EU Cyber Resilience Act reporting obligations applying since September 11, 2026, how quickly must manufacturers issue an early warning of an actively exploited vulnerability?

> A. Within 72 hours, followed by a final report within thirty days
>
> B. Without undue delay once the breach has been confirmed
>
> C. Within seven days of a patch becoming available
>
> D. Within 24 hours, followed by a fuller notification within 72 hours

**12.** What does the chapter identify as the step that distinguishes an exit plan from a hope?

> A. Obtaining a bridge letter from the supplier each year
>
> B. Periodically testing that exported data can be loaded elsewhere
>
> C. Including a termination-for-convenience clause in the contract
>
> D. Listing alternative suppliers in the third-party register

**13.** An organization's appetite does not permit critical suppliers lacking any independent assurance report. How does the chapter recommend expressing this in procurement?

> A. As a disqualifying criterion stated in the RFP before selection
>
> B. As a weighted scoring item worth five percent of the evaluation
>
> C. As a negotiation point raised after the preferred vendor is chosen
>
> D. As a remediation commitment written into the first renewal

**14.** What is the main governance value of holding SBOMs for critical software?

> A. They prove the software was built without tampering from source
>
> B. They replace the need for vulnerability exploitability information
>
> C. They certify that the supplier follows the SSDF practices
>
> D. They let the organization determine exposure to component vulnerabilities in hours

**15.** A Canadian organization's supplier contract allows thirty days to report confirmed breaches. Why does the chapter consider such a clause inadequate?

> A. Because contracts should never specify notification timelines
>
> B. Because only regulators may set supplier notification deadlines
>
> C. Because the organization cannot meet its own regulatory deadlines and suspected compromises go unreported
>
> D. Because notification clauses are unenforceable against cloud providers

**16.** A critical SaaS provider offers only standard, non-negotiable terms that fall short of requirements. What does the chapter recommend?

> A. Refuse the provider, since unnegotiated contracts are never acceptable
>
> B. Document the gap, apply compensating controls, and obtain executive risk acceptance
>
> C. Accept the terms silently, because large providers are generally secure
>
> D. Ask the security team to waive the requirement on the business's behalf

**17.** A supplier could expose millions of customer records, yet its contract caps liability at twelve months of fees. How should management view this?

> A. As standard practice that requires no executive attention
>
> B. As potentially inadequate for the risk, requiring executive attention
>
> C. As acceptable, because service credits will cover breach costs
>
> D. As irrelevant, because liability caps do not affect security

**18.** What does the chapter identify as the main limitation of security ratings for supplier monitoring?

> A. They see only the external surface and say nothing about internal controls
>
> B. They require on-site audits that suppliers rarely permit
>
> C. They are only available for suppliers holding SOC 2 reports
>
> D. They are updated too rarely to detect exposed services

**19.** Which governance question does the chapter draw from the July 2024 global IT outage caused by a defective security software update?

> A. Whether the organization should avoid well-regarded security vendors
>
> B. Whether endpoint security products are necessary on Windows systems
>
> C. Whether cyber insurance policies cover outages caused by vendors
>
> D. Which suppliers can push changes into production without the organization's approval

**20.** Why did Nordet's board conclude that its governance, rather than the marketing firm's negligence, was the main problem?

> A. Because the firm had refused to apply the vendor's patch
>
> B. Because Nordet had no incident response team in place
>
> C. Because tiering ignored data sensitivity, leaving weak evidence and contract terms
>
> D. Because the file-transfer product had no known vulnerabilities
