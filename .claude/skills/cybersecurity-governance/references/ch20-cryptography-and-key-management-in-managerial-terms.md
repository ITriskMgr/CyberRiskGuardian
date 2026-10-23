<!-- Léger, M.-A. (2026). Introduction to Cybersecurity Governance for Business Technology Management (3rd ed., v2.51). Chapter 20: Cryptography and Key Management in Managerial Terms. CC BY-NC 4.0 -->

# Chapter 20: Cryptography and Key Management in Managerial Terms

## Learning outcomes

After completing this chapter, you should be able to:

1.  Distinguish the confidentiality, integrity, and authenticity objectives of cryptography and explain what encryption does and does not protect against.

2.  Apply an information classification scheme to define cryptographic expectations for data in transit, at rest, in use, and in logs and copies.

3.  Explain the key lifecycle, key hierarchies, HSMs and key management services, and the role of separation of duties and dual control.

4.  Compare provider-managed keys, customer-managed keys, BYOK, dedicated HSMs, and HYOK in terms of control, responsibility, and new failure modes.

5.  Evaluate key escrow, recovery, and lawful access arrangements as documented trade-offs among continuity, insider risk, and legal compliance.

6.  Analyze certificate and PKI governance in light of shrinking certificate lifetimes, and recommend ownership, inventory, and automation measures.

7.  Design a post-quantum migration program with a cryptographic inventory, prioritization, crypto-agility, supplier requirements, and board reporting against public milestones.

8.  Recommend evidence, key risk indicators, and privacy integration measures that make cryptographic claims defensible to auditors and regulators.

In most organizations, encryption appears in policy as a one-line requirement: sensitive data must be encrypted. The sentence sounds reassuring, and auditors often accept it at face value. Yet it hides the decisions that actually determine whether the organization is protected: which data, in which locations and states, using whose technology, under whose control, with what recovery arrangements, and under what conditions the organization will decrypt information for a court, a regulator, or an internal investigation. A policy that answers none of these questions is not a control. It is an aspiration.

Cryptography is not only a technical safeguard; it is a governance instrument. It shapes who can see what, under what circumstances, and with what evidentiary trail. It also underpins trust between machines, software, and people. Key management is the practical mechanism through which that power is exercised. Keys determine whether information is accessible, recoverable, or irreversibly lost. They also determine whether the organization can comply with lawful requests, preserve evidence, demonstrate safeguards to a privacy regulator, or respond to fraud and misuse by insiders.

For business technology managers, the central message of this chapter is that cryptography and key management require explicit governance choices. Those choices affect privacy, regulatory compliance, operational resilience, insider risk, cloud dependency, and the organization's relationship with regulators and law enforcement. When they are left entirely to technical teams or to vendors' default settings, the organization may find itself compliant on paper but fragile or overexposed in practice. Some of the most disruptive cryptographic failures organizations experience are not caused by broken algorithms at all; they are caused by an expired certificate that nobody owned, a key that could not be recovered, or a secret left in a source code repository.

The chapter builds on several earlier discussions. Chapters 1 and 4 framed risk as the interaction of threat, vulnerability, and impact—the risk triangle—and Chapters 5 and 6 showed how to express that interaction as scenarios that managers can assess and prioritize. Chapter 10 treated identity and access management, which is closely related to cryptography because keys and certificates are, in effect, the identities of machines. Chapter 15 examined the shared responsibility model in the cloud, and Chapter 16 addressed data protection and privacy obligations. This chapter does not repeat those discussions; it shows where cryptographic decisions sit within them and what managers must decide, own, and evidence.

The chapter proceeds in a sequence that mirrors the questions a manager should ask. It begins with what encryption does and does not achieve, then turns to what should be encrypted and in which data states. It explains key management in managerial language, clarifies ownership and accountability, and examines customer-managed keys in cloud environments. It then addresses escrow, recovery, and lawful access; certificates and machine trust, including the industry's move to much shorter certificate lifetimes; and the transition to post-quantum cryptography, which is now a multi-year governance program with public milestones. It closes with measurement and assurance, the links to privacy and cross-border obligations, and a synthesis with questions that managers and boards should ask.

## What encryption does, and what it does not do

To govern cryptography, managers do not need to understand the mathematics of algorithms. They do need a clear and realistic grasp of what cryptographic controls achieve, because both overconfidence and underuse are expensive. Overconfidence produces the comfortable belief that encrypted data is safe data, which leads organizations to neglect access control and monitoring. Underuse produces avoidable exposure when lost laptops, stolen backups, or intercepted traffic reveal information that could easily have been protected.

### Three jobs cryptography performs

Cryptography supports three distinct security objectives, and it helps managers to keep them separate. The first is **confidentiality**: encryption transforms readable data into a form that only holders of the right key can reverse. The second is **integrity**: cryptographic hashes and message authentication codes allow a recipient to detect whether data has been altered in storage or in transit. The third is **authenticity and non-repudiation**: digital signatures allow a party to prove that a message, document, transaction, or software package came from a specific key holder and has not been changed since it was signed.

These objectives map to different business needs. Confidentiality protects customer records, trade secrets, and personal information. Integrity protects financial transactions, configuration files, and evidence that may later be presented to a court or regulator. Authenticity protects software supply chains, electronic approvals, and the connections between systems. A policy that speaks only of "encryption" therefore addresses one of three jobs. When managers review cryptographic standards, they should ask whether integrity and authenticity requirements have been defined with the same care as confidentiality requirements, particularly for payment flows, software releases, and audit logs.

### What encryption protects against

When applied correctly, encryption prevents people who do not have the key from reading data, even if they obtain the physical storage, intercept network traffic, or gain access to underlying infrastructure. It is particularly powerful against certain classes of incident: theft or loss of devices, compromise or improper disposal of storage media, interception over untrusted networks such as public Wi-Fi, and some forms of hosting-layer or cloud infrastructure compromise. In these scenarios, encryption changes the impact side of the risk triangle dramatically. An attacker who steals an encrypted disk without the key obtains, in practical terms, nothing of value.

This is why privacy regulators and breach notification regimes often treat properly encrypted data differently. If the keys were not compromised, the probability that a loss leads to harm is substantially reduced. As discussed later in this chapter, that treatment depends entirely on the quality of key management. Encryption whose key sits on the same stolen laptop, in a configuration file next to the database, or in an administrator's unprotected notes offers far less than it appears to.

### The limits of encryption

Encryption does not make data safe against everything. It does not protect against misuse by authorized users, including insiders acting maliciously or negligently, because authorized users see decrypted data by design. It does not protect against attacks that exploit application logic while the application is legitimately decrypting data; an injection flaw in a web application will return plaintext records regardless of how the database is encrypted underneath. It does not compensate for weak identity and access management, poorly secured endpoints, or badly designed business processes. And it does not, by itself, guarantee availability: ransomware criminals use strong encryption too, and an organization that loses its own keys has effectively attacked itself.

There is also a timing limit that has become a governance concern in its own right. Encryption protects data for as long as the algorithm and key length remain strong enough. Information that must remain confidential for decades, such as health records, government identifiers, intellectual property, or defence information, can be captured today and decrypted later if algorithms weaken. This is the logic behind the post-quantum transition discussed later in the chapter.

| Encryption moves risk rather than eliminating it: it concentrates the protection of data into the protection of keys, so **whoever controls the keys controls the data**. |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

### Two anchoring questions

A useful discipline is to ask two questions for each significant system or dataset. First: against which threats do we expect cryptography to protect us? Second: at what points must the data be in usable, decrypted form, and what protections exist at those points? The first question ties cryptography to scenario-based risk assessment, so that encryption is justified by specific threats rather than by habit. The second question exposes the places where encryption ends and other controls must take over, such as access control, monitoring, data loss prevention, and application security.

Applied consistently, these questions keep cryptography anchored in risk reasoning rather than in assumption. They also sharpen vendor conversations: a supplier who claims that "all data is encrypted" should be able to explain which threats that addresses, where data is decrypted, and who holds the keys. If the explanation is vague, the assurance is weak.

## What should be encrypted? A classification-driven view

In principle, all sensitive information should be encrypted in transit and at rest. In practice, organizations must still prioritize, because some data stores, processes, and integrations are more consequential than others, and because cryptographic controls carry operational cost: key management effort, performance considerations, compatibility constraints, and the risk of self-inflicted outages. The most manageable approach is to derive cryptographic expectations from the organization's information classification scheme, which Chapter 16 treated as a foundation of data protection.

### Aligning cryptographic expectations with classification

Organizations usually distinguish several levels of information: public, internal, confidential, and restricted or highly confidential. Cryptographic requirements should be expressed for each level, so that system owners, architects, and suppliers can determine what is expected without negotiating every case. The table below illustrates one way of expressing those expectations; actual requirements should reflect the organization's risk appetite, regulatory context, and technical environment.

| **Classification** | **Typical examples**                                                | **Cryptographic expectation**                                                         |
|--------------------|---------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| Restricted         | Card data, health records, government identifiers, keys and secrets | Strong encryption everywhere; keys in HSM or managed KMS; strict separation of duties |
| Confidential       | Customer data, HR records, contracts, internal financials           | Encryption in transit and at rest; managed keys; logged key use                       |
| Internal           | Operational documents, internal communications                      | Platform encryption by default; standard transport encryption                         |
| Public             | Published reports, marketing content                                | Integrity and authenticity protection for published content                           |

Restricted information should be subject to strong encryption by default wherever it resides or moves. This usually includes payment card data, bank account details, government identifiers, health information, sensitive personal information as understood under Québec Law 25 and other privacy laws, trade secrets and proprietary algorithms, critical system configuration data, encryption keys and authentication secrets themselves, and certain categories of logs, such as those containing personal identifiers or session tokens. Confidential information should also be encrypted, especially at rest in multi-tenant or cloud environments, because the organization does not control who else shares the underlying infrastructure.

Internal information may warrant encryption depending on context. In modern environments, the marginal cost of encrypting storage volumes and network traffic is low, so organizations increasingly treat encryption at rest and in transit as the default even for less sensitive categories. Public information is a useful reminder that cryptography is not only about secrecy: a published annual report, a software download, or a public data feed benefits from integrity and authenticity protection, so that recipients can trust it has not been altered.

### Backups, archives, and secrets

Two domains require special attention because they are frequently overlooked. The first is backups and archives. These often contain the most complete historical picture of sensitive data and are prime targets in ransomware and data extortion scenarios, as discussed in Chapter 12. If backups are not encrypted, properly isolated, and protected by keys that are themselves recoverable, the organization's last line of defence is weaker than its production systems. A particular trap is the backup whose key is stored only in the environment being backed up.

The second domain is machine-to-machine secrets: application programming interface (API) keys, service account passwords, access tokens, database connection strings, and configuration files that embed credentials. These secrets are cryptographic material in everything but name. They should be stored in dedicated secrets management systems, encrypted at rest, rotated, and handled with the same seriousness as privileged user credentials. Exposed secrets in code repositories, container images, and collaboration tools remain a common entry point for attackers, and they are easy to measure, which makes them a good candidate for a key risk indicator.

### Encryption by default and governed exceptions

The managerial principle is straightforward: **if a category of data would cause serious harm or regulatory exposure if exfiltrated, it should be encrypted by design wherever it resides and moves, not as a discretionary option**. In practice, this translates into a small number of governance mechanisms. Encryption requirements should be embedded in security baselines and configuration standards (Chapter 14) so that they apply automatically to new systems. Architecture reviews should verify that designs meet the requirement before deployment. And exceptions should be handled through the same process as other control exceptions: documented, risk-assessed, approved by the accountable risk owner, given compensating controls, and assigned an expiry date.

Exceptions are not rare in practice. Legacy systems may not support modern encryption; industrial and medical devices may have limited processing capacity (see Chapter 11); some analytics workloads may be designed around plaintext. The governance failure is not the existence of such exceptions but their invisibility. An organization that knows where its unencrypted restricted data lives, why, and until when, is in a far more defensible position than one that simply assumes its policy is being followed.

## Where encryption matters: in transit, at rest, in use, and in logs

Cryptography operates across several data states, each with its own threats, technologies, and governance questions. Managers do not need to configure these technologies, but they should understand the distinctions well enough to recognize when a claim of "encrypted" covers only one state and leaves others exposed. The figure below summarizes the four domains discussed in this section.

<span id="_Toc243635580" class="anchor"></span>Figure 81: Data states and the governance questions encryption raises

### In transit

Encryption in transit protects data as it moves over networks between users, systems, and services. The managerial decision is no longer whether to use it; modern practice is that all external and most internal traffic should be protected. The real questions concern consistency and assurance. Managers should expect that web and API traffic uses current versions of Transport Layer Security (TLS), with older protocol versions disabled; that administrative and remote access sessions are protected; that email between organizations is encrypted where possible; and that communication between services in cloud and container environments is either encrypted or appropriately constrained and monitored.

Two weaknesses recur in practice. The first is the internal network treated as trusted, where traffic between data centre systems or between microservices travels in plaintext because it never leaves the corporate perimeter. Zero trust architectures, discussed in Chapter 13, reject that assumption. The second is inspection: many organizations decrypt traffic at proxies or security gateways to inspect it for threats, then re-encrypt it. This is legitimate, but it creates points where data is in plaintext and powerful keys are concentrated. Those points need to be identified, justified, and governed, including from a privacy standpoint when employee traffic is inspected.

### At rest

Encryption at rest protects data stored on disks, in databases, in file systems, and in object stores. Many platforms provide transparent encryption, where data is encrypted on disk but decrypted automatically for any application or user with access. This is valuable for certain scenarios, such as lost disks, stolen backups, improper media disposal, and some infrastructure compromises. It does not protect against misuse by applications or authenticated users, because the platform decrypts data for them.

For this reason, managers should understand storage encryption as a **necessary baseline, not the endpoint**. For restricted data, additional layers may be appropriate: database-level or field-level encryption that keeps the most sensitive attributes encrypted even from database administrators, application-level encryption where only the application holds the keys, or tokenization that replaces sensitive values with surrogates. Each additional layer increases protection against particular insider and application-level threats and also increases complexity, cost, and the consequences of key loss. The right choice depends on the scenario being mitigated, which is precisely why risk assessment should drive the decision.

### In use: confidential computing

Data must normally be decrypted in memory to be processed, which means that anyone controlling the underlying hardware, hypervisor, or operating system could in principle observe it. **Confidential computing** addresses this gap. The Confidential Computing Consortium, a Linux Foundation project, defines it as protecting "data in use by performing computation in a hardware-based, attested Trusted Execution Environment." In practical terms, specialized processor features isolate a workload so that even the cloud provider's administrators or a compromised host cannot read its memory, and remote attestation allows the customer to verify cryptographically that the workload runs in a genuine protected environment before releasing keys to it.

Confidential computing is now offered by the major cloud providers and is used for workloads such as multi-party analytics on sensitive data, processing of regulated financial or health information, and protection of AI models and the data used to train or query them (see Chapter 18). It is, however, not a universal solution. The European Data Protection Supervisor's technology monitoring notes that protection against side-channel attacks, sophisticated physical attacks, and supply-chain compromise varies by technology, and that trust ultimately rests on the hardware manufacturer. For managers, the governance point is that confidential computing is a credible option for high-sensitivity workloads where infrastructure-level access is a material threat, and that adopting it should be a risk-based decision with clear evidence requirements, notably attestation records.

### Logs, telemetry, and copies

Logs and telemetry present a genuine tension. They are essential for security monitoring, audit, investigation, and operations, yet they can leak sensitive data if not handled carefully. Applications frequently record personal information, session tokens, or even passwords in debug logs; security tools collect large volumes of user activity; and logs are often shipped to third-party platforms. A governance-aware approach requires that logs containing personal information or secrets be protected in storage, that access be restricted and monitored, that retention follow privacy and legal requirements, and that logging standards prohibit the recording of secrets in the first place.

Logs also have an integrity dimension. If an attacker or a dishonest insider can alter logs, the organization loses its ability to reconstruct events and to present credible evidence. Cryptographic integrity protection, write-once storage, and forwarding to a separately administered platform make logs more trustworthy. The same reasoning applies to the many copies of data that organizations create for testing, analytics, and reporting. A production database that is carefully encrypted but copied in plaintext to a development environment is not, in practice, protected.

In each of these domains, the critical managerial questions are the same: what is protected, who can decrypt, where are the keys stored, and how is access to keys governed and evidenced?

## Key management fundamentals in managerial language

Encryption without key management is an illusion. Keys are the practical expression of cryptographic power: whoever holds a key can read what it protects, sign in the name of its owner, or, by losing it, make information permanently inaccessible. If keys are poorly generated, stored, or managed, the strength of the algorithms and protocols does not translate into real protection. Attackers rarely break modern algorithms; they steal, misuse, or exploit poorly managed keys.

The authoritative reference for key management is NIST SP 800-57 Part 1, *Recommendation for Key Management: General*. Revision 5, published in 2020, remains the current final version at the time of writing; NIST released an initial public draft of Revision 6 in December 2025 that adds the post-quantum algorithms, treats keys for storage and for key establishment separately, and adds guidance on storing keying material. ISO/IEC 27002:2022 addresses the same ground in control 8.24, *Use of cryptography*, which expects organizations to define and implement rules for the effective use of cryptography, including key management. Managers do not need to read these documents in detail, but they should know that they exist and expect internal standards to be traceable to them.

### Keys are assets

Keys deserve classification, ownership, and protection comparable to, or stronger than, the data they protect. This is particularly true for root or master keys that can decrypt large volumes of data or derive other keys, for certificate authority signing keys that underpin trust across an entire environment, and for code-signing keys that allow software to be trusted by customers. Most organizations use key hierarchies: a small number of highly protected master keys encrypt many data encryption keys, which in turn encrypt the data itself. This design, often called envelope encryption, makes rotation and revocation manageable, but it also means that the compromise or loss of a master key has a very large blast radius.

Treating keys as assets has concrete implications. Keys should appear in an inventory with an owner, a purpose, a classification, a location, a cryptoperiod (the time during which a key is authorized for use), and dependencies on systems and data. Without such an inventory, the organization cannot answer basic questions after an incident, such as which data a compromised key protected, or before a migration, such as which systems depend on an algorithm that must be retired. As the post-quantum section of this chapter shows, the inventory has become a strategic necessity rather than a technical nicety.

### The key lifecycle

Key management follows a lifecycle, and each stage presents distinct risks. The figure below presents the lifecycle as a cycle, because keys that reach the end of their cryptoperiod are replaced by new keys that begin the cycle again.

<span id="_Toc243635581" class="anchor"></span>Figure 82: The cryptographic key lifecycle as a governed cycle

- **Generation** must use approved algorithms and sufficient randomness; predictable keys are as weak as no keys at all, and keys generated on ordinary servers or developer workstations are harder to protect than keys generated inside dedicated hardware.

- **Distribution and storage** must prevent interception or misdirection; keys should be transmitted only in protected form and stored in hardware security modules, managed key services, or dedicated secrets managers rather than in files, scripts, or source code.

- **Use** should be limited to the intended purpose and to authorized systems, so that a key created to encrypt backups cannot also be used to sign software, and every use of a high-value key leaves a log record.

- **Rotation** limits the amount of data exposed by a single key and the period during which a stolen key remains useful; rotation schedules should follow the cryptoperiods defined in the organization's standard and be automated wherever possible.

- **Revocation and suspension** allow a compromised or misused key to be withdrawn quickly; this capability must be tested, because a revocation that breaks critical systems will not be executed under pressure.

- **Backup and recovery** ensure that the loss of a key does not mean the loss of data, under controls strong enough that the backup does not become the easiest path to compromise.

- **Archival and destruction** ensure that keys no longer needed are retained only as long as data encrypted under them must remain readable, and then destroyed in a verifiable way, including copies in backups and configuration files.

### HSMs, key management services, and validated modules

Specialized systems exist to manage keys. **Hardware security modules (HSMs)** are tamper-resistant devices designed to generate and store keys and to perform cryptographic operations without exposing key material, so that an application can request "decrypt this" or "sign this" without ever seeing the key. Cloud **key management services (KMS)** provide similar functions as a managed service, often backed by provider-operated HSMs, with policy controls and detailed usage logs. Secrets managers handle passwords, tokens, and API keys for applications and automation.

Many regulators, customers, and public-sector buyers expect cryptographic modules to be independently validated. In the United States and Canada, the reference is the Cryptographic Module Validation Program (CMVP), run jointly by NIST and the Canadian Centre for Cyber Security, which validates modules against FIPS 140-3, *Security Requirements for Cryptographic Modules*. The transition from the older FIPS 140-2 is now complete: NIST's transition schedule placed all remaining FIPS 140-2 certificates on the historical list in September 2026, which means they should no longer be relied upon for new procurements. The Government of Canada's October 2025 Security Policy Implementation Notice on post-quantum migration goes further, requiring that new federal contracts with a digital component include clauses calling for CMVP-validated modules and cryptographic agility. Managers should treat validation status as procurement evidence, while remembering that a validated module can still be configured or used badly.

### Separation of duties and dual control

Separation of duties, introduced in earlier chapters as a general governance principle, applies with particular force to keys. The people who administer storage systems should not be able to export master keys. The people who administer HSMs should not be able to use them silently, without logging. Application developers should not hard-code keys into source code, and no single administrator should be able to both change a key policy and approve that change. For the most sensitive operations, such as creating a root certificate authority, restoring a master key from backup, or exporting key material, mature organizations use **dual control** and split knowledge, so that two or more authorized individuals must act together and no single person ever holds the complete secret.

These arrangements can feel bureaucratic, but they address a real scenario: a single compromised or malicious privileged account that can decrypt everything. When managers review a key management design, they should ask what one person, acting alone, can do. If the honest answer is "decrypt all customer data without anyone noticing," the design does not reflect any reasonable risk appetite.

### Key management generates evidence

Key management also produces evidence. Logs of key creation, use, rotation, and administrative change become critical in incident response, regulatory inquiries, and audits. They allow investigators to determine whether a stolen dataset could have been decrypted, whether an insider accessed keys outside normal patterns, and whether rotation policies were actually followed. Without such evidence, claims of strong encryption are difficult to substantiate, and the organization may be unable to argue that a breach of encrypted data posed no real risk.

The managerial takeaway is that when evaluating systems and vendors, leaders should ask as much about key management as about encryption features. **Knowing where the keys are and who controls them is often more important than knowing whether the data is encrypted.**

## Ownership and accountability: who owns keys?

Key ownership is not a trivial question. In many organizations, critical keys are effectively owned by whichever administrator created them, whichever team happens to run the HSM, or whichever vendor configured the platform. That arrangement works until something goes wrong: the administrator leaves, the team is reorganized, the vendor relationship ends, or a regulator asks who approved a decryption. Governance requires that ownership be deliberate.

### Three dimensions of ownership

Key ownership has three dimensions that should be distinguished. **Legal ownership** refers to who has the right to control keys and decide on their use. For data entrusted to a cloud provider or software-as-a-service (SaaS) platform, the organization usually remains the controller of the data even when keys are technically managed by the provider, and contracts should state that keys associated with customer data are used only according to the customer's instructions and applicable law.

**Operational control** refers to who can act on keys: generate, rotate, revoke, or use them for decryption and signing. In on-premises environments, this might be the internal security or infrastructure team operating HSMs. In cloud environments, operational control is shared between the provider's KMS and the customer in proportions that depend on the model chosen, as the next section explains.

**Governance accountability** refers to who answers for whether key management aligns with risk appetite, regulatory obligations, and internal policy. This is typically shared among the CISO function, which is accountable for technical soundness; data and system owners, who are accountable for business relevance and for accepting residual risk; and privacy and legal functions, which define the conditions for lawful access and disclosure. The NIST Cybersecurity Framework 2.0 Govern function, with its emphasis on roles, responsibilities, and authorities, provides a natural home for documenting these allocations.

### An operating model for cryptography

A clear operating model distinguishes roles so that each decision has an owner. In a typical arrangement:

- **The board and executive risk committee** set risk appetite for data exposure and availability, approve the cryptographic transition program, and receive periodic reporting on key risks and exceptions.

- **The CISO or security architecture function** defines cryptographic standards, including approved algorithms, key lengths, cryptoperiods, which systems must use HSMs, and how separation of duties is implemented.

- **Infrastructure, platform, and cloud teams** operate HSMs, KMS, secrets managers, and certificate management tooling according to those standards, and produce the evidence that they are doing so.

- **Data and system owners** approve key usage in line with business processes, accept residual risk for exceptions, and ensure that systems they own appear in the cryptographic inventory.

- **Legal and privacy functions** define the conditions under which keys may be used to respond to lawful requests or internal investigations, and ensure that cryptographic choices are reflected in privacy impact assessments.

- **Internal audit**, as the third line, periodically tests whether the standards are implemented and whether reporting to the board is accurate.

### Cryptographic policy and standards

The operating model should be anchored in a short, stable cryptographic policy and a more detailed, regularly updated standard. The policy states principles: cryptography is required according to classification, keys are assets with owners, approved algorithms are defined centrally, exceptions are time-limited, and the organization maintains the ability to change algorithms. The standard specifies the technical details that change over time, such as approved algorithms and parameters, protocol versions, cryptoperiods, and certificate profiles. OSFI Guideline B-13, which applies to federally regulated financial institutions, captures this dual expectation: it calls on institutions to implement and maintain strong cryptographic technologies, to protect encryption keys from unauthorized access, usage, and disclosure across their lifecycle, and to assess their cryptography standards and technologies regularly.

Without this clarity, the organization may find that critical keys are effectively owned by nobody in particular, or by a vendor whose contractual obligations are vague. **That is a governance failure, not merely a technical one**, and it tends to surface at the worst possible moment.

## Customer-managed keys in the cloud: BYOK, HYOK, and HSM services

Customer-managed keys have become a common term in cloud and SaaS marketing. The phrase suggests greater customer control and sometimes implies stronger confidentiality. In reality it covers several different models, and the operational implications differ considerably. Chapter 15 discussed the shared responsibility model in general terms; this section focuses on how key management choices shift responsibilities between the provider and the customer.

### A spectrum of control

It is helpful to think of key control as a spectrum rather than a binary choice.

- **Provider-managed keys** are generated, stored, rotated, and used by the cloud provider, with encryption enabled by default and few decisions left to the customer. This is simple, resilient, and adequate for much data, but the provider retains substantial technical capability to decrypt.

- **Customer-managed keys (CMK)** are created and governed by the customer through the provider's KMS. The customer sets access policies, rotation schedules, and permissions, and can disable or revoke keys; the provider's services use the keys to encrypt data on the customer's behalf, and every use is logged in the customer's account.

- **Bring your own key (BYOK)** means the customer generates key material in its own environment, typically an on-premises HSM, and imports it into the provider's KMS. This gives the customer assurance about how the key was generated and a copy outside the provider, but once imported the key is used inside the provider's infrastructure.

- **Dedicated or single-tenant cloud HSMs** give the customer exclusive control of HSM partitions hosted by the provider, often with FIPS 140-3 validated hardware, for workloads where regulation or risk appetite requires that key material never be shared with other tenants.

- **Hold your own key (HYOK)**, sometimes implemented as external key management, keeps master keys in an HSM or key service outside the primary provider, which must call out to the customer's environment for each decryption. The primary provider never holds the master key, and the customer can cut off access entirely.

Moving along this spectrum increases customer control and, in some respects, confidentiality against the provider. It also increases customer responsibility, complexity, and the consequences of error.

### What changes operationally

Customer-managed models shift part of the **availability risk** to the customer. If the customer mismanages keys, rotates them incorrectly, deletes them, or loses access to an external key environment, it may lose access to its own data even while the provider's infrastructure is functioning normally. In a HYOK design, an outage of the customer's key service becomes an outage of every cloud application that depends on it.

These models also require disciplined lifecycle management. Configurations must be documented, changes controlled, and rotation coordinated with dependent systems and backups. In exchange, they improve the customer's ability to enforce access policies, to revoke access rapidly, and to demonstrate control to regulators and clients. However, customer-managed keys do not automatically prevent every access route. Depending on the architecture and jurisdiction, providers may retain capabilities needed to operate the service or to respond to legal orders, and SaaS applications must decrypt data to process it, so the provider's application layer still sees plaintext.

| Customer-managed keys are a transfer of control **and** of responsibility: the organization gains the power to lock others out, and with it the power to lock itself out. |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

### Illustrative scenario: a key that could not be found

A mid-sized Québec manufacturer of industrial components migrated its engineering document management system to a SaaS platform. To satisfy a defence-sector customer, the project team enabled the provider's BYOK option and generated the master key on a small HSM appliance in the company's data centre. The engineer who configured the arrangement documented it in a personal notebook and left the company eighteen months later. When the company replaced its data centre equipment, the HSM was decommissioned with other hardware; nobody connected it to the SaaS platform.

The problem surfaced when the provider announced a mandatory key rotation that required the customer to generate fresh key material on its own HSM and import it. The company no longer had the HSM, and the backup of its contents could not be located. After several anxious weeks, the provider's support team confirmed that data remained accessible under the existing imported key, which had not yet expired, and the company was able to export its documents and re-establish a new key hierarchy. The cost was measured in consulting fees, a delayed product program, and an uncomfortable disclosure to the defence customer, whose contract required notification of material security events.

The post-incident review found no technical flaw. The failures were managerial: no named owner for the key, no entry in an asset inventory, no link between the HSM and the business service it supported, no tested recovery procedure, and a decommissioning process that did not ask about cryptographic dependencies. The corrective actions were equally managerial. The company added keys and HSMs to its configuration management database, assigned ownership of each key hierarchy to a named system owner, required dual control and documented escrow for all imported key material, and added a cryptographic dependency check to its change and decommissioning procedures.

### Questions to put to providers

For managers, the key questions about customer-managed keys are concrete. Exactly what is under our control, and what remains with the provider? What new failure modes do we introduce by taking this control, and how will we detect and recover from them? Where are the keys physically and legally located, and how does that interact with data residency commitments and lawful access in the provider's jurisdictions? What evidence can we obtain, such as key usage logs, attestation reports, and HSM validation certificates, to demonstrate control to regulators and clients? And what happens to our keys and data when the contract ends? These questions belong in procurement and third-party risk assessment (Chapter 19) as much as in architecture review.

## Key escrow, recovery, and lawful access

**Key escrow** is the practice of storing copies of cryptographic keys with a trusted party, internal or external, so that keys can be recovered or used under defined conditions. It is a sensitive governance topic because it sits at the intersection of three concerns that pull in different directions: business continuity, insider risk, and lawful access.

### Recovery as a continuity requirement

From a business continuity standpoint, some form of key recovery is usually necessary. If all copies of a critical key are lost, the data it protects may become unrecoverable, which can be catastrophic, particularly for regulated records that must remain accessible for legal retention periods measured in years. Escrow and other recovery mechanisms, such as master keys backed up under dual control or split among custodians, mitigate this risk. Recovery procedures should be part of the business continuity and disaster recovery plans discussed in Chapter 12, and they must be tested. A recovery procedure that has never been exercised is a hypothesis, not a control.

Endpoint encryption is the most common everyday example. When employees' laptops are encrypted, the organization normally escrows recovery keys in its device management or directory service so that help desk staff can unlock a device after a forgotten password or hardware failure. That escrow is a powerful asset: anyone who can query it can unlock any corporate laptop. Access to it should therefore be restricted, logged, and reviewed like any other privileged capability.

### Escrow as concentrated risk

From an insider risk standpoint, escrow creates another concentrated asset. If escrow arrangements are weakly governed, individuals with access to escrowed keys can potentially decrypt large volumes of data without detection. Mature recovery designs therefore use multi-person control, tamper-evident logging, strict procedural safeguards, and independent review of every recovery operation. The objective is to make legitimate recovery possible and illegitimate recovery both difficult and visible.

### A lawful access position

Organizations may receive legally binding orders to produce data, and they may need to decrypt information for internal investigations, litigation holds, or regulatory examinations. Whether and how this is possible depends on architecture and key control. In some models, the organization can comply by using its own keys to decrypt data. In others, particularly where external providers operate the encryption environment, lawful access may involve those providers, possibly under the laws of another jurisdiction. In designs where the organization deliberately cannot decrypt certain data, such as end-to-end encrypted messaging with keys held only by users, compliance options are more limited.

Governance requires an explicit position on lawful access, formulated with legal counsel and privacy leadership, and the technical design must make that position implementable. The position should address at least the following questions:

- **Legal conditions:** under what legal authorities and conditions will the organization provide decrypted data, and how will it verify the validity and scope of a request?

- **Authority:** who is authorized to approve decryption for legal or investigative purposes, and what role do legal counsel and the privacy officer play?

- **Records:** what records and audit trails are required for each decryption, and who reviews them?

- **Architecture:** which system architectures and provider arrangements are compatible or incompatible with this stance, including where providers might be compelled to act without the organization's knowledge?

- **Notification:** when, and to whom, will affected individuals or clients be informed, subject to legal constraints?

### Documenting the trade-off

Key escrow decisions should be documented and risk-assessed like any other significant control choice. For some data categories, strong recovery and lawful access requirements will justify carefully controlled escrow. For others, the organization may accept that the loss of keys will render data unrecoverable, as a deliberate trade-off for stronger confidentiality. What matters is that these trade-offs are conscious, approved by the appropriate risk owner, aligned with legal obligations and risk appetite, and revisited when circumstances change. A defensible decision is one the organization can explain after the fact, not one it discovers it made by default.

## Certificates, PKI, and machine trust

Much of the cryptography in an organization is not about encrypting stored data; it is about establishing trust between machines, services, and people. This is the realm of certificates and **public key infrastructure (PKI)**. It is also the area where cryptographic failures most often become visible to customers, because a certificate problem typically appears as a service that simply stops working.

### What certificates and PKI do

Certificates are digital credentials that bind an identity, such as a domain name, a service, a device, or a person, to a public key. They are used in TLS for web and API connections, in virtual private networks, in device and workload authentication, in email signing, in code signing, and in many internal systems. A PKI is the combination of certificate authorities, policies, procedures, and technology that issues, manages, renews, and revokes these certificates. Publicly trusted certificates for internet-facing services come from commercial or non-profit certificate authorities whose practices are governed by the CA/Browser Forum's Baseline Requirements and by browser and operating system vendors' root programs. Internal certificates usually come from an organization's own private PKI.

Poorly governed PKI creates serious operational and security risk. Expired certificates cause outages; mis-issued certificates enable impersonation; weakly protected certificate authority keys threaten entire environments; and certificates that cannot be revoked quickly leave compromised systems trusted. Conversely, a well-governed PKI makes secure connectivity and identity assurance possible at scale. Managers should expect clarity on who operates each certificate authority, defined processes for requesting, approving, issuing, renewing, and revoking certificates, a complete inventory of certificates with their owners and expiry dates, and procedures for revoking certificates when keys are compromised or systems are decommissioned.

### Shrinking certificate lifetimes

The governance of certificates is changing quickly because the public certificate ecosystem has decided to make certificates much shorter-lived. In April 2025, the CA/Browser Forum adopted ballot SC-081, which sets a schedule for reducing the maximum validity of publicly trusted TLS certificates. Certificates issued on or after March 15, 2026 may be valid for no more than 200 days; from March 15, 2027, no more than 100 days; and from March 15, 2029, no more than 47 days. The ballot also progressively shortens the period during which a certificate authority may reuse earlier domain validation, down to ten days by the end of the schedule.

The rationale is sound: short-lived certificates limit the damage from compromised keys and mis-issuance, reduce reliance on revocation mechanisms that have never worked well, and force the ecosystem toward automation, which will also make future algorithm changes easier. The managerial consequence is equally clear. An organization that renews certificates manually, using spreadsheets and calendar reminders, will move from roughly one renewal per certificate per year to around eight. **Certificate renewal can no longer be a manual task; it must become an automated, monitored process with clear ownership.** Automation through standard protocols such as ACME, discovery tools that find certificates the organization did not know it had, and monitoring that alerts long before expiry are now baseline expectations rather than refinements.

### Machine identities and secrets sprawl

Certificates are one form of a broader category that practitioners call **machine identities**: the certificates, keys, tokens, and service credentials that allow workloads, devices, containers, and automated processes to authenticate to one another. In modern cloud and DevOps environments, machine identities vastly outnumber human identities, and they are created and destroyed continuously by automation. Chapter 10 discussed the governance of non-human identities from the access management perspective; the cryptographic perspective adds the requirement that the keys behind those identities be generated, stored, rotated, and revoked properly.

PKI governance also creates dependencies that cross organizational boundaries. A SaaS provider that pins a customer's certificate, a partner that hard-codes an intermediate certificate authority, or an industrial device that cannot accept a new root can each turn a routine renewal into an incident. These dependencies should be recorded in the same inventory as the certificates themselves.

### Illustrative scenario: the certificate nobody owned

A Canadian credit union with about 200,000 members operated an online banking platform, a mobile application, and an integration with a payments network for real-time transfers. On a Monday morning in early spring, members began reporting that the mobile application would not connect. Within an hour, the contact centre was overwhelmed, and transfers to other institutions were failing. The cause was a single expired TLS certificate on an API gateway that served both the mobile application and the payments integration.

The certificate had been purchased three years earlier by a project team that had since been disbanded. Renewal notices were sent to a shared mailbox that nobody monitored. The certificate did not appear in the infrastructure team's inventory, because it had been installed on a gateway managed by an external integrator. Restoring service required obtaining a new certificate, which was quick, and installing it on the gateway, which was not: the integrator's change process required approvals, and the mobile application pinned the old intermediate certificate, so a new application version had to be pushed through the app stores. The mobile service was degraded for most of two days. The credit union reported the incident to its regulator, issued member communications, and absorbed contact-centre overtime, reputational harm, and fee waivers.

Viewed through the lens of Chapter 5, this was a realistic and foreseeable scenario: a threat event (certificate expiry) exploiting vulnerabilities (no owner, no inventory, manual renewal, certificate pinning, third-party dependency) to produce a material availability impact on a critical business service. The post-incident review produced managerial decisions rather than technical ones. The board's risk committee asked management to treat certificate management as part of operational resilience. The CIO assigned ownership of the certificate inventory to the infrastructure team and ownership of each certificate to a named system owner. Contracts with integrators were amended to require certificate inventory reporting. A certificate lifecycle management tool was funded to discover and automatically renew certificates, with a key risk indicator reporting the number of certificates expiring within 30 days without an automated renewal path. With the SC-081 schedule in view, the credit union also set a target that all public certificates be renewed automatically before validity fell to 100 days in 2027.

The scenario illustrates a broader lesson. The organization's encryption was strong; its algorithms were current; its auditors had found no weakness in its cryptographic standard. What failed was ownership, inventory, and process, which are precisely the things governance is supposed to provide.

## Post-quantum cryptography and crypto-agility as a governance program

The most significant cryptographic change facing organizations in the coming decade is the transition to post-quantum cryptography. It is not a product upgrade or a patch cycle. It is a multi-year program that touches almost every system, many suppliers, and the organization's long-term data protection commitments, and it now has public milestones set by standards bodies and governments. For managers, it is best understood as a governance program with an owner, a budget, a roadmap, and reporting to the board.

### Why quantum computing matters to managers

Most public-key cryptography in use today, including the RSA and elliptic curve algorithms used for key exchange, digital signatures, and certificates, relies on mathematical problems that a sufficiently powerful quantum computer could solve efficiently. Such a machine, often called a cryptographically relevant quantum computer, does not exist publicly at the time of writing, and experts disagree on when one might. Symmetric encryption and hashing are much less affected; using sufficiently long keys largely preserves their strength. The vulnerable part is the public-key layer that protects how keys are exchanged and how identities and signatures are verified.

Two features of this threat make it a present-day governance concern despite the uncertainty about timing. The first is **harvest now, decrypt later**: an adversary can record encrypted traffic or steal encrypted data today and store it until it can be decrypted. For information that must remain confidential for many years, such as health records, personal information, trade secrets, legal files, and government or defence data, the exposure begins at the moment of interception, not at the moment a quantum computer appears. The second is migration time. Replacing cryptography across applications, devices, protocols, partners, and suppliers historically takes many years; past transitions away from weak hash functions and older protocol versions took a decade or more in many organizations. A simple reasoning test, usually credited to the quantum-safety researcher Michele Mosca, captures the logic: if the number of years data must remain secret, plus the number of years it will take to migrate, exceeds the number of years before a quantum threat materializes, the organization is already late.

| For long-lived sensitive data, the quantum risk window opens when data is intercepted, not when a quantum computer is built, which is why **post-quantum migration is a present-day governance decision**. |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

### The new standards

NIST published the first three post-quantum cryptography standards on August 13, 2024. FIPS 203 specifies ML-KEM, a module-lattice-based key-encapsulation mechanism intended as the primary algorithm for establishing shared keys. FIPS 204 specifies ML-DSA, a module-lattice-based digital signature algorithm intended as the primary general-purpose signature scheme. FIPS 205 specifies SLH-DSA, a stateless hash-based signature algorithm that relies on different mathematics and serves as a conservative alternative. A fourth standard, FIPS 206, will specify FN-DSA, a signature algorithm derived from the Falcon submission that produces compact signatures; at the time of writing, FIPS 206 has been drafted and submitted for approval but not yet published as a final standard, so organizations should watch for its release rather than plan around a specific date.

In March 2025, NIST selected HQC, a code-based key-encapsulation mechanism, as a backup to ML-KEM, so that a different mathematical foundation is available if a weakness in lattice-based schemes were discovered. NIST indicated that a draft standard would follow roughly a year after selection and a final standard was expected in 2027. NIST also published SP 800-227, *Recommendations for Key-Encapsulation Mechanisms*, in September 2025, which guides the secure use of KEMs, including hybrid combinations of classical and post-quantum algorithms. For managers, the point is not to memorize algorithm names but to understand that standards now exist, that more are coming, and that internal cryptographic standards and procurement requirements should reference them.

### Transition timelines

Several authorities have now published timelines that give the transition a calendar. NIST IR 8547, *Transition to Post-Quantum Cryptography Standards*, released as an initial public draft in November 2024, proposes that quantum-vulnerable public-key algorithms such as RSA and elliptic curve cryptography at the 112-bit security level be deprecated after 2030 and that quantum-vulnerable algorithms be disallowed in NIST standards after 2035, with high-risk systems transitioning earlier. At the time of writing, IR 8547 remains a draft, but its 2030 and 2035 milestones have been widely adopted as planning anchors by vendors and regulators.

In Canada, the Canadian Centre for Cyber Security published ITSM.40.001, *Roadmap for the migration to post-quantum cryptography for the Government of Canada*, in June 2025. It describes three overlapping phases, preparation, identification, and transition, and sets two completion targets: migration of high-priority systems by the end of 2031 and of remaining systems by the end of 2035. A Treasury Board Security Policy Implementation Notice issued in October 2025 translated the roadmap into obligations for federal departments: an initial departmental migration plan and annual progress reporting starting April 2026, procurement clauses requiring validated modules and cryptographic agility in new contracts from that date, an updated plan for the identification phase by April 2027, and identification of high-priority systems, updated system records, and the start of transition by April 2028. Internationally, the European Union's coordinated implementation roadmap, published in June 2025, asks member states to begin transition activities by the end of 2026, to migrate high-risk use cases by the end of 2030, and to complete as much of the remaining transition as feasible by 2035.

These timelines formally bind governments and their agencies, but their influence is much broader. Suppliers to the federal government will face procurement requirements; financial regulators and critical infrastructure authorities tend to align with national guidance; and large customers will cascade expectations to their suppliers. A private organization in Canada that intends to sell to governments, banks, or critical infrastructure operators should therefore treat 2031 and 2035 as realistic external deadlines.

### The cryptographic inventory and CBOM

Every authoritative roadmap begins with the same step: know where cryptography is used. A **cryptographic inventory** records, for each system, which algorithms, key lengths, protocols, certificates, libraries, and hardware modules it uses, for what purpose, which data it protects, how long that data must remain confidential, who owns the system, and which suppliers control the relevant components. Without this inventory, the organization cannot prioritize, estimate cost, or demonstrate progress.

Building the inventory is harder than it sounds, because cryptography is embedded in operating systems, libraries, firmware, network equipment, SaaS platforms, and partner connections, often invisibly. Organizations combine several sources: automated discovery tools that scan networks, endpoints, and code; certificate and key management systems; configuration management databases; and supplier attestations. A useful emerging artifact is the **cryptographic bill of materials (CBOM)**, which extends the software bill of materials concept discussed in Chapter 19 to describe the cryptographic assets within a software product. The CycloneDX standard supports CBOM as part of its bill of materials format. Requesting CBOMs, or at minimum structured cryptographic disclosures, from key suppliers is a practical way to bring the supply chain into the inventory.

The inventory then supports prioritization. Systems that protect long-lived sensitive data and are exposed to interception, such as external connections, virtual private networks, and data exchanges with partners, rank high because of harvest-now-decrypt-later risk. Long-lived roots of trust, such as firmware signing, code signing, and certificate authorities, also rank high, because forged signatures would undermine everything downstream. Custom applications rank high because no vendor will migrate them on the organization's behalf.

### Crypto-agility

**Crypto-agility** is the ability to change cryptographic algorithms, parameters, and implementations quickly and safely, without redesigning systems. It matters beyond the quantum transition: algorithms are periodically weakened, implementations contain flaws, and regulators change requirements. The post-quantum migration will not be the last cryptographic change organizations face, and the investments made now should leave the organization able to make the next change faster.

In governance terms, crypto-agility rests on a few concrete practices. Cryptography should be consumed through centrally managed services, libraries, and configuration rather than hard-coded into applications. Algorithms and parameters should be selected by policy and configuration, so that a change does not require new code. Inventory and automation, as in the certificate case, make change measurable and repeatable. Procurement should require suppliers to support current standards, provide a credible roadmap for post-quantum algorithms, and commit to timely updates. Many organizations will also pass through a period of **hybrid** cryptography, in which classical and post-quantum algorithms are combined so that a connection remains secure as long as either is unbroken; major browsers and cloud providers have already deployed hybrid key exchange for TLS. Hybrid approaches reduce risk during the transition but add complexity, and decisions about where to use them belong in the cryptographic standard.

### A migration roadmap

The figure below presents a migration roadmap in five steps that aligns with the phases of the CCCS roadmap and can be adapted to any organization.

<span id="_Toc243635582" class="anchor"></span>Figure 83: A post-quantum migration roadmap as a governance program

1.  **Govern and plan:** appoint an accountable executive and a migration lead, establish a steering committee, integrate the program into the risk register and investment plan, and set board reporting.

2.  **Discover and inventory:** build the cryptographic inventory, request CBOMs and roadmaps from suppliers, and record data confidentiality lifetimes for each system.

3.  **Assess and prioritize:** rank systems by harvest-now-decrypt-later exposure, signature and trust-anchor criticality, and migration difficulty, using scenario-based risk assessment rather than technical preference alone.

4.  **Migrate and test:** upgrade, replace, tunnel, or isolate systems according to priority, using hybrid modes where appropriate, with impact assessments, testing, and rollback plans under normal change management.

5.  **Verify and sustain:** measure progress against milestones, confirm through testing that deprecated algorithms are no longer in use, retire exceptions on schedule, and embed crypto-agility in architecture and procurement standards.

The benefits of such a program are largely invisible, its costs are spread across many budgets, and much of the work depends on vendors' timelines. Chapter 17's treatment of investment governance applies directly: the program should be justified by risk scenarios that quantify exposure, especially for long-lived data, and funded as a multi-year commitment with milestones rather than as an annual discretionary item.

### Suppliers and procurement

For most organizations, the majority of their cryptography is supplied by others: operating system vendors, cloud providers, SaaS platforms, network equipment manufacturers, payment processors, and managed service providers. Post-quantum migration therefore depends heavily on supplier readiness. Procurement requirements should ask suppliers to state which post-quantum algorithms they support or plan to support, by when, and in which products; whether their cryptographic modules are or will be validated under FIPS 140-3; whether they can provide a CBOM or equivalent disclosure; and how they will notify customers of cryptographic changes. Contracts for systems with long service lives, such as industrial equipment, medical devices, and core banking platforms, deserve particular scrutiny, because a device purchased in 2027 may still be in service well after 2035.

## Measuring and assuring cryptographic controls

Because cryptographic choices appear technical, it is easy for boards and executives either to ignore them or to accept assurances without verification. A governance-oriented program establishes ways to measure and assure the effectiveness of encryption and key management, using the evidence-based approach that runs through this book.

### Forms of evidence

Several forms of evidence are particularly useful:

- **Configuration evidence** shows where encryption is enabled and how, including storage encryption settings for databases, file systems, and object stores; transport encryption settings for external and internal services; and KMS or HSM configurations for key use.

- **Coverage evidence** demonstrates that critical systems and datasets fall within the scope of encryption standards, for example reports showing that all systems containing payment data use strong encryption at rest, or that all administrative remote access channels use approved protocols.

- **Key management evidence** includes logs of key creation, rotation, and use; evidence that rotation policies are enforced; records of role assignments and separation of duties; and documented, dated results of key recovery tests.

- **Inventory evidence** covers certificates, keys, secrets, and algorithms, with ownership and expiry, and is increasingly the foundation for both operational resilience and post-quantum reporting.

- **Testing evidence** comes from penetration tests, configuration audits, and post-incident reviews, for example assessments confirming that stolen disks cannot be decrypted without keys, that attackers cannot trivially extract keys from application servers, or that TLS configurations reject obsolete protocol versions.

- **Certification and validation evidence**, such as FIPS 140-3 certificates for cryptographic modules and SOC 2 or ISO/IEC 27001 reports from providers, is useful but supplementary: it shows that suppliers have controls, not that those controls match the organization's specific threats and requirements.

### Key risk indicators for cryptography

Cryptography lends itself to measurable indicators, which makes it a good domain for the KRI discipline discussed in earlier chapters. Useful indicators include the percentage of restricted-data systems meeting the encryption standard; the number of certificates expiring within 30 days without an automated renewal path; the percentage of public certificates renewed automatically; the number of exposed secrets detected in code repositories and the median time to revoke them; the number of keys past their cryptoperiod; the number of active cryptographic exceptions and how many are past expiry; the date of the last successful key recovery test for each critical key hierarchy; and, for the post-quantum program, the percentage of high-priority systems inventoried, assessed, and migrated against the roadmap.

Each indicator should have an owner and a threshold linked to risk appetite and tolerance. A target of zero expired certificates on critical services, for example, is a clear tolerance statement that the board can understand and that management can be held to. Indicators should be reported in business terms: not "1,200 certificates," but "four member-facing services at risk of outage within 30 days."

### Assurance across the three lines

The three lines model provides a natural structure for cryptographic assurance. The first line, the operational teams that run key management, PKI, and platforms, produces configuration and key management evidence as a by-product of automated operations. The second line, security risk management and compliance, defines standards, monitors indicators, challenges exceptions, and maps cryptographic controls to frameworks such as ISO/IEC 27001:2022, NIST CSF 2.0, and the data protection safeguards of CIS Controls v8.1. The third line, internal audit, periodically tests whether controls work as described, for instance by sampling key recovery tests, verifying that HSM administrators cannot use keys without logging, or confirming that the cryptographic inventory is complete for a sample of systems.

For managers, the practical approach is to ask regular, structured questions. For our most critical systems, can we show that encryption is properly enabled? Do we know where keys are and who can operate them? Have we tested key recovery under realistic disaster conditions? Have recent incidents revealed gaps between our encryption claims and reality? Answers to these questions are far more meaningful than a generic statement that the organization uses strong encryption, which is often a form of compliance theatre.

## Integration with privacy, data protection, and cross-border obligations

Cryptography and key management are deeply intertwined with privacy and data protection, particularly in jurisdictions that emphasize both safeguarding and demonstrable accountability. Chapter 16 covered privacy obligations in depth; this section highlights where cryptographic decisions change the analysis.

### Encryption and breach notification

Encryption is frequently cited in privacy laws and guidance as an appropriate security safeguard, and it often plays a role in deciding whether a breach must be notified. Under Québec Law 25, organizations must assess whether a confidentiality incident presents a risk of serious injury, considering the sensitivity of the information, the anticipated consequences of its use, and the likelihood that it will be misused. Under PIPEDA, the test is whether the breach creates a real risk of significant harm, assessed in light of sensitivity and probability of misuse. Under the EU GDPR, communication to affected individuals is not required when the data was rendered unintelligible to unauthorized persons, for example through encryption. In all three regimes, properly encrypted data whose keys were not compromised generally weighs toward lower risk.

That treatment is contingent on strong key management and on evidence. If keys were stored alongside the data, if logs show that attackers may have obtained decryption capability, or if the organization cannot show who had access to keys, regulators will not treat encryption as a mitigating factor. Notification to individuals is also only one obligation; recording the incident and reporting to the Commission d'accès à l'information or the Office of the Privacy Commissioner of Canada remain separate duties. **The ability to prove that keys were not compromised is often what makes encryption count in a breach assessment.**

### Key location and jurisdiction

Data residency and cross-border concerns also intersect with cryptography. Some organizations rely on encryption and key location, for example keeping master keys only in Canada or in a customer-controlled HSM, as part of their cross-border risk management strategy. Québec Law 25 requires a privacy impact assessment before personal information is communicated outside Québec, which must consider the protection the information will receive, and encryption with customer-controlled keys is a relevant factor in that assessment. For the approach to be meaningful, the architecture must ensure that providers cannot decrypt data outside the intended jurisdiction without the organization's cooperation, and contractual and legal arrangements must support that stance.

Managers should be realistic about what key location achieves. If a SaaS application must decrypt data to process it, the provider's application layer will see plaintext wherever it operates, so key location is strongest for storage and backups and weakest for data a provider actively processes.

### Pseudonymization and tokenization

Pseudonymization and tokenization, discussed in Chapter 16, have important cryptographic dimensions. Tokenization schemes that replace identifiers with tokens may rely on key-protected mapping tables or on keyed cryptographic functions. If keys or mapping tables are compromised, pseudonymized data reverts to identifiable personal information, both technically and in the eyes of regulators. Governance must therefore treat such systems as part of key management, with the same inventory, ownership, separation of duties, and evidence requirements, rather than as simple data transformations performed by an analytics team.

### Privacy impact assessments

Privacy impact assessments for new systems should explicitly consider cryptographic choices: what is encrypted, in which states, under which key management model, with what recovery and lawful access arrangements, and with what residual risk. For systems that will hold long-lived sensitive personal information, the assessment should also consider the post-quantum horizon, because data collected now may need protection well beyond 2035. Cryptography cannot replace compliance, but when properly governed it becomes one of the strongest technical arguments for due diligence and proportionality.

## Bringing it together: cryptography and key management in the governance story

This chapter has treated cryptography and key management as instruments of governance. The central ideas are straightforward, even if their implementation is demanding.

First, encryption is only as good as key management. Claims of strong encryption are empty if keys are uncontrolled, mismanaged, or concentrated in ways inconsistent with risk appetite. Keys are assets that need owners, inventories, lifecycles, separation of duties, and evidence. Second, cryptographic choices embody governance decisions about who can access data, under what conditions, with what audit trail, and how recovery and lawful access are handled. Those decisions must be made consciously by accountable risk owners, not left to vendor defaults or the habits of individual administrators.

Third, customer-managed keys and cloud key management models give organizations more control, but they also impose greater responsibility for availability, lifecycle management, and lawful access. They are governance options, not automatic improvements. Fourth, key escrow and recovery sit at the intersection of continuity, insider risk, and legal compliance, and they require clear policy, multi-party controls, tested procedures, and transparent reasoning about trade-offs.

Fifth, cryptography is not only about data at rest; it is also about machine trust, PKI, and the identities of the workloads and devices that make up modern environments. The move to 47-day public certificates by 2029 turns certificate management from an administrative chore into an automation and resilience discipline. Failures here produce outages and impersonation even when data encryption looks sound.

Sixth, the post-quantum transition has moved from research to governance. Standards exist, timelines have been published by NIST, the Canadian Centre for Cyber Security, and the European Union, and harvest-now-decrypt-later risk means that long-lived sensitive data is already exposed. The organizations that will meet the 2031 and 2035 milestones are those that build a cryptographic inventory now, invest in crypto-agility, bring suppliers into the program, and report progress to the board like any other strategic initiative.

### Questions managers and boards should ask

- **Inventory:** do we have a current inventory of our keys, certificates, secrets, and algorithms, with an owner, purpose, and expiry for each, and how complete is it?

- **Classification:** are our cryptographic requirements derived from our information classification scheme, and can we show which restricted-data systems do not yet meet them?

- **Control:** for our most critical data, who holds the keys, what can a single administrator do alone, and how would we know if keys were misused?

- **Cloud and SaaS:** which key management model have we chosen for each major provider, what responsibilities and failure modes did that choice create, and what evidence do we receive?

- **Recovery:** when did we last successfully test recovery of each critical key hierarchy, including backup keys, under realistic disaster conditions?

- **Lawful access:** do we have an approved position on decrypting data for legal requests and investigations, and do our architectures and contracts make that position implementable?

- **Certificates:** how many of our certificates renew automatically, how many critical services could fail because of an expiry in the next 30 days, and are we ready for 100-day certificates in 2027?

- **Post-quantum:** who is accountable for our post-quantum migration, which systems face harvest-now-decrypt-later risk, and what is our progress against the 2031 and 2035 milestones?

- **Suppliers:** have our key suppliers provided post-quantum roadmaps, validated module evidence, and cryptographic disclosures such as CBOMs, and are these requirements in our contracts?

- **Exceptions:** how many cryptographic exceptions are active, who approved them, and how many have passed their expiry date?

- **Breach readiness:** if encrypted data were stolen tomorrow, could we prove to a regulator that the keys were not compromised?

Finally, in the broader orientation of this book, cryptography and key management are cross-cutting enabling controls. They support privacy engineering, cloud and third-party governance, identity and access management, incident management and resilience, and the economic logic of risk reduction. When managers can ask and answer the right questions, namely what is encrypted, who holds the keys, what happens if we lose them, what happens if we are compelled to use them, how quickly we can change them, and how we can demonstrate control, cryptography stops being a mysterious technical requirement and becomes an integral part of defensible cybersecurity governance.

## Chapter summary

A policy stating that sensitive data must be encrypted hides the decisions that actually determine protection: which data, in which states, under whose control, with what recovery arrangements, and under what conditions the organization will decrypt. This chapter treats cryptography as a governance instrument and key management as the mechanism through which its power is exercised. It explains what encryption achieves and where it ends, derives requirements from classification, and examines key ownership, cloud key models, escrow and lawful access, certificates and machine trust, and the post-quantum transition. Its central message is that encryption is only as good as key management, and that cryptographic choices must be made and evidenced by accountable owners.

### Key points to retain

- Cryptography serves three distinct objectives, confidentiality, integrity, and authenticity, and it moves risk rather than eliminating it by concentrating protection in the keys.

- Encryption protects against lost devices, stolen media, interception, and some infrastructure compromises, but not against authorized misuse, application-logic attacks, or loss of one's own keys.

- Cryptographic expectations should be derived from information classification, embedded in baselines, and enforced by default, with exceptions documented, owned, compensated, and given an expiry date.

- Each data state, in transit, at rest, in use, and in logs and copies, raises its own question of who can decrypt and where keys are held.

- Keys are assets that need an inventory, owners, cryptoperiods, a governed lifecycle, protection in HSMs or managed services, and separation of duties with dual control for the most sensitive operations.

- Key ownership has legal, operational, and governance dimensions, which an operating model and a short policy with a regularly updated standard must allocate explicitly.

- Customer-managed key models transfer both control and responsibility: the organization gains the power to lock others out and also the power to lock itself out.

- Key escrow and recovery must balance continuity, insider risk, and lawful access, supported by an explicit lawful access position, multi-person control, and tested procedures.

- Certificates are machine identities, and the move to 47-day public certificates by 2029 makes automated, owned, and monitored certificate lifecycle management a resilience requirement.

- Post-quantum migration is a present-day governance program because of harvest-now-decrypt-later risk and long migration times, anchored by new NIST standards and the 2031 and 2035 milestones.

- Cryptographic assurance relies on configuration, coverage, key management, inventory, testing, and validation evidence, plus KRIs with owners and thresholds reported in business terms.

- Encryption weighs toward lower risk in breach assessments under Law 25, PIPEDA, and the GDPR only when the organization can prove the keys were not compromised.

### Key terms

- **Integrity protection** — The use of cryptographic hashes and message authentication codes to detect whether data has been altered in storage or transit.

- **Digital signature** — A cryptographic mechanism allowing a party to prove that a message, document, or software package came from a specific key holder and has not changed since signing.

- **Envelope encryption** — A key hierarchy in which a few highly protected master keys encrypt many data encryption keys, which in turn encrypt the data.

- **Cryptoperiod** — The time during which a specific key is authorized for use.

- **Hardware security module (HSM)** — A tamper-resistant device that generates and stores keys and performs cryptographic operations without exposing key material.

- **Dual control and split knowledge** — Arrangements requiring two or more authorized individuals to act together so that no single person ever holds a complete secret.

- **Confidential computing** — Protecting data in use by performing computation in a hardware-based, attested Trusted Execution Environment.

- **Bring your own key (BYOK)** — A model in which the customer generates key material in its own environment and imports it into the provider's key management service.

- **Hold your own key (HYOK)** — A model that keeps master keys outside the primary provider, which must call the customer's environment for each decryption.

- **Key escrow** — Storing copies of cryptographic keys with a trusted internal or external party so they can be recovered or used under defined conditions.

- **Public key infrastructure (PKI)** — The certificate authorities, policies, procedures, and technology that issue, manage, renew, and revoke certificates.

- **Harvest now, decrypt later** — An adversary's practice of capturing encrypted data today and storing it until it can be decrypted, for example by a future quantum computer.

- **Crypto-agility** — The ability to change cryptographic algorithms, parameters, and implementations quickly and safely without redesigning systems.

- **Cryptographic bill of materials (CBOM)** — An extension of the SBOM concept describing the cryptographic assets within a software product.

- **Cryptographic inventory** — A record of the algorithms, key lengths, protocols, certificates, libraries, and modules each system uses, with purpose, data protected, confidentiality lifetime, owner, and supplier.

## Review questions

### Essay questions

1.  A vendor tells your organization that all customer data in its SaaS platform is encrypted. Using the chapter's two anchoring questions and the four data states, analyze what this claim may and may not mean, and formulate the questions you would put to the vendor before accepting it.

2.  Analyze the Québec manufacturer's BYOK incident. Identify the managerial failures that allowed the key to be lost, evaluate the corrective actions taken, and recommend any further measures drawn from the chapter's treatment of keys as assets and of customer-managed key models.

3.  Choose an organization you know (or a fictional one) and evaluate where it should sit on the spectrum from provider-managed keys to HYOK for one critical cloud service. Justify the choice in terms of threats, availability risk, lawful access, data residency, and operational capability.

4.  The credit union's certificate outage occurred despite strong algorithms and a clean audit of its cryptographic standard. Using the risk triangle and the SC-081 certificate lifetime schedule, evaluate why governance rather than technology failed, and assess whether the board's response is sufficient for 2027 and 2029.

5.  Design the first two years of a post-quantum migration program for a Canadian organization that sells to the federal government. Explain how you would build the cryptographic inventory, prioritize systems, bring suppliers into the program, fund the work, and report progress to the board.

### Multiple-choice questions

Choose the best answer. Answers are listed in the appendix "Answers to multiple-choice questions".

**1.** A software publisher signs each release so customers can verify its origin. Which cryptographic objective does this primarily serve?

> A. Confidentiality, by preventing others from reading the software
>
> B. Availability, by ensuring the software can always be downloaded
>
> C. Authenticity, by proving the package came from a specific key holder
>
> D. Privacy, by limiting what personal data the software collects

**2.** A customer database is fully encrypted at rest. Which incident would this encryption do little to mitigate?

> A. An injection flaw in the web application that returns plaintext records
>
> B. Theft of a decommissioned disk that was improperly disposed of
>
> C. Loss of a backup tape during transport to an off-site vault
>
> D. A compromise of the hosting layer exposing raw storage volumes

**3.** Health records must stay confidential for 10 years, migration will take 7 years, and a quantum threat is expected in 12 years. What does Mosca's reasoning test conclude?

> A. The organization has five years of slack before it must start
>
> B. The organization can wait until a quantum computer is announced
>
> C. The test does not apply to health records, only to government data
>
> D. The organization is already late, because 10 plus 7 exceeds 12

**4.** In the chapter's sample classification table, what cryptographic expectation applies to restricted information such as card data and keys?

> A. Platform encryption by default with standard transport encryption
>
> B. Strong encryption everywhere, keys in an HSM or managed KMS, and strict separation of duties
>
> C. Integrity and authenticity protection for published content only
>
> D. Encryption in transit only, with plaintext storage permitted internally

**5.** Under the Canadian Centre for Cyber Security roadmap ITSM.40.001, by when should Government of Canada high-priority systems complete migration to post-quantum cryptography?

> A. By the end of 2026
>
> B. By the end of 2031
>
> C. By the end of 2030
>
> D. By the end of 2035

**6.** Which backup arrangement does the chapter describe as a particular trap?

> A. Backups encrypted with keys held under dual control
>
> B. Backups isolated from the production network
>
> C. Backups whose recovery keys are tested every year
>
> D. Backups whose key is stored only in the environment being backed up

**7.** Which dimension of key ownership answers for whether key management aligns with risk appetite, regulatory obligations, and internal policy?

> A. Governance accountability, shared among the CISO, data owners, and legal and privacy
>
> B. Operational control, held by whoever can generate, rotate, or revoke keys
>
> C. Legal ownership, held by the party with the right to control key use
>
> D. Technical custody, held by the vendor that configured the platform

**8.** What feature defines confidential computing, according to the Confidential Computing Consortium definition cited in the chapter?

> A. Field-level encryption that hides data from database administrators
>
> B. Tokenization that replaces sensitive values with surrogate tokens
>
> C. Computation in a hardware-based, attested Trusted Execution Environment
>
> D. End-to-end encryption with keys held only by individual users

**9.** An organization uses envelope encryption with a single master key protecting thousands of data keys. What governance consequence follows?

> A. Rotation and revocation become impractical across the hierarchy
>
> B. Data keys no longer need inventory entries or owners
>
> C. Separation of duties is no longer needed for administrators
>
> D. Loss or compromise of the master key has a very large blast radius

**10.** How does the chapter recommend reporting a certificate-expiry KRI to the board?

> A. As the total number of certificates held, for example 1,200 certificates
>
> B. In business terms, such as four member-facing services at risk of outage within 30 days
>
> C. As the percentage of certificates issued by private rather than public authorities
>
> D. As a technical appendix listing every certificate's expiry date

**11.** Why do mature organizations use dual control and split knowledge for operations such as restoring a master key from backup?

> A. So that no single person can perform the operation or hold the complete secret
>
> B. So that the operation can be completed faster during an incident
>
> C. So that the HSM vendor can perform the operation on the customer's behalf
>
> D. So that key usage logs are no longer required for these operations

**12.** NIST's transition schedule placed all remaining FIPS 140-2 certificates on the historical list in September 2026. What should managers conclude?

> A. Modules validated under FIPS 140-2 are now prohibited from operation
>
> B. Validation status no longer matters for cryptographic procurement
>
> C. FIPS 140-2 certificates should not be relied upon for new procurements
>
> D. Only Canadian modules require FIPS 140-3 validation from now on

**13.** An organization adopts HYOK for several SaaS applications, keeping master keys in its own external key service. What new failure mode does this introduce?

> A. The provider can now decrypt data without the customer's knowledge
>
> B. An outage of the customer's key service disables every dependent cloud application
>
> C. Key usage can no longer be logged in the customer's own account
>
> D. The provider's application layer can no longer process any data

**14.** Laptop recovery keys are escrowed in the organization's device management service. How should access to that escrow be governed?

> A. Restricted, logged, and reviewed like any other privileged capability
>
> B. Open to all help desk staff without logging, to speed support
>
> C. Delegated to the device vendor, which holds legal ownership of the keys
>
> D. Disabled entirely, accepting that forgotten passwords mean lost devices

**15.** In the Québec manufacturer scenario, the BYOK master key could not be found. What did the post-incident review identify as the root cause?

> A. A weakness in the SaaS provider's key management algorithm
>
> B. An attack that deliberately destroyed the on-premises HSM
>
> C. The defence customer's unrealistic encryption requirements
>
> D. Managerial failures: no owner, no inventory entry, and no cryptographic dependency check at decommissioning

**16.** Under SC-081, public TLS certificate validity falls to 47 days by 2029. Roughly how many renewals per certificate per year does this imply, compared with one today?

> A. About two renewals per year
>
> B. About four renewals per year
>
> C. About eight renewals per year
>
> D. About twelve renewals per year

**17.** Under CA/Browser Forum ballot SC-081, what is the maximum validity of a publicly trusted TLS certificate issued on or after March 15, 2027?

> A. 200 days
>
> B. 47 days
>
> C. 100 days
>
> D. 398 days

**18.** Why does harvest-now-decrypt-later make post-quantum migration a present-day concern?

> A. Because current symmetric encryption is already broken by quantum computers
>
> B. Because exposure for long-lived data begins when it is intercepted, not when a quantum computer appears
>
> C. Because NIST has already disallowed RSA and elliptic curve algorithms
>
> D. Because quantum computers can alter digital signatures retroactively

**19.** Which NIST standard specifies the algorithm intended as the primary mechanism for establishing shared keys in the post-quantum transition?

> A. FIPS 203, specifying ML-KEM
>
> B. FIPS 204, specifying ML-DSA
>
> C. FIPS 205, specifying SLH-DSA
>
> D. FIPS 206, specifying FN-DSA

**20.** After encrypted customer data is stolen, what most determines whether encryption counts as a mitigating factor in a Law 25, PIPEDA, or GDPR breach assessment?

> A. Whether the encryption algorithm was selected by the vendor
>
> B. Whether the data was stored in a Canadian data centre
>
> C. Whether the organization holds an ISO/IEC 27001 certificate
>
> D. Whether the organization can prove the keys were not compromised
