<!-- Introduction to Cybersecurity Governance, 3rd ed., v2.1g (Aug 2026) - (c) Marc-André Léger. Licensed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/). Third-party standards, frameworks and quotations remain under their owners' terms. -->

# Chapter 21: Cryptography and Key Management in Managerial Terms

In most organizations, encryption appears in policies as a one-line requirement: sensitive data must be encrypted. This sounds reassuring, but it hides the decisions that truly matter which data, in which locations, using whose technology, under whose control, and with what recovery and lawful access conditions.

Cryptography is not only a technical control. It is a governance instrument. It shapes who can see what, under what circumstances, and with what evidentiary trail. Key management, in turn, is the practical mechanism by which that power is exercised. Keys determine whether information is accessible, recoverable, or irreversibly lost. They also determine whether the organization can comply with lawful requests, preserve evidence, or respond to internal fraud and misuse.

For business technology managers, the central message is that cryptography and key management require explicit governance choices. Those choices affect privacy, regulatory compliance, operational resilience, insider risk, cloud dependency, and the organization’s relationship with law enforcement and regulators. If those decisions are left entirely to technical teams or to vendors’ default settings, the organization may find itself compliant on paper but fragile or overexposed in practice.

This chapter therefore focuses on managerial questions: what should be encrypted; what protection encryption provides; who should own and operate keys; what key escrow and lawful access really mean; and how customer-managed keys change responsibilities in cloud and SaaS environments.

## What encryption does, and what it does not do

To govern cryptography, managers do not need to understand algorithms, but they do need a clear grasp of what encryption does and does not achieve.

Encryption protects data against unauthorized access on the path or at rest. When applied correctly, it prevents people who do not have the key from reading the data, even if they obtain the physical storage, intercept the traffic, or access infrastructure. It is particularly powerful against some classes of breach: theft or loss of devices, compromise of storage media, interception over untrusted networks, and certain cloud or hosting-layer compromises.

At the same time, encryption does not make data safe against everything. It does not protect against misuse by authorized users, including insiders acting maliciously or negligently. It does not protect against attacks that exploit application logic when the application is legitimately decrypting the data. It does not compensate for weak identity and access management, poor endpoint security, or badly designed business processes. And it does not, by itself, guarantee integrity or availability; those require additional controls and operational discipline.

A useful way to think about encryption is to ask two questions for each system or dataset. First: against which threats are we expecting encryption to protect us? Second: at what points must the data be in usable form, and what protections exist at those points? This keeps cryptography anchored rather than in magical thinking.

## What should be encrypted? A classification-driven view

In principle, all sensitive information should be encrypted in transit and at rest. In practice, organizations must still prioritize and design around the most critical impact areas, because some data stores and processes are more consequential than others.

A classification-driven view is the most manageable. Organizations usually distinguish several levels of information: public or non-sensitive, internal, confidential, and restricted or highly confidential. Cryptographic expectations are then aligned with these levels.

Highly confidential and restricted information should be subject to strong encryption by default. This usually includes payment card data, bank account details, government identifiers, health and medical information, sensitive personal information under privacy laws, trade secrets and proprietary algorithms, critical system configuration data, encryption keys and authentication secrets themselves, and certain categories of logs (for example, those containing personal identifiers or security tokens).

Confidential information should also be encrypted, especially at rest in multi-tenant or cloud environments. This tends to include most customer data, internal financial data, contract details, HR records, and operational logs that could reveal sensitive business patterns or internal structure.

Internal information may still warrant encryption depending on context. In modern environments, the marginal cost of encrypting storage volumes and network traffic is often low, so organizations increasingly treat encryption at rest and in transit as the default even for less sensitive categories.

Two domains require special attention. The first is backups and archives. These often contain the most complete historical picture of sensitive data and are prime targets in ransomware and data extortion scenarios. If backups are not encrypted and properly isolated, the organization’s last line of defense is weaker than its production systems. The second is machine-to-machine secrets: API keys, service accounts, tokens, and configuration files that embed credentials. These should be encrypted at rest and handled with the same seriousness as user credentials.

The managerial principle is straightforward: if a category of data would cause serious harm or regulatory exposure if exfiltrated, it should be encrypted by design wherever it resides and moves, not as a discretionary option.

## Where encryption matters: in transit, at rest, in use, and in logs

Cryptography operates in several domains, each with its own governance questions.

In transit, encryption protects data as it moves over networks between users, systems, and services. The managerial decision here is not whether to use it—modern practice is that all external and most internal traffic should be protected—but how consistently and with what assurance. Managers should expect that web and API traffic use modern transport encryption (for example, TLS), that administrative and remote access sessions are protected, and that inter-service communication in cloud environments is either encrypted or appropriately constrained and monitored. Exceptions should be rare, justified, and documented.

At rest, encryption protects data stored on disks, in databases, in file systems, and in object stores. Here, the nuance is that many platforms provide transparent encryption, where the platform encrypts data on disk but decrypts it automatically for applications. This is valuable for certain threat scenarios (lost disks, stolen backups, some infrastructure compromises) but does not protect against misuse by applications or authenticated users. Managers should understand that storage encryption enabled is a necessary baseline, not the endpoint.

In use encryption—such as confidential computing and related technologies—aims to protect data while it is being processed by keeping it protected against the underlying infrastructure. This is still emerging for many use cases, but for particularly sensitive workloads it may become a relevant governance option. For now, managers mainly need to know that encryption in use exists as a possible control, not to design it themselves.

Logs and telemetry present a tension. On the one hand, they are essential for security, audit, and operations. On the other hand, they can leak sensitive data if not handled carefully. A governance-aware approach to cryptography requires that logs containing personal information or secrets be protected in storage, that their access be restricted and monitored, and that retention be controlled according to privacy and legal requirements.

In each of these domains, the critical managerial questions are: what is protected, who can decrypt, where are keys stored, and how is access to keys governed and evidenced?

## Key management fundamentals in managerial language

Encryption without key management is an illusion. Keys are the practical expression of cryptographic power. If keys are poorly generated, stored, or managed, the apparent strength of algorithms and protocols does not translate into real protection.

For managers, several key concepts are important.

First, keys are assets. They deserve classification, ownership, and protection comparable to—or stronger than—the data they protect. This is particularly true for root or master keys that can decrypt large volumes of data or generate other keys.

Second, key management follows a lifecycle: generation, distribution, storage, use, rotation, revocation, backup, and destruction. Each stage presents risk. Poorly generated keys may be predictable; poorly distributed keys may be intercepted or misdirected; keys without rotation may be misused for long periods; keys without proper destruction may linger in backups or configuration files.

Third, specialized systems exist for key management. Hardware security modules (HSMs) and key management services (KMS) are designed to protect keys from extraction, enforce controlled operations (for example, decrypt or sign without exposing keys), and log key usage. In cloud environments, provider KMS services are central components of the shared responsibility model.

Fourth, separation of duties is a governance concept applied to keys. The people who administer storage systems should not be able to export master keys; the people who administer HSMs should not be able to silently use them without logging; application developers should not hard-code keys in source code. Proper separation prevents a single insider or a single compromised role from undermining security.

Finally, key management generates evidence. Logs of key use, rotation, and administrative changes become critical in incident response, regulatory inquiries, and audits. Without such evidence, claims about strong encryption are difficult to substantiate.

The managerial takeaway is that when evaluating systems and vendors, leaders should ask as much about key management as about encryption features. The question of knowing where the keys are, and who controls them is often more important than determining if the data is encrypted.

## Ownership and accountability: who owns keys?

Key ownership is not trivial. It has several dimensions: legal ownership, operational control, and governance accountability.

Legal ownership refers to who has the legal right to control keys and decide on their use. For data entrusted to a cloud provider or SaaS platform, the organization usually remains the controller of the data even if keys are technically managed by the provider. Contracts should clarify that keys associated with the customer’s data are used only according to the customer’s instructions and applicable law.

Operational control refers to who can operate on keys: generate new keys, rotate them, revoke them, or use them for decryption and signing. In traditional on-premises environments, this might be the internal security or infrastructure team operating HSMs. In cloud environments, operational control may be shared between the provider’s KMS and the customer, depending on the model.

Governance accountability refers to who is responsible for ensuring that key management aligns with risk appetite, regulatory obligations, and internal policies. This is typically a shared responsibility between the CISO function (for technical soundness), data or system owners (for business relevance), and privacy or legal functions (for lawful access and compliance).

A clear operating model often distinguishes roles. The CISO or security architecture defines key management standards, including which systems must use HSMs, how keys are rotated, and how separation of duties is implemented. Operations teams implement those standards through KMS, HSM, and configuration management. Business owners approve key usage in line with business processes and risk. Legal and privacy functions define conditions under which keys can be used in response to lawful requests.

Without this clarity, the organization may find that critical keys are effectively owned by whichever administrator created them, or by a vendor whose contractual obligations are vague. That is a governance failure, not just a technical one.

## Customer-managed keys in the cloud: what it means

Customer-managed keys (CMK) has become a common term in cloud and SaaS marketing. It suggests greater customer control and sometimes implies stronger confidentiality. It covers several different models, and the operational implications are significant.

At one end of the spectrum, provider-managed keys are fully handled by the cloud provider. The provider generates, stores, rotates, and uses keys, exposing only configuration options to the customer. This is simple to operate but leaves the provider with substantial control.

Customer-managed keys, in a common sense, mean that keys used to encrypt customer data in provider systems are created, controlled, and managed by the customer via a KMS interface. The provider uses those keys (or derived keys) to encrypt data but cannot use them for any other purpose and cannot access keys outside of defined operations. In many designs, the provider’s KMS executes encryption and decryption operations inside a protected environment, and the customer controls policies, rotation schedules, and access via their own accounts.

Some providers and architectures go further into hold your own key models, where master keys are stored in on-premises HSMs or in a separate provider, and the primary cloud provider never sees actual key material. In these models, loss of access to the external key infrastructure can make data permanently unrecoverable.

Operationally, customer-managed keys change responsibilities in several ways.

First, they shift part of the availability risk to the customer. If the customer mismanages keys, rotates them incorrectly, or loses access to the key management environment, they may lose access to their data even when the provider’s infrastructure is functioning normally.

Second, they require disciplined key lifecycle management. CMK configurations must be documented, changes must be controlled, and rotation events must be coordinated with dependent systems. This increases operational complexity.

Third, they improve the customer’s ability to enforce certain access policies. For example, the customer can revoke keys to render stored data inaccessible or can require explicit approval for certain key operations. However, this does not automatically prevent all access routes; providers may retain capabilities to comply with legal orders or to operate the service, depending on architecture and jurisdiction.

For managers, the key questions about CMK are: what exactly is under our control; what failure modes do we introduce by taking this control; how do CMKs interact with data residency and lawful access; and what evidence can we provide to regulators and clients about our control over keys?

## Key escrow, recovery, and lawful access

Key escrow is the practice of storing copies of cryptographic keys with a trusted party to enable recovery or access under defined conditions. It is a sensitive governance topic because it sits at the intersection of business continuity, insider risk, and lawful access.

From a business continuity standpoint, some form of key recovery is often necessary. If all copies of a critical key are lost, data may become unrecoverable, which can be catastrophic, particularly for regulated data that must remain accessible for legal retention periods. Escrow or other recovery mechanisms (for example, key hierarchies where master keys can derive new keys) can mitigate this risk.

From an insider risk standpoint, escrow creates another concentrated asset. If escrow arrangements are weakly governed, individuals with access to escrowed keys can potentially decrypt large volumes of data without detection. This is why mature key recovery designs use multi-person control, audit logging, and strict procedural safeguards for any recovery operation.

From a lawful access standpoint, organizations may receive legally binding orders to provide access to data. Whether and how this is possible depends on architecture and key control arrangements. In some models, the organization can comply by using its own keys to decrypt data or to produce data in plaintext form. In others, particularly where external providers operate the encryption environment, lawful access may involve those providers. In designs where the organization deliberately cannot decrypt certain data (for example, end-to-end encryption with keys held only by end users), compliance options are more limited.

Governance requires a position on lawful access. It should be formulated with legal counsel and privacy leadership, but the technical design must make it implementable. The position should address at least four questions: under what legal conditions will the organization provide decrypted data; who is authorized to approve such actions; what records and audit trails are required; and what system architectures are compatible or incompatible with that stance.

Key escrow decisions should be documented and risk-assessed. For some data categories, strong recovery and lawful access requirements will justify carefully controlled escrow. For others, the organization may accept that loss of keys will render data unrecoverable, as a trade-off for stronger confidentiality. What matters is that these trade-offs are conscious and aligned with obligations and risk appetite.

## Certificates, PKI, and machine trust: the identity side of cryptography

Much of cryptography in organizations is not about encrypting stored data; it is about establishing trust between machines and services. This is the realm of certificates and public key infrastructure (PKI).

Certificates are digital credentials that bind identities (domains, services, users, devices) to cryptographic keys. They are used in protocols like TLS for secure web and API connections, in VPNs, in device authentication, and in many internal systems. A PKI is the system that issues, manages, and revokes these certificates.

From a managerial perspective, poorly governed PKI creates serious operational risk. Expired certificates cause outages; mis-issued certificates enable impersonation; weakly protected certificate authority keys threaten entire environments. Conversely, a well-governed PKI simplifies secure connectivity and identity assurance at scale.

Managers should expect that their organizations have at least the following: clarity on who operates certificate authorities; defined processes for requesting, approving, issuing, and renewing certificates; visibility into certificate inventories and expiry; and procedures for revoking certificates when keys are compromised or systems are decommissioned. For externally visible services, they should also expect that certificates are acquired from trusted public authorities and configured correctly.

PKI governance also intersects with cloud and third-party security. Many providers rely on certificates for internal trust relationships. When organizations extend their networks or identity systems into cloud and SaaS environments, PKI and certificate management decisions can either enforce strong boundaries or create brittle dependencies.

The key message is that cryptography is not only about encrypting data. It is also about establishing and managing trust relationships among machines and systems. Governance of PKI is therefore part of overall key management and should be treated with similar seriousness.

## Measuring and assuring cryptographic controls

Because cryptographic choices are technical, it is easy for boards and executives to either ignore them or accept assurances without verification. A governance-oriented program establishes ways to measure and assure the effectiveness of encryption and key management.

Several forms of evidence are particularly useful:

- **Configuration evidence** shows where encryption is enabled and how. This includes storage encryption settings for databases, file systems, and object stores; transport encryption settings for external and internal services; and KMS or HSM configurations for key use.

- **Coverage evidence** demonstrates that critical systems and data sets are within the scope of encryption standards. For example, reports that all systems containing payment data use strong encryption at rest, or that all administrative remote access channels use secure protocols.

- **Key management evidence** includes logs of key creation, rotation, and usage; evidence that rotation policies are enforced; records of role assignments and separation of duties; and documented key recovery tests.

- **Testing evidence** comes from penetration tests, configuration audits, and incident post-mortems. For example, assessments that confirm that stolen disks cannot be decrypted without keys; that attackers cannot trivially extract keys from application servers; or that TLS configurations resist known downgrade attacks.

- **Compliance and certification evidence** may also be relevant, particularly in cloud contexts, but it should be considered supplementary. Certifications indicate that providers have controls, not that they are perfectly aligned with the organization’s specific threats and requirements.

For managers, the practical approach is to ask regular, structured questions. For our most critical systems, can we show that encryption is properly enabled? Do we know where keys are and who can operate them? Have we tested key recovery and disaster scenarios? Have recent incidents revealed gaps between our encryption claims and reality? Answers to these questions are more meaningful than a generic claim that we use strong encryption.

## Integration with privacy, data protection, and cross-border obligations

Cryptography and key management are deeply intertwined with privacy and data protection, particularly in jurisdictions that emphasize both safeguarding and demonstrable accountability.

Encryption is frequently mentioned in privacy laws and guidance as an appropriate safeguard, and in some cases as a factor in assessing whether a breach triggers notification obligations. Properly encrypted data may be considered lower risk if keys are not compromised. However, this is contingent on strong key management. If keys are poorly controlled or if logs show that attackers likely obtained decryption capability, regulators will not treat encryption as a mitigating factor.

Data residency and cross-border concerns also intersect with cryptography. Some organizations rely on encryption and key location (for example, keeping keys only in certain jurisdictions) as part of their cross-border risk management strategy. For this to be meaningful, the architecture must ensure that providers cannot decrypt data outside the intended jurisdiction without the organization’s cooperation, and that legal arrangements support this stance. Again, the key question is who controls keys and where.

Anonymization and pseudonymization, discussed in earlier chapters, have cryptographic dimensions as well. Tokenization schemes that replace sensitive identifiers with tokens may rely on key-protected mapping tables. If keys or mapping tables are compromised, pseudonymized data may revert to personal data in the eyes of regulators. Governance must therefore treat such systems as part of key management, not as simple data transformation.

Finally, privacy impact assessments for new systems should explicitly consider cryptographic choices: what is encrypted, where, with which key management model, and what residual risk remains. Cryptography cannot replace compliance, but when properly governed it becomes one of the strongest technical arguments for due diligence and proportionality.

## Bringing it together: cryptography and key management in the governance story

This chapter has treated cryptography and key management not as mathematical topics, but as instruments of governance. The central ideas are straightforward.

First, encryption is only as good as key management. Claims of strong encryption are empty if keys are uncontrolled, mismanaged, or concentrated in ways that are inconsistent with risk appetite.

Second, cryptographic choices embody governance decisions about who can access data, under what conditions, with what audit trail, and how recovery and lawful access are handled. Those decisions must be made consciously, not left to default vendor configurations or ad hoc practices.

Third, customer-managed keys and cloud KMS models give organizations more control, but they also impose greater responsibility for availability, lifecycle management, and lawful access decisions. They are governance options, not automatic improvements.

Fourth, key escrow and recovery mechanisms sit at the intersection of continuity, insider risk, and legal compliance. They require clear policies, multi-party controls, and transparent reasoning about trade-offs.

Fifth, cryptography is not only about data at rest; it is also about machine trust, PKI, and the secure operation of identity and access systems. Failures here can produce outages and impersonation even when data-at-rest encryption looks sound.

Finally, in the broader governance orientation of this book, cryptography and key management are cross-cutting enabling controls. They support privacy engineering, cloud and third-party governance, incident management and resilience, and the economic logic of risk reduction. When managers can ask and answer the right questions—what is encrypted; who holds the keys; what happens if we lose them; what happens if we are compelled to use them; and how we can demonstrate control—cryptography stops being a mysterious technical requirement and becomes an integral part of defensible cybersecurity governance.

Haut du formulaire

Bas du formulaire
