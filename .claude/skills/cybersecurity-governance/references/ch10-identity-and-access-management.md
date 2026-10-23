<!-- Léger, M.-A. (2026). Introduction to Cybersecurity Governance for Business Technology Management (3rd ed., v2.51). Chapter 10: Identity and Access Management. CC BY-NC 4.0 -->

# Chapter 10: Identity and Access Management

## Learning outcomes

After completing this chapter, you should be able to:

1.  Distinguish Identity and Access Management (IAM) from Privileged Access Management (PAM) and explain why privileged access is delegated organizational authority.

2.  Explain the structural reasons why identity weaknesses initiate, enable, or amplify so many cybersecurity incidents.

3.  Design an IAM operating model that assigns decision rights across central identity, business ownership, and risk and assurance functions, including governed exceptions.

4.  Apply joiner–mover–leaver lifecycle discipline to human and non-human identities and identify the controls that prevent privilege creep and orphaned access.

5.  Recommend an authentication strategy that matches assurance to consequence, using phishing-resistant methods, NIST SP 800-63-4 assurance levels, conditional access, and secure recovery.

6.  Compare RBAC and ABAC authorization models and evaluate whether access reviews and segregation-of-duties controls provide real assurance rather than completed paperwork.

7.  Evaluate PAM, break-glass, third-party, and non-human identity controls against the principle that authority must be minimized, attributable, monitored, and revocable.

8.  Justify a risk-based IAM and PAM roadmap supported by reliable evidence and governance-grade metrics that measure exposure rather than administrative activity.

In modern enterprises and cloud environments, identity has become one of the most important cybersecurity control planes. Networks remain essential, but they are no longer the sole—or necessarily the primary—boundary for access control. Users, applications, devices, workloads, data, service accounts, APIs, and third-party integrations now operate across SaaS platforms, cloud services, remote-work environments, partner ecosystems, and hybrid infrastructures. The central cybersecurity question is therefore no longer simply how the network is segmented, but **who or what may access which resources, under what conditions, with what level of privilege, and how that access is monitored, reviewed, and evidenced over time**.

For managers, **Identity and Access Management (IAM)** and **Privileged Access Management (PAM)** are not merely technical or administrative functions. They are fundamental governance control domains. They determine who can act within organizational systems, approve transactions, view or modify sensitive information, administer infrastructure, change configurations, and override normal safeguards. In this sense, IAM and PAM directly shape the organization's exposure to fraud, sabotage, ransomware, espionage, insider misuse, privacy and regulatory breaches, and operational disruption.

When IAM fails, the organization loses control of routine access. When PAM fails, it loses control of its most powerful capabilities. Many significant cyber incidents are initiated, enabled, or amplified by weaknesses in identity governance: stolen credentials, weak authentication, incomplete deployment of multi-factor authentication, excessive privileges, poor separation between standard and administrative accounts, unmanaged service accounts, exposed API keys and secrets, ineffective joiner–mover–leaver processes, and insufficient monitoring of privileged activity. Attacks that appear highly technical on the surface are often, at their core, failures of authentication, authorization, or privilege governance. An attacker may first obtain a legitimate user's identity, then acquire additional privileges, then gain administrative authority, and ultimately act using permissions the organization itself has granted. At that point, the distinction between an external attacker and an authorized user becomes operationally blurred.

Cloud and SaaS environments intensify this reality because identity often serves as the primary control plane. Administrative identities may control management consoles, databases, storage, virtual networks, encryption keys, logging systems, security controls, and workloads. A compromised cloud administrator account can therefore allow an attacker to alter configurations, expose or delete data, disable monitoring, manipulate encryption keys, establish persistence, create new identities, or move across interconnected environments—often across a large portion of the technology estate at once.

The same governance challenge applies to **non-human identities**: service accounts, workloads, automation scripts, API credentials, tokens, secrets, certificates, and machine-to-machine identities. These identities are increasingly numerous and often hold substantial privileges because they enable automated business and technical processes, yet they may receive less governance attention than human accounts. Credentials may be long-lived, ownership may be unclear, permissions may accumulate, and access may persist long after the original requirement has disappeared. As organizations rely more on cloud services, APIs, automation, and artificial intelligence, governing these identities becomes an increasingly important part of cybersecurity governance.

IAM and PAM warrant dedicated attention because they bridge governance intent and technical execution. Risk appetite, compliance obligations, segregation of duties, privacy requirements, least privilege, operational resilience, and accountability all depend on the organization's ability to govern identity. Policies may state that access must be restricted, privileges minimized, and sensitive actions monitored, but those objectives become meaningful only when identity controls translate them into enforceable system behaviour. Authentication, authorization, provisioning, access reviews, privileged-session management, secrets management, logging, and evidence collection are therefore not isolated technical tasks. Together, they determine who or what may act, what actions may be performed, under which conditions, with how much authority, and how management knows whether that access remains appropriate.

This chapter builds on the governance foundations of Chapters 2 and 4, the scenario-based risk assessment of Chapters 5 and 6, and the control frameworks of Chapter 9. It defines IAM and PAM and the boundary between them, explains why identity failures dominate the incident landscape, and sets out governance objectives, decision rights, and the identity lifecycle. It then examines authentication, authorization, segregation of duties, third-party access, privileged access, emergency access, and non-human identities, before turning to evidence, metrics, and a practical implementation roadmap. Effective identity governance is not simply about controlling access to technology. It is about controlling the exercise of organizational authority within digital systems.

## IAM and PAM: definitions and the managerial boundary

Identity and Access Management encompasses the policies, processes, technologies, roles, and accountability mechanisms an organization uses to govern digital identities and their access to information, systems, applications, and other resources. IAM covers both human and non-human identities, including employees, contractors, partners, customers, devices, applications, service accounts, and workloads. It governs how these identities are established, authenticated, authorized, monitored, reviewed, modified, and ultimately removed.

In practical terms, IAM includes identity lifecycle management, authentication, authorization, provisioning and deprovisioning, identity federation, single sign-on, access reviews, and the evidence needed to demonstrate that access remains appropriate. Its purpose is not simply to grant access, but to ensure that access aligns with a legitimate business need, reflects the individual's or system's current role, and changes as responsibilities, employment status, contractual relationships, or operational requirements change.

Privileged Access Management is a specialized domain within identity governance focused on identities, permissions, credentials, and activities that confer elevated or exceptional authority. **Privileged access** is access that can materially affect systems, security configurations, data protection, business processes, financial outcomes, or other users' access rights. It includes administering infrastructure, modifying configurations, creating or deleting identities, changing permissions, disabling security controls, accessing sensitive information at scale, executing high-impact transactions, or intervening in systems during emergencies. Typical PAM mechanisms include credential vaulting, administrative account separation, just-in-time and just-enough privilege, approval workflows, privileged-session monitoring and recording, periodic privileged-access reviews, break-glass procedures, and controls for privileged non-human identities. These controls matter because privileged access can bypass, modify, or disable the safeguards that constrain ordinary users.

A useful managerial distinction is therefore the following.

| **IAM governs access across the organization, while PAM governs exceptional privileges within it.** |
|-----------------------------------------------------------------------------------------------------|

IAM asks whether a person, device, application, or service has appropriate access to perform legitimate work. PAM asks whether access that could lead to disproportionate consequences is minimized, justified, tightly controlled, monitored, and documented. The distinction is not absolute—PAM is closely integrated with IAM—but it is useful because the consequences of privileged access failures are generally much greater. The figure below summarizes the relationship.

<span id="_Toc243635549" class="anchor"></span>Figure 50: IAM and PAM

Both domains are essential, but PAM warrants greater scrutiny because privileged identities can alter the environment in which other controls operate. Mismanaged ordinary access can expose sensitive information, facilitate fraud, or violate segregation-of-duties requirements. Mismanaged privileged access can enable an attacker or malicious insider to change the organization's security posture itself. Compromising a domain administrator, cloud management account, database administrator, highly privileged service account, or automation credential may allow an adversary to create identities, elevate privileges, disable monitoring, modify or delete logs, exfiltrate large volumes of information, deploy malicious software, destroy backups, or establish persistent access. Because privileged identities can modify the safeguards intended to constrain activity, their compromise can undermine several layers of protection simultaneously.

This is why privileged access is more than a stronger form of ordinary access. It is delegated organizational authority within digital systems. A privileged identity may be able to perform actions that would otherwise require formal organizational authority: approving changes, modifying financial or operational records, altering security policies, granting access to others, or interrupting critical services. PAM therefore governs not only credentials but the controlled exercise of this delegated authority.

For Business Technology Management professionals, the two domains raise distinct governance questions. IAM governance asks whether access is based on legitimate business requirements, aligned with roles and responsibilities, periodically reviewed, adjusted as circumstances change, and promptly removed when no longer required. PAM governance asks whether exceptional authority has been minimized, separately assigned, specifically approved, limited in duration and scope where practicable, isolated from routine activity, closely monitored, and subject to independent review.

Together, IAM and PAM operationalize several fundamental governance principles. **Least privilege** requires that identities receive no more authority than necessary. **Segregation of duties** prevents a single identity from exercising incompatible powers without oversight. **Accountability** requires that significant actions be attributable to identifiable actors or controlled services. **Lifecycle governance** ensures that access evolves as organizational circumstances change. **Monitoring and evidence** allow management to determine whether access controls function as intended, rather than merely assuming they do. The managerial objective is to ensure that digital authority is deliberately granted, proportionate to legitimate need, constrained by appropriate controls, observable, periodically reassessed, and withdrawn when no longer justified.

## Why identity failures dominate the incident landscape

Managers should understand why identity-related weaknesses appear so frequently in cybersecurity incidents. Compromising a legitimate identity allows attackers to enter and operate within the organization through the same pathways used by employees, administrators, applications, devices, and suppliers. Once an attacker obtains valid credentials, a usable session, or another trusted identity, malicious activity may no longer resemble a conventional intrusion. To the systems involved, it may initially look like legitimate work performed by an authorized user or service. Five structural reasons explain why.

- **Credentials are scalable attack targets.** Attackers can steal, phish, purchase, reuse, guess, or extract passwords, tokens, session cookies, and API keys from users and compromised systems. Credential stuffing exploits password reuse across services, phishing exploits trust and urgency, and information stolen from compromised devices or browsers can be used to hijack authenticated sessions. This combination of availability, reuse, and automation makes identity compromise an efficient initial-access vector.

- **Identity relationships extend beyond the organization's boundaries.** Contractors, suppliers, consultants, managed service providers, cloud platforms, software integrations, and APIs all rely on some form of digital trust relationship. Each creates business value, but each is also a potential pathway into organizational resources. The governance problem is no longer which employees should have access, but which people and systems across an interconnected ecosystem should be trusted, to what extent, and under what conditions.

- **Permissions accumulate over time.** Access is granted quickly because business activity depends on it, while removal receives less attention because it rarely delivers an immediate benefit. People change roles, join projects, and leave; consultants complete assignments; temporary exceptions persist; service accounts are reused for new purposes. This **privilege creep** enlarges the attack surface, weakens least privilege, and can create segregation-of-duties conflicts.

- **Privileged pathways may operate outside normal controls.** Shared administrator accounts, long-lived service-account passwords, unmanaged API keys, permanent local-administrator rights, embedded credentials in scripts, and broadly assigned cloud-administrator roles may escape monitoring, rotation, review, or ownership. Because privileged identities can alter the environment itself, their compromise can turn a limited intrusion into an enterprise-wide incident.

- **Identity-based attacks resemble legitimate activity.** A firewall may permit the connection, an application may accept the credentials, a cloud platform may validate the token, and a database may authorize the query. From each control's perspective the activity is permissible because the identity holds valid rights. Technical authorization does not mean that the person or process exercising it is legitimate.

This last point highlights a key limitation of perimeter-focused security. Knowing that an authenticated identity is permitted to perform an action does not establish that the activity is appropriate in the current circumstances. Organizations increasingly need additional context: the device being used, location, authentication strength, the sensitivity of the resource, time of access, historical behaviour, and whether the action is consistent with the identity's legitimate role. This is why phishing-resistant authentication, conditional access, behavioural analytics, and privileged-session monitoring have become increasingly important.

From a governance perspective, IAM and PAM are high-leverage control domains because they influence both the likelihood and the consequences of incidents; in the terms of the risk triangle, they reduce exploitable vulnerability and constrain impact. They reduce the likelihood of successful compromise by strengthening authentication, controlling identity creation, limiting unnecessary access, and managing credentials and lifecycles. They reduce the impact of compromise by enforcing least privilege, segregation of duties, administrative-account separation, conditional access, and just-in-time privilege. They also strengthen accountability by showing who or what had access, who approved it, when it was exercised, and whether it remained justified. This is why identity weaknesses feature so prominently in the risk scenarios developed in Chapters 5 and 6: a stolen administrator credential or an orphaned supplier account is often the hinge on which a scenario turns from minor to severe.

The business case is therefore much broader than protecting login processes. Strong IAM and PAM can reduce exposure to fraud, limit the blast radius of ransomware, constrain insider misuse, reduce privacy risk, strengthen control over supplier access, and improve auditability. Every digital identity represents some degree of delegated organizational authority: IAM determines how broadly that authority is distributed, and PAM determines how the organization controls its most consequential digital powers. **When identity governance fails, attackers do not necessarily need to defeat the organization's controls—they may instead inherit the authority to use, modify, or disable them.**

## Governance objectives for IAM and PAM: what good looks like

A mature IAM and PAM program should deliver clear, observable, and defensible outcomes. Its purpose is not merely to administer accounts or automate access requests, but to ensure that digital access is appropriate, justified, proportionate, monitored, and continuously aligned with legitimate business needs. Effective governance should enable management to demonstrate not only that access controls exist, but that they operate consistently and that deviations are identified and corrected. Five objectives are especially important.

### Verified identity

Before granting access, the organization must establish, with an appropriate level of confidence, that the person, device, application, workload, or service requesting access is the identity it claims to be. For human users, this includes appropriate identity proofing, strong authentication, and higher-assurance requirements for sensitive or privileged roles. The required assurance should be proportionate to the consequences of unauthorized access rather than applied uniformly. For non-human identities, verification means confidence that the workload, API client, automation process, or device presenting credentials is the intended system and has not been replaced, impersonated, or compromised. Authentication is therefore not simply about passwords or users; it is about establishing sufficient confidence in any actor permitted to exercise digital authority.

### Least privilege and need-to-know by default

Access should be granted only to the extent necessary to perform a legitimate role, task, process, or system function—not because it might be useful later, because similar users already have it, or because broad permissions are easier to administer. Privileged access requires even greater discipline: elevated permissions should be minimized, kept separate from ordinary access, limited in scope and duration where practicable, explicitly approved, and removed when the requirement ends. Least privilege reduces both the likelihood of inappropriate access and the blast radius if an identity is compromised.

### Segregation of duties and controlled decision rights

High-risk activities should not rely on the unchecked authority of a single individual or identity. Where the consequences justify it, the stages of a sensitive process should be distributed across separate roles so that no one actor can initiate, approve, execute, and conceal the same high-impact action. The principle applies to payments, procurement, payroll, identity administration, privileged-access approval, security configuration, change management, and incident recovery. It is examined in detail later in this chapter.

### Identity lifecycle discipline

Access should follow the legitimate lifecycle of the business relationship it represents. It must be created, modified, suspended, reviewed, and removed as people join, change responsibilities, take extended leave, or leave—and as contractors complete assignments, applications are retired, integrations are discontinued, and credentials expire. Without disciplined lifecycle governance, access persists after its justification has disappeared, making lifecycle failures one of the most predictable sources of excessive access.

### Auditability and evidence

The organization must be able to reconstruct and demonstrate how access was governed: who or what had access, which permissions were assigned, why, who approved them, when they were used, which significant actions were performed, and when access was reviewed or removed. The requirement is strongest for privileged access, because privileged actors can potentially interfere with the evidence used to investigate their own actions. Auditability should not, however, be confused with generating large volumes of logs. Evidence has governance value only when it is complete, reliable, protected, retained, and reviewable enough to support decisions, investigations, and assurance.

### The desired state

Together, these objectives define the desired state for IAM and PAM governance:

- Identity is not simply trusted; it is verified.

- Access is not simply granted; it is justified and periodically reassessed.

- Privilege is not simply assigned; it is minimized and controlled.

- Authority is not unnecessarily concentrated; it is separated where risk requires it.

- Activity is not merely logged; it is made observable and supported by credible evidence.

Policies, risk appetite, and compliance obligations remain largely aspirational unless they are enforced through identity controls. IAM and PAM are the mechanisms that translate governance intent into controlled, auditable digital authority.

## IAM governance in practice: decision rights, operating model, and accountability

IAM is inherently a cross-functional governance system. IT alone cannot govern it, and cybersecurity cannot implement it effectively on its own. Decisions about identity and access affect business operations, fraud exposure, privacy, regulatory compliance, operational resilience, third-party risk, and incident response. The central governance question is therefore not simply how access is provisioned, but **who has the authority to determine what access is appropriate, under what conditions, and with what accountability**.

Effective IAM governance rests on clearly defined decision rights. The organization must establish who owns the access-control model, defines business roles, approves access, sets authentication requirements, reviews permissions, maintains IAM policies and standards, operates identity platforms, authorizes exceptions, and is accountable when access decisions create unacceptable risk. Without these, IAM easily deteriorates into a ticket-processing function in which access is granted because someone requested it rather than because an accountable authority determined it was necessary.

### Three layers of responsibility

A practical IAM operating model typically comprises three interdependent layers, which map naturally onto the three lines model introduced in Chapter 1.

- **A central identity and access function**, usually within cybersecurity, IT security, or enterprise technology, provides the technical and procedural foundation. It designs the IAM architecture, operates identity platforms, sets technical standards, implements authentication and federation, supports provisioning workflows, manages conditional access, integrates applications, and ensures that controls generate sufficient evidence. Its role is to make governance requirements technically enforceable and operationally repeatable.

- **Business ownership** determines which roles exist, what access those roles legitimately require, which information and processes are sensitive, and who may approve access within each area. Technology teams can administer permissions, but they generally cannot judge whether a particular employee, contractor, or service genuinely requires access to a specific business resource. Access is a business risk decision implemented through technology, not a technical request fulfilled by IT.

- **Risk, compliance, privacy, and assurance oversight** provides challenge and verification. These functions assess whether policies are appropriate, approvals are justified, segregation-of-duties conflicts are addressed, reviews are completed, privileged access is controlled, and exceptions are documented, all against regulatory obligations, contractual commitments, and the organization's risk appetite. Internal audit then provides independent assurance that the model operates as intended.

A model in which **"IT grants access when asked"** is insufficient governance: it reflects operational execution without decision authority, risk ownership, or independent oversight. A mature model deliberately separates these responsibilities and prevents the same function from requesting, approving, implementing, and validating access without challenge. The degree of separation should match the risk. Routine, low-risk access may be largely automated under predefined policies, whereas sensitive, conflicting, or privileged access may require explicit approval, stronger authentication, time limits, additional monitoring, or independent review.

### Policy anchors and exceptions

The major control frameworks discussed in Chapter 9 give this operating model its reference points. ISO/IEC 27002:2022 addresses access control, identity management, authentication information, and access rights (controls 5.15 to 5.18), as well as privileged access rights (8.2) and secure authentication (8.5). CIS Controls v8.1 devotes Control 5 to account management and Control 6 to access control management, and the NIST CSF 2.0 Protect function includes identity management, authentication, and access control as a distinct category (PR.AA). These references help managers frame policy and test completeness; they do not substitute for decisions about who owns which access.

Exceptions require equally clear governance. Temporary access, emergency permissions, unusual business requirements, and legacy-system limitations may occasionally make standard controls impractical. Such situations should not be handled informally. Each exception should have an identified owner, a documented justification, a defined scope, compensating controls where necessary, an expiry or review date, and approval commensurate with the risk. Otherwise, temporary exceptions quietly become permanent sources of excessive access.

The practical objective is to make access decisions traceable, explainable, and defensible. For every significant role, sensitive system, important data set, or privileged capability, the organization should be able to say who owns the access, who may approve it, why it is required, what conditions constrain its use, which conflicting permissions must be prevented, how often it is reviewed, how it is revoked, and what evidence shows that these controls operate. When these answers are consistently available, IAM moves beyond account administration and becomes a system for assigning, constraining, monitoring, and holding people accountable for digital authority.

## Identity lifecycle: the most underestimated control

The **joiner–mover–leaver (JML) process** underpins effective access discipline. Many identity-related failures do not stem from a lack of sophisticated technology. They stem from something more fundamental: identities and permissions are created, changed, and removed inconsistently as people and systems evolve. A mature IAM program treats lifecycle management as a core governance control rather than an administrative HR or IT process. The underlying principle is straightforward: **digital access should consistently reflect organizational reality**.

### Joiners

For joiners, the objective is to establish identity appropriately and grant access aligned with an approved business role. Onboarding should not rely on informal requests, permissions copied from another employee, or broad default access granted for convenience; these shortcuts are expedient but make it impossible to show that access is justified. A mature process uses defined roles, standardized access profiles where appropriate, documented approvals, and clear business ownership. Identity information should come from authoritative sources such as the HR system or the contractor register, and higher-risk access should receive greater scrutiny. The aim is not to minimize access to the point that people cannot work, but to ensure that each person receives the access needed for their legitimate responsibilities—and no more.

### Movers

For movers, the principal risk is privilege creep. Employees change jobs, join projects, assume temporary duties, and transfer between departments. New access is granted because the new work must begin immediately, while permissions tied to previous responsibilities remain. Over time, the account accumulates combinations of authority that management never explicitly approved, and the risk remains largely invisible because each individual permission was once legitimate. A role change should therefore trigger both provisioning and deprovisioning. The governance question is not only "What new access does this person require?" but also "What existing access is no longer justified?" Temporary permissions should carry expiry dates or review points, and significant role changes should trigger reassessment of sensitive and privileged access.

### Leavers

For leavers, timeliness and completeness are critical. Once an employment, contractual, or supplier relationship ends, continued access is no longer justified, and delayed deprovisioning creates opportunities for misuse, fraud, or data theft. Urgency should reflect the circumstances of the departure, but organizations should establish clear procedures and service levels rather than rely on informal coordination. A complete leaver process may include disabling accounts, revoking active sessions and tokens, removing remote and privileged access, recovering devices and authenticators, removing delegated permissions, terminating access through federated or third-party systems, and rotating shared secrets the individual knew. Modern users have access through many pathways—SaaS applications, cloud platforms, remote-access services, administrative accounts, development environments, supplier portals, and API credentials—so deprovisioning must address the individual's actual access, not merely their most visible account.

### Non-human identities in the lifecycle

Lifecycle discipline must also apply to non-human identities. Unlike employees, service accounts, API credentials, certificates, and workload identities do not resign, transfer, or appear on termination reports; without explicit governance, they persist indefinitely. Every significant non-human identity should have an owner and a legitimate purpose, with requirements for rotation, expiry, periodic review, and decommissioning. When an application is retired, a supplier relationship ends, or an integration is replaced, the associated identities and credentials should be reviewed and removed. This topic is developed further later in the chapter.

### Lifecycle as assurance

Lifecycle controls are also an important compliance and assurance mechanism, and they are relatively easy to test. Auditors can examine onboarding approvals, role mappings, access-change histories, termination notifications, disablement timestamps, expired temporary permissions, and exception approvals to determine whether documented policies operate in practice. Managers should monitor lifecycle performance rather than assume it is reliable, using indicators such as the percentage of leavers disabled within the required period, orphaned accounts, access persisting after role changes, expired contractor accounts still active, and non-human identities without owners.

The managerial lesson is simple yet consequential: **access should follow legitimate authority throughout its lifecycle**. It should be established when a valid relationship begins, adjusted when responsibilities change, reviewed while active, and withdrawn when the relationship that justified it ends. A well-governed JML process keeps digital authority aligned with organizational reality, reduces excessive privilege, reinforces segregation of duties, and limits the consequences of compromised identities.

## Authentication strategy: MFA, phishing resistance, and conditional access

Managers should treat **multi-factor authentication (MFA)** as a baseline control for significant organizational access, not an optional enhancement. Password-only authentication provides insufficient assurance in most modern environments, especially for cloud services, remote work, sensitive information, financial processes, and administrative functions. However, the managerial objective is not simply to "turn on MFA." A mature authentication strategy determines how much confidence the organization requires in an identity under different circumstances and selects controls proportionate to that risk.

### Not all MFA is equal

Using two factors does not necessarily make authentication resistant to phishing, interception, or session compromise. SMS codes may be exposed through attacks on the subscriber's mobile account or the communications channel. Push-based approvals can be abused when attackers generate repeated requests until a user accepts one, a pattern known as **MFA fatigue**. One-time passwords can be captured and relayed in real time by adversary-in-the-middle phishing sites. **MFA should therefore not be treated as a single homogeneous control.** Management should know which mechanisms are deployed, which threats they mitigate, and where stronger methods are required.

For high-risk access, organizations should favour **phishing-resistant authentication**, in which the protocol itself, rather than the user's vigilance, prevents an attacker from capturing and replaying an authenticator output. Phishing-resistant mechanisms rely on cryptography that binds the authenticator to the legitimate service or session. Examples include FIDO2/WebAuthn security keys and platform authenticators, properly implemented passkeys, and PKI-based smart cards. The U.S. Cybersecurity and Infrastructure Security Agency (CISA), in its 2022 fact sheet *Implementing Phishing-Resistant MFA*, identifies FIDO/WebAuthn and PKI-based methods as the forms that resist phishing. In Canada, the Treasury Board of Canada Secretariat's *Guideline on Multi-Factor Authentication* (2025) likewise treats passwords, one-time codes, and push approvals as not phishing-resistant and requires highly privileged federal users to use at least one phishing-resistant authenticator. Such methods matter most for identities with substantial consequences: privileged and identity administrators, cloud-management accounts, security personnel, financial approvers, users with production access, and those who reach highly sensitive information. **Authentication strength should increase as the potential consequences of identity compromise rise.**

### Assurance levels and NIST SP 800-63-4

Managers can use the NIST Digital Identity Guidelines, **SP 800-63-4**, finalized in July 2025 with its companion volumes SP 800-63A-4, 63B-4, and 63C-4, as a reference for structuring these decisions. The NIST model distinguishes three related assurance problems.

- **Identity proofing** concerns the confidence with which an organization establishes that a person is the real-world individual they claim to be at enrolment.

- **Authentication** concerns the confidence that the person using a digital identity controls the authenticators bound to it.

- **Federation** concerns the mechanisms that allow identity information and authentication results to be trusted across systems, relying parties, and identity providers.

These distinctions matter because strong authentication cannot compensate for weak proofing, and strong proofing cannot compensate for weak authentication later. Federating an identity extends trust to other systems and therefore requires governance of the identity provider, assertions, and relationships that convey that trust.

Where proofing or authentication relies on biometrics, privacy law adds its own conditions. In Québec, biometric information is sensitive personal information; using it to verify or confirm identity requires the person's express consent under the *Act to establish a legal framework for information technology*, and any database of biometric characteristics must be disclosed to the Commission d'accès à l'information at least 60 days before it enters service. Convenience alone rarely justifies biometrics, so managers should expect a privacy impact assessment and a documented necessity argument before approving such a design (see Chapter 16).

NIST expresses authentication requirements as **Authentication Assurance Levels (AALs)**. The managerial value lies less in the terminology than in the principle that different situations warrant different levels of assurance. Revision 4 makes the expectations concrete: at AAL2, verifiers must offer at least one phishing-resistant option, and at AAL3 a phishing-resistant, non-exportable authenticator is required. The revision also recognizes syncable authenticators such as synced passkeys as acceptable at AAL2 but not at AAL3, a useful distinction for organizations deciding where consumer-style passkeys are adequate and where hardware-bound keys are needed. Authentication requirements should therefore be selected based on role, resource sensitivity, threat exposure, and potential business consequences, rather than through a single policy applied everywhere.

### Conditional and adaptive access

Authentication strategy should extend beyond the initial login. A user who authenticated at the start of a session is not trustworthy indefinitely: devices can be compromised, tokens stolen, sessions hijacked, and risk conditions can change. Mature organizations therefore complement authentication with **conditional access**, which applies predefined policies to decide whether access should be granted and on what terms. Decisions may consider the identity's role, authentication strength, device posture, network and geographic context, session risk, unusual behaviour, threat indicators, and the sensitivity of the resource.

Routine access to a low-sensitivity application from a managed device may require little friction. The same user reaching sensitive information from an unmanaged device may be required to use stronger authentication or be prevented from downloading data. An administrator accessing a cloud-management environment under unusual conditions may be required to reauthenticate with a phishing-resistant method from a compliant device. Depending on circumstances, a policy may restrict the session, shorten its duration, force reauthentication, block access entirely, or raise an event for investigation.

This approach reflects zero-trust principles, as described in NIST SP 800-207, *Zero Trust Architecture* (2020), and relevant to the security architecture discussed in Chapter 13. Trust is not inherited from network location or a single successful login; it is conditional and continuously reassessed against identity, device, context, and the requested resource. Conditional access also links risk appetite to operational enforcement. Governance may decide, for example, that privileged administration must never occur from unmanaged devices or that certain information cannot be accessed in specified contexts; conditional-access rules turn those decisions into repeatable technical behaviour.

### Proportionality, tiers, and recovery

Proportionality remains essential. Stronger authentication provides greater assurance but imposes cost and friction; excessive friction prompts unsafe workarounds and overwhelms support teams, while controls designed for convenience leave high-consequence access exposed. Governance should therefore establish explicit authentication tiers: defined minimums for routine access, stronger assurance for sensitive access, and the highest protection for privileged, administrative, and mission-critical access. These tiers should be documented and applied consistently rather than left to individual system owners.

The strategy must also address authentication recovery, because recovery processes can become alternative pathways around strong controls. An organization may deploy phishing-resistant authentication and remain vulnerable if an attacker can persuade a help desk to reset an account using weak identity verification. Processes for lost authenticators, account recovery, and credential reset should provide assurance proportionate to the access they restore. **Recovery should never become the weakest path into a strongly protected identity.**

Authentication governance should therefore cover minimum requirements, higher-assurance tiers, approved authenticator types, conditional-access policies, recovery procedures, exceptions, monitoring, user education, and periodic reassessment as threats, technologies, and regulations change. Authentication does not prove that an actor will behave appropriately, nor guarantee that an identity has not been compromised. Its role is to provide a defensible level of confidence that the actor using an identity is entitled to do so—to align confidence with consequence.

## Authorization models: role engineering, RBAC/ABAC, and access reviews

Authorization is the decision logic that determines **what an authenticated identity may do**. Authentication answers, "Is this actor who or what they claim to be?" Authorization answers the next question: "Given that identity, which resources, functions, transactions, and data may it access?" Strong authentication does not determine whether an identity should be able to view a data set, approve a payment, administer a server, or change another user's permissions. An organization can have strong MFA and still have weak access control if authenticated users hold excessive permissions.

Many organizations mistake the presence of roles and permissions for effective authorization governance. An application may include roles, groups, profiles, and entitlements, but their presence alone does not prove that the access model is coherent, current, risk-based, or aligned with real responsibilities. Authorization becomes governable only when the organization can explain why specific permissions exist, who should receive them, who owns the decision, what risks they create, and how their continued appropriateness is verified.

### Role engineering: translating business responsibility into access

**Role engineering** is the discipline of designing and maintaining roles that reflect legitimate organizational responsibilities and mapping the necessary entitlements to those roles. Without it, least privilege becomes very difficult to sustain at scale: access is granted per user in response to individual requests, exceptions accumulate, inherited permissions become opaque, temporary access becomes permanent, and managers lose the ability to explain why identities hold particular privileges.

A mature model begins with business responsibilities rather than system permissions. The organization first determines what a person or system must accomplish, then identifies the minimum entitlements required. Business managers understand which responsibilities belong to which jobs and which combinations of authority would be inappropriate; cybersecurity and IT understand technical dependencies, privilege relationships, and threat exposure. Effective role design integrates both perspectives. Roles must also be treated as living governance objects, not static configurations. Processes change, applications are replaced, organizations restructure, and automation alters job functions, so significant roles need owners, periodic entitlement reviews, and procedures for change and retirement. The figure below compares the main authorization models.

<span id="_Toc243635550" class="anchor"></span>Figure 51: Authorization models

### Role-based access control

**Role-Based Access Control (RBAC)** assigns permissions to defined roles and then associates identities with those roles. Instead of deciding individually whether each employee should hold dozens of permissions, the organization defines roles such as accounts-payable clerk, HR advisor, database administrator, or security analyst, each with an approved permission set. RBAC can standardize provisioning, simplify onboarding and role changes, reduce arbitrary assignments, support segregation of duties, and make access reviews easier to understand. It works best where groups of people perform stable, clearly defined functions.

RBAC is only as effective as the roles it depends on. Poorly engineered roles become large bundles that institutionalize excessive access. Conversely, creating ever more specialized roles to accommodate exceptions leads to **role explosion**, in which hundreds or thousands of narrowly differentiated roles become almost as hard to govern as individual permissions. Roles should remain meaningful, understandable, owned, and linked to identifiable business responsibilities.

### Attribute-based access control

More dynamic environments may require **Attribute-Based Access Control (ABAC)**, which uses attributes of the identity, resource, action, and context to decide whether access is permitted. Relevant attributes may include department, employment status, location, device status, data classification, project membership, transaction value, time, authentication strength, or current risk indicators. An organization might allow a department to view a category of information only from managed devices and only while employment status is active, or require a specific job function, stronger authentication, and additional approval for transactions above a threshold. ABAC enables precise, context-sensitive decisions, particularly in cloud, data-centric, and zero-trust architectures.

Greater flexibility, however, increases governance complexity. Overlapping attributes and conditional policies can be hard to predict, test, and audit, and small changes may affect access unexpectedly. The managerial objective is not to choose the most sophisticated model but one that remains appropriate, explainable, enforceable, testable, and auditable. Complexity that the people accountable for the resulting risk cannot understand is a governance problem in itself. In practice, many organizations combine the two: roles establish a baseline, and attributes constrain how, when, and from where those permissions may be exercised.

Authorization is also the primary mechanism for enforcing segregation of duties. Governance must consider not only whether each permission is legitimate, but whether the combination of permissions held by one identity produces inappropriate authority. This is the subject of the next section.

### Access reviews: testing whether authorization remains appropriate

Access reviews allow the organization to reassess periodically whether previously approved authorizations remain justified, since legitimate decisions become inappropriate as roles, systems, and risks change. In practice, reviews are often weak. Managers receive long lists of users and technical entitlement names and are asked to approve them. They may not understand what the access enables, why it was granted, or whether it has been used. Faced with uncertainty and volume, many simply approve. The organization can then show that a review occurred, but the review provides little assurance. This illustrates a dangerous gap between control completion and control effectiveness: a completed certification proves that a process ran, not that access risk was competently assessed.

A governance-grade review provides reviewers with decision-useful information: what the access allows, why it was granted, which role justifies it, when it was last used, whether the user still performs the function, whether it is privileged or sensitive, and whether it conflicts with other permissions. Reviews should also be risk-based. Privileged and administrative access, financial approvals, regulated or sensitive data, production systems, cloud management, external identities, dormant accounts, non-human identities, and conflicting permissions warrant more frequent review, stronger evidence, or independent validation than routine access.

The value of a review ultimately depends on what happens next. Removing excessive permissions, resolving conflicts, disabling dormant accounts, correcting ownership, updating role definitions, and documenting justified exceptions are integral to the control. A review that identifies inappropriate access but does not lead to removal, remediation, or formally authorized risk acceptance offers little protection. Governance should therefore track whether findings are resolved on time, not just whether reviews are completed; organizations can achieve high completion rates while retaining substantial excessive access.

### Authorization as continuous governance

Authorization is not a one-time configuration exercise. Roles evolve, users move, systems change, and risk conditions shift, so a decision appropriate at approval does not remain so indefinitely. Roles must be deliberately designed and maintained, entitlements must have clear purposes and owners, conflicting permissions must be controlled, exceptions must be limited, and reviews must give accountable decision-makers meaningful information that leads to action. Authorization is where least privilege, segregation of duties, privacy, and fraud prevention become technically enforceable. Authentication establishes confidence in the actor; authorization determines the extent of organizational authority that identity may exercise.

## Segregation of duties: IAM as fraud prevention and control assurance

**Segregation of duties (SoD)** is one of the clearest points where cybersecurity governance intersects with financial control, internal audit, compliance, and fraud prevention. In many high-impact processes, the risk is not only that an external attacker may compromise a system, but that a single person—or a single compromised identity—may initiate, approve, execute, conceal, or reverse a consequential action without independent oversight. This matters in payments, procurement, payroll, supplier management, financial reporting, identity administration, security configuration, cloud administration, and production change control. SoD is therefore not merely an accounting principle; it is a cybersecurity control because it limits the concentration of digital authority.

### Toxic combinations

The underlying principle is that no individual identity should hold permissions that allow it to complete a high-risk process from beginning to end without meaningful challenge. Individual permissions may each be legitimate, yet become dangerous together; such pairings are often called **toxic combinations**. A person who creates a new supplier should not ordinarily also approve that supplier, initiate payments to it, and reconcile those payments. A developer should not be able to approve and deploy their own change to production. An identity administrator may create accounts but should not approve their own privileged access. The same logic applies to security administration: an administrator who can create privileged identities, assign elevated permissions, disable monitoring, modify audit logs, and review the resulting evidence holds excessive authority. If that identity is misused or compromised, the access that enables the harmful action may also allow it to be concealed.

### Preventive and detective enforcement

IAM must make SoD requirements enforceable in practice. This requires identifying incompatible roles and permissions, defining conflict rules, preventing inappropriate combinations where possible, detecting violations, and establishing controlled exception processes. In mature organizations, access requests are evaluated against conflict rules before permissions are granted, and potential conflicts are escalated rather than discovered months later during an audit. These controls should be both preventive and detective. Preventive controls stop conflicting access from being assigned; detective controls find conflicts that arise later because roles changed, permissions accumulated, applications were modified, or emergency access was granted. Access reviews should accordingly examine combinations of permissions, especially for privileged users, financial processes, production systems, and identities that span several applications.

### Dual control and compensating controls

When full separation is impractical—in small organizations, specialized technical teams, emergencies, or privileged administration—dual control, additional approval, enhanced monitoring, or other compensating controls may be necessary. Dual control requires more than one authorized person to participate in a high-impact action: one administrator prepares a critical change while another approves or executes it, or transactions above a threshold require two approvers. Compensating controls do not eliminate the conflict; they reduce or monitor its risk. They should therefore be governed like any other exception, with documentation of why the conflict exists, who owns the resulting risk, which safeguards apply, and when the arrangement will be reassessed.

### Fraud, blast radius, and assurance

SoD also connects to the **fraud triangle** introduced in Chapter 1. Of the three conditions associated with fraud, opportunity is the one controls can most directly reduce, and excessive access, weak oversight, and concentrated authority create it. SoD reduces that opportunity by requiring collaboration, independent approval, or separate execution of sensitive activities. The same mechanism constrains external attackers. An account that can initiate but not approve a transaction, or administer a system but not disable its monitoring, offers far less to an adversary than one with end-to-end authority. SoD thus limits the blast radius of an identity compromise.

For managers, SoD also provides assurance. A control environment is more credible when high-risk actions require identifiable decision points, independent participation, and evidence of approval. Organizations should be able to identify significant SoD conflicts, explain which roles create them, demonstrate how they are prevented or detected, and show how approved exceptions are monitored and reassessed. The managerial lesson is that access rights must be evaluated collectively, not one entitlement at a time, and that IAM and PAM programs must make SoD visible, enforceable, reviewable, and verifiable.

## Third-party and partner access governance: the extended identity problem

Organizations increasingly rely on contractors, managed service providers, cloud consultants, software vendors, outsourcing partners, auditors, implementation partners, and business-to-business integrations. These relationships create identity dependencies essential to modern operations, yet they are often harder to govern than employee access. A vendor account with long-standing privileged access can provide an unintended pathway into critical systems. A contractor whose access remains active after an engagement ends is a predictable lifecycle failure. A partner integration with broader permissions than required may expose data or infrastructure beyond the intended purpose. External identities may reach cloud consoles, source-code repositories, customer information, financial platforms, remote-access tools, and administrative interfaces, sometimes while facing weaker onboarding, less frequent review, and unclear internal ownership. Third-party identity is therefore an enterprise risk and governance issue, not a secondary access-management task. The broader governance of suppliers is the subject of Chapter 19; this section focuses on the identity dimension.

### External access must be owned internally

The fundamental principle is that external access must be owned internally. Each significant third-party account, federated identity, or integration should have an identifiable internal sponsor accountable for the business justification, the scope of access, the sensitivity of the resources involved, the duration of the relationship, and periodic confirmation that access remains necessary. Without explicit ownership, access persists after projects end, personnel change, or contracts expire, simply because no one is responsible for removing it.

Third-party access should therefore be purpose-specific and time-limited wherever practicable. Access should be granted for a defined purpose and period, and renewal should require confirmation that the need persists. Contract dates, statement-of-work periods, supplier personnel changes, and vendor offboarding should be linked to identity lifecycle controls, and expiry dates should be embedded in provisioning where technically feasible. Long-lived external access may sometimes be justified, but it should be deliberately governed rather than assumed.

### Least privilege and PAM for suppliers

Least privilege applies to third parties as rigorously as to employees. Broad administrator permissions should not be granted merely because they simplify support. When a supplier requires privileged access, the organization should apply the same or stronger PAM controls used for internal administrators: separate administrative identities, just-in-time elevation, approval workflows, phishing-resistant authentication, credential vaulting, restricted administrative pathways, limited session duration, and session monitoring or recording. This is especially important for managed service providers whose personnel may administer infrastructure for many customers. The organization should know whether access is assigned to named individuals, how the supplier manages its own privileged identities, whether shared accounts exist, and how quickly access can be revoked when supplier personnel change.

Monitoring should be proportionate to the risk of the relationship. External access to production environments, cloud administration, identity platforms, security tooling, backups, financial applications, or critical operational systems should never be invisible to the organization's monitoring and assurance processes. Management should be able to determine not only that a supplier had access, but when it was used and which significant actions were performed.

### Federation, integrations, and contracts

Third-party governance must also cover federated and machine-to-machine access. Partners may authenticate through their own identity providers, and applications may exchange data using API credentials, tokens, certificates, or service principals. These arrangements reduce administrative effort but create dependencies on external controls. Federation does not eliminate governance: the organization must understand which attributes and assertions it trusts, what happens when a partner employee leaves, how access is revoked, and what assurance it has that the partner's identity processes remain reliable.

Identity governance must therefore be integrated with procurement, contracting, third-party risk management, and supplier offboarding, and access conditions should be evaluated before a supplier is granted connectivity. Contracts should, where appropriate, set expectations for authentication, privileged access, logging, timely notification of personnel changes, incident notification, access termination, subcontractors, and cooperation with audits. Identity controls cannot indefinitely compensate for weak commercial arrangements: if a supplier is not obliged to report changes in its privileged personnel, the organization may not know when access should be revoked. Ending a relationship should trigger a coordinated review of all associated accounts, federated identities, administrative roles, remote-access permissions, API credentials, certificates, tokens, and shared secrets; disabling a single account is rarely sufficient.

The crucial governance principle is the following.

| **Delegating access does not transfer accountability.** The identity may belong to someone outside the organization, but the access rights fall within the organization's risk boundary. |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

Outsourcing an activity may transfer operational responsibility, but it does not eliminate the need for governance, oversight, evidence, and risk ownership. Third-party identities, privileged vendor access, federated trust, and machine-to-machine integrations should therefore be incorporated into the organization's IAM, PAM, monitoring, and risk-management processes rather than managed as exceptions at their edge.

## PAM: the crown jewel controls managers must demand

Privileged access is qualitatively different from ordinary access. Ordinary access allows users and systems to perform approved work within established boundaries. Privileged access allows users, administrators, services, and automation tools to change those boundaries: create or disable accounts, assign permissions, change security settings, access data at scale, alter or delete logs, manage encryption keys, deploy software, and interrupt operations. Managers should assume that privileged identities and administrative pathways will be targeted. The governance objective is not to assume that privileged compromise can always be prevented, but to make privilege difficult to obtain, tightly constrained when granted, observable during use, and rapidly revocable.

PAM is therefore a **crown-jewel control domain**, not an optional enhancement for technical teams. A firewall, endpoint platform, backup system, logging platform, identity provider, or encryption infrastructure is only as trustworthy as the privileged-access model that secures its administration. An attacker who gains sufficient administrative authority may be able to disable the very controls intended to detect or contain them.

### Separate privileged and ordinary identities

A baseline PAM design begins with strong separation between ordinary activity and privileged administration. Administrators should not browse the web, read email, or open collaboration messages using identities that also hold administrative authority. A separate privileged identity, used only when administrative authority is required, means that compromising routine activity does not immediately yield privileged access. This separation is more than a naming convention: it creates distinct trust boundaries and allows stronger authentication, monitoring, and restrictions for administrative identities than would be practical for everyday use.

### Reduce standing privilege

Mature PAM seeks to minimize **standing privilege**. Users should not hold permanent administrative authority simply because they occasionally need it; the longer privilege remains available, the larger the window for misuse. **Just-in-time (JIT) access** is the key pattern: a user operates with ordinary permissions, requests elevation when needed, receives it for an approved purpose, and loses it automatically when the task or time window ends. The related principle of **just-enough privilege** grants only the capabilities necessary for the specific task. Together, these reduce the number of permanently privileged identities, narrow the period during which authority can be exploited, and reduce the value of stolen credentials.

### Protect privileged credentials and secrets

Credential vaulting and secrets management are central PAM capabilities. Privileged passwords, local administrator credentials, service-account passwords, API keys, certificates, and tokens should not be stored in spreadsheets, scripts, shared folders, source-code repositories, personal notes, or unsecured configuration files. They should be held in controlled systems that enforce access restrictions and, where applicable, rotation, expiry, approval, and logging. In a mature design, users may reach a privileged function without ever seeing the underlying secret: the PAM platform brokers the session, injects credentials, rotates them after use, or issues temporary secrets that expire. The principle is that **privileged authority should be governed independently of possession of a reusable credential**. The cryptographic side of this problem—keys, certificates, and their custody—is treated in Chapter 20.

### Monitor privileged sessions and actions

Privileged-session management strengthens accountability. High-risk administrative sessions may be brokered, monitored, or recorded where the risk justifies it, not for surveillance's sake but to create reliable evidence, deter misuse, and support investigation. Monitoring should establish which person or process initiated the activity, why it was approved, which resources were accessed, what significant actions were performed, and when the privilege ended. Shared administrative accounts are particularly problematic because logs show what the account did but not who did it. Where they cannot be eliminated, PAM should associate each use with a named individual, approval record, session, and business purpose. Privileged activity should always be attributable to an accountable person or a clearly governed non-human process.

### Control administrative pathways

PAM must also govern where privileged actions can originate and how sessions reach sensitive systems. Strong credential controls can be undermined if administration is permitted from ordinary workstations, unmanaged devices, or insecure remote-access services. Mature designs constrain administrative pathways through hardened privileged-access workstations, dedicated management environments, segmented administrative networks, device-compliance requirements, conditional access, phishing-resistant authentication, and session brokering. Privileged access should require not only a privileged identity but a trusted administrative path, so that malware on a routine endpoint or a stolen browser session does not become a route to administrative systems.

### Non-human privilege and emergency access

PAM must also encompass non-human identities. Service accounts, deployment pipelines, cloud roles, and workload identities may hold administrative permissions even though no person logs in with them, and they are often long-lived, hard to rotate, and over-permissioned because teams fear disrupting automated processes. The objective is to avoid tightly governed human administrators alongside powerful machine identities that remain effectively invisible; this is developed in its own section below. PAM must likewise provide for emergency or break-glass access when normal identity systems fail, without creating a hidden permanent bypass—the subject of the next section.

### PAM as a governance and assurance mechanism

Managers should be able to obtain credible answers to a small set of questions: Who holds privileged authority, and over which systems and data? Why is it required, and is it permanent or temporary? Who approved it, and through which pathways can it be exercised? How are privileged secrets protected and significant activity monitored? When was privileged access last reviewed, and how quickly can it be revoked? These questions move PAM beyond product deployment. An organization may own a PAM platform and still have weak PAM governance if administrator accounts bypass it, service accounts remain unmanaged, passwords are shared informally, permanent cloud roles proliferate, or session records are never reviewed.

The managerial lesson is that privileged authority must be **minimized, justified, separated, time-bounded where practicable, strongly authenticated, exercised through controlled pathways, monitored, attributable, and evidenced**. PAM is not merely a mechanism for protecting administrator passwords; it is a governance safeguard designed to prevent an ordinary compromise from becoming a catastrophic one.

## Break-glass accounts: emergency access without permanent weakness

Emergency access, commonly called **break-glass access**, is necessary because normal identity and administrative pathways can fail. Identity providers may become unavailable, federation may be disrupted, administrators may be locked out, or urgent recovery may be needed during an incident. A well-governed organization must preserve the ability to regain control of critical systems when ordinary mechanisms are unavailable—a resilience requirement that connects directly to the incident and continuity planning of Chapter 12.

The governance challenge is that the independence that makes emergency access useful can also make it dangerous. Break-glass identities that bypass normal workflows can become persistent backdoors if they are shared, weakly protected, poorly documented, rarely tested, or excluded from monitoring. The objective is to achieve two outcomes simultaneously: **emergency access must remain available when normal controls fail, and it must remain tightly controlled when no emergency exists**. Break-glass mechanisms should be few, explicitly justified, assigned to defined systems or recovery purposes, and never used for convenience, routine troubleshooting, or avoiding normal privileged-access workflows.

### Preserve independence without creating a backdoor

Emergency access should be independent of the controls whose failure it is meant to overcome. A recovery mechanism that depends on the same identity provider, authentication service, network path, or PAM platform that has failed provides little resilience. Yet independence must not mean absence of control. Governance should define which normal dependencies may be bypassed during an emergency and which safeguards must remain in force.

| **Break-glass access should bypass failed dependencies, not governance itself.** |
|----------------------------------------------------------------------------------|

### Protect credentials and define activation conditions

Break-glass credentials and authenticators should receive protection proportionate to the authority they confer: strong authentication, secure storage, controlled custody, restricted knowledge, and physical protection of hardware authenticators. Arrangements in which several administrators casually know a shared emergency password, or in which credentials sit in accessible documents or personal password stores, should be avoided. Where practical, retrieval may require more than one authorized person. The aim is for emergency access to be recoverable without becoming routinely accessible.

Management should also define when break-glass access may legitimately be used—failure of normal identity infrastructure, loss of ordinary privileged access, or a critical incident in which delay would cause significant harm—together with who may initiate it, who must be notified, what approval is required where circumstances permit, and what documentation must follow. The process must remain practical in a real crisis. A mechanism that depends on unavailable personnel, inaccessible systems, or an elaborate approval chain may fail exactly when it is needed.

### Monitor, test, and review every use

Break-glass accounts should be excluded from routine use but included in monitoring and alerting. Activation should be treated as a high-significance security event, generating immediate alerts to security operations, identity administrators, system owners, and relevant management. The organization should be able to determine who invoked the mechanism, when, on which system, why normal access was unavailable, what was done, and when emergency access ended. Because some emergencies involve failure of normal monitoring, critical evidence may need collection mechanisms that do not depend on the affected system. **Emergency authority should never become invisible authority.**

Emergency access should be tested periodically through controlled exercises. Testing verifies that credentials and authenticators work, that designated personnel understand the procedure, that monitoring detects the event, and that recovery can be performed without unsafe improvisation. It should also confirm that the mechanism has not quietly lost its independence through changes to identity infrastructure, cloud configuration, or networks, and that it has not accumulated unnecessary permissions or custodians. Break-glass testing is therefore both a resilience test and a control-assurance test.

Every activation should trigger a formal post-use review. This does not imply that emergency use is suspicious, but that it represents an intentional departure from normal controls and warrants heightened accountability. The review should confirm why access was needed, whether its use was authorized and proportionate, what changes were made, and whether normal mechanisms have been restored. Credentials should be rotated or resecured, temporary changes reviewed, and unnecessary permissions removed. If administrators had to improvise because recovery procedures were inadequate, the organization should treat that as a governance lesson rather than simply closing the record.

### Ownership and the managerial test

Each significant emergency identity or recovery pathway should have a clear owner and a record of the system it protects, who may invoke it, how its credentials are protected, what permissions it holds, how activation is detected, and when it was last tested. Periodic review should confirm that it is still required and that its permissions, custodians, monitoring, and documentation remain current. Obsolete break-glass identities are especially dangerous: administrators assume no one uses them, while attackers specifically seek neglected administrative pathways.

The managerial question is straightforward: **can the organization regain administrative control during a crisis without creating a permanent weakness during normal operations?** A mature program can answer yes because its emergency access is available when normal mechanisms fail, independent of the dependencies it is designed to recover from, strongly protected while dormant, limited to defined purposes, attributable, alerted on every use, periodically tested, reviewed after each activation, and resecured once the emergency ends. Break-glass access is not a bypass around governance; it is a governed resilience mechanism.

## Non-human identities: service accounts, API keys, tokens, and machine-to-machine trust

A major blind spot in many IAM and PAM programs is the governance of non-human identities. Significant compromises do not always begin with an employee account. They may begin with a leaked API key, a long-lived token, an overprivileged service account, a poorly managed certificate, a secret embedded in source code, or credentials exposed through a CI/CD pipeline. These identities can hold substantial authority over systems, cloud resources, data, deployment processes, and security controls—sometimes broader, more persistent, and less visible than that of any human user.

Managers must therefore recognize a fundamental principle: **workloads are identities, and machine-to-machine trust is an access-control relationship**. Applications authenticate to databases, microservices call one another, infrastructure-as-code platforms create cloud resources, backup platforms read large volumes of data, pipelines build and deploy software, and security tools quarantine endpoints or block accounts. Each activity requires a mechanism through which one system is trusted to act upon another, whether a service account, API key, token, certificate, workload identity, or cloud role. If these relationships are not governed, powerful access pathways accumulate outside employee-focused IAM processes and remain invisible to managers and reviewers.

### Inventory, ownership, and least privilege

Governance begins with visibility: an organization cannot govern machine identities it does not know exist. Significant non-human identities should be inventoried and linked to the application, workload, integration, or business process that depends on them and to an accountable owner. Ownership matters because machine identities do not participate naturally in human lifecycle processes. A service account created for a project may keep operating long after the project ends; an API key may remain active after its application is replaced; a certificate may continue to authorize a system no one remembers owning. Each significant non-human identity should have a defined owner, legitimate purpose, authorized scope, appropriate privilege level, credential-management requirements, monitoring expectations, and retirement criteria.

Least privilege applies equally to machines. A reporting service that reads a limited data set should not hold database administration rights; a pipeline that publishes one application should not control unrelated production environments. This matters more for machines than for people because an overprivileged automated process may run thousands of times a day and be trusted implicitly by many systems. Machine identities should therefore be evaluated not only by what they do, but by what their permissions would allow if compromised. Identities able to deploy production software, modify cloud infrastructure, administer databases, manage backups, change identity configurations, or issue credentials to other systems should be classified as privileged and governed within the PAM model.

### Reduce dependence on long-lived secrets

Static credentials are attractive to attackers because they can be copied and reused without triggering the behavioural indicators associated with compromised human accounts. Passwords, API keys, private keys, and tokens should not be embedded in source code, scripts, container images, configuration files, or build logs. Secrets-management platforms provide stronger protection through access control, logging, rotation, versioning, and expiry. Better still, where technically feasible, organizations should reduce reliance on long-lived reusable secrets altogether through short-lived credentials, managed identities, workload identity federation, certificate-based authentication, and dynamically issued tokens. Protecting a secret indefinitely is inherently difficult; reducing credential lifetime limits how long a stolen secret remains useful. Where long-lived credentials remain necessary because of legacy constraints, they should be treated as controlled exceptions with justification, compensating safeguards, and periodic review.

### Govern the machine identity lifecycle

Non-human identities need a lifecycle just as human ones do: deliberate creation, assignment to an owner, appropriate permissions, periodic review, modification when requirements change, and retirement when the purpose ends. Triggers include application retirement, project completion, supplier termination, infrastructure replacement, platform migration, certificate expiry, and redesign of automated processes. The governance question is the same one applied elsewhere in IAM: does this identity still have a legitimate reason to exist, and does it still require the authority it holds? Dormant, orphaned, or unexplained identities should be investigated and disabled or removed.

### Supply-chain, CI/CD, and cloud identities

Non-human identity governance intersects directly with software supply-chain security. Source-code repositories, build servers, artifact repositories, package managers, signing services, and deployment pipelines trust one another through machine identities. A compromised CI/CD identity may allow an attacker to alter source code, introduce malicious dependencies, tamper with builds, obtain further credentials, or push unauthorized software into production, and its activity may be assumed legitimate because it appears to come from normal development processes. Pipeline identities should therefore be governed according to the authority they can exercise, not classified as ordinary technical accounts.

Cloud environments amplify the issue because workloads routinely obtain permissions to create resources, modify storage permissions, alter network configurations, manage encryption services, and create further identities. Cloud roles and workload identities should have clearly defined trust relationships: management should understand not only which permissions an identity holds, but which systems are permitted to assume or invoke it, especially where trust policies allow another workload, account, tenant, pipeline, or external service to obtain its privileges. Cloud-specific identity architecture is discussed further in Chapter 15.

### Monitor behaviour and avoid anonymous authority

Machine identities can in some respects be easier to monitor than people because automated systems behave predictably. A service account that normally reads one database should attract attention if it begins querying unrelated systems; a deployment identity that starts creating administrative users, or a backup account that starts deleting data, warrants investigation. Monitoring should look for unexpected resources, unusual API operations, new execution locations, privilege escalation, dormant-identity use, and activity outside the intended purpose. The objective is to determine whether the machine identity is behaving consistently with the purpose for which its authority was granted.

Accountability also suffers when one generic service account is shared across unrelated applications. Shared machine identities simplify administration but weaken attribution and enlarge the impact of credential compromise. Where practicable, distinct workloads should use distinct identities so that permissions can be tailored, credentials rotated or revoked independently, and activity attributed to the responsible system. The goal is to avoid anonymous machine authority, in which powerful credentials are widely shared but no one can say which application used them or why.

### Machine identity as delegated digital authority

Modern organizations increasingly delegate operational authority to software. Applications execute transactions, automation modifies infrastructure, pipelines deploy production systems, security tools take containment actions, and APIs transfer organizational data. These are exercises of digital authority even when no person is directly involved at the moment of action. Non-human identities must therefore be inventoried, owned, classified, minimized, protected, monitored, reviewed, and ultimately retired. Managers should be able to ask of a machine identity the same questions they ask of a human one: what it is, who owns it, why it exists, what it can reach, how it authenticates, where its credentials are protected, how compromise would be detected, and what will cause it to be retired. AI agents that act through tools are the newest and fastest-growing form of this delegated authority; Chapter 18 explains why they should be governed as non-human identities with least privilege, accountable owners, and approval gates for consequential actions. Non-human identity governance is not simply secrets management; it is the governance of organizational authority exercised by software.

## Auditability and evidence: turning access control into credible assurance

One of the most important lessons for managers is that **access control without evidence is difficult to distinguish from an assumption**. An organization may believe that MFA is enforced, leavers are deprovisioned promptly, privileged access is monitored, and reviews are completed. Unless it can produce reliable evidence, such claims are hard to verify and defend. Evidence is the difference between "we believe the control operates" and "we can demonstrate that it operates." Chapter 3 discusses how such evidence supports compliance; here the focus is on what IAM and PAM evidence must contain.

Auditability requires both decision traceability and activity traceability: the organization should be able to reconstruct how access was granted, modified, exercised, reviewed, and removed. This does not mean elaborate documentation for every routine action. It means that access decisions and high-impact activity leave sufficient evidence for accountability, assurance, and investigation. At a minimum, the organization should be able to evidence authentication requirements and MFA enforcement, assigned roles and permissions, requests and approvals, access changes, privileged logins and significant privileged actions, review results, lifecycle events, exception approvals, and formal acceptance of residual access risk. The depth of evidence should be proportionate to the authority involved.

### Lifecycle and privileged evidence

For human identities, the organization should be able to show when an individual joined, what access was granted, which role justified it, and who approved it; which permissions were added and removed when responsibilities changed; and when access was disabled after departure, including sessions, remote access, privileged accounts, and delegated permissions for higher-risk identities. This answers a central assurance question: did digital authority change when organizational authority changed? For non-human identities, comparable evidence shows who owns the identity, why it exists, what it can do, what depends on it, when credentials were issued or rotated, when access was last reviewed, and how it will be decommissioned. Without such evidence, machine identities can remain active for years with no one able to explain why.

PAM requires the strongest evidence because privileged access represents exceptional authority. Relevant records include privileged-access requests and justifications, approvals, JIT elevation windows, credential retrieval, session start and end, administrative activity, session recordings where justified, and break-glass activations and reviews. The purpose is not documentation for its own sake but to make consequential actions reconstructable and attributable: management should know which person or governed process exercised the authority, why, under what approval, and whether the activity remained within the authorized purpose—even when the underlying system only records that a generic administrative account acted.

### Evidence must be reliable

A record is not automatically reliable evidence. Logs and approval records that can be altered or deleted by the same privileged identities whose actions they document provide limited assurance. High-value evidence should be protected against unauthorized alteration and deletion, access to audit records should be restricted, and sensitive environments may require separation between those who administer systems and those who control audit evidence. Retention should reflect operational, investigative, legal, contractual, privacy, and regulatory needs; keeping identity data indefinitely is not appropriate, but keeping it too briefly can make investigations and compliance demonstrations impossible. Evidence governance should answer not only what is logged, but who can modify it, where it is stored, how long it is kept, and whether it will be available when auditors, investigators, or regulators need it.

### IAM, PAM, and security operations

Identity evidence should not remain isolated within access-management platforms. Authentication logs, privilege elevations, account changes, token use, and anomalous machine-identity behaviour should be correlated with evidence from endpoints, cloud platforms, applications, ticketing, and change management, as discussed in Chapters 7 and 8. Context distinguishes legitimate activity from misuse: a privileged login outside normal hours may be harmless if it matches an approved emergency change, and highly significant if no change record exists, the device is unfamiliar, and the identity then modifies logging. Identity-related detection, now often packaged by vendors as **identity threat detection and response (ITDR)**, may cover repeated authentication failures, unexpected MFA prompts, suspicious account recovery, unusual privilege elevation, dormant-account use, access from unmanaged devices, high-risk permission changes, abnormal API activity, and emergency-account activation. Not every anomaly is an incident, but significant alerts need defined ownership, triage criteria, and escalation. **Evidence that is collected but never examined provides little control value.**

### From logging to assurance

Managers do not need to understand log formats or authentication protocols. They do need to insist that, for significant access, the organization can establish who or what had access, why, who approved it, what it permitted, when and from where it was used, what significant actions were performed, whether the activity matched the identity's role or purpose, whether it was reviewed, whether inappropriate permissions were removed, and whether the organization could reconstruct events if the identity were later suspected of compromise. These are governance questions, not logging questions. An organization may collect billions of identity events and still have weak assurance if those records cannot answer them.

Reliable evidence also supports the three lines model. Operational teams use it to administer access and demonstrate that procedures were followed; risk, compliance, and privacy functions use the same evidence to monitor whether access remains within policy and risk appetite; and internal audit uses it to test design and operating effectiveness independently. **The function performing a control should not be the only source of assurance that the control works.** Without reliable evidence, audits become interviews, investigations become reconstructions from fragments, and compliance claims depend largely on trust. With it, management can identify excessive access, challenge exceptions, verify deprovisioning, investigate suspicious activity, and demonstrate that identity controls operate as intended. The objective is not to "log everything," but to produce **decision-useful, reliable, protected, and reviewable evidence** of how digital authority is granted, exercised, changed, and withdrawn.

## Measurement: IAM/PAM metrics that matter to governance

IAM and PAM are governance domains in part because their effectiveness can be observed and measured. The most useful indicators are not those that count activity—accounts created, passwords reset, tickets processed. Such figures help operational management but say little about whether identity risk is controlled. Governance-grade metrics should show whether access exposure is rising or falling, whether high-risk identities and privileges are controlled, whether lifecycle processes are reliable, whether exceptions are being reduced, and whether evidence exists to demonstrate that access remains appropriate. As discussed in Chapter 4, the most important of these should be framed as key risk indicators with thresholds tied to risk appetite.

Good measurement combines several types of indicators. **Coverage metrics** show whether controls are deployed across the relevant population. **Performance metrics** show whether they operate within expected standards. **Risk indicators** highlight conditions that increase exposure, such as orphaned accounts or standing privilege. **Remediation metrics** show whether identified weaknesses are being corrected. The table below gives representative examples.

| **Governance area**               | **Example metric**                                                          | **Why it matters**                                      |
|-----------------------------------|-----------------------------------------------------------------------------|---------------------------------------------------------|
| MFA coverage                      | % of active workforce identities protected by MFA                           | Shows whether baseline protection is broadly enforced   |
| Phishing-resistant authentication | % of privileged and high-risk identities using phishing-resistant methods   | Concentrates strong assurance where impact is greatest  |
| Conditional access                | % of sensitive applications and privileged paths under approved policies    | Shows whether access decisions use context              |
| Leaver deprovisioning             | % of leavers disabled within the required service level                     | Tests a critical lifecycle control                      |
| Mover remediation                 | % of role changes with obsolete access removed on time                      | Measures control of privilege creep                     |
| Standing privilege                | Number of permanently privileged identities, with trend                     | Shows whether persistent authority is shrinking         |
| Just-in-time privilege            | % of eligible privileged access granted through time-limited elevation      | Shows whether privilege is temporary and justified      |
| Privileged-session accountability | % of high-risk sessions attributable to a named person or governed process  | Shows whether admin activity can be reconstructed       |
| Access-review remediation         | % of inappropriate access removed within the required period                | Measures review effectiveness, not completion           |
| SoD conflicts                     | Unresolved high-risk conflicts, and % of exceptions with owner and expiry   | Shows whether authority concentrations are governed     |
| Break-glass readiness             | % of emergency mechanisms tested; % of activations reviewed                 | Confirms emergency access works and stays accountable   |
| Dormant and orphaned accounts     | Accounts beyond inactivity threshold or without valid owner                 | Identifies identities that escaped lifecycle governance |
| Third-party access                | % of external identities with sponsor, purpose, and expiry                  | Ties external access to a live relationship             |
| Non-human identities              | % of significant machine identities with owner and privilege classification | Shows whether machine authority is visible              |
| Secret hygiene                    | % of machine credentials meeting rotation or short-lived requirements       | Tracks reduction of long-lived secrets                  |
| Identity exceptions               | Active exceptions and % past review or expiry date                          | Shows whether exceptions stay temporary                 |
| Evidence completeness             | % of sampled high-risk access decisions with full evidence                  | Tests whether controls are demonstrable                 |

### Interpreting metrics and trends

Metrics should not be read mechanically. A rise in access-review findings may indicate poor governance, or it may show that an improved review process is finally exposing excess that was previously hidden. A fall in privileged user accounts may reflect better least privilege, or a transfer of privilege into unmanaged service accounts and cloud roles. A high MFA coverage figure can create false confidence if the uncovered remainder includes administrators; a 99 percent result matters less than knowing which one percent remains outside the control. Reporting should therefore be segmented—ordinary versus privileged users, employees versus third parties, human versus non-human identities, routine versus critical systems—because enterprise averages hide concentrations of exposure.

Most identity measures are more meaningful as trends than as isolated values. Management learns more from whether standing privilege is declining and being replaced by controlled elevation than from a single count. Each significant indicator should therefore carry a current value, a target, a trend, an escalation threshold, and an accountable owner; a metric without an expected range informs but does not drive governance action. A balanced framework also combines leading indicators, such as phishing-resistant coverage, standing privilege, overdue reviews, and identities without owners, which reveal deteriorating conditions before an incident, with lagging indicators, such as confirmed identity compromises, privileged-account misuse, and access-related audit findings, which show whether controls are preventing the outcomes they were designed to address.

### Measure effectiveness, not administrative activity

Managers should distinguish carefully between activity metrics and effectiveness metrics. "Ten thousand access reviews completed" measures activity. "Ninety-eight percent of high-risk access reviewed on time" measures process performance. "Twenty-three percent of reviewed permissions were removed" says something about the quality of the existing access model, and "95 percent of inappropriate permissions remediated within 30 days" measures corrective effectiveness. Similarly, "500 privileged-access requests processed" is operational information, whereas "standing administrative access declined by 30 percent without an increase in unmanaged privileged identities" is evidence of risk reduction. Measurement should focus on whether exposure is changing, not on whether IAM teams are busy.

Executives do not need every technical indicator. The dashboard should let them answer a smaller set of consequential questions: whether fewer identities hold unnecessary access, whether privilege is increasingly temporary, whether people and suppliers lose access promptly when relationships change, whether SoD conflicts are declining, whether machine and third-party identities are visible and owned, whether exceptions are genuinely temporary, and whether privileged actions can be reconstructed. The objective is not to maximize every metric, but to understand whether **digital authority is becoming more controlled or less controlled over time**.

## Implementation roadmap: a practical sequence for managers

Organizations often approach IAM and PAM as large technology transformation programs and then struggle with legacy applications, inconsistent identity data, fragmented ownership, unclear decision rights, and resistance to change. A more governable approach builds capability progressively, with each stage reducing meaningful exposure, improving visibility, strengthening evidence, and laying a foundation for the next. The roadmap is not a rigid technical sequence; activities overlap, and some controls must be accelerated where risk is high. The principle is that **IAM and PAM maturity should progress according to risk and dependency, not according to software deployment schedules**.

### Stage 1: establish a reliable identity and authentication foundation

The first stage defines authoritative identity sources, improves the quality of identity records, reduces duplicate identity stores, and consolidates identity providers where feasible. Later controls—automated provisioning, access reviews, role assignment, conditional access, and PAM—become unreliable if identity data is incomplete or disconnected from authoritative business information. Authentication is strengthened at the same time: MFA is enforced broadly for significant access, phishing-resistant methods are prioritized for privileged administrators, identity administrators, cloud management, remote administration, and financial approvals, and conditional-access policies progressively incorporate device posture, identity risk, and resource sensitivity. The objective is greater confidence that the identities exercising digital authority are legitimate and appropriately authenticated.

### Stage 2: strengthen identity lifecycle discipline

The second stage standardizes joiner–mover–leaver processes and integrates them with HR, contractor, and supplier processes so that identity changes originate from authoritative events rather than manual requests. Deprovisioning often yields immediate risk reduction: leavers, expired contractor accounts, obsolete external identities, and dormant accounts should be disabled promptly. Mover processes should remove access tied to former responsibilities, not merely add new access. Measurable expectations—deprovisioning service levels, the share of lifecycle events processed through approved workflows, and orphaned identities—should be introduced here. The objective is to keep digital authority synchronized with organizational reality.

### Stage 3: bring privileged authority under control

The third stage identifies where privileged authority exists and begins with the systems and identities capable of the most significant consequences. Ordinary and administrative identities are separated, standing privilege is reduced, privileged credentials are vaulted, and just-in-time or just-enough privilege is introduced where feasible. High-risk sessions are made attributable and monitored, administrative pathways are constrained, and break-glass mechanisms are designed, protected, tested, and reviewed. The objective is not to place every administrative account into a PAM platform immediately, but to ensure that the most consequential forms of digital authority are progressively brought under stronger control.

### Stage 4: strengthen authorization and role governance

The fourth stage improves the quality of authorization decisions, starting with high-risk systems, sensitive information, financial processes, production environments, cloud platforms, and functions where SoD conflicts could have material consequences. Business-owned roles are defined, entitlements rationalized, unnecessary access removed, and RBAC, ABAC, or hybrid models applied where they improve consistency. Access reviews mature from administrative certifications into decision-useful processes, and findings lead to remediation. The objective is to demonstrate that authorization remains justified and that excessive authority is actually reduced.

### Stage 5: extend governance to third parties and non-human identities

As human and privileged-access governance mature, comparable discipline is extended to external and machine identities. Third-party access receives internal sponsors, documented justification, defined scope, appropriate authentication, and expiry or renewal mechanisms, with privileged supplier access governed like internal administration. Service accounts, workload identities, API credentials, certificates, cloud roles, and pipeline identities are inventoried and assigned owners; their privileges are assessed by risk; static secrets are replaced where feasible with managed identities, workload federation, or short-lived credentials; and high-risk machine identities are brought within PAM. The objective is to prevent identity governance from stopping at the human workforce while significant external and automated authority remains unmanaged. The figure below summarizes the sequence.

<span id="_Toc243635551" class="anchor"></span>Figure 52: IAM implementation roadmap

### Build monitoring, evidence, and measurement into every stage

Monitoring and evidence should be designed into every stage rather than deferred to the end. Identity and privilege events should be logged, protected, retained, and integrated with security operations so that management can see whether controls actually operate rather than relying on implementation status reports. Metrics should evolve with the program, from MFA coverage, dormant accounts, and deprovisioning timeliness early on to standing-privilege reduction, review remediation, SoD conflicts, non-human identity ownership, and exception ageing later. This matters because risk migrates: reducing human administrator accounts achieves little if equivalent privilege moves into unmanaged cloud roles, and improving employee deprovisioning while ignoring external identities leaves comparable exposure elsewhere. Measurement should show that identity exposure is being reduced rather than merely relocated.

### Define milestones as risk-reduction outcomes and prioritize by risk

Managers should require each stage to produce observable governance outcomes rather than technology milestones. "We purchased an IAM platform" and "the PAM system is live" are implementation statements; they do not show that identity risk has fallen. More meaningful milestones might include the following, with thresholds set according to the organization's risk appetite.

- MFA enforced for nearly all in-scope users, with every privileged user on approved phishing-resistant authentication.

- Leavers disabled within the approved service level in at least 99 percent of cases.

- Standing privileged accounts substantially reduced, with most eligible administrative access converted to time-limited elevation.

- All break-glass accounts tested successfully during the review period.

- All high-risk third-party identities and service accounts assigned accountable owners and expiry, review, or credential requirements.

- All privileged activity on critical systems attributable to a named individual or governed process.

Organizations should also resist implementing controls uniformly across every system from the outset. A risk-based roadmap prioritizes the identities and access paths capable of the greatest business consequences. A high-risk cloud administrator account may justify immediate phishing-resistant authentication and PAM controls while broader role engineering is still incomplete; a legacy application holding highly sensitive data may warrant accelerated lifecycle and review controls even if interim manual processes are required. The scenario methods of Chapters 5 and 6 are well suited to making these priorities explicit and defensible.

### Illustrative scenario: prioritizing privileged access at a Québec manufacturer

Consider a mid-sized Québec manufacturer with 1,200 employees, a cloud-based ERP system, a growing portfolio of SaaS applications, and an outsourced IT provider that administers its servers. The board has approved a two-year IAM program, and the vendor proposes beginning with a full role-engineering exercise across all applications. The CISO instead runs two scenarios with the risk owners: ransomware entering through a compromised provider administrator account, and fraudulent supplier payments through an ERP user who can both create suppliers and release payments. Both scenarios rate as high residual risk, well above the appetite the board has set for operational disruption and financial loss.

The resulting roadmap reorders the work. In the first quarter, the outsourced provider's personnel move to named, phishing-resistant administrative identities brokered through just-in-time elevation, and the shared domain administrator password is retired. Two break-glass accounts are created, stored under dual custody, alerted on use, and tested. The ERP owner commissions a targeted SoD analysis of the purchase-to-pay process, which identifies fourteen users holding the supplier-creation and payment-release combination; eleven are remediated, and three are approved as time-limited exceptions with a second-approver control and a named risk owner. Broader role engineering is scheduled for the second year.

Quarterly reporting to the audit committee shows standing privileged accounts, the share of provider sessions attributable to named individuals, and open SoD exceptions against their expiry dates. The program can show the board within months that the two most consequential scenarios have moved inside appetite, which is a stronger basis for continued investment than a report that the platform has been deployed.

### Manage IAM and PAM as a governance transformation

IAM and PAM implementation should be treated as a risk-reduction and governance transformation program, not a software deployment project. Technology is necessary, but success also depends on authoritative data, business ownership, clear decision rights, effective processes, integration with HR and supplier management, meaningful access models, monitoring, and credible assurance. Each stage should remove known weaknesses, reduce unnecessary digital authority, improve management's visibility, and establish the foundation for the next. A successful roadmap is defined not by how many identity technologies have been deployed, but by whether **access becomes more accurate, privilege more constrained, evidence more credible, exceptions more controlled, and accountability more explicit over time**.

## Concluding integration: identity governance as the core of modern cybersecurity

In identity-centric enterprises, cybersecurity governance depends heavily on the ability to govern digital identities and the authority attached to them. IAM and PAM determine whether access is appropriately granted, whether excessive privilege is constrained, whether insider and third-party risk is reduced, whether attackers can escalate after an initial compromise, and whether management can produce credible evidence for investigations, audits, and assurance.

The central conclusion of this chapter is straightforward.

| **When identity governance is weak, many other cybersecurity controls become fragile.** |
|-----------------------------------------------------------------------------------------|

Organizations invest in segmentation, endpoint protection, cloud security, monitoring, data protection, backups, and incident response. Yet a strong firewall offers limited protection if a compromised administrator can change its rules; logging provides limited assurance if a privileged identity can disable it; secure cloud storage can still leak data if an overprivileged account can change its access policy; and backups offer little resilience if administrative credentials allow an attacker to delete them. Identity governance affects not only who enters the environment, but **who or what has the authority to change the environment once inside it**.

Conversely, disciplined identity governance strengthens many other investments. Strong authentication increases confidence that the actor using an identity is legitimate. Least privilege reduces the authority available for misuse. Segregation of duties prevents concentrations of authority that enable fraud and concealment. Lifecycle governance keeps access aligned with employment, contracts, and business requirements. PAM constrains exceptional authority so that an ordinary compromise does not become a catastrophic one. Third-party governance ties external identities to legitimate relationships and internal accountability, and non-human identity governance extends the same principles to workloads, automation, and APIs. Monitoring and evidence make all of this observable and defensible.

In this sense, IAM and PAM operationalize principles developed throughout this book. Policies become enforceable through authentication and authorization rules. Risk appetite becomes concrete through conditional-access policies and authentication tiers. Accountability becomes demonstrable through attribution and evidence. Risk treatment becomes actionable when excessive access is removed, privileged pathways are constrained, and exceptions are deliberately governed with owners and expiry dates. Identity also spans the whole cybersecurity lifecycle. Before an incident, it reduces unnecessary access and strengthens authentication; during an intrusion, it limits lateral movement and privilege escalation; in detection, it surfaces anomalous authentication and privilege use; in response, it allows sessions to be terminated, credentials rotated, and tokens revoked; and in recovery, governed emergency access lets authorized personnel regain control without abandoning governance.

For Business Technology Management professionals, the practical implication is that IAM and PAM must be governed as continuous risk-management disciplines, not implemented once as account-administration projects.

### Questions managers and boards should ask

The following questions reveal whether identity governance exists in practice or merely in policy.

- **Ownership:** Who owns identity and access decisions for each significant system, and who is authorized to approve sensitive and privileged access?

- **Legitimacy:** What constitutes legitimate access, and can we explain why each high-risk entitlement exists?

- **Authentication:** How is authentication strength matched to risk, and which privileged or high-risk identities still lack phishing-resistant methods?

- **Recovery:** Could an attacker bypass strong authentication through the help desk or account-recovery process?

- **Least privilege:** How much standing privilege remains, and is it declining in favour of time-limited elevation?

- **Segregation of duties:** Which toxic combinations exist in our critical processes, and how are approved exceptions monitored and retired?

- **Lifecycle:** How quickly does access change when roles and relationships change, including for contractors and suppliers?

- **Third parties:** Which external parties hold privileged access, who sponsors them internally, and when does that access expire?

- **Machine identities:** Do we know how many non-human identities we have, who owns them, and which are privileged?

- **Emergency access:** When were our break-glass accounts last tested, and would we know immediately if one were used?

- **Evidence:** Could we reconstruct, from protected evidence, every significant privileged action on a critical system last quarter?

- **Assurance:** How does management know, independently of the teams operating the controls, that the IAM and PAM environment works as intended?

The ultimate objective is not to eliminate every identity-related risk. Organizations must delegate authority to employees, administrators, suppliers, applications, and automated processes in order to operate. The governance challenge is to ensure that this authority is **deliberate, proportionate, constrained, observable, reviewable, and revocable**. IAM determines how digital authority is distributed throughout the organization; PAM applies stronger governance to its most consequential forms. Together, they are one of the essential foundations through which cybersecurity governance becomes operational, measurable, and defensible.

## Chapter summary

Identity has become one of the most important cybersecurity control planes, because users, workloads, suppliers, and APIs now act across cloud, SaaS, and hybrid environments where the network is no longer the main boundary. The chapter argues that IAM and PAM are governance domains, not administrative functions: every digital identity carries delegated organizational authority, IAM determines how broadly that authority is distributed, and PAM controls its most consequential forms. When identity governance fails, attackers do not need to defeat controls; they inherit the authority to use, modify, or disable them. Managers must therefore ensure that digital authority is deliberate, proportionate, constrained, observable, reviewable, and revocable.

### Key points to retain

- IAM governs access across the organization, while PAM governs exceptional privileges within it; privileged access can alter the safeguards that constrain everyone else.

- Identity failures dominate incidents because credentials are scalable targets, trust extends beyond the organization, permissions accumulate, privileged pathways escape controls, and misuse resembles legitimate activity.

- A mature program delivers verified identity, least privilege, segregation of duties, lifecycle discipline, and auditable evidence; policies stay aspirational until identity controls enforce them.

- Access is a business risk decision implemented through technology, so decision rights must be shared among a central identity function, business owners, and risk, compliance, and audit oversight.

- The joiner–mover–leaver process should keep digital access aligned with organizational reality, removing obsolete access at every role change and departure, including for non-human identities.

- Not all MFA is equal: high-consequence identities need phishing-resistant authentication, conditional access, explicit authentication tiers, and recovery processes that are never the weakest path.

- Authorization determines what an authenticated identity may do; RBAC and ABAC must stay explainable, and access reviews matter only when findings lead to removal or formal risk acceptance.

- Segregation of duties prevents toxic combinations of permissions, uses preventive and detective enforcement, and relies on governed compensating controls such as dual control when separation is impractical.

- Third-party access must be owned internally, purpose-specific, time-limited, and governed by PAM; delegating access does not transfer accountability.

- PAM minimizes standing privilege, separates administrative identities, vaults secrets, monitors sessions, and constrains administrative pathways; break-glass access must bypass failed dependencies, not governance.

- Workloads are identities: service accounts, API keys, tokens, pipelines, and AI agents need inventories, owners, least privilege, short-lived credentials, monitoring, and retirement.

- Reliable, protected evidence and trend-based metrics tied to risk appetite show whether exposure is falling, and a risk-based roadmap should define milestones as risk-reduction outcomes.

### Key terms

- **Identity and Access Management (IAM)** — The policies, processes, technologies, roles, and accountability mechanisms used to govern human and non-human digital identities and their access to resources throughout their lifecycle.

- **Privileged Access Management (PAM)** — The specialized domain of identity governance that controls identities, credentials, and activities conferring elevated authority capable of materially affecting systems, data, security settings, or business outcomes.

- **Least privilege** — The principle that an identity receives only the access needed for a legitimate role, task, or function, and no more.

- **Privilege creep** — The gradual accumulation of permissions as people and systems change roles or purposes without obsolete access being removed.

- **Joiner–mover–leaver (JML)** — The lifecycle process that creates, adjusts, and removes access as people join, change responsibilities, and leave the organization.

- **Phishing-resistant authentication** — Authentication in which cryptography binds the authenticator to the legitimate service so the protocol, not user vigilance, prevents capture and replay, as with FIDO2/WebAuthn keys or PKI smart cards.

- **Authentication Assurance Level (AAL)** — A NIST SP 800-63-4 level of confidence that the person using an identity controls its authenticators; AAL3 requires a phishing-resistant, non-exportable authenticator.

- **Conditional access** — Predefined policies that decide whether and on what terms access is granted, using role, authentication strength, device posture, context, risk, and resource sensitivity.

- **Role-Based Access Control (RBAC)** — An authorization model that assigns approved permission sets to defined roles and then associates identities with those roles.

- **Attribute-Based Access Control (ABAC)** — An authorization model that decides access using attributes of the identity, resource, action, and context, such as device status or data classification.

- **Toxic combination** — A set of individually legitimate permissions that together allow one identity to complete and conceal a high-risk process without independent challenge.

- **Just-in-time (JIT) access** — A PAM pattern in which privilege is granted on request for an approved purpose and removed automatically when the task or time window ends.

- **Break-glass access** — Governed emergency access that lets authorized personnel regain control when normal identity or administrative pathways fail, without becoming a permanent bypass.

- **Non-human identity** — A service account, workload identity, API key, token, certificate, pipeline, or similar machine credential through which software exercises delegated authority.

- **Identity threat detection and response (ITDR)** — Vendor term for identity-related detection, such as unusual privilege elevation, suspicious recovery, dormant-account use, or emergency-account activation, correlated with other security evidence.

## Review questions

### Essay questions

1.  The chapter argues that privileged access is 'delegated organizational authority within digital systems' rather than a stronger form of ordinary access. Analyze this claim and explain how it changes the way managers should govern administrators, suppliers, and machine identities.

2.  Evaluate the statement that 'a completed certification proves that a process ran, not that access risk was competently assessed.' Design an access review and segregation-of-duties process that would provide genuine assurance in a payments or purchase-to-pay process.

3.  An organization reports 99 percent MFA coverage and concludes that its authentication risk is low. Using the chapter, critique this conclusion and recommend an authentication strategy, including tiers, phishing resistance, conditional access, and recovery.

4.  Apply the chapter to an organization of your choice (or to the Québec manufacturer in the illustrative scenario). Identify its most consequential identity risks, propose a prioritized IAM and PAM roadmap, and define three milestones expressed as risk-reduction outcomes.

5.  Non-human identities and AI agents increasingly exercise organizational authority with no person involved at the moment of action. Assess why they are often a governance blind spot and propose the controls, evidence, and metrics management should require.

### Multiple-choice questions

Choose the best answer. Answers are listed in the appendix "Answers to multiple-choice questions".

**1.** According to the chapter, what is the most useful managerial distinction between IAM and PAM?

> A. IAM governs access across the organization, while PAM governs exceptional privileges within it
>
> B. IAM governs employee accounts, while PAM governs all supplier and contractor accounts
>
> C. IAM is a business process, while PAM is a purely technical function owned by IT
>
> D. IAM covers cloud services, while PAM covers on-premises servers and network devices

**2.** Why does the chapter describe privileged access as more than a stronger form of ordinary access?

> A. It always requires hardware tokens that ordinary users are never issued
>
> B. It is delegated organizational authority that can change the safeguards constraining other users
>
> C. It is legally owned by the IT department rather than by business units
>
> D. It is used only during emergencies and is otherwise disabled by default

**3.** Which of the following is one of the five structural reasons the chapter gives for identity failures dominating the incident landscape?

> A. Encryption algorithms protecting stored passwords are routinely broken by attackers
>
> B. Firewalls cannot inspect traffic once users connect from remote locations
>
> C. Permissions accumulate over time because removal receives less attention than granting
>
> D. Identity platforms are rarely included in vendor support contracts

**4.** In the chapter's three-layer IAM operating model, which layer should judge whether a particular employee genuinely needs access to a business resource?

> A. The central identity function, because it operates the identity platforms and workflows
>
> B. Internal audit, because it provides independent assurance over access decisions
>
> C. The help desk, because it receives and processes most access requests
>
> D. Business ownership, because access is a business risk decision implemented through technology

**5.** An analyst transfers from accounts payable to treasury. Under joiner–mover–leaver discipline, what should the role change trigger?

> A. Provisioning of treasury access while keeping prior access in case it is needed
>
> B. A full account deletion and new identity, with no review of earlier approvals
>
> C. No change until the next annual access certification campaign is completed
>
> D. Provisioning of treasury access and removal of accounts-payable access no longer justified

**6.** Why does the chapter warn that push-based MFA approvals are not phishing-resistant?

> A. Push notifications cannot be delivered to mobile devices outside the corporate network
>
> B. Push approvals store the user's password in plain text on the mobile device
>
> C. Attackers can send repeated requests until a user accepts one, a pattern called MFA fatigue
>
> D. Push methods are not recognized by NIST SP 800-63-4 at any assurance level

**7.** Under NIST SP 800-63-4 as described in the chapter, how are synced passkeys treated?

> A. Acceptable at AAL3 because they are phishing-resistant and convenient for users
>
> B. Acceptable at AAL2 but not at AAL3, which needs a phishing-resistant, non-exportable authenticator
>
> C. Not acceptable at any level because they can be copied between devices
>
> D. Acceptable only for identity proofing, not for authentication or federation

**8.** An organization deploys phishing-resistant keys for administrators, but the help desk resets accounts after a caller answers basic personal questions. What weakness does this illustrate?

> A. Recovery has become an alternative path around strong authentication controls
>
> B. Conditional access policies have been configured too strictly for administrators
>
> C. Role explosion has made administrator roles too difficult to review
>
> D. Federation has extended trust to an unmanaged external identity provider

**9.** Which statement best describes the governance risk of role explosion in RBAC?

> A. Roles become so broad that every user ends up receiving administrator permissions by default
>
> B. So many narrow roles exist that they are almost as hard to govern as permissions
>
> C. Roles are replaced entirely by attributes, making access decisions unpredictable and hard to audit
>
> D. Roles are deleted automatically whenever business processes change or the organization restructures

**10.** According to the chapter, what is the main governance concern with ABAC?

> A. Overlapping attributes and conditional policies can be hard to predict, test, and audit
>
> B. It cannot use context such as device status, location, or authentication strength
>
> C. It requires every user to be assigned to exactly one job role at all times
>
> D. It is prohibited by ISO/IEC 27002:2022 for systems holding sensitive information

**11.** A business unit reports 100 percent completion of its quarterly access certifications, yet almost no access was removed. What does the chapter suggest management should conclude?

> A. The access model is clean, so reviews can safely move from quarterly to annual
>
> B. Reviewers are performing well, so remediation tracking is no longer necessary
>
> C. Review frequency is too high and is creating unnecessary administrative work
>
> D. Completion shows the process ran, but not that access risk was competently assessed

**12.** Which combination is the clearest example of a toxic combination in a purchase-to-pay process?

> A. One user can view supplier records and also print purchase-order reports
>
> B. One user can submit expense claims and also view their own payment history
>
> C. One user can create a new supplier and also release payments to that supplier
>
> D. One user can update their own contact details and also reset their password

**13.** How does the chapter link segregation of duties to the fraud triangle introduced in Chapter 1?

> A. SoD reduces rationalization by requiring employees to sign annual ethics statements
>
> B. SoD reduces pressure by distributing workloads more evenly across finance teams
>
> C. SoD reduces opportunity, the fraud condition that controls can most directly influence
>
> D. SoD removes all three fraud conditions once preventive conflict rules are in place

**14.** What principle does the chapter state about third-party access governance?

> A. Suppliers become fully accountable for any access granted under a signed contract
>
> B. Federated partner identities need no internal governance once trust is configured
>
> C. External access should be permanent to avoid disrupting essential supplier support
>
> D. Delegating access does not transfer accountability; external access stays within our risk boundary

**15.** Which practice best reflects the PAM goal of reducing standing privilege?

> A. Granting time-limited elevation for an approved task and removing it automatically afterwards
>
> B. Giving all administrators permanent rights but recording their sessions on video
>
> C. Sharing one administrator password among the team and rotating it every year
>
> D. Assigning administrative rights to users' ordinary email and browsing identities

**16.** Which statement captures the chapter's principle for break-glass emergency access?

> A. It should rely on the main identity provider so that it stays fully consistent
>
> B. It should bypass failed dependencies, not governance itself
>
> C. It should be known to all administrators so it is always available in a crisis
>
> D. It should be excluded from monitoring so attackers cannot detect its existence

**17.** Why does the chapter argue that least privilege matters even more for machine identities than for people?

> A. An overprivileged automated process may run thousands of times daily and be trusted implicitly
>
> B. Machine identities are always exempt from logging, so their activity cannot be observed
>
> C. Machine identities can never be classified as privileged under the PAM model
>
> D. Machine identities always use phishing-resistant authentication, so they are rarely compromised

**18.** Logs of privileged activity are stored on a platform that the same domain administrators can modify and delete. What is the main assurance problem?

> A. The logs are probably too small to support a full forensic investigation
>
> B. The logs will be retained too long and breach every privacy requirement
>
> C. The evidence can be altered by the identities whose actions it documents
>
> D. The logs record activity in a format that managers cannot read directly

**19.** Which of the following is an effectiveness metric rather than an activity metric, in the chapter's terms?

> A. Ten thousand access reviews completed during the year
>
> B. 95 percent of inappropriate permissions remediated within 30 days
>
> C. Five hundred privileged-access requests processed this quarter
>
> D. Two thousand password resets handled by the service desk

**20.** In the Québec manufacturer scenario, why did the CISO reorder the vendor's proposed IAM roadmap?

> A. The board refused to fund any role-engineering work across its applications
>
> B. Regulators required PAM deployment before any other identity control could proceed
>
> C. The outsourced provider could not support the ERP system's role-based access model
>
> D. Two scenarios exceeded risk appetite, so the most consequential access paths were addressed first
