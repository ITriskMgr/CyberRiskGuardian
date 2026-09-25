<!-- Introduction to Cybersecurity Governance, 3rd ed., v2.1g (Aug 2026) - (c) Marc-André Léger. Licensed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/). Third-party standards, frameworks and quotations remain under their owners' terms. -->

# Chapter 10: Identity and Access Management

In modern enterprises and cloud environments, identity has become one of the most important cybersecurity control planes. Networks remain essential, but they are no longer the sole—or necessarily the primary—boundary for access control. Users, applications, devices, workloads, data, service accounts, APIs, and third-party integrations now operate across SaaS platforms, cloud services, remote-work environments, partner ecosystems, and hybrid infrastructures. In this environment, the central cybersecurity question is no longer simply how the network is segmented, but rather **who or what may access which resources, under what conditions, with what level of privilege, and how that access is monitored, reviewed, and evidenced over time**.

For managers, **Identity and Access Management (IAM)** and **Privileged Access Management (PAM)** are not merely technical or administrative functions. They are fundamental governance control domains. They determine who can act within organizational systems, approve transactions, view or modify sensitive information, administer infrastructure, change configurations, and override normal safeguards. In this sense, IAM and PAM directly shape the organization’s exposure to fraud, sabotage, ransomware, espionage, insider misuse, privacy and regulatory breaches, and operational disruption.

When IAM fails, the organization loses control of routine access. When PAM fails, it loses control of its most powerful capabilities. Many significant cyber incidents are initiated, enabled, or amplified by weaknesses in identity governance: stolen credentials, weak authentication, incomplete deployment of multi-factor authentication, excessive privileges, poor separation between standard and administrative accounts, unmanaged service accounts, exposed API keys and secrets, ineffective joiner–mover–leaver processes, and insufficient monitoring of privileged activity. Attacks that appear highly technical on the surface are often, at their core, failures of authentication, authorization, or privilege governance. An attacker may first obtain a legitimate user's identity, then acquire additional privileges, then gain administrative authority, and ultimately perform actions using permissions the organization itself has granted. At that point, the distinction between an external attacker and an authorized system user becomes operationally blurred.

Cloud and SaaS environments intensify this reality because identity often serves as the primary control plane. Administrative identities may control management consoles, configurations, databases, storage services, virtual networks, encryption keys, logging systems, security controls, and workloads. A compromised cloud administrator account can therefore enable an attacker to alter configurations, expose or delete data, disable monitoring, weaken security controls, manipulate encryption keys, establish persistence, create new identities, or move across interconnected environments. The potential impact is especially significant because highly privileged identities often span multiple services and can affect large portions of the organization’s technology estate.

The same governance challenge applies to **non-human identities**, including service accounts, workloads, automation scripts, API credentials, tokens, secrets, certificates, and machine-to-machine identities. These identities are increasingly numerous and often have substantial privileges because they enable automated business and technical processes. Yet they may receive less governance attention than human accounts. Credentials may be long-lived, ownership may be unclear, permissions may accumulate over time, and access may persist long after the original business requirement has disappeared. As organizations rely more on cloud services, APIs, automation, and artificial intelligence, governing non-human identities becomes an increasingly important part of cybersecurity governance.

IAM and PAM therefore warrant dedicated attention because they bridge **governance intent and technical execution**. Risk appetite, compliance obligations, segregation of duties, privacy requirements, least privilege, operational resilience, and accountability all depend on the organization’s ability to govern identity effectively. Policies may state that access must be restricted, privileges minimized, and sensitive actions monitored, but those governance objectives become meaningful only when identity controls translate them into enforceable system behaviour.

Authentication, authorization, access provisioning, access reviews, privileged-session management, credential and secret management, logging, monitoring, and evidence collection should not be viewed as isolated technical tasks. Together, they form the mechanisms through which the organization determines **who or what may act, what actions may be performed, under which conditions, how much authority to grant, and how management can determine whether that access remains appropriate over time**. Effective identity governance is therefore not simply about controlling access to technology. It is about controlling the exercise of organizational authority within digital systems.

## IAM and PAM: Definitions and the Managerial Boundary

Identity and Access Management (IAM) encompasses the policies, processes, technologies, roles, and accountability mechanisms an organization uses to govern digital identities and their access to information, systems, applications, and other resources. IAM covers both human and non-human identities, including employees, contractors, partners, customers, devices, applications, service accounts, and workloads. It governs how these identities are established, authenticated, authorized, monitored, reviewed, modified, and ultimately removed.

In practical terms, IAM encompasses identity lifecycle management, authentication, authorization, access provisioning and deprovisioning, identity federation, single sign-on, access reviews, and evidence generation to demonstrate that access remains appropriate. Its purpose is not simply to grant access, but to ensure that access aligns with a legitimate business need, reflects the individual or system's current role, and changes as responsibilities, employment status, contractual relationships, or operational requirements change.

Privileged Access Management (PAM) is a specialized domain within identity governance focused on identities, permissions, credentials, and activities that confer elevated or exceptional authority. Privileged access is access that can materially affect systems, security configurations, data protection, business processes, financial outcomes, or other users' access rights. This includes administering infrastructure, modifying system configurations, creating or deleting identities, changing permissions, disabling security controls, accessing sensitive information at scale, executing high-impact transactions, or intervening in systems during emergencies.

PAM therefore applies tighter controls over these powerful capabilities. Typical PAM mechanisms include privileged identity governance, credential vaulting, administrative account separation, just-in-time and just-enough privilege, approval workflows, privileged-session monitoring and recording, command or activity monitoring, periodic privileged-access reviews, emergency or break-glass procedures, and controls for privileged non-human identities, such as service accounts, automation tools, scripts, API credentials, tokens, and secrets. These controls are important because privileged access can bypass, modify, or disable safeguards that constrain ordinary users.

A useful managerial distinction is therefore:

> **IAM governs access across the organization, while PAM governs exceptional privileges within it.**

IAM asks whether a person, device, application, or service has appropriate access to perform legitimate work. PAM asks whether access that could lead to disproportionately significant consequences is minimized, justified, tightly controlled, monitored, and documented. The distinction is not absolute—PAM is closely integrated with IAM—but it is useful because the consequences of privileged access failures are generally much greater.

<img src="media/image50.png" style="width:6in;height:4in" />

Figure 50: IAM and PAM

Both IAM and PAM are essential to cybersecurity governance, but PAM warrants greater scrutiny because privileged identities have the authority to alter the environment in which other controls operate. Mismanaged ordinary access can expose sensitive information, facilitate fraud, violate segregation-of-duties requirements, or create unnecessary operational risk. Mismanaged privileged access can enable an attacker or malicious insider to alter the organization's security posture.

For example, compromising an administrator account, cloud management account, domain administrator, database administrator, highly privileged service account, or automation credential may allow an adversary to create new identities, elevate privileges, change configurations, disable monitoring, modify or delete logs, access or exfiltrate large volumes of information, deploy malicious software, interfere with recovery capabilities, destroy backups, or establish persistent access. Because privileged users and services can modify the safeguards intended to constrain activity, the compromise of a privileged identity can undermine multiple layers of protection simultaneously.

This is why privileged access is more than a stronger form of ordinary access. It is **delegated organizational authority within digital systems**. A privileged identity may be able to make decisions or perform actions that would otherwise require formal organizational authority: approving changes, modifying financial or operational records, altering security policies, granting access to others, or interrupting critical services. PAM therefore governs not only credentials but also the controlled exercise of this delegated authority.

For Business Technology Management professionals, the distinction between IAM and PAM raises distinct governance questions. IAM governance asks whether access is based on legitimate business requirements, aligned with roles and responsibilities, periodically reviewed, adjusted as circumstances change, and promptly removed when no longer required. PAM governance asks whether exceptional authority has been minimized, separately assigned, specifically approved, limited in duration and scope where practicable, isolated from routine user activity, closely monitored, and subject to independent review.

Together, IAM and PAM operationalize several fundamental cybersecurity governance principles. **Least privilege** requires that identities receive no more authority than necessary. **Segregation of duties** prevents a single identity from exercising incompatible or excessive powers without oversight. **Accountability** requires that significant actions be attributable to identifiable actors or controlled services. **Lifecycle governance** ensures that access evolves as organizational circumstances change. **Monitoring and evidence** allow management to determine whether access controls function as intended, rather than merely assuming they do.

The managerial objective is therefore not merely to ensure that users can log in or that administrators can perform their work. It is to ensure that **digital authority is deliberately granted, proportionate to legitimate need, constrained by appropriate controls, continuously observable, periodically reassessed, and withdrawn when no longer justified**. This is where IAM and PAM function as governance mechanisms rather than merely technical access-control functions.

## Why Identity Failures Dominate the Incident Landscape

Managers should understand why identity-related weaknesses appear so frequently in cybersecurity incidents. Identity is particularly attractive to attackers because compromising a legitimate identity allows them to enter and operate within the organization through many of the same pathways used by employees, administrators, applications, devices, and suppliers. Once an attacker obtains valid credentials, a usable session, or another trusted identity, malicious activity may no longer resemble a conventional external intrusion. To the systems involved, it may initially appear to be legitimate work performed by an authorized user or service.

- First, **credentials are scalable attack targets**. Attackers can steal, phish, purchase, reuse, guess, or extract passwords, authentication tokens, session cookies, API keys, and other credentials from users and compromised systems. Credential stuffing exploits password reuse across services, while phishing and social engineering exploit trust, urgency, authority, and human error. Credentials exposed in one breach may later be tested against unrelated services, and information from compromised devices or browsers can be used to hijack authenticated sessions. This combination of availability, reuse, and automation makes identity compromise an efficient and attractive initial-access vector.

- Second, **modern organizations increasingly extend identity relationships beyond their traditional boundaries**. Employees are only one category of identity that requires access. Contractors, suppliers, consultants, partners, managed service providers, cloud platforms, software integrations, and APIs all rely on some form of digital trust relationship. Each relationship can create business value, but it can also create an additional pathway into organizational resources. Identity governance must therefore extend beyond the workforce to encompass third-party identities, federated identities, supplier-managed accounts, delegated administration, service accounts, workloads, applications, and other non-human identities. The governance problem is no longer simply determining which employees should have access; it is determining which people and systems across an increasingly interconnected ecosystem should be trusted, to what extent, and under what conditions.

- Third, **permissions tend to accumulate over time**. Access is often granted quickly because business activity depends on it, while removal receives less attention because it rarely delivers an immediate operational benefit. Employees change roles, join projects, assume temporary responsibilities, move between departments, or leave the organization. Consultants complete assignments, suppliers change personnel, and temporary exceptions persist beyond their intended duration. Service accounts created for one purpose may later be reused for other purposes. Unless access is systematically reviewed and adjusted, privileges can accumulate beyond legitimate business requirements. This phenomenon, often described as **privilege creep**, increases the organization's attack surface, weakens least-privilege principles, and can create conflicts with segregation-of-duties requirements.

- Fourth, **privileged pathways may operate outside the controls governing ordinary access**. Shared administrator accounts, long-lived service-account passwords, unmanaged API keys, permanent local-administrator rights, emergency accounts, embedded credentials in scripts, and broadly assigned cloud-administrator roles may receive inadequate monitoring, rotation, review, or ownership. These weaknesses are particularly consequential because privileged identities can alter the environment itself. They may modify configurations, grant additional permissions, disable security controls, establish persistence, access sensitive information at scale, interfere with backup and recovery mechanisms, or alter or delete evidence. A privileged identity compromise can therefore amplify an initially limited intrusion into an enterprise-wide incident.

- Fifth, **identity-based attacks can be difficult to distinguish from legitimate activity**. A firewall may permit the connection. An application may accept the credentials. A cloud platform may validate the authentication token. A database may authorize the requested query. From each control's perspective, the activity may appear permissible because the identity has valid access rights. The cybersecurity problem is that technical authorization does not necessarily mean the person or process exercising it is legitimate. An attacker using stolen credentials, a hijacked session, or a compromised service account can perform malicious actions through apparently valid access channels.

This distinction highlights a key limitation of traditional perimeter-focused security. Knowing that an authenticated identity is permitted to perform an action does not establish that the activity is appropriate in the current circumstances. Organizations increasingly need additional context: the device being used, location, authentication strength, the sensitivity of the requested resource, time of access, historical behaviour, risk indicators, and whether the requested action is consistent with the identity's legitimate role. This is why mechanisms such as phishing-resistant authentication, conditional access, behavioural analytics, privileged-session monitoring, and continuous identity assurance have become increasingly important.

From a governance perspective, IAM and PAM are **high-leverage control domains** because they can influence both the likelihood and the consequences of cyber incidents. They reduce the likelihood of successful compromise by strengthening authentication, controlling identity creation, limiting unnecessary access, and managing credentials and identity lifecycles. They reduce the potential impact of compromise by enforcing least privilege, segregation of duties, administrative-account separation, conditional access, just-in-time privilege, and restrictions on high-risk activities. They also strengthen accountability by showing who or what had access, who approved it, when it was exercised, what actions were performed, and whether the access remained justified over time.

The business case for effective identity governance is therefore much broader than protecting login processes. Strong IAM and PAM can reduce exposure to fraud, limit the blast radius of ransomware and other intrusions, constrain insider misuse, reduce privacy and data-protection risks, strengthen control over supplier access, improve auditability, and reduce the likelihood that a compromised identity will cause widespread operational disruption. They also provide management with evidence that access decisions are deliberate, reviewed, proportionate, and defensible.

In a modern digital organization, identity governance is a clear point where cybersecurity controls intersect with managerial accountability. **Every digital identity represents some degree of delegated organizational authority. IAM determines how broadly that authority is distributed; PAM determines how the organization controls its most consequential digital powers.** When identity governance fails, attackers do not necessarily need to defeat the organization's controls—they may instead inherit the authority to use, modify, or disable them.

## Governance Objectives for IAM and PAM: What Good Looks Like

A mature IAM and PAM governance program should deliver clear, observable, and defensible outcomes. Its purpose is not merely to administer accounts or automate access requests, but to ensure that digital access is **appropriate, justified, proportionate, controlled, monitored, and continuously aligned with legitimate business needs**. Effective governance should enable management to demonstrate not only that access controls exist, but also that they operate consistently and that deviations are identified and corrected. Five governance objectives are especially important.

- The first objective is **verified identity**. Before granting access, the organization must establish, with an appropriate level of confidence, that the person, device, application, workload, or service requesting access is the identity it claims to be. For human users, this includes appropriate identity proofing, strong authentication, multi-factor authentication, and higher-assurance requirements for sensitive or privileged roles. The required level of assurance should be proportionate to the consequences of unauthorized access, rather than applied uniformly to every user and resource.

For non-human identities, verification requires confidence that the application, workload, service account, API client, automation process, or device presenting credentials is the intended system and has not been replaced, impersonated, or compromised. As organizations increasingly rely on cloud workloads, APIs, automation, and machine-to-machine interactions, this aspect of identity governance becomes more important. Authentication is therefore not simply about passwords or users; it is about establishing sufficient confidence in the identity of any actor permitted to exercise digital authority.

- The second objective is to apply **least privilege and need-to-know by default**. Grant access only to the extent necessary to perform a legitimate role, task, business process, or system function. It should not be granted merely because it might be useful later, because similar users already have it, or because broad permissions are administratively easier to manage. Access decisions should instead reflect a clear business requirement and the sensitivity of the resources involved.

Privileged access requires even greater discipline. Minimize elevated permissions, keep them separate from ordinary user access, limit their scope, restrict their duration where practicable, approve them explicitly when appropriate, and remove them when the underlying requirement ends. Just-in-time and just-enough-access approaches can further reduce standing privilege available for misuse or compromise. Least privilege therefore reduces both the likelihood of inappropriate access and the potential **blast radius** if an identity is compromised.

- The third objective is **segregation of duties and controlled decision rights**. High-risk activities should not rely on the unchecked authority of a single individual or identity. Where the consequences justify it, the stages of a sensitive process should be distributed across separate roles so that one actor cannot initiate, approve, execute, and conceal the same high-impact action without oversight.

This principle is especially important in financial transactions, procurement, payroll, identity administration, privileged-access approval, security configuration, sensitive data management, change management, and incident recovery. Segregation of duties is broader than a cybersecurity control. It serves as a fraud-prevention mechanism, an internal-control principle, a compliance requirement, and a governance mechanism to prevent excessive concentrations of digital authority. In technical environments, it may be reinforced through approval workflows, separation of administrative roles, dual authorization, and restrictions on conflicting permissions.

- The fourth objective is **identity lifecycle discipline**. Access should align with the identity’s legitimate lifecycle and the business relationship it represents. It must be created, modified, suspended, reviewed, and removed as employees join the organization, change responsibilities, take on temporary assignments, move between departments, take extended leave, change employment status, or leave. These joiner–mover–leaver processes should be reliable enough to translate changes in organizational reality into access changes without unnecessary delay.

The same principle applies beyond employees. Contractors and suppliers complete assignments; partners change responsibilities; applications are retired; workloads are replaced; devices are decommissioned; API integrations are discontinued; service accounts become obsolete; and tokens, keys, certificates, and other credentials expire or no longer serve a legitimate purpose. Without disciplined lifecycle governance, access tends to persist even after its justification has disappeared. Over time, orphaned identities and accumulated permissions increase exposure silently, making lifecycle failures one of the most predictable sources of excessive access and privilege creep.

- The fifth objective is **auditability and evidence**. The organization must be able to reconstruct and demonstrate how access was governed. Management should be able to determine who or what had access, which permissions were assigned, why access was required, who approved it, when it became effective, when it was used, which significant actions were performed, when it was reviewed, and when it was modified or removed.

For privileged access, the evidentiary requirement is especially important because privileged actors can alter systems, grant access to others, modify sensitive data, change security configurations, disable controls, and potentially interfere with the evidence used to investigate their own actions. Appropriate logging, privileged-session monitoring, approval and access-review records, credential-management evidence, and tamper-resistant audit trails therefore provide an essential basis for accountability.

Auditability should not, however, be confused with merely generating large volumes of logs. Evidence has governance value only when it is sufficiently complete, reliable, protected, retained, and reviewable to support decision-making and assurance. Without credible evidence, an organization may have controls in principle but cannot demonstrate that they operated as intended. It will also be less able to investigate incidents, attribute actions, support disciplinary or legal proceedings, satisfy auditors and regulators, or demonstrate compliance with internal and external requirements.

Together, these five objectives define the desired state for effective IAM and PAM governance:

- Identity is not simply trusted; it is verified.

- Access is not simply granted; it is justified and periodically reassessed.

- Privilege is not simply assigned; it is minimized and controlled.

- Authority is not unnecessarily concentrated; it is separated where risk requires it.

- Activity is not merely logged; it is made observable and supported by credible evidence.

In this sense, IAM and PAM translate governance principles into day-to-day digital behaviour. Policies, risk appetite, least privilege, segregation of duties, accountability, and compliance obligations remain largely aspirational unless they are enforced through identity controls. IAM and PAM are therefore key mechanisms for translating **governance intent into controlled, auditable digital authority**.

## IAM Governance in Practice: Decision Rights, Operating Model, and Accountability

Identity and Access Management is inherently a cross-functional governance system. IT alone cannot govern it, and cybersecurity cannot implement it effectively on its own. Decisions about identity and access affect business operations, fraud exposure, privacy, regulatory compliance, segregation of duties, operational resilience, third-party risk, and incident response. The central governance question is therefore not simply how access is provisioned, but **who has the authority to determine what access is appropriate, under what conditions, and with what accountability**.

Effective IAM governance rests on clearly defined **decision rights**. The organization must establish who owns the access-control model, defines business roles, approves access, determines authentication requirements, reviews permissions, maintains IAM policies and standards, operates identity platforms, authorizes exceptions, and is accountable when access decisions create unacceptable risk. Without clearly assigned decision rights, IAM can easily deteriorate into a ticket-processing function in which access is granted because someone requested it rather than because an accountable authority determined it was necessary and appropriate.

A practical IAM operating model typically comprises three interdependent layers of responsibility.

- The first is a **central identity and access function**, typically located within cybersecurity, IT security, enterprise technology, or a shared governance function. This group provides the technical and procedural foundation for the IAM program. It designs and maintains the IAM architecture, operates or oversees identity platforms, establishes technical standards, implements authentication and federation mechanisms, supports provisioning and deprovisioning workflows, manages conditional-access capabilities, integrates applications with identity services, and ensures that identity-related controls generate sufficient evidence for monitoring and assurance. Its role is to make governance requirements technically enforceable and operationally repeatable.

- The second layer is **business ownership**. Business managers and information owners must determine which roles exist, what access is legitimately required to perform those roles, which information and processes are particularly sensitive, and who has authority to approve access within their areas of responsibility. This distinction is fundamental. Technology teams can administer permissions, but they generally cannot determine whether a particular employee, contractor, application, or service genuinely requires access to a specific business resource. That judgment depends on business context, operational responsibilities, regulatory obligations, and risk. Access is therefore a business risk decision implemented through technology, not merely a technical request fulfilled by IT.

- The third layer comprises **risk, compliance, privacy, and assurance oversight**. These functions provide independent or semi-independent challenge and verification. They assess whether access-control policies are appropriate, whether approvals are supported by legitimate business requirements, whether segregation-of-duties conflicts are identified and addressed, whether periodic access reviews are completed, whether privileged access is properly controlled, and whether exceptions are documented and justified. They also help determine whether IAM practices remain aligned with regulatory obligations, privacy requirements, contractual commitments, internal-control expectations, and the organization's risk appetite. Internal audit may subsequently provide independent assurance that these governance and management processes operate as intended.

Managers should insist on explicit accountability for high-risk access. A model in which **“IT grants access when asked”** is insufficient governance. It reflects operational execution without clearly defined decision authority, risk ownership, or independent oversight. A mature IAM model deliberately separates these responsibilities: business owners determine legitimate access requirements; cybersecurity defines and maintains appropriate control patterns; IT and identity teams operate reliable provisioning and authentication processes; risk and compliance functions provide oversight; and internal audit independently evaluates whether the governance model is functioning effectively.

This division of responsibility also helps prevent a common governance failure: allowing the same function to request, approve, implement, and validate access without sufficient challenge. The degree of separation should match the risk. Routine, low-risk access may be largely automated under predefined policies, whereas sensitive, conflicting, or privileged access may require explicit approval, stronger authentication, additional monitoring, time limits, or independent review. Governance should therefore apply greater scrutiny to access that confers greater organizational authority or potential harm.

Exceptions require equally clear governance. Temporary access, emergency permissions, unusual business requirements, and legacy-system limitations may occasionally make standard IAM controls impractical. Do not handle these situations informally. Each exception should have an identified owner, a documented justification, a defined scope, compensating controls where necessary, an expiration or review date, and appropriate approval commensurate with the level of risk. Otherwise, temporary exceptions can quietly become permanent sources of excessive access.

The practical objective is to ensure that access decisions **are traceable, explainable, and defensible**. For every significant role, sensitive system, important data set, or privileged capability, the organization should be able to answer a consistent set of questions: Who owns this access? Who is authorized to approve it? Why is it required? What conditions constrain its use? What conflicting permissions must be prevented? How often is this access reviewed? How is it revoked when no longer needed? What evidence demonstrates that these controls are operating effectively?

When these questions can be answered consistently and supported by reliable evidence, IAM moves beyond account administration. It becomes a governance system that enables the organization to assign digital authority, constrain its use, monitor its ongoing appropriateness, and hold accountable those who make and execute access decisions.

## Identity Lifecycle: The Most Underestimated Control

The **joiner–mover–leaver (JML) process** underpins effective access discipline. Many identity-related failures do not stem from a lack of sophisticated security technologies. They stem from something more fundamental: identities and permissions are created, changed, and removed inconsistently as people and systems evolve within the organization. A mature IAM program must therefore treat identity lifecycle management as a core governance control rather than as an administrative human resources or IT process.

The underlying principle is straightforward: **digital access should consistently reflect organizational reality**. When a person joins the organization, changes responsibilities, assumes a temporary assignment, moves between business units, or leaves, the corresponding digital authority should be updated. The same principle applies when contractors complete engagements, suppliers change personnel, applications are retired, workloads are replaced, or automated processes no longer require specific credentials. The governance challenge is to ensure that changes to legitimate business authority are reliably and promptly reflected in digital access.

For **joiners**, the objective is to establish identity appropriately and grant access aligned with an approved business role. Onboarding should not rely on informal requests, copied permissions from another employee, or broad default access granted for convenience. While such approaches may be operationally expedient, they make it difficult to determine whether access is justified.

A mature onboarding process uses defined roles, standardized access profiles where appropriate, documented approvals, and clearly identified business ownership. Identity information should come from authoritative sources, and higher-risk access should receive greater scrutiny. The objective is not to minimize access to the point that people cannot do their work, but to ensure employees, contractors, suppliers, partners, and other authorized users receive **the access they need to perform their legitimate responsibilities—and no more**.

For **movers**, the principal risk is privilege creep. Employees routinely change jobs, participate in projects, assume temporary duties, relocate, transfer between departments, or take on new responsibilities. New access is often granted because the new work must begin immediately, while permissions tied to previous responsibilities remain unchanged. Over time, this leads to **privilege creep**: an accumulation of permissions that no longer reflects the user's current role.

Privilege creep creates several governance problems. It weakens least privilege, increases the number of resources exposed if an account is compromised, can create segregation-of-duties conflicts, and may allow individuals to exercise combinations of authority that management never explicitly approved. The risk can remain largely invisible because each individual permission may once have been legitimate.

A role change should therefore trigger both **provisioning and deprovisioning**. The appropriate governance question is not simply, “What new access does this person require?” It must also be, **“What existing access is no longer justified?”** Temporary permissions should have defined expiration dates or review points, where practicable. Significant role changes should trigger reassessment of sensitive and privileged access rather than merely adding new entitlements to an existing account.

For **leavers**, timeliness and completeness are especially important. Once an employment, contractual, supplier, or other trusted relationship ends, continued access may no longer be justified. Delayed or incomplete deprovisioning can create opportunities for unauthorized access, insider misuse, fraud, data theft, or post-employment activity. The degree of urgency should reflect the circumstances and risk of the departure, but organizations should establish clear procedures and service expectations rather than relying on informal coordination.

A complete leaver process may include disabling relevant accounts, revoking active authentication sessions and tokens, removing remote and privileged access, recovering organizational devices and credentials where applicable, removing delegated permissions, terminating access through federated or third-party systems, and addressing any shared secrets or credentials the departing individual accessed. The process may also require coordination among management, human resources, IT, cybersecurity, physical security, legal, records management, and other functions, depending on the role and the nature of the departure.

The objective is not simply to close the individual's primary user account. Modern users may have access through multiple pathways: SaaS applications, cloud platforms, VPN or remote-access services, administrative accounts, collaboration systems, development environments, supplier portals, API credentials, shared resources, or delegated permissions. Effective deprovisioning must therefore address the individual's actual **access**, not merely their most visible identity.

Lifecycle discipline must also apply to **non-human identities**. Service accounts, workloads, applications, API credentials, automation scripts, tokens, certificates, devices, secrets, and machine-to-machine identities are often created to support legitimate technical or business requirements and then remain in place long after those requirements change. Unlike employees, these identities do not resign, transfer departments, or trigger an HR termination process. Without explicit governance, they can persist indefinitely.

Every significant non-human identity should therefore have an identifiable owner and a legitimate purpose. Where appropriate, governance should establish requirements for credential rotation, expiration, periodic review, privilege reassessment, and decommissioning. When an application is retired, a supplier relationship ends, an integration is replaced, or an automated process changes, the associated identities and credentials should be reviewed and removed if no longer required. Otherwise, forgotten non-human identities can become persistent, poorly understood access paths into organizational systems.

Identity lifecycle management is also an important **compliance and assurance mechanism**. Lifecycle controls govern access to personal information, financial systems, regulated records, intellectual property, cloud environments, administrative functions, and critical operational assets. They provide evidence that the organization is translating employment, contractual, and business decisions into appropriate access-control actions.

Lifecycle controls are also relatively straightforward to test. Auditors and assurance functions can examine onboarding approvals, identity records, role mappings, access-change histories, termination notifications, account-disablement timestamps, privileged-access changes, periodic access reviews, expired temporary permissions, and exception approvals. This evidence helps management determine whether documented IAM policies are operating in practice.

Managers should therefore monitor lifecycle performance rather than assume the process is reliable. Useful indicators may include the percentage of terminated identities disabled within the required period, the number of orphaned accounts, overdue access reviews, dormant privileged identities, access that persists after role changes, expired contractor accounts, non-human identities without documented owners, and temporary privileges that remain active beyond their approved duration. These measures make lifecycle discipline visible and help identify recurring weaknesses before they contribute to an incident.

The managerial lesson is simple yet consequential: **access should follow legitimate authority throughout its lifecycle**. It should be established when a valid relationship begins, adjusted when responsibilities change, reviewed while active, and withdrawn when the relationship that originally justified it ends. Access should not accumulate by accident, persist because no one remembered to remove it, or outlast the termination of the relationship that originally justified it.

A well-governed joiner–mover–leaver process therefore does more than manage accounts. It keeps digital authority aligned with organizational reality. In doing so, it reduces excessive privilege, reinforces segregation of duties, improves auditability, eliminates unnecessary access paths, and limits the consequences of compromised or misused identities.

## Authentication Strategy: MFA, Phishing Resistance, and Conditional Access

Managers should treat **multi-factor authentication (MFA)** as a baseline control for significant organizational access, not an optional enhancement. Password-only authentication provides insufficient assurance in most modern enterprise environments, especially when users access cloud services, remote-work platforms, sensitive information, financial processes, administrative functions, or critical operational systems. However, the managerial objective should not be simply to “turn on MFA.” A mature authentication strategy determines **how much confidence the organization requires in an identity under different circumstances and selects authentication controls proportionate to that risk**.

Not all MFA methods provide the same level of protection. Using two factors does not necessarily make the authentication process resistant to phishing, social engineering, interception, or session compromise. Some mechanisms still depend heavily on the user's ability to recognize malicious requests.

SMS-based authentication codes, for example, may be exposed through attacks on the subscriber's mobile account, social engineering, or other weaknesses in the communications channel. Push-based authentication can be abused when attackers repeatedly generate approval requests in the hope that a user will eventually accept one, a pattern commonly described as **MFA fatigue** or **push fatigue**. One-time passwords can also be captured and immediately relayed via adversary-in-the-middle phishing infrastructure when users are deceived into entering them on fraudulent websites.

The governance implication is important: **MFA should not be treated as a single homogeneous control**. Management should understand which authentication mechanisms are deployed, the threats they mitigate, and where stronger authentication methods are required.

For high-risk access, organizations should increasingly favour **phishing-resistant authentication**. Phishing resistance reduces reliance on a user's ability to determine whether a login page or authentication request is legitimate. Instead, the authentication protocol itself helps prevent an attacker from capturing an authenticator output and replaying or relaying it to the legitimate service.

Modern phishing-resistant mechanisms generally rely on cryptographic authentication, in which the authenticator is bound to the legitimate verifier or to the authentication session. Examples include FIDO2/WebAuthn authenticators, appropriately implemented passkeys, smart cards, and other cryptographic authenticators. These approaches provide substantially stronger protection against conventional credential-phishing attacks than passwords paired with manually entered one-time codes.

Phishing-resistant authentication is especially important for identities that could have substantial organizational consequences. These include privileged administrators, identity administrators, cloud-management accounts, security personnel, financial approvers, users with production access, remote administrators, and individuals accessing highly sensitive information or critical systems. The principle is not that every identity must always use the strongest available mechanism, but that **authentication strength should increase as the potential consequences of identity compromise rise**.

Managers can use the **NIST Digital Identity Guidelines, SP 800-63-4**, as an important reference for structuring these decisions. The NIST model distinguishes three related but distinct assurance problems.

- **Identity proofing** concerns the confidence with which an organization establishes that a person is the real-world individual they claim to be at the time of enrolment.

- **Authentication** concerns the confidence that the person attempting to use a digital identity controls the authenticators associated with that identity.

- **Federation** concerns the mechanisms and assurance that enable trust in identity information and authentication results across systems, relying parties, and identity providers.

These distinctions matter because strong authentication cannot compensate for weak identity proofing, just as strong initial identity proofing cannot compensate for weak authentication later. Similarly, federating an identity extends trust to other systems and therefore requires governance of the identity provider, assertions, protocols, and relationships that convey that trust.

NIST further defines authentication requirements through **Authentication Assurance Levels (AALs)**. The managerial value of this model lies not in the terminology itself but in the principle behind it: different situations warrant different levels of authentication assurance. Routine, lower-risk access may not require the same controls as access that can change security configurations, authorize significant financial transactions, or control critical infrastructure. Authentication requirements should therefore be selected based on role, resource sensitivity, threat exposure, transaction significance, and potential business consequences, rather than applying a single authentication policy across the organization.

Authentication strategy should also extend beyond the initial login. A user who authenticated at the start of a session is not necessarily trustworthy indefinitely. Devices can be compromised, tokens can be stolen, sessions can be hijacked, user behaviour can change, and risk conditions can evolve. Mature organizations therefore complement authentication with **conditional and adaptive access controls**.

Conditional access applies predefined policies and risk criteria to determine whether access should be granted and under what conditions. Decisions may consider factors such as the identity's role, authentication strength, device security posture, network characteristics, geographic context, session risk, unusual behaviour, known threat indicators, and the sensitivity of the requested resource.

For example, routine access to a low-sensitivity application from a managed organizational device may require relatively little additional friction. The same user attempting to access sensitive information from an unmanaged device may be required to use stronger authentication or be prevented from downloading information. An administrator attempting to access a cloud-management environment under unusual conditions may be required to reauthenticate using a phishing-resistant method, use a compliant device, or satisfy additional controls before access is granted.

Conditional access therefore makes authentication requirements **context-sensitive rather than static**. Depending on the circumstances, a policy may require additional authentication, restrict the session, prohibit specific activities, shorten session duration, require a managed or compliant device, prevent downloads, force reauthentication, block access entirely, or generate an event for investigation.

This approach aligns with **zero-trust principles**. Authentication should not confer permanent or unlimited trust simply because a user logs in or because a connection originates from a traditionally trusted network. Instead, access decisions should consider identity, device, context, the requested resource, and current risk conditions. Trust becomes conditional and is continuously reassessed rather than being implicitly inherited from network location or a single successful authentication event.

Conditional access also links **risk appetite to operational enforcement**. Governance may, for example, determine that highly privileged administrative activity must never occur on unmanaged devices, that certain sensitive information cannot be accessed in specified contexts, or that unusual authentication behaviour requires additional assurance. Conditional-access mechanisms translate those decisions into repeatable technical rules.

**Proportionality** remains an essential managerial principle. Stronger authentication generally provides greater assurance, but it can also impose financial, operational, and usability costs. Excessive friction may prompt users to adopt unsafe workarounds, overwhelm support functions, or impede legitimate business processes. Conversely, authentication controls designed primarily for convenience can leave high-consequence access inadequately protected.

Governance should therefore establish **explicit authentication tiers**. Routine organizational access should have defined minimum requirements. Sensitive access should require stronger assurance. Privileged, administrative, and mission-critical access should receive the highest level of protection, justified by the organization's risk environment. Document these requirements and apply them consistently, rather than letting individual system owners determine them informally.

The strategy must also address **authentication recovery**, as these processes can become alternative pathways around otherwise strong controls. An organization may deploy phishing-resistant authentication and remain vulnerable if an attacker can persuade a help desk to reset an account using weak identity-verification procedures. Processes for lost authenticators, account recovery, authenticator replacement, emergency access, and credential reset should therefore provide assurance proportionate to the access they can restore. Recovery should not become the weakest path into a strongly protected identity.

Authentication governance should define more than the technologies used at login. It should establish minimum authentication requirements, higher-assurance requirements for sensitive and privileged access, approved authenticator types, conditional-access policies, recovery procedures, exception processes, monitoring expectations, user education, and periodic review. It should also ensure that changes in threats, technologies, business activities, and regulatory requirements trigger reassessment of the authentication strategy.

The managerial objective is to **align confidence with consequence**. Authentication does not prove that an actor will behave appropriately, nor can it guarantee that an identity has not been compromised. Its role is to provide a defensible level of confidence that the actor using a digital identity is legitimately entitled to do so. By combining appropriate identity proofing, risk-tiered authentication, phishing-resistant methods, conditional access, secure recovery, and continuous reassessment, the organization makes that confidence substantially more reliable and its access decisions more defensible.

## Authorization Models: Role Engineering, RBAC/ABAC, and Access Reviews

Authorization is the decision logic that determines **what an authenticated identity may do**. Authentication answers, “Is this actor who or what they claim to be?” Authorization then answers the next question: “Given that identity, what resources, functions, transactions, and data should the actor be allowed to access?”

This distinction is fundamental. Strong authentication can establish confidence in an identity, but it does not determine whether that identity should be able to view a particular data set, approve a payment, administer a server, modify a production application, or change another user’s permissions. An organization can therefore have strong MFA and still have weak access control if authenticated users have excessive or inappropriate permissions.

Many organizations struggle at this stage because they mistake the presence of roles and permissions for effective **authorization governance**. An application may include roles, security groups, profiles, privileges, permissions, and entitlements, but their presence alone does not prove the access model is coherent, current, risk-based, or aligned with actual business responsibilities. Authorization becomes governable only when the organization can explain why specific permissions exist, who should receive them, who owns the decision, what risks they create, and how it verifies their continued appropriateness.

### Role Engineering: Translating Business Responsibility into Access

**Role engineering** is the discipline of designing and maintaining roles that align with legitimate organizational responsibilities and of mapping the necessary permissions or entitlements to those roles. It bridges business responsibilities and technical access.

Without effective role engineering, maintaining least privilege becomes increasingly difficult at scale. Access is often granted on a per-user basis, typically in response to individual requests. Exceptions accumulate, inherited permissions become hard to understand, temporary access becomes permanent, and managers gradually lose the ability to explain why particular identities have specific privileges.

A mature authorization model therefore begins with business responsibilities rather than with system permissions. The organization should first determine what a person or system must be able to accomplish, then identify the minimum set of entitlements required to support those activities.

Role engineering requires meaningful participation from business owners. Business managers understand which responsibilities belong to specific jobs, which transactions employees should be able to perform, which information they require, and which combinations of authority would be inappropriate. Cybersecurity and IT bring a different perspective: they understand technical dependencies, privilege relationships, system limitations, threat exposure, and control requirements. Effective role design integrates these perspectives.

Roles must also be treated as **living governance objects,** not static configurations. Business processes change, applications are replaced, organizations restructure, responsibilities shift between departments, activities are outsourced, automation alters job functions, and regulatory requirements evolve. A role that provided appropriate access at creation may become excessive or insufficient over time.

Consequently, governance should establish clear ownership of significant roles, conduct periodic reviews of their associated entitlements, and define procedures for changing or retiring roles when business requirements change.

<img src="media/image51.png" style="width:6in;height:4.5in" />

Figure 51: Authorization models

### Role-Based Access Control

**Role-Based Access Control (RBAC)** assigns permissions to defined roles and then associates identities with those roles. Instead of determining independently whether each employee should have dozens or hundreds of individual permissions, the organization can define a role such as accounts-payable clerk, human-resources advisor, database administrator, or security analyst and associate an approved set of permissions with that role.

RBAC can significantly improve access governance. It can standardize provisioning, streamline onboarding and role changes, reduce arbitrary access assignments, improve consistency, facilitate segregation of duties, and make access reviews easier to understand. It is particularly useful when groups of people perform stable, clearly defined functions.

However, RBAC is only as effective as the roles it depends on. Poorly engineered roles may simply become large bundles of permissions that institutionalize excessive access. Organizations may also create too many increasingly specialized roles to accommodate exceptions, eventually leading to **role proliferation or role explosion**. When hundreds or thousands of narrowly differentiated roles exist, the model can become almost as difficult to understand and govern as individualized permissions.

Governance should therefore resist the assumption that creating more roles necessarily improves access control. Roles should remain meaningful, understandable, owned, and linked to identifiable business responsibilities.

### Attribute-Based Access Control

More dynamic environments may require **Attribute-Based Access Control (ABAC)**. Rather than relying primarily on predefined roles, ABAC uses attributes of the identity, resource, requested action, and surrounding context to determine whether access is permitted.

Relevant attributes may include department, employment status, job classification, geographic location, device status, data classification, project membership, transaction value, time of access, authentication strength, network characteristics, or current risk indicators. Policies can combine multiple attributes to enable more granular decisions.

For example, an organization might permit employees in a specific department to view a particular category of information only when accessing it from managed devices and when their employment status is active. A financial transaction above a defined threshold might require a specific job function, stronger authentication, and additional approval. Access to particularly sensitive data might be permitted only to users with defined responsibilities and from devices that meet specific security requirements.

ABAC can therefore enable more precise, context-sensitive access decisions, particularly in cloud, data-centric, distributed, and zero-trust architectures where static roles alone may be insufficient.

Greater flexibility, however, increases governance complexity. A large collection of overlapping attributes and conditional policies can be difficult to understand, predict, test, and audit. Small changes to an attribute, policy, or dependency may unexpectedly affect access.

The managerial objective is therefore not to choose the most technically sophisticated authorization model. It is to ensure that the model remains **appropriate, explainable, enforceable, testable, and auditable**. Complexity that the people accountable for the resulting risk cannot understand is a governance problem in itself.

In practice, organizations often combine RBAC and ABAC. Roles may establish a baseline set of permissions, while attributes and contextual conditions further constrain how, when, and from where those permissions may be exercised. This hybrid approach can combine the administrative simplicity of roles with the contextual precision of attribute-based policies.

### Authorization and Segregation of Duties

Authorization is also a primary mechanism for enforcing **segregation of duties (SoD)**. The organization should identify permission combinations that create unacceptable concentrations of authority.

For example, the same individual may not be permitted to create a supplier account and approve payments to that supplier. A developer may be restricted from unilaterally approving and deploying changes to production. An identity administrator may be permitted to create accounts but not to independently approve their own privileged access. A person responsible for security operations may require additional oversight before disabling certain monitoring controls.

These combinations are sometimes called **toxic combinations** because individual permissions may seem reasonable in isolation but create significant fraud, security, or control risk when combined.

Authorization governance must therefore consider not only whether each permission is legitimate but also whether the **combination of permissions** assigned to an identity results in inappropriate authority.

### Access Reviews: Testing Whether Authorization Remains Appropriate

Access reviews provide a mechanism for organizations to periodically assess whether previously approved authorizations remain justified. This is necessary because legitimate access decisions can become inappropriate over time as roles, responsibilities, systems, and risks change.

In practice, however, organizations often conduct access reviews poorly. Organizations may conduct periodic certification exercises in which managers receive long lists of users and technical permissions and are asked to approve them. Reviewers may not recognize entitlement names, understand what the access enables, recall why it was originally granted, or know whether it has been used recently.

Faced with uncertainty and large volumes of information, reviewers may simply approve existing access. The organization can then demonstrate that an access review occurred, but the process may provide little meaningful assurance.

This creates a dangerous distinction between **control completion and control effectiveness**. A completed certification is evidence that a process occurred; it is not necessarily evidence that access risk was competently assessed.

A governance-grade access review should therefore provide reviewers with **decision-useful information**. Reviewers should be able to understand what the access allows the identity to do, why it was originally granted, which role or responsibility justifies it, when it was last exercised, whether the user still performs the relevant function, whether the permission is privileged or sensitive, and whether it conflicts with other permissions.

Reviews should also be **risk-based**. Not every entitlement requires the same frequency or level of scrutiny. Generally prioritize privileged access, administrative roles, financial approvals, access to sensitive or regulated information, production systems, cloud management environments, external identities, dormant accounts, non-human identities, and permissions that can create segregation-of-duties conflicts.

High-risk access may warrant more frequent review, stronger evidence, or independent validation than routine, low-risk access does.

### Review Must Lead to Remediation

The effectiveness of an access review ultimately depends on what happens after you identify inappropriate access.

Removing excessive permissions, resolving segregation-of-duties conflicts, disabling dormant accounts, correcting ownership information, updating role definitions, terminating obsolete identities, and documenting justified exceptions are integral to the control. A review that identifies inappropriate access but does not result in removal, remediation, or formally authorized risk acceptance offers little protection.

Governance should therefore monitor not only whether access reviews were completed but also whether findings were resolved within appropriate timeframes. Useful indicators may include overdue certifications, the percentage of high-risk access reviewed on schedule, the number of permissions removed after review, unresolved segregation-of-duties conflicts, dormant privileged accounts, orphaned accounts, and exceptions that remain beyond their approved duration.

This distinction matters because organizations can achieve high review completion rates while still retaining substantial excessive access. Effective governance assesses the **quality and consequences of the review**, not merely the completion rate.

### Authorization as Continuous Governance

The managerial lesson is that authorization is not a one-time configuration exercise. An access decision that was appropriate at the time of approval does not remain so indefinitely. Roles evolve, users move, systems change, new information is created, organizational structures shift, and risk conditions change.

Authorization must therefore be a continuous governance process. Organizations must deliberately design and maintain roles. Entitlements must have clear purposes and owners. Access must be based on legitimate requirements. Organizations must identify and control conflicting permissions. Exceptions must be documented and limited. Reviews must provide meaningful information to accountable decision-makers, and findings must lead to action.

Authorization is the point at which several key governance principles become technically enforceable. Least privilege determines the authority level an identity receives. Segregation of duties determines which powers should not be combined. Privacy and data governance determine which information an identity may access. Fraud prevention constrains sensitive transactions. Access reviews verify that prior decisions remain justified.

Authentication establishes confidence in the actor's identity. Authorization determines the **extent of organizational authority that identity may exercise**. Effective authorization governance ensures that this authority remains proportionate to legitimate need, is understandable to accountable managers, and is subject to ongoing review throughout the identity lifecycle.

## Segregation of Duties: IAM as Fraud Prevention and Control Assurance

Segregation of duties (SoD) is one of the clearest points where cybersecurity governance intersects with financial control, internal audit, compliance, fraud prevention, and organizational accountability. In many high-impact processes, the risk is not only that an external attacker may compromise a system. It is also that a single person—or a single compromised identity—may initiate, approve, execute, conceal, or reverse a consequential action without adequate independent oversight.

This matters in processes such as payments, procurement, payroll, supplier management, financial reporting, identity administration, cybersecurity configuration, cloud administration, privileged access, sensitive data management, and production change control. When a single identity can perform incompatible activities within the same process, the organization creates an opportunity for fraud, error, insider misuse, unauthorized changes, or concealment of inappropriate activity.

Segregation of duties is therefore not merely an accounting or financial control principle. It is an important **cybersecurity governance control because it limits the concentration of digital authority**.

The underlying principle is straightforward: no individual identity should hold a combination of permissions that allows it to complete a high-risk process from beginning to end without meaningful challenge or oversight. Individual permissions may each be legitimate when considered separately, yet become dangerous when combined. The governance problem therefore lies not only in determining whether each entitlement is appropriate, but also in determining whether the **combination of entitlements creates excessive authority**.

IAM must be designed to make segregation-of-duties requirements enforceable in practice. This requires the organization to identify incompatible roles and permissions, define conflict rules, prevent inappropriate combinations where possible, detect violations when they occur, and establish controlled exception processes where complete separation cannot reasonably be achieved.

For example, a person responsible for creating a new supplier should not ordinarily be able to approve that supplier, initiate payments to it, and independently reconcile those payments. Each activity may be a legitimate organizational function, but concentrating them in a single identity removes independent oversight and creates an opportunity for fraud or concealment.

The same principle applies to cybersecurity administration. An administrator who can create privileged identities, assign elevated permissions, approve their use, disable monitoring, modify audit logs, and independently review the resulting evidence holds excessive authority. If that identity is misused or compromised, the same access that enables the harmful action may also allow the actor to conceal it.

In mature organizations, segregation-of-duties rules are embedded directly within identity-governance processes. Access requests can be evaluated against predefined conflict rules before permissions are granted. Organizations can escalate potential conflicts for review rather than discovering them months later during an audit. High-risk exceptions may require explicit business justification, a defined duration, compensating controls, and approval from an accountable authority.

These controls may be both **preventive and detective**. Preventive controls aim to prevent conflicting access from being assigned in the first place. Detective controls identify inappropriate combinations that already exist, perhaps because roles changed, permissions accumulated over time, applications were modified, or emergency access was granted. Both are necessary because authorization environments are dynamic, and a configuration that was appropriate at the time of approval may later create a conflict.

Periodic access reviews should therefore assess not only whether individual permissions remain necessary but also whether the combination of permissions assigned to an identity creates inappropriate concentrations of authority. This is especially important for privileged users, financial processes, production systems, security administration, and identities that span multiple applications or business processes.

When structural separation is impractical, **dual control, additional approval, enhanced monitoring, or other compensating controls** may be necessary. This can arise in small organizations, specialized technical teams, emergency situations, privileged administration, production changes, financial approvals, or sensitive operational activities where staffing or system limitations make full separation difficult.

Dual control generally requires more than one authorized person to participate or approve a high-impact action before it is completed. For example, one administrator may prepare a critical production change, while another independently approves or executes it. A financial transaction above a defined threshold may require two authorized approvers. Emergency privileged access may require subsequent independent review.

Compensating controls do not eliminate the underlying SoD conflict. Rather, they reduce or monitor the associated risk when complete separation is not reasonably achievable. For that reason, the organization should govern compensating arrangements. The organization should document why the conflict exists, who owns the resulting risk, which additional safeguards apply, how long the exception will remain valid, and when it will be reassessed.

Segregation of duties also relates to the **fraud triangle** introduced earlier in this book. One of the three conditions associated with fraud is opportunity. Excessive access, weak oversight, and concentrated decision authority create such opportunities. SoD reduces that opportunity by requiring collaboration, independent approval, or separate execution of sensitive activities.

The same mechanism also constrains external attackers. If an adversary compromises one identity, that identity's permissions determine what the attacker can accomplish. An account that can initiate but not approve a transaction, administer one component but not disable its monitoring, or access information but not alter authorization policies presents a smaller attack surface than an identity with unrestricted end-to-end authority.

Segregation of duties therefore helps limit the **blast radius of an identity compromise**. A compromised account is dangerous; a compromised account with multiple conflicting permissions and no independent oversight is substantially more dangerous.

For managers, SoD also provides important assurance. A control environment is more credible when high-risk actions require identifiable decision points, independent participation, and evidence of approval. Audit trails should demonstrate not only that an activity occurred but also that the required separation was observed or that an approved exception and a compensating control were in place.

Organizations should consequently be able to identify significant SoD conflicts, explain which roles and permissions create them, demonstrate how they are prevented or detected, identify approved exceptions, and show how those exceptions are monitored and reassessed periodically.

The managerial lesson is that access rights must be evaluated collectively**, not one entitlement at a time**. A permission that seems reasonable in isolation may become unacceptable when combined with another permission that allows the same identity to approve, execute, conceal, or reverse a sensitive action.

IAM and PAM programs must therefore make segregation of duties **visible, enforceable, reviewable, and verifiable**. By limiting concentrations of digital authority, SoD strengthens fraud prevention, cybersecurity resilience, internal control, accountability, and assurance.

## Third-Party and Partner Access Governance: The Extended Identity Problem

<span id="_Toc215626101" class="anchor"></span>Organizations increasingly rely on third parties, including contractors, managed service providers, cloud consultants, software vendors, outsourcing partners, auditors, implementation partners, and business-to-business integrations. These relationships create identity dependencies essential to modern operations, yet they are often more difficult to govern than employee access.

A third-party identity can become a significant source of exposure when access is persistent, excessive, poorly monitored, or no longer justified. A vendor account with long-standing privileged access can provide an unintended pathway into critical systems. A contractor whose access remains active after an engagement ends is a predictable lifecycle failure. A partner integration with broader permissions than required may expose data, applications, or infrastructure beyond the intended business purpose.

Third-party identity should therefore not be treated as a secondary access-management issue. It is an **enterprise risk and governance issue**. External identities may access internal systems, cloud consoles, source-code repositories, ticketing environments, customer information, financial platforms, production systems, remote-access tools, security platforms, or administrative interfaces. In some cases, third-party personnel or services may exercise substantial operational authority while facing weaker onboarding, less frequent access reviews, different employment controls, or unclear internal ownership.

The fundamental governance principle is that external access must be owned internally.

Each significant third-party account, federated identity, service relationship, or integration should have an identifiable internal sponsor or business owner. That owner should be accountable for the business justification, the scope of access, the sensitivity of the resources involved, the duration of the relationship, and periodic confirmation that continued access remains necessary.

Without explicit ownership, third-party access can persist long after the circumstances that originally justified it have changed. Projects end, supplier personnel change, consultants complete assignments, contracts expire, services are replaced, and responsibilities shift between organizations. If no internal party is accountable for reassessing the associated access, the identity may remain active simply because no one is responsible for removing it.

Third-party access should therefore be **purpose-specific and time-limited wherever practicable**. Access should be granted for a defined business purpose and an appropriate period, rather than indefinitely by default. Renewal should require confirmation that the underlying business need persists.

Contract dates, project milestones, statement-of-work periods, supplier personnel changes, and vendor offboarding processes should be linked to identity lifecycle controls. Where technically feasible, expiration dates should be embedded directly in access provisioning to prevent temporary access from silently becoming permanent. Long-lived external access may sometimes be justified, but it should be deliberately governed rather than assumed.

The principle of **least privilege** applies to third parties as rigorously as to employees. Access should align with the specific services the external party is expected to perform. Do not grant broad administrator permissions merely because they simplify support or reduce the number of access requests.

When a supplier requires privileged access, the organization should apply the same or stronger PAM controls as those used for internal administrators. Depending on risk, these may include just-in-time privilege elevation, separate administrative identities, approval workflows, restricted administrative pathways, phishing-resistant authentication, credential vaulting, limited session duration, activity monitoring, and privileged-session recording.

This is particularly important for managed service providers and technology vendors whose personnel may administer infrastructure for multiple customers. The organization should understand whether access is assigned to identifiable individuals, how the supplier manages its privileged identities, how authentication is performed, whether shared accounts exist, and how quickly access can be revoked when supplier personnel change.

Monitoring should also be **proportionate to the risk posed by the external relationship**. Higher-risk third-party access may warrant stronger controls, such as hardened access points, managed devices, network or application restrictions, ticket-based approval, session recording, detailed activity logging, enhanced alerting, or retrospective review of privileged actions.

External access to production environments, cloud administration, identity platforms, sensitive information, cybersecurity tooling, backup environments, financial applications, or critical operational systems should never be invisible to the organization’s monitoring and assurance processes. The organization should be able to determine not only that a supplier had access but also when that access was exercised and which significant actions were performed.

Third-party identity governance should also cover **federated and non-human access**. Business partners may authenticate via their own identity providers rather than through locally managed accounts. Applications may exchange data using API credentials, tokens, certificates, service principals, or other machine identities. These arrangements can reduce administrative effort, but they also create dependencies on external identity and security controls.

Federation does not eliminate the need for governance. The organization must understand which attributes and authentication assertions it trusts, what happens when a partner employee leaves, how access is revoked, and what assurance there is that the partner's identity processes remain reliable. Similarly, machine-to-machine integrations should have defined ownership, limited permissions, appropriate credential protection, rotation or expiration requirements, and decommissioning procedures.

Third-party identity governance must therefore be integrated with **procurement, contracting, third-party risk management, and supplier offboarding**. Access conditions should be evaluated before a supplier is granted technical connectivity, not after access has already been granted.

Contracts and related agreements should, where appropriate, define expectations for authentication, privileged access, access approval, logging, data handling, incident notification, personnel changes, credential management, access termination, and cooperation with audit or assurance activities. Higher-risk relationships may also warrant requirements for subcontractors or other parties through which the supplier delivers its services.

This integration is important because identity governance cannot indefinitely compensate for weak commercial arrangements. For example, if a supplier's contractual obligations do not require timely notification of changes to privileged personnel, the organization may have difficulty determining when access should be revoked. Procurement and contracting therefore set the conditions under which effective IAM and PAM controls can operate.

Third-party offboarding warrants particular attention. Ending a commercial or contractual relationship should trigger a coordinated review of accounts, federated identities, administrative roles, remote-access permissions, API credentials, certificates, tokens, shared secrets, devices, delegated access, and other trust relationships associated with that supplier. Disabling a single user account is rarely sufficient when the relationship has created multiple technical access paths.

The crucial governance principle is that **delegating access does not transfer accountability**. An organization may allow a vendor to administer infrastructure, operate a platform, manage applications, or provide cybersecurity services, but it remains accountable for the risks arising from the authority it delegates.

This principle is especially important when third parties perform functions the organization itself cannot easily observe. Outsourcing an activity may transfer operational responsibility, but it does not eliminate the need for governance, oversight, evidence, and risk ownership.

A mature organization should be able to consistently answer a set of questions about external identities: Which third parties have access? What business purpose justifies that access? Who internally owns the relationship? What systems, data, and functions can they access? Which permissions are privileged? How is authentication performed? What conditions constrain their access? When will it expire or be reviewed? How will access be revoked if the relationship changes? How is significant activity monitored and documented?

Third-party access is ultimately an extension of the organization’s identity environment. **The identity may belong to someone outside the organization, but the access rights fall within the organization’s risk boundary.** Effective governance therefore requires that third-party identities, privileged vendor access, federated trust, and machine-to-machine integrations be incorporated into the organization’s IAM, PAM, monitoring, assurance, and risk-management processes rather than managed as exceptions at the edge of those processes.

## PAM: The Crown Jewel Controls Managers Must Demand

Privileged access is qualitatively different from ordinary access. Ordinary access allows users and systems to perform approved work within established boundaries. Privileged access allows users, administrators, services, and automation tools to **change those boundaries**. It can create or disable accounts, modify configurations, assign permissions, change security settings, access sensitive information at scale, alter or delete logs, manage encryption keys, deploy software, modify infrastructure, recover systems, or interrupt operations.

For this reason, managers should assume that privileged identities and administrative pathways will be attractive targets for attackers and that attempts will eventually be made to obtain, misuse, or abuse them. The governance objective is not to assume that privileged compromise can always be prevented, but to make privilege difficult to obtain, tightly constrained when granted, observable during use, and rapidly revocable when no longer justified.

**Privileged Access Management (PAM)** makes privileged access harder to obtain and misuse, easier to monitor, and easier to revoke. PAM should not be treated as an optional enhancement for technical teams. It is a **crown-jewel control domain** because compromising privileged access can undermine many of the safeguards on which the organization depends.

A firewall, endpoint security platform, backup system, cloud environment, logging platform, identity provider, encryption infrastructure, or security monitoring system is only as trustworthy as the privileged-access model that secures its administration. If an attacker gains sufficient administrative authority, they may be able to modify or disable the controls intended to detect or contain them.

### Separate Privileged and Ordinary Identities

A baseline PAM design begins with strong separation between ordinary user activity and privileged administration. Administrators should not routinely browse the web, read email, open collaboration messages, or perform other everyday activities using identities that also possess administrative authority.

Separate administrative identities reduce the likelihood that compromise of routine user activity immediately produces privileged access. An administrator may therefore have one identity for everyday work and a separate privileged identity used only when administrative authority is required.

This separation is more than an account-management convention. It creates distinct trust boundaries, permits stronger authentication and monitoring for privileged activity, and allows organizations to apply more restrictive conditions to administrative identities than would be practical for routine user access.

### Reduce Standing Privilege

Mature PAM also seeks to minimize **standing privilege**. Users should not retain permanent administrative authority simply because they occasionally require it. The longer privileged access remains continuously available, the larger the opportunity window for misuse or compromise.

Where technically and operationally feasible, privileged access should instead be granted through controlled elevation mechanisms. **Just-in-time (JIT) access** is an important pattern: a user operates with ordinary permissions by default, requests elevated access when required, receives it for a defined and approved purpose, and automatically loses that privilege when the task, session, or authorized time period ends.

A related principle is **just-enough privilege**: the elevated identity should receive only the administrative capabilities necessary for the specific task rather than broad unrestricted authority.

Together, these approaches reduce the number of identities possessing permanent administrative rights, narrow the period during which privileged authority can be exploited, and reduce the value of stealing a credential that does not itself provide continuous administrative access.

### Protect Privileged Credentials and Secrets

Credential vaulting and secrets management are also central PAM capabilities. Privileged passwords, local administrator credentials, service-account passwords, API keys, certificates, tokens, cryptographic secrets, and automation credentials should not be stored in spreadsheets, scripts, shared folders, source-code repositories, personal notes, unsecured configuration files, or other unmanaged locations.

Privileged secrets should instead be stored and managed through controlled systems capable of enforcing appropriate access restrictions and, where applicable, rotation, expiration, approval, logging, and recovery procedures.

The governance objective is to reduce the number of people and systems that directly know or possess reusable privileged credentials. In a mature design, users may sometimes obtain access to a privileged function without ever seeing the underlying password or secret. The PAM platform can broker access, inject credentials into an approved session, rotate credentials after use, or provide temporary secrets that expire automatically.

This distinction matters because **privileged authority should be governed independently from possession of a reusable credential**. If administrative capability can be granted temporarily without distributing long-lived secrets, the organization reduces opportunities for credential theft, copying, reuse, and uncontrolled sharing.

### Monitor Privileged Sessions and Actions

Privileged-session management strengthens accountability. High-risk administrative sessions may be proxied, brokered, monitored, or recorded when justified by the level of risk. The objective is not surveillance for its own sake, but to create reliable evidence, deter misuse, support incident investigation, and establish accountability for actions performed through powerful identities.

Monitoring should focus not merely on whether a privileged account was used, but on **which person or process initiated the activity, why the access was approved, when it began, which resources were accessed, what significant actions were performed, and when the privilege ended**.

Shared administrative accounts are particularly problematic because they weaken attribution. If several people know the same administrator password, subsequent logs may show what the account did without showing which individual was responsible. Where shared accounts cannot be eliminated, PAM should provide mechanisms that associate each use with a named individual, authentication event, approval record, session, and business purpose.

This is a fundamental assurance requirement. Privileged activity should be attributable to an accountable human identity or a clearly governed non-human process.

### Control Administrative Pathways

PAM governance must also address **where privileged actions can originate and how administrative sessions reach sensitive systems**. Strong credential controls can be undermined if privileged access is permitted from ordinary workstations, unmanaged devices, broad networks, insecure remote-access services, or untrusted locations.

A mature design therefore constrains administrative pathways according to risk. Controls may include hardened or privileged administrative workstations, dedicated management environments, segmented administrative networks, private management endpoints, device-compliance requirements, conditional access, phishing-resistant authentication, session brokering, network restrictions, and stronger controls for remote administration.

The principle is that privileged access should not merely require a privileged identity; it should occur through a **trusted administrative path**. This reduces the likelihood that malware on a routine endpoint, stolen browser sessions, compromised personal devices, or insecure networks become pathways to administrative systems.

### Govern Non-Human Privilege

PAM must also encompass **non-human identities**. Service accounts, automation platforms, deployment pipelines, scripts, API credentials, cloud roles, workload identities, certificates, tokens, and machine-to-machine services may possess administrative or high-impact permissions even though no human user interactively logs in with them.

These identities can create significant governance challenges because they are often long-lived, deeply integrated into applications, difficult to rotate, and poorly understood outside the technical teams that created them. They may also accumulate excessive privileges because reducing permissions can create fear of disrupting automated processes.

A mature PAM program should therefore assign ownership and business or technical purpose to significant non-human privileged identities. Their permissions should follow least-privilege principles, credentials or secrets should be protected appropriately, and access should be reviewed, monitored, rotated, expired, or decommissioned where relevant.

The objective is to avoid a situation in which human administrator accounts are tightly governed while powerful machine identities remain effectively invisible.

### Emergency and Break-Glass Access

PAM governance must also account for exceptional circumstances. Organizations may require **break-glass or emergency access** when ordinary identity systems, authentication services, network paths, or approval workflows are unavailable.

Emergency access should not become a hidden permanent bypass around normal controls. Break-glass identities should be tightly restricted, protected with strong authentication and credential-management measures, used only for defined emergency purposes, and monitored closely. Their use should trigger notification and subsequent review, and credentials should be changed or resecured after use where appropriate.

Managers should view emergency access as a resilience mechanism rather than an exception from governance. The organization must be capable of recovering critical systems when normal controls fail, but it must do so without creating uncontrolled administrative pathways that persist indefinitely.

### PAM as a Governance and Assurance Mechanism

Managers should be able to obtain credible answers to several fundamental questions: Who possesses privileged authority? Which systems and data can they control? Why is that privilege required? Is it permanent or temporary? Who approved it? Through which administrative pathways can it be exercised? How are privileged secrets protected? How is significant activity monitored? How are exceptions handled? How does emergency access work? When was privileged access last reviewed? How quickly can it be revoked?

These questions move PAM beyond product deployment. An organization may own a privileged-access platform and still have weak PAM governance if administrator accounts bypass it, service accounts remain unmanaged, passwords are shared informally, permanent cloud roles proliferate, or session monitoring is rarely reviewed.

The managerial lesson is that privileged authority must be **minimized, justified, separated, time-bounded where practicable, strongly authenticated, exercised through controlled pathways, monitored, attributable, reviewable, and evidenced**.

PAM is therefore not merely a technical mechanism for protecting administrator passwords. It is an architectural and governance safeguard designed to prevent an ordinary compromise from becoming a catastrophic one. Effective PAM constrains the identities and mechanisms capable of changing the organization's control environment itself—and provides management with evidence that this exceptional digital authority remains under control.

## Break-Glass Accounts: Emergency Access Without Permanent Weakness

Emergency access, commonly referred to as **break-glass access**, is necessary because normal identity and administrative pathways can fail. Identity providers may become unavailable, federation services may be disrupted, administrators may be locked out, authentication mechanisms may fail, or urgent recovery actions may be required during a cybersecurity incident. A well-governed organization must therefore preserve the ability to regain control of critical systems when ordinary access mechanisms are unavailable.

The governance challenge is that the very independence that makes emergency access useful can also make it dangerous. Break-glass identities that bypass normal workflows, dependencies, or approval mechanisms can become persistent backdoors if they are shared, weakly protected, poorly documented, rarely tested, or excluded from monitoring.

The objective is therefore to achieve two outcomes simultaneously: **emergency access must remain available when normal controls fail, and it must remain tightly controlled when no emergency exists**.

Break-glass access should be exceptional by design. Accounts and other emergency-access mechanisms should be few in number, explicitly justified, assigned to clearly defined systems or recovery purposes, and subject to documented governance. They should not become alternative administrator accounts used for convenience, routine troubleshooting, faster access, or avoidance of normal privileged-access workflows.

Their purpose is **exceptional authority in exceptional circumstances**.

### Preserve Independence Without Creating a Backdoor

Emergency access should be sufficiently independent from the controls whose failure it is intended to overcome. For example, a recovery mechanism that depends entirely on the same identity provider, authentication service, network path, or PAM platform that has failed may provide little resilience during an actual crisis.

At the same time, independence must not mean absence of control. Break-glass mechanisms still require strong protection, accountability, and oversight. Governance should define which normal dependencies may appropriately be bypassed during an emergency and which safeguards must remain in force.

This creates an important architectural principle: **break-glass access should bypass failed dependencies, not governance itself**.

### Protect Emergency Credentials

Break-glass credentials and authenticators should receive protection proportionate to the authority they provide. Depending on the environment, this may involve strong authentication, secure credential storage, controlled custody, restricted knowledge of credentials, physical protection of authenticators, or other mechanisms designed to prevent unauthorized use.

Where credentials are stored for emergency retrieval, access to them should itself be tightly controlled and evidenced. Organizations should avoid arrangements in which multiple administrators casually know a shared emergency password or where credentials are stored in easily accessible documents, scripts, personal password stores, or unsecured locations.

Where practical, access to emergency credentials may require participation by more than one authorized individual, particularly when the account provides extensive control over critical systems.

The objective is to ensure that emergency access remains **recoverable without becoming routinely accessible**.

### Define Explicit Activation Conditions

Management should define the circumstances under which break-glass access may legitimately be used. Examples may include failure of the normal identity infrastructure, loss of ordinary privileged-access capability, a critical incident requiring immediate intervention, or a recovery situation in which delay would create significant additional harm.

The authorization model should also specify who may initiate emergency access, who must be notified, what level of approval is required where circumstances permit, and what documentation must follow the event.

The process must remain practical during a real crisis. An emergency mechanism that requires unavailable personnel, inaccessible systems, or an overly complex approval chain may fail precisely when it is most needed. Governance must therefore balance control with operational resilience.

### Monitor Every Use

Break-glass accounts should be excluded from routine administrative use but **included in monitoring and alerting**. Their activation should be treated as an unusual and high-significance security event.

Where technically feasible, use of emergency access should generate immediate alerts to appropriate functions such as security operations, identity or PAM administrators, system owners, and relevant management. The organization should be able to determine who invoked the emergency mechanism, when access began, which system was accessed, why normal authentication or administration was unavailable, what significant actions were performed, and when emergency access ended.

Logging and evidence should be designed carefully because some emergencies may involve failure of the organization's normal monitoring infrastructure. Critical emergency-access evidence may therefore require collection or protection mechanisms that do not depend entirely on the affected system.

The objective is that **emergency authority should never become invisible authority**.

### Test Break-Glass Access Before It Is Needed

Emergency access should be tested periodically through controlled exercises. A break-glass mechanism that exists only in documentation may not work when an actual crisis occurs.

Testing can verify that credentials remain usable, authenticators function correctly, designated personnel understand the procedure, required systems can be reached, monitoring detects the event, alerts reach the appropriate recipients, and recovery actions can be performed without unsafe improvisation.

Testing should also verify that emergency access has not quietly lost its independence. Changes to identity infrastructure, cloud configurations, networks, authentication technologies, or security policies can unintentionally make a previously functional break-glass process dependent on systems it was intended to bypass.

Conversely, testing may reveal that an emergency identity has acquired unnecessary permissions or that too many individuals can obtain access.

Break-glass testing is therefore both a **resilience test and a control-assurance test**.

### Review Every Activation

Use of break-glass access should trigger a formal post-use review. This does not imply that emergency use is inherently suspicious or that every activation constitutes a security incident. Rather, emergency access represents an intentional departure from normal administrative controls and therefore warrants heightened evidence and accountability.

After use, the organization should confirm why emergency access was required, whether its use was authorized and proportionate, which actions were performed, whether any unintended or unauthorized changes occurred, and whether normal privileged-access mechanisms have been restored.

Credentials, authentication factors, tokens, or other emergency secrets should be rotated, reset, or otherwise resecured where appropriate. Temporary changes made during the emergency should be reviewed, and any elevated permissions that are no longer required should be removed.

The post-use process should also identify whether the emergency exposed weaknesses in normal identity, PAM, recovery, or operational procedures. If administrators were forced to improvise because documented recovery processes were inadequate, the organization should treat that as a governance lesson rather than simply closing the emergency-access record.

### Govern Ownership and Periodic Review

Break-glass mechanisms require clear ownership. For each significant emergency identity or recovery pathway, management should be able to identify the system it protects, the business or technical owner, who may invoke it, how its credentials are protected, what permissions it possesses, how activation is detected, and when it was last tested.

Periodic review should confirm that the account is still required, that its permissions remain appropriate, that authorized custodians remain valid, that monitoring remains effective, and that documentation reflects the current environment.

Emergency accounts should not survive indefinitely merely because they are rarely used. An obsolete break-glass identity may become particularly dangerous because administrators may assume that no one uses it while attackers specifically seek neglected administrative pathways.

### The Managerial Objective

The managerial question is straightforward: **can the organization regain administrative control during a crisis without creating a permanent weakness during normal operations?**

A mature PAM program should be able to answer yes. Emergency access should be:

- available when normal administrative mechanisms fail;

- independent of the specific dependencies it is designed to recover from;

- strongly protected while dormant;

- limited to clearly defined emergency purposes;

- attributable to authorized individuals;

- monitored and alerted whenever activated;

- periodically tested;

- reviewed after every use; and

- resecured immediately once the emergency ends.

Break-glass access is therefore not a bypass around governance. It is a **governed resilience mechanism**. Properly designed, it allows the organization to recover control when ordinary IAM and PAM mechanisms fail while ensuring that the exceptional authority created for emergencies does not become a permanent path around the very controls it is intended to protect.

## Non-Human Identities: Service Accounts, API Keys, Tokens, and Machine-to-Machine Trust

<span id="_Toc215808028" class="anchor"></span>A major blind spot in many modern IAM and PAM programs is the governance of **non-human identities**. Significant compromises do not always begin with an employee or administrator account. They may begin with a leaked API key, a long-lived authentication token, an overprivileged service account, an expired or poorly managed certificate, a secret embedded in source code, or credentials exposed through a development or CI/CD pipeline.

These identities do not represent individual employees, but they can possess substantial authority over systems, applications, cloud resources, data, deployment processes, security controls, and administrative functions. From a governance perspective, the absence of a human user does not reduce the importance of the access. In some cases, machine identities possess broader, more persistent, and less visible privileges than human users.

Managers must therefore recognize a fundamental principle: **workloads are identities, and machine-to-machine trust is an access-control relationship**.

Applications authenticate to databases. Microservices call other microservices. Automation scripts modify systems. Infrastructure-as-code platforms create and configure cloud resources. Monitoring systems collect telemetry across the environment. Backup platforms access large volumes of organizational data. CI/CD pipelines build, test, sign, and deploy software. Security tools may quarantine endpoints, block accounts, or modify network controls. Each of these activities requires some mechanism through which one system is trusted to act upon another.

That trust may be expressed through service accounts, API keys, tokens, certificates, secrets, workload identities, cloud roles, service principals, or other authentication mechanisms. If these relationships are not governed, the organization may accumulate powerful access pathways that exist outside traditional employee-focused IAM processes and remain largely invisible to managers and access reviewers.

### Inventory and Ownership

Effective governance begins with visibility. An organization cannot govern machine identities that it does not know exist.

Significant non-human identities should therefore be inventoried and associated with identifiable ownership. Management should be able to determine which application, workload, integration, platform, or business process depends on each identity and who is accountable for its continued use.

Ownership is especially important because machine identities do not naturally participate in human lifecycle processes. They do not resign, change departments, or appear on termination reports. A service account created for a project may continue operating long after the project has ended. An API key may remain active after the application that used it has been replaced. A certificate may continue to authorize a system no one remembers owning.

Without explicit ownership, obsolete machine identities can persist indefinitely.

Each significant non-human identity should therefore have a defined owner, legitimate purpose, authorized scope, expected systems or resources, appropriate privilege level, credential-management requirements, monitoring expectations, and retirement criteria.

### Apply Least Privilege to Machines

The principle of **least privilege** applies equally to machine identities. An application should receive only the permissions required to perform its intended function. A reporting service that needs to read a limited data set should not have administrative database permissions. A deployment pipeline that publishes one application should not automatically possess broad control over unrelated production environments.

This is particularly important because machine identities often operate continuously and automatically. An excessive permission assigned to a human account may be exercised occasionally. An overprivileged automated process may operate thousands of times per day and may be trusted implicitly by multiple systems.

Machine identities should therefore be evaluated not only according to what they currently do, but according to **what their permissions would allow them to do if compromised**.

High-impact machine identities should be classified as privileged identities and governed accordingly. Examples may include identities capable of deploying production software, modifying cloud infrastructure, administering databases, managing backups, changing identity configurations, accessing sensitive information at scale, altering security controls, or issuing credentials to other systems.

These identities belong within the PAM governance model even though no human administrator interacts with them directly.

### Reduce Dependence on Long-Lived Secrets

Secrets management is central to non-human identity governance. Static credentials are attractive to attackers because they can often be copied and reused without triggering the same behavioural indicators associated with compromised human accounts.

Passwords, API keys, private keys, tokens, and other sensitive machine credentials should not be embedded directly in source code, scripts, container images, configuration files, spreadsheets, shared folders, build logs, or other unmanaged locations.

Controlled vaults and secrets-management platforms can provide stronger protection by supporting access control, logging, rotation, versioning, expiration, policy enforcement, and integration with application and deployment workflows.

Where technically feasible, organizations should reduce their dependence on long-lived reusable secrets altogether. Alternatives may include **short-lived credentials, managed identities, workload identity federation, certificate-based authentication, dynamically issued tokens, or other mechanisms that allow systems to authenticate without permanently storing a reusable shared secret**.

This represents an important governance improvement because protecting a secret indefinitely is inherently difficult. Reducing the lifetime and reuse of credentials limits the period during which a stolen credential remains valuable.

### Govern the Machine Identity Lifecycle

Non-human identities require a lifecycle just as human identities do.

They should be deliberately created, assigned to an owner, configured with appropriate permissions, periodically reviewed, modified when requirements change, and retired when the underlying system or business purpose ends.

Lifecycle triggers may include application retirement, project completion, supplier termination, infrastructure replacement, changes to cloud architecture, migration to a new platform, expiration of a certificate, replacement of an integration, or redesign of an automated process.

The governance question should be the same one applied elsewhere in IAM: **does this identity still have a legitimate reason to exist, and does it still require the authority it currently possesses?**

Dormant, unused, orphaned, or unexplained identities should therefore be investigated and, where appropriate, disabled or removed.

Rotation and expiration should also reflect the type of credential and risk involved. Long-lived credentials may occasionally be necessary because of legacy-system limitations, but they should be treated as controlled exceptions with documented justification, compensating safeguards, and periodic review.

### Software Supply-Chain and CI/CD Risk

Non-human identity governance also intersects directly with **software supply-chain security**.

Modern software-development and deployment environments depend heavily on automated trust relationships. Source-code repositories, build servers, artifact repositories, package managers, deployment pipelines, signing services, cloud platforms, infrastructure-as-code tools, and production environments may all exchange credentials or trust one another through machine identities.

A compromised CI/CD or deployment identity may therefore provide an attacker with capabilities far beyond ordinary system access. Depending on its permissions, the attacker may be able to alter source code, introduce malicious dependencies, manipulate build processes, modify deployment artifacts, change infrastructure, obtain additional credentials, weaken security controls, or deploy unauthorized software into production.

The risk is amplified when automated identities are highly trusted because their activities may be assumed to originate from legitimate development or operational processes.

Machine identities associated with development and deployment pipelines should consequently be treated according to the authority they can exercise rather than simply classified as technical service accounts.

### Cloud and Automation Identities

Cloud environments further increase the importance of machine identity governance because applications and workloads routinely obtain permissions to interact with infrastructure and platform services.

An automation identity may be authorized to create virtual resources, modify storage permissions, alter network configurations, manage encryption services, create additional identities, access databases, or change security policies. If compromised, that identity may provide an attacker with an opportunity to establish persistence, expose information, disable monitoring, or escalate privileges across interconnected services.

Cloud roles and workload identities should therefore have clearly defined trust relationships. Management should understand not only which permissions an identity possesses, but also **which systems are permitted to assume or invoke that identity**.

This is particularly important where trust policies allow one workload, cloud account, tenant, pipeline, or external service to obtain another identity's privileges.

### Monitor Machine Behaviour

Machine identities require monitoring appropriate to their expected behaviour. In some respects, they can be easier to monitor than humans because automated systems often perform relatively predictable activities.

A service account that normally accesses one database from a particular workload should warrant attention if it suddenly begins querying unrelated systems. An automation identity that normally performs deployments during controlled workflows should be investigated if it begins creating administrative users. A backup account that normally performs scheduled read operations should attract scrutiny if it begins deleting or modifying data.

Monitoring may therefore consider unexpected resources, unusual API operations, changes in access patterns, abnormal transaction volume, new execution locations, unexpected privilege escalation, dormant identity use, or activity outside the identity's intended purpose.

The objective is not simply to detect that a credential was used. It is to determine whether **the machine identity is behaving consistently with the purpose for which its authority was granted**.

High-risk machine identities should receive monitoring comparable to other privileged identities, including appropriate logging, alerting, behavioural analysis, and investigation of anomalous activity.

### Avoid Anonymous Machine Authority

Accountability is more difficult when machine identities are shared across unrelated applications or processes. A single generic service account used by multiple systems may make administration easier, but it weakens attribution and expands the impact of credential compromise.

Where practicable, distinct workloads and applications should use distinct identities. This allows permissions to be tailored more precisely, credentials to be rotated or revoked independently, and activity to be attributed to the specific system or process responsible.

The governance objective is to avoid creating **anonymous machine authority** in which powerful credentials are broadly shared but no one can determine which application used them or why.

### Machine Identity as Delegated Digital Authority

The managerial lesson is that identity governance cannot stop with employees, administrators, contractors, and partners. Modern organizations increasingly delegate operational authority to software.

Applications make transactions. Automation modifies infrastructure. Pipelines deploy production systems. Security tools take containment actions. Backup platforms access critical information. APIs transfer organizational data. These are exercises of digital authority even when no human user is directly involved at the moment an action occurs.

Non-human identities must therefore be **inventoried, owned, classified, minimized, protected, monitored, reviewed, rotated or renewed where appropriate, and ultimately retired**.

Managers should be able to ask the same fundamental questions of a machine identity that they ask of a human one: What is this identity? Who owns it? Why does it exist? What can it access? How much authority does it possess? How does it authenticate? Where are its credentials protected? When was its access last reviewed? How would compromise be detected? How can it be revoked? What will cause it to be retired?

Without this discipline, an organization may have sophisticated controls over employees and administrators while some of its most consequential access paths remain unmanaged. Non-human identity governance therefore not simply secrets management. It is the governance of organizational authority exercised by software, workloads, devices, and automated processes.

## Auditability and Evidence: Turning Access Control into Credible Assurance

One of the most important governance lessons for managers is that **access control without evidence is difficult to distinguish from an assumption**. An organization may believe that multi-factor authentication is enforced, terminated users are deprovisioned promptly, privileged access is monitored, segregation-of-duties conflicts are controlled, or periodic access reviews are completed. Unless it can produce reliable evidence that these controls are operating as intended, however, such claims remain difficult to verify and defend.

Evidence creates the difference between **“we believe the control operates” and “we can demonstrate that it operates.”**

Auditability in IAM and PAM therefore requires both **decision traceability and activity traceability**. The organization should be able to reconstruct how access was granted, modified, exercised, reviewed, and removed. This does not mean that every routine action requires elaborate documentation. It means that access decisions and high-impact identity activity should leave sufficient evidence to support accountability, assurance, and investigation.

At a minimum, an organization should be capable of producing evidence concerning authentication requirements, MFA deployment and enforcement, assigned roles and permissions, access requests and approvals, changes to access, privileged logins, significant privileged actions, periodic access-review results, identity lifecycle events, exception approvals, and formal acceptance of residual access risk where applicable.

The level of evidence should be **proportionate to the authority and risk involved**. Routine low-risk access may require relatively simple records. Privileged or high-impact access requires stronger evidence because those identities may be able to modify systems, change permissions, alter configurations, disable controls, access sensitive information at scale, manipulate logs, or affect the access rights of other users.

### Evidence of the Identity Lifecycle

Lifecycle evidence is particularly important because access changes continuously as organizational relationships change.

For human identities, the organization should be able to reconstruct when an individual joined, what access was initially granted, which role or business requirement justified it, who approved the access, and when it became effective. When responsibilities change, evidence should show which permissions were added, which were removed, who authorized the changes, and whether any segregation-of-duties or privileged-access implications were identified.

When an individual leaves, management should be able to determine when the termination or departure became effective and when relevant access was disabled or revoked. For higher-risk identities, evidence may also be required for session termination, remote-access revocation, privileged-account disablement, delegated permissions, and other access pathways associated with the individual.

This allows management to answer an important assurance question: **did digital authority change when organizational authority changed?**

For non-human identities, lifecycle evidence requires a different but comparable structure. The organization should be able to determine who owns the service account, application identity, workload identity, API credential, certificate, token, or other machine identity; why it exists; what permissions it possesses; what systems depend on it; when credentials were issued or changed; whether the identity remains active; when its access was last reviewed; and how it will eventually be decommissioned.

Without this evidence, machine identities can remain active for years without anyone being able to demonstrate why they still exist or whether their authority remains appropriate.

### Evidence of Privileged Authority

PAM requires particularly strong evidence because privileged access represents exceptional digital authority.

Evidence may include privileged-access requests, approval records, business justification, privilege-elevation events, just-in-time access windows, credential retrieval or use, session initiation and termination, administrative activity, privileged-session recordings where justified, emergency-access activation, break-glass reviews, and actions taken after privileged use.

The purpose is not to create documentation for its own sake. The purpose is to make consequential actions **reconstructable and attributable**.

Management should be able to determine not merely that an administrator account was used, but which person or governed process exercised that authority, why the access was required, whether it was properly approved, what significant actions were performed, and whether the resulting activity remained within the authorized purpose.

This is especially important when shared or technical privileged identities cannot be eliminated. The underlying system may record that a generic administrative account performed an action, but PAM controls should, where possible, provide the additional evidence required to associate that use with a named individual, approved workflow, session, or automated process.

### Evidence Must Be Reliable

The existence of a record does not automatically make it reliable evidence.

IAM and PAM evidence should therefore be subject to controls that protect its **integrity, availability, and appropriate retention**. Logs and approval records that can be altered or deleted by the same privileged identities whose actions they are intended to document provide limited assurance.

High-value evidence should be protected against unauthorized alteration and deletion. Access to audit records should itself be restricted, and particularly sensitive environments may require separation between those who administer systems and those who control or review audit evidence.

Retention requirements should also reflect operational, investigative, legal, contractual, privacy, and regulatory needs. Retaining identity data indefinitely is not necessarily appropriate, but retaining it for too short a period may make investigations, audits, or compliance demonstrations impossible.

Evidence governance should therefore answer not only **what is logged**, but also who can modify the evidence, where it is stored, how long it is retained, and whether it will still be available when management, auditors, investigators, or regulators need it.

### IAM, PAM, and Security Operations

Auditability also creates an important connection between IAM, PAM, and **security operations**.

Identity evidence should not remain isolated within access-management platforms if it is relevant to detecting and investigating suspicious activity. Authentication logs, privilege-elevation events, administrative actions, account changes, token use, and anomalous machine-identity behaviour may need to be centralized or correlated with operational evidence from endpoints, cloud platforms, applications, networks, ticketing systems, change-management processes, and other security controls.

This broader context helps distinguish legitimate activity from misuse.

A privileged login outside an expected window may be harmless if it corresponds to an approved emergency change. The same event may be highly significant if no change record exists, the device is unfamiliar, and the identity subsequently modifies logging controls. Evidence becomes substantially more useful when events can be examined together rather than as isolated records.

Identity-related monitoring may therefore include events such as repeated failed authentication, unexpected MFA activity, suspicious account recovery, unusual privileged elevation, dormant-account use, access from unmanaged devices, changes to high-risk permissions, abnormal API activity, unusual workload behaviour, emergency-account activation, or privileged activity outside approved conditions.

Not every anomaly represents an incident. Governance should ensure, however, that significant alerts have defined ownership, triage criteria, escalation procedures, and response expectations. **Evidence that is collected but never examined provides limited control value.**

### From Logging to Assurance

Managers do not need to understand every identity log field, authentication protocol, event identifier, or technical data format. They do need to insist that the organization can answer fundamental accountability questions.

For significant access, management should be able to determine:

- Who or what had access?

- Why was that access required?

- Who approved it?

- What permissions were provided?

- When did the access become effective?

- When and from where was it used?

- What significant actions were performed?

- Was the activity consistent with the identity's role or system purpose?

- Was the access periodically reviewed?

- Were inappropriate permissions removed?

- Were exceptions explicitly approved?

- Can the organization reconstruct what happened if the identity is later suspected of compromise or misuse?

These are governance questions, not logging questions.

An organization may collect billions of identity events and still have weak assurance if those records cannot answer these basic questions. Conversely, a well-designed evidence model focuses on information that supports accountability, risk management, control testing, investigation, and decision-making.

### Evidence and the Three Lines of Assurance

IAM and PAM evidence also supports different layers of organizational assurance.

Operational teams use evidence to administer access, investigate anomalies, and demonstrate that required procedures were completed. Risk, compliance, privacy, and security-governance functions can use the same evidence to monitor whether access remains within policy and risk expectations. Internal audit can independently test whether those controls are designed appropriately and operating effectively.

This distinction reinforces an important governance principle: **the function performing the control should not be the only source of assurance that the control works**.

Reliable IAM and PAM evidence allows other management and assurance functions to challenge, test, and verify access-control claims rather than relying entirely on interviews or representations from the teams that operate the systems.

### The Managerial Objective

Credible IAM and PAM assurance ultimately depends on the organization's ability to move from assertion to demonstration.

Without reliable evidence, audits can become interviews, investigations can become reconstruction exercises based on incomplete information, and compliance claims may depend largely on trust. Management may know what policies require without knowing whether those requirements are consistently enforced.

With reliable evidence, access control becomes governable. Management can identify excessive access, challenge exceptions, verify deprovisioning, investigate suspicious activity, evaluate privileged actions, measure control performance, and demonstrate to auditors and other stakeholders that identity-related controls operate as intended.

The managerial lesson is therefore broader than simply **“log everything.”** The objective is to produce **decision-useful, reliable, protected, and reviewable evidence** that demonstrates how digital authority is granted, exercised, monitored, changed, and withdrawn.

In this sense, auditability converts IAM and PAM from collections of technical controls into sources of **credible assurance**. It enables the organization not merely to state that access is controlled, but to demonstrate that control through evidence that can withstand management review, audit, investigation, and challenge.

## Measurement: IAM/PAM Metrics That Matter to Governance

IAM and PAM are governance domains in part because their effectiveness can be observed and measured. The most useful indicators are not those that simply count activity, such as the number of accounts created, password resets completed, or access tickets processed. Those measures may be useful for operational management, but they say relatively little about whether identity-related risk is being controlled.

**Governance-grade metrics should help management determine whether access exposure is increasing or decreasing, whether high-risk identities and privileges are adequately controlled, whether lifecycle processes are reliable, whether exceptions are being reduced, and whether sufficient evidence exists to demonstrate that access remains appropriate.**

Good IAM and PAM measurement therefore combines several types of indicators. **Coverage metrics** show whether important controls have been deployed across the relevant population. **Performance metrics** show whether those controls operate within expected standards. **Risk indicators** highlight conditions that may increase exposure, such as orphaned accounts or standing privilege. **Outcome and remediation metrics** show whether identified weaknesses are being corrected.

The following table provides examples of IAM and PAM metrics that can support executive oversight, risk reporting, audit preparation, and continuous improvement.

| **Governance area**                   | **Example metric**                                                                                                                                      | **Why it matters**                                                                                                                                              |
|---------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **MFA coverage**                      | Percentage of active workforce identities protected by MFA                                                                                              | Shows whether baseline authentication protection is broadly enforced.                                                                                           |
| **Phishing-resistant authentication** | Percentage of privileged and high-risk identities using approved phishing-resistant authentication                                                      | Indicates whether stronger authentication is concentrated on identities with the greatest potential impact.                                                     |
| **Conditional access**                | Percentage of sensitive applications and privileged access paths governed by approved conditional-access policies                                       | Shows whether access decisions incorporate relevant context such as device posture, identity risk, location, authentication strength, and resource sensitivity. |
| **Joiner provisioning**               | Percentage of new identities provisioned through approved role-based or standardized workflows                                                          | Indicates whether onboarding follows governed access models rather than informal or ad hoc requests.                                                            |
| **Mover remediation**                 | Percentage of significant role changes for which obsolete access is removed within the defined period                                                   | Measures whether privilege accumulation and access inherited from previous responsibilities are being controlled.                                               |
| **Leaver deprovisioning**             | Percentage of terminated identities disabled within the required service level                                                                          | Measures the reliability and timeliness of one of the most important identity lifecycle controls.                                                               |
| **Privileged access coverage**        | Percentage of identified privileged human accounts governed through approved PAM controls                                                               | Indicates whether privileged authority is governed rather than administered outside controlled processes.                                                       |
| **Standing privilege**                | Number or percentage of identities with permanent privileged access, including trend over time                                                          | Shows whether unnecessary persistent administrative authority is being reduced.                                                                                 |
| **Just-in-time privilege**            | Percentage of eligible privileged access granted through time-limited elevation                                                                         | Indicates whether administrative authority is temporary, justified, and automatically removed when no longer required.                                          |
| **Privileged-session accountability** | Percentage of high-risk privileged sessions attributable to a named user or governed process and appropriately logged, brokered, monitored, or recorded | Demonstrates whether consequential administrative activity can be reconstructed and reviewed.                                                                   |
| **Privileged-access review**          | Percentage of privileged identities reviewed within the required review period                                                                          | Shows whether exceptional authority is periodically reassessed.                                                                                                 |
| **Access-review completion**          | Percentage of required access certifications completed within the defined period                                                                        | Provides a basic measure of review-process execution.                                                                                                           |
| **Access-review remediation**         | Percentage of identified inappropriate access removed or otherwise resolved within the required period                                                  | Measures the effectiveness of access reviews rather than merely their completion.                                                                               |
| **Segregation-of-duties conflicts**   | Number of unresolved high-risk SoD conflicts and trend over time                                                                                        | Shows whether combinations of permissions are creating excessive concentrations of authority.                                                                   |
| **SoD exceptions**                    | Percentage of approved SoD exceptions with documented owner, compensating controls, review date, and expiry where applicable                            | Indicates whether unavoidable conflicts are governed rather than informally tolerated.                                                                          |
| **Break-glass readiness**             | Percentage of required break-glass mechanisms tested successfully during the review period                                                              | Confirms that emergency access is available and functional when needed.                                                                                         |
| **Break-glass use**                   | Number of break-glass activations and percentage subjected to post-use review                                                                           | Shows whether emergency access remains exceptional, observable, and accountable.                                                                                |
| **Dormant accounts**                  | Number or percentage of active accounts exceeding the approved inactivity threshold                                                                     | Identifies unnecessary identities that may increase attack surface.                                                                                             |
| **Orphaned accounts**                 | Number of accounts without a valid owner, employment relationship, contract, application, or business justification                                     | Highlights identities that may have escaped normal lifecycle governance.                                                                                        |
| **Third-party access**                | Percentage of active external identities with an assigned internal sponsor, documented purpose, and defined review or expiry date                       | Shows whether external access has internal ownership and remains tied to a legitimate relationship.                                                             |
| **Non-human identity ownership**      | Percentage of significant service accounts, workload identities, API credentials, certificates, tokens, and machine identities with assigned owners     | Indicates whether machine identities are visible and accountable.                                                                                               |
| **Non-human privilege**               | Number or percentage of machine identities classified as privileged and governed through appropriate PAM or equivalent controls                         | Shows whether powerful automated identities receive governance comparable to privileged human identities.                                                       |
| **Credential and secret hygiene**     | Percentage of in-scope machine credentials meeting defined rotation, expiration, or short-lived credential requirements                                 | Indicates whether dependence on long-lived or unmanaged machine secrets is being reduced.                                                                       |
| **Identity exceptions**               | Number of active IAM/PAM exceptions, percentage past review or expiry date, and trend over time                                                         | Shows whether temporary departures from policy are being governed or becoming permanent weaknesses.                                                             |
| **Identity-related security events**  | Number and trend of significant identity anomalies investigated within required service levels                                                          | Indicates whether suspicious authentication, privilege, or machine-identity activity is being detected and acted upon.                                          |
| **Evidence completeness**             | Percentage of sampled high-risk access decisions for which required approval, justification, review, and activity evidence can be produced              | Tests whether IAM and PAM controls are demonstrable rather than merely asserted.                                                                                |

### Metrics Must Be Interpreted, Not Merely Reported

IAM and PAM metrics should not be interpreted mechanically. A number is rarely meaningful without context, trend, scope, and an understanding of how the underlying process operates.

For example, a high number of access-review findings may indicate poor access governance. It may also indicate that a newly improved review process is finally identifying excessive permissions that previously remained hidden. In the short term, more findings may therefore represent stronger detection rather than deteriorating control.

Similarly, a reduction in privileged user accounts may indicate improved least privilege, but it may also conceal a transfer of privilege into unmanaged service accounts, cloud roles, automation identities, or shared credentials. A high MFA coverage percentage can create false confidence if the remaining unprotected identities include administrators or other high-impact users. A 99-percent compliance figure may therefore be less important than understanding **which one percent remains outside the control**.

Governance reporting should consequently include appropriate segmentation. Management may need to distinguish between ordinary users and administrators, employees and third parties, human and non-human identities, routine applications and critical systems, or low-risk and high-risk access. Enterprise-wide averages can otherwise hide significant concentrations of exposure.

### Trends Matter More Than Isolated Numbers

Many IAM and PAM measures are most useful when viewed as trends.

A single count of standing privileged accounts provides limited information. Management gains more value from understanding whether the number is increasing or decreasing, whether newly created privileged accounts are justified, and whether permanent privilege is gradually being replaced by controlled elevation.

The same applies to orphaned accounts, SoD conflicts, access-review findings, expired exceptions, non-human identities without owners, and delayed deprovisioning. The trend helps management determine whether control maturity is improving, remaining stable, or deteriorating.

Where appropriate, metrics should therefore include:

> **Current value → target → trend → threshold → accountable owner.**

For significant indicators, management should also define when deviation becomes sufficiently important to require investigation or escalation. A metric without an expected range or escalation threshold may inform management but does not necessarily support governance action.

### Leading and Lagging Indicators

A mature IAM/PAM measurement framework should contain both **leading and lagging indicators**.

Leading indicators describe conditions that influence future risk. Examples include MFA coverage, phishing-resistant authentication adoption, standing privilege, JIT utilization, overdue access reviews, orphaned identities, unresolved SoD conflicts, third-party accounts without sponsors, or machine identities without defined owners. These measures can reveal deteriorating control conditions before a material incident occurs.

Lagging indicators describe events or consequences that have already occurred. Examples may include confirmed identity compromise, misuse of privileged accounts, access-related audit findings, policy violations, unauthorized access incidents, or security events caused by delayed deprovisioning.

Leading indicators are particularly valuable for governance because they provide an opportunity to intervene before control weaknesses produce significant consequences. Lagging indicators remain important because they help management determine whether existing controls are preventing the outcomes they were designed to address.

### Measure Effectiveness, Not Administrative Activity

Managers should distinguish carefully between **activity metrics and effectiveness metrics**.

For example:

- “10,000 access reviews completed” measures activity.

- “98 percent of high-risk access reviewed on time” measures process performance.

- “23 percent of reviewed permissions were removed” provides information about the quality of the existing access model.

- “95 percent of identified inappropriate permissions were remediated within 30 days” measures corrective effectiveness.

Likewise:

- “500 privileged-access requests processed” is operational information.

- “82 percent of eligible privileged access is now time-limited” is a governance indicator.

- “Standing administrative access declined by 30 percent without an increase in unmanaged privileged identities” provides stronger evidence of risk reduction.

Measurement should therefore focus increasingly on whether controls change the organization's exposure, not simply whether IAM and PAM teams are busy.

### Executive Questions Behind the Metrics

Executives do not need to review every technical identity indicator. The dashboard should enable them to answer a smaller number of consequential governance questions:

- Are fewer identities holding unnecessary or excessive access?

- Are privileged identities increasingly governed through controlled and temporary mechanisms?

- Are employees, contractors, and suppliers losing access promptly when their relationship or responsibilities change?

- Are high-risk roles protected by stronger authentication?

- Are segregation-of-duties conflicts declining or being appropriately controlled?

- Are third-party identities owned, time-bounded, and reviewed?

- Are machine identities visible, appropriately privileged, and governed throughout their lifecycle?

- Are exceptions genuinely temporary?

- Are access reviews identifying problems and producing remediation?

- Can significant privileged actions be reconstructed and attributed?

- Are identity anomalies being detected, investigated, and resolved?

- Can management produce credible evidence that the IAM and PAM control environment is operating as intended?

These questions transform identity measurement from operational reporting into **governance intelligence**.

The managerial objective is not to maximize every IAM or PAM metric. It is to maintain a balanced set of indicators that allows management to understand whether **digital authority is becoming more controlled or less controlled over time**.

Effective measurement makes identity risk visible. It enables trends to be challenged, thresholds to trigger action, control weaknesses to be prioritized, investments to be justified, and management to determine whether IAM and PAM are reducing exposure rather than merely administering access.

## Implementation Roadmap: A Practical Sequence for Managers

Organizations often approach IAM and PAM as large-scale technology transformation programs and then struggle with complexity, legacy applications, inconsistent identity data, fragmented ownership, unclear decision rights, and organizational resistance. A more governable approach is to build capability progressively, using a sequence in which each stage reduces meaningful exposure, improves visibility, strengthens evidence, and creates a stronger foundation for the next.

The roadmap should not be interpreted as a rigid technical sequence. Some activities will overlap, and organizations may need to accelerate specific controls where risk is especially high. The underlying principle is that **IAM and PAM maturity should progress according to risk and dependency, not according to software deployment schedules**.

### Stage 1: Establish a Reliable Identity and Authentication Foundation

The first stage is to establish a reliable identity foundation. The organization should define authoritative identity sources, improve the quality and consistency of identity records, reduce unnecessary duplication of identity stores, and consolidate identity providers where feasible.

This foundation matters because later controls depend on accurate information about who or what an identity represents. Automated provisioning, access reviews, role assignment, conditional access, and privileged-access governance become unreliable if identity records are incomplete, inconsistent, or disconnected from authoritative business information.

Authentication should also be strengthened at this stage. MFA should be broadly enforced for significant organizational access, with stronger and phishing-resistant authentication prioritized for high-risk identities and access paths such as privileged administrators, identity administrators, cloud-management environments, remote administration, financial approvals, and sensitive systems.

Conditional-access policies can then progressively incorporate additional context such as authentication strength, device posture, network characteristics, identity risk, and resource sensitivity.

The objective is not simply to deploy an identity provider or MFA technology. It is to increase the organization's confidence that **the identities exercising digital authority are legitimate and appropriately authenticated**.

### Stage 2: Strengthen Identity Lifecycle Discipline

Once the identity foundation is sufficiently reliable, the organization should strengthen lifecycle controls.

Joiner–mover–leaver processes should be standardized, integrated with relevant human-resources, contractor, supplier, and business processes, and supported by clearly defined responsibilities. Where feasible, identity changes should originate from authoritative events rather than depend primarily on manual requests and informal communication.

Deprovisioning deserves particular attention because it can produce immediate risk reduction. Terminated employees, expired contractor accounts, obsolete external identities, dormant accounts, and other access relationships that no longer have a legitimate purpose should be disabled or removed promptly.

Role changes are equally important. A mover process should not simply add access required for the new position. It should also identify and remove permissions associated with responsibilities that no longer exist.

This stage should introduce measurable expectations such as deprovisioning service levels, percentage of lifecycle events processed automatically or through approved workflows, number of orphaned identities, and access remaining after significant role changes.

The objective is to keep **digital authority synchronized with organizational reality**.

### Stage 3: Bring Privileged Authority Under Control

The third stage is to establish stronger governance over privileged identities, credentials, sessions, and administrative pathways.

The organization should identify where privileged authority exists and begin with the systems and identities capable of producing the most significant consequences. Ordinary user accounts should be separated from administrative identities. Standing privilege should be reduced. Privileged credentials and secrets should be protected through appropriate vaulting or secrets-management mechanisms, and just-in-time or just-enough privilege should be introduced where technically and operationally feasible.

High-risk privileged sessions should be attributable, monitored, and evidenced. Administrative pathways should be constrained through stronger authentication, controlled devices, hardened administrative environments, session brokering, network restrictions, or other mechanisms appropriate to the environment.

Break-glass and emergency-access mechanisms should also be designed, protected, tested, monitored, and reviewed.

The objective is not necessarily to place every administrative account into a PAM platform immediately. It is to ensure that the **most consequential forms of digital authority are identified and progressively brought under stronger control**.

### Stage 4: Strengthen Authorization and Role Governance

The fourth stage is to improve the quality of authorization decisions.

This work should begin with high-risk systems, sensitive information, financial processes, production environments, cloud platforms, and functions in which segregation-of-duties conflicts could produce material consequences.

Business-owned roles should be defined and mapped to legitimate responsibilities. Entitlements should be rationalized, unnecessary access removed, and role ownership established. RBAC, ABAC, or hybrid authorization models can then be applied where they improve consistency and control.

Access reviews should also mature from administrative certification exercises into meaningful governance processes. Reviewers should receive understandable information about what access enables, why it was granted, whether it remains aligned with the user's role, when it was last exercised, and whether it creates conflicts with other permissions.

Review findings must lead to remediation. Inappropriate access should be removed, role models corrected, conflicts resolved, and exceptions documented and governed.

The objective is not to demonstrate that access-review forms were completed. It is to demonstrate that **authorization remains justified and that excessive authority is actually reduced**.

### Stage 5: Extend Governance to Third Parties and Non-Human Identities

As human identity and privileged-access governance mature, the organization should extend comparable discipline to external and machine identities.

Third-party access should have explicit internal ownership, documented business justification, defined scope, appropriate authentication, review requirements, and expiration or renewal mechanisms. Privileged supplier access should be governed through the same principles applied to internal administrators.

Non-human identities should also be inventoried and governed systematically. Service accounts, workload identities, API credentials, tokens, certificates, cloud roles, automation scripts, machine identities, and CI/CD pipeline identities should have defined owners and legitimate purposes.

Their privileges should be assessed according to risk. Static secrets should be reduced where feasible in favour of managed identities, workload federation, short-lived credentials, or other mechanisms that reduce dependence on long-lived reusable secrets. High-risk machine identities should be brought within PAM or equivalent privileged-governance controls.

Dormant, orphaned, unexplained, or obsolete non-human identities should be investigated and retired when no longer required.

The objective is to prevent identity governance from stopping at the human workforce while significant external and automated authority remains unmanaged.

<img src="media/image52.png" style="width:6in;height:4.5in" />

Figure 52: IAM Implementation roadmap

### Build Monitoring, Evidence, and Measurement into Every Stage

Monitoring and evidence should not be deferred until the end of the roadmap. They should be designed into every stage.

Identity and privilege events should be logged appropriately, protected against unauthorized alteration, retained according to defined requirements, and integrated with security operations where relevant. Management should be able to determine whether important controls are actually operating rather than relying primarily on implementation status reports.

Metrics should evolve with the program. Early measures may focus on MFA coverage, dormant accounts, deprovisioning timeliness, and privileged-account inventory. Later measures may include standing-privilege reduction, JIT adoption, access-review remediation, SoD conflicts, third-party sponsorship, non-human identity ownership, credential lifecycle compliance, exception aging, and privileged-session accountability.

This is important because risk can migrate. Reducing human administrator accounts provides limited benefit if equivalent privilege moves into unmanaged cloud roles or service accounts. Improving employee deprovisioning while ignoring external identities can leave comparable exposure elsewhere.

Measurement should therefore help management determine whether **overall identity exposure is being reduced rather than merely relocated**.

### Define Milestones as Risk-Reduction Outcomes

Managers should require each stage to produce observable security and governance outcomes rather than simply technology milestones.

“We purchased an IAM platform” or “the PAM system is live” are implementation statements. They do not demonstrate that identity risk has been reduced.

More meaningful milestones might include:

- MFA enforced for 98 percent of in-scope users, with all privileged users protected by approved phishing-resistant authentication;

- terminated identities disabled within the approved service level in at least 99 percent of cases;

- standing privileged accounts reduced by 60 percent;

- 90 percent of eligible administrative access converted to time-limited elevation;

- all break-glass accounts tested successfully during the review period;

- all high-risk third-party identities assigned an internal sponsor and expiry or review date;

- all high-risk service accounts assigned accountable owners and governed credential requirements;

- 95 percent of inappropriate access identified through reviews remediated within the approved period; and

- all privileged activity on critical systems attributable to a named individual or governed process.

The specific thresholds will vary by organization and risk appetite. The important principle is that milestones should describe **improved control conditions**, not merely completion of technical tasks.

### Prioritize According to Risk

Organizations should also resist the temptation to implement IAM controls uniformly across every system from the beginning.

Legacy systems, low-risk applications, critical infrastructure, cloud platforms, financial systems, and modern SaaS environments may require different treatment. A risk-based roadmap should prioritize identities and access paths capable of producing the greatest business consequences.

A high-risk cloud administrator account may justify immediate phishing-resistant authentication and PAM controls even while broader role engineering remains incomplete. A legacy application containing highly sensitive information may warrant accelerated lifecycle and access-review controls even if integration with the central IAM platform requires manual interim processes.

The roadmap should therefore provide strategic direction without preventing management from addressing urgent concentrations of risk.

### Manage IAM and PAM as a Governance Transformation

IAM and PAM implementation should ultimately be treated as a **risk-reduction and governance transformation program**, not simply as a software deployment project.

Technology is necessary, but successful implementation also depends on authoritative data, business ownership, clear decision rights, effective processes, integration with HR and supplier management, meaningful access models, operational monitoring, and credible assurance.

Progress should therefore be assessed through outcomes such as:

- more accurate access;

- stronger authentication;

- fewer unnecessary identities;

- reduced standing privilege;

- faster deprovisioning;

- fewer unresolved SoD conflicts;

- greater visibility of machine and third-party identities;

- stronger evidence of privileged activity;

- fewer unmanaged exceptions; and

- clearer accountability for identity-related decisions.

The managerial lesson is that IAM and PAM maturity develops incrementally. Each stage should remove known weaknesses, reduce the amount of unnecessary digital authority, improve management's visibility, and establish the control foundation required for the next level of maturity.

A successful roadmap is therefore not defined by how many identity technologies have been deployed. It is defined by whether **access becomes more accurate, privilege more constrained, evidence more credible, exceptions more controlled, and accountability more explicit over time**.

## Concluding Integration: Identity Governance as the Core of Modern Cybersecurity

In increasingly identity-centric enterprises, cybersecurity governance depends heavily on the organization's ability to govern digital identities and the authority attached to them. IAM and PAM determine whether access is appropriately granted, whether excessive privilege is constrained, whether insider and third-party risk is reduced, whether attackers can escalate after an initial compromise, and whether management can produce credible evidence for investigations, audits, compliance, and assurance.

The central managerial conclusion of this chapter is straightforward: **when identity governance is weak, many other cybersecurity controls become fragile**.

An organization may invest in network segmentation, endpoint protection, cloud-security technologies, monitoring platforms, data-protection controls, backup systems, and incident-response capabilities. Yet these safeguards can be undermined if attackers, insiders, compromised applications, or unmanaged machine identities acquire permissions that allow them to bypass, modify, or disable those controls.

A strong firewall provides limited protection if a compromised administrator can change its rules. Effective logging provides limited assurance if a privileged identity can disable or alter the logs. Secure cloud storage can still expose sensitive information if an overprivileged account can modify its access policies. Reliable backups may provide little resilience if administrative credentials allow an attacker to delete or encrypt them. Identity governance therefore affects not only who enters the environment, but also **who or what has the authority to change the environment once inside it**.

Conversely, disciplined identity governance strengthens the effectiveness of many other cybersecurity investments.

- **Strong authentication** increases confidence that the actor exercising an identity is legitimate.

- **Least privilege** reduces the amount of unnecessary authority available for misuse or compromise.

- **Segregation of duties** prevents excessive concentrations of authority and reduces opportunities for fraud, error, and concealment.

- **Lifecycle governance** ensures that digital access changes as employment, contractual relationships, roles, applications, and business requirements change.

- **Privileged Access Management** constrains exceptional administrative authority and reduces the likelihood that an ordinary compromise becomes a catastrophic one.

- **Third-party access governance** ensures that external identities remain tied to legitimate business relationships and internal accountability.

- **Non-human identity governance and secrets management** extend these same principles to applications, workloads, services, automation, APIs, and machine-to-machine trust.

- **Monitoring, logging, and evidence** make identity decisions and activity observable, reviewable, and defensible.

Together, these mechanisms turn identity from an administrative concern into a system of **controlled digital authority**.

IAM and PAM are therefore not auxiliary technical disciplines. They are structural governance domains because they translate management expectations into enforceable access decisions. Risk appetite may establish how much exposure the organization is willing to tolerate, but identity controls determine which actors are actually permitted to reach particular resources, under which conditions, with what level of authority, and with what oversight.

In this sense, IAM and PAM operationalize several of the governance principles developed throughout this book. Policies become enforceable through authentication and authorization rules. Least privilege becomes meaningful through role design and privilege management. Segregation of duties becomes operational through entitlement controls and approval workflows. Accountability becomes demonstrable through attribution, logging, and evidence. Risk treatment becomes actionable when excessive access is removed, privileged pathways are constrained, and exceptions are deliberately governed.

Identity governance also creates an important bridge between **prevention, detection, response, and recovery**.

Before an incident, it reduces unnecessary access and strengthens authentication. During an intrusion, it can limit an attacker's ability to move laterally, escalate privilege, or access sensitive resources. Detection capabilities can identify suspicious authentication, anomalous privilege use, or unexpected machine-identity behaviour. During response, identities and sessions can be disabled, credentials rotated, tokens revoked, and administrative pathways restricted. During recovery, controlled emergency access allows authorized personnel to regain control of systems without abandoning governance.

Identity is therefore relevant throughout the cybersecurity lifecycle rather than only at the moment of login.

For Business Technology Management professionals, the practical implication is that IAM and PAM must be governed as **continuous risk-management disciplines**, not implemented once as account-administration projects.

- Management should be able to explain:

- Who owns identity and access decisions?

- What constitutes legitimate access?

- Who is authorized to approve sensitive and privileged access?

- How is authentication strength matched to risk?

- How are permissions constrained through least privilege?

- How are segregation-of-duties conflicts identified and managed?

- How does access change when organizational roles and relationships change?

- How are privileged identities, credentials, sessions, and administrative pathways controlled?

- How are third-party and non-human identities governed?

- How are exceptions approved, monitored, and retired?

- How does management know that the control environment continues to operate as intended?

These questions reveal whether identity governance exists in practice or merely in policy.

The ultimate objective is not to eliminate every identity-related risk. Modern organizations must delegate authority to employees, administrators, suppliers, applications, workloads, and automated processes to operate. The governance challenge is to ensure that this authority is **deliberate, proportionate, constrained, observable, reviewable, and revocable**.

That is the central role of IAM and PAM.

Modern cybersecurity increasingly depends on the organization's ability to govern **who or what may exercise digital authority, over which resources, under which conditions, for how long, and with what evidence and accountability**.

IAM determines how that authority is distributed throughout the organization. PAM applies stronger governance to its most consequential forms. Together, they provide one of the essential foundations through which cybersecurity governance becomes operational, measurable, and defensible.
