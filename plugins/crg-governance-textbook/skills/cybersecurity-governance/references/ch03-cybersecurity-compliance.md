<!-- Léger, M.-A. (2026). Introduction to Cybersecurity Governance for Business Technology Management (3rd ed., v2.51). Chapter 3: Cybersecurity Compliance. CC BY-NC 4.0 -->

# Chapter 3: Cybersecurity Compliance

## Learning outcomes

After completing this chapter, you should be able to:

1.  Explain the role of compliance within the Governance-Risk-Compliance continuum and why compliance is where governance becomes auditable.

2.  Identify the four main sources of cybersecurity compliance obligations and describe how ISO 37301:2021 expects them to be identified, documented, and kept current.

3.  Explain why compliance should be run as a management system, including scope, alignment with strategy, culture, leadership commitment, and the independence of the compliance function.

4.  Distinguish the main Canadian and international regimes that shape cybersecurity compliance and explain why a repeatable process matters more than memorizing a list of laws.

5.  Apply compliance risk assessment to identify where an organization could fail to meet its obligations, and distinguish real compliance from compliance theatre.

6.  Compare compliance assessment approaches and types of assurance, such as self-assessment, certification, attestation, technical testing, and continuous monitoring.

7.  Design a PDCA-aligned compliance management system for cybersecurity, including an obligations register and exceptions with expiry.

8.  Evaluate an organization's compliance arrangements using the chapter's illustrative scenario and the questions managers and boards should ask.

This chapter examines **cybersecurity compliance** as a central pillar of cybersecurity governance. As established in the previous chapter, cybersecurity governance can be understood as a continuum of **Governance, Risk Management, and Compliance (GRC)**. Governance sets direction, clarifies priorities, and defines the organization’s ethical and strategic intent. Risk management supports decision-making by identifying unacceptable risks, analyzing trade-offs, and selecting appropriate treatments. Compliance completes the triad by ensuring that the organization can **demonstrate** it meets its obligations, that its controls and procedures function as intended, and that deviations are detected and corrected in a disciplined manner.

In practice, compliance is where cybersecurity governance becomes auditable: organizations translate intentions into verifiable evidence. A cybersecurity program that cannot demonstrate which obligations apply, which controls exist, how performance is measured, and how issues are handled will struggle to defend its due diligence when facing regulators, customers, insurers, partners, or courts. This chapter is inspired by ISO 37301:2021, the international standard for a Compliance Management System (CMS), which replaced the earlier guidance standard ISO 19600:2014 and serves as a backbone for cybersecurity compliance. ISO 37301 requires organizations to establish, implement, maintain, and continually improve a CMS, and it explicitly anchors the CMS in the organization’s strategy, values, objectives, and compliance risks. Because ISO 37301 is a requirements standard rather than guidance only, an organization can have its CMS certified as evidence that it follows recognized practice. However, certification has a cost, so the decision to obtain certification needs to be carefully examined. Nevertheless, organizations can follow a recognized “best practice” without getting the formal certification.

## What compliance means in cybersecurity

At its core, compliance is straightforward: organizations must **follow rules** and be able to **prove** they do so. Those rules come from multiple sources: laws and regulations; contractual obligations, which may include cybersecurity clauses imposed by customers or suppliers; formal standards that the organization chooses to adopt and, in some cases, seeks certification for; and internal policies that codify expected behaviour and operational discipline.

<span id="_Toc243635514" class="anchor"></span>Figure 15: cybersecurity compliance

Under ISO 37301, the organization needs to systematically identify its compliance obligations arising from its activities, products, and services; assess their impact on operations; maintain documented information about those obligations; and keep those obligations current by identifying new or changed requirements and updating the organization’s management approach accordingly.

In cybersecurity terms, this means that compliance is not a one-time check-the-box effort. It is a living capability that must evolve as technologies, threats, business models, and stakeholder expectations change. Compliance is not primarily about being perfect, but about being **systematic, traceable, and improvable**. Compliance is the organizational habit of continuously knowing what applies, doing what is required, verifying what is done, and correcting what is found.

## Compliance as a management system

Compliance should be understood as an organization-wide **management system**. The harmonized structure shared by ISO management system standards, including ISO 37301, defines a management system as a “set of interrelated or interacting elements of an organization to establish policies and objectives, as well as processes to achieve those objectives.” This approach matters for cybersecurity because compliance is not sustained by a single team or technology. It requires consistent decision-making, disciplined execution, and organizational learning.

Implementing a CMS begins with **context and scope**. ISO 37301 requires the organization to determine the scope of its CMS, considering internal and external issues and the requirements of interested parties, and to make that scope available as documented information. In cybersecurity, scope typically answers questions such as:

- Which business units are covered?

- Which systems and data types are in scope?

- Which legal jurisdictions and regulators apply?

- Which third parties, cloud providers, or processors sit inside the compliance boundary?

A CMS must also reflect the organization’s strategic DNA: ISO 37301 requires it to reflect organizational values, objectives, strategy, and compliance risks. This prevents a common failure mode: implementing compliance controls that are technically sound but strategically misaligned, operationally unrealistic, or culturally rejected.

## Why compliance fails in practice: culture, leadership, and governance

Organizations often attempt to force compliance through rules, surveillance, and punishment. That approach can improve short-term adherence but tends to generate long-term fragility: workarounds, resentment, quiet noncompliance, and superficial reporting. A more sustainable approach is to build a **culture of compliance**, where conforming to obligations becomes the normal way of working rather than a periodic disruption.

According to ISO 37301, operating a successful CMS requires the organization to develop, maintain, and promote a culture of compliance at all levels. It also requires the governing body and management to demonstrate an active and sustained commitment to the standard of conduct expected throughout the organization.

| This aligns directly with cybersecurity realities: **human behaviour is the most variable and most decisive factor in day-to-day cybersecurity outcomes.** |
|------------------------------------------------------------------------------------------------------------------------------------------------------------|

The standard also expects the governing body and top management to demonstrate commitment by ensuring the compliance policy and objectives are compatible with the strategic direction, integrating compliance requirements into business processes, providing resources, communicating the importance of compliance, ensuring the CMS achieves intended results, promoting continual improvement, and supporting relevant roles.

Finally, ISO 37301 calls for compliance governance safeguards. These include principles such as direct access of the compliance function to the governing body, independence of the compliance function, and appropriate authority and competence.

These requirements exist because compliance can become politically inconvenient: it may reveal underfunding, risky strategic trade-offs, or uncomfortable truths about operational practices. Independence and direct access reduce the risk that compliance findings are suppressed, diluted, or ignored.

## Compliance obligations in cybersecurity: mapping the landscape

Cybersecurity compliance obligations generally fall into overlapping families:

- First, there are **legal and regulatory obligations**, especially in privacy and incident reporting. The same organization may be subject to multiple regimes depending on where it operates and where its users, customers, or students are located.

- Second, there are **contractual obligations**, such as customer requirements for cybersecurity attestations, audit rights, breach notifications within specific timeframes, encryption requirements, or supplier risk controls.

- Third, there are **standards and frameworks** (such as ISO/IEC 27001 and ISO/IEC 27002, NIST publications, COBIT, or sector-specific standards) that an organization adopts to structure its controls, increase maturity, obtain certification, or satisfy partners and insurers.

- Fourth, there are **internal policies and procedures**, including cybersecurity policies, acceptable use, access management, and change management, which translate obligations into operational expectations.

> This landscape is to be managed systematically: organizations must maintain documented records of their compliance obligations and have processes to identify new or changed obligations and evaluate their impact. In other words, the compliance landscape is not merely a list; it is a monitored environment that triggers change. Because one control often satisfies several obligations, many organizations maintain a single mapped set of controls, so that a well-evidenced control (for example, multi-factor authentication for remote access) can be tested once and reported against several requirements.
>
> In Canada, a realistic compliance landscape for a regulated organization typically includes privacy statutes and guidance, sectoral regulatory requirements (such as those issued by financial regulators), and contractual requirements from business partners. The essential point for students is methodological rather than encyclopedic: the goal is not to memorize a changing list of laws, but to learn how to build a **repeatable process** that identifies what applies and keeps it up to date.
>
> That said, a few regimes recur so often in Canadian practice that managers should recognize them. Federally, the Personal Information Protection and Electronic Documents Act (PIPEDA) has required organizations since November 2018 to report breaches of security safeguards that create a real risk of significant harm. In Québec, Law 25 amended the Act respecting the protection of personal information in the private sector, adding, among other obligations, mandatory reporting of confidentiality incidents to the Commission d’accès à l’information. Federally regulated financial institutions must meet OSFI Guidelines B-10 (third-party risk management) and B-13 (technology and cyber risk management), and the Critical Cyber Systems Protection Act, enacted through Bill C-8 in June 2026, will require designated operators in federally regulated critical sectors to maintain cybersecurity programs and report incidents as its provisions come into force. Organizations operating internationally may also face the EU General Data Protection Regulation (GDPR), the NIS2 Directive, the Digital Operational Resilience Act (DORA) for financial entities, and the US SEC cybersecurity disclosure rules, while contractual regimes such as the Payment Card Industry Data Security Standard (PCI DSS) apply wherever payment cards are processed. Chapters 12, 16, and 19 examine the incident-reporting, privacy, and third-party dimensions of these regimes.

<span id="_Toc243635515" class="anchor"></span>Figure 16: Compliance obligations landscape

## Compliance risk assessment: where obligations meet operations

ISO 37301 contains an explicit requirement for compliance risk assessment: the organization must identify, analyze, and evaluate compliance risks, relating compliance obligations to activities, products, services, and relevant aspects of operations. This is especially important in cybersecurity because compliance failures are frequently operational rather than ideological. Compliance failures often result from misconfigurations, inconsistent access management, untrained staff, and incomplete vendor oversight to name only a few.

Compliance risks must also be assessed periodically for **outsourced and third-party processes**, and when there are material changes in circumstances or organizational context, and documented information must be retained regarding the risk assessment and actions taken. This requirement is particularly aligned with modern cybersecurity, where cloud services, managed cybersecurity services, payment processors, SaaS platforms, and data-sharing partnerships often sit at the centre of operational reality.

For students, a practical interpretation is this: compliance risk assessment is the discipline of asking, repeatedly and systematically:

| **Where could we fail to meet obligations, why, and with what consequences?** |
|-------------------------------------------------------------------------------|

It is also where organizations confront the uncomfortable fact that having a policy does not mean being compliant. Compliance exists only when controls operate effectively and repeatedly, under real conditions, with real people, across real exceptions. The opposite pattern, often called compliance theatre, produces polished documentation and passed audits while the underlying risk is unchanged; governance should guard against it by asking whether the evidence shows that controls actually reduce the risks the obligations were meant to address.

## Demonstrating compliance: evidence, reporting, and the ability to raise concerns

Because compliance must be demonstrable, it must be documented. ISO 37301 requires key elements to be documented, including the scope of the CMS, the register of compliance obligations, and the compliance risk assessment with associated actions. In cybersecurity practice, documentation is not bureaucracy for its own sake; it is operational memory. It allows accountability, continuity through staff turnover, and credibility in audits and investigations.

Leadership must ensure the organization has a system for raising and addressing concerns; ISO 37002:2021 on whistleblowing management systems offers detailed guidance on designing one. This is critical in cybersecurity compliance because employees often see near misses before incidents occur: suspicious vendor practices, informal data exports, unsecured workarounds, pressure to bypass approvals, or hidden shadow IT. A compliance system that discourages speaking up will fail quietly until it fails publicly.

## Compliance assessment approaches in cybersecurity

Organizations use multiple, complementary approaches to assess cybersecurity compliance. Internal self-assessments establish a baseline and support continuous improvement. External audits, such as ISO/IEC 27001 certification audits or SOC 2 examinations, provide independent assurance and are sometimes mandated by regulators or contracts. Technical testing, such as vulnerability assessments and penetration testing, can be used to evaluate the effectiveness of controls against plausible attack paths and can expose gaps that policy reviews miss when results are explicitly mapped to applicable requirements. Technical testing does not, by itself, prove legal compliance. Continuous monitoring, often supported by SIEM, log analytics, and detection engineering, reduces the time between deviation and correction. Finally, training and awareness assessments evaluate whether compliance expectations are realistically understood and practiced.

The essential lesson is that compliance assessment is not an annual ritual. It is an ecosystem of feedback loops. An organization’s CMS must be structured precisely to support these loops through operation, performance evaluation, and improvement, including internal audit, management review, and corrective action.

## Implementing a PDCA-aligned compliance management system for cybersecurity

A CMS is not simply installed; it is built, embedded, and matured over time. Organizations need to establish, implement, maintain, and continually improve the CMS and its interacting processes.

<span id="_Toc243635516" class="anchor"></span>Figure 17: compliance management PDCA cycle

A practical implementation approach, consistent with the ISO management system logic, is to structure work as a continuous improvement cycle.

- In the **Plan** phase, the organization defines scope, understands the internal and external context, identifies interested parties, establishes compliance obligations, and conducts a documented compliance risk assessment that includes third parties.

- In the **Do** phase, the organization implements controls and procedures that translate obligations into operational behaviour, supported by training, communication, and resources, and guided by leadership commitment.

- In the **Check** phase, the organization monitors performance, conducts audits, gathers feedback, and evaluates whether the CMS achieves intended results—using objective indicators rather than perception.

- In the **Act** phase, the organization corrects nonconformities, improves processes, updates documentation, and strengthens controls and training to make compliance more robust under real operational pressure.

This cycle is particularly valuable for cybersecurity because the environment changes constantly: new threats, new systems, new vendors, new legal interpretations, new business priorities. A CMS creates the organizational habit of adapting without improvising.

### Illustrative scenario: a Montréal billing-platform provider builds its compliance management system

Pontvert Systems (a fictional organization) is a Montréal software-as-a-service scale-up with about 350 employees. Its platform handles billing and payment collection for utilities, property managers, insurers, and, since last year, two federally regulated banks. It holds personal information on several million end customers across Canada and processes card payments on its clients' behalf. In three years, Pontvert had collected an ISO/IEC 27001 certificate, a SOC 2 report, and a thick folder of customer security questionnaires. What it did not have was a single view of what it had promised to whom.

The problem surfaced when a new compliance director, hired from a financial institution, tried to answer a bank's question: "List every obligation you have to notify us of an incident, and the deadline for each." She found three spreadsheets maintained by security, legal, and sales, none complete, and contracts that set four different notification deadlines for essentially the same event. Her first recommendation to the executive committee was to build a compliance management system inspired by ISO 37301:2021, starting with the obligations register. She also recommended deferring formal ISO 37301 certification: no customer or regulator required it, and the money was better spent first on making the system work.

**The obligations register.** Over a quarter, the team recorded each obligation with its source, the activity it applied to, an accountable owner, the controls that satisfied it, the evidence that proved it, and a review date. The main families were:

- **Québec Law 25:** obligations for Québec residents' personal information, including privacy impact assessments, the register of confidentiality incidents, and reporting incidents that present a risk of serious injury to the Commission d'accès à l'information.

- **PIPEDA:** obligations for personal information handled in commercial activity in other provinces, including reporting breaches of security safeguards that create a real risk of significant harm.

- **OSFI expectations, by contract:** OSFI does not supervise Pontvert, but its bank clients must meet Guidelines B-10 and B-13, and they pass those expectations down through audit rights, incident-notification clauses, subcontractor disclosure, and exit provisions.

- **PCI DSS:** because Pontvert stores, processes, or transmits cardholder data for its clients, it must be assessed against PCI DSS as a service provider, as its acquiring bank and clients require.

- **Customer contracts:** notification deadlines, data-location commitments, encryption requirements, and audit rights negotiated deal by deal.

Mapping the controls showed that a small number of well-evidenced controls—multi-factor authentication, logging, encryption, access reviews—satisfied most obligations, so they could be tested once and reported against many requirements. The contracts were the messiest family, so sales and legal agreed on standard security clauses, and any deviation now required the compliance director's sign-off before signature.

**Three kinds of assurance.** The board's audit committee asked a sharp question: if Pontvert was already certified, why did it need all this? The compliance director's answer became a standing slide. ISO/IEC 27001 *certification* means an accredited certification body has judged that the ISMS conforms to the standard within a defined scope. A SOC 2 report is an *attestation*: an independent auditor's opinion on whether described controls were suitably designed and, for a Type II report, operated effectively over a period. A *regulatory examination* is something a supervisor performs on a regulated entity; in Pontvert's case, OSFI examines the banks, and the banks then exercise their contractual audit rights over Pontvert. Each answers a different question, for a different audience, over a different scope. None of them, alone, proves that Pontvert meets its obligations.

**Exceptions with expiry.** The compliance risk assessment found that a legacy reporting server used by two insurer clients could not support multi-factor authentication, which conflicted with both a contract clause and Pontvert's own policy. Rather than ignore the gap or quietly mark it compliant, the vice-president of product, as risk owner, approved a formal exception: access restricted to a jump host with strong authentication, enhanced logging reviewed weekly, a 90-day expiry, and a migration plan. When the 90 days ran out, the exception was not renewed automatically; it returned to the risk owner and the compliance director with evidence of progress, and the committee saw the count of open and expired exceptions each quarter.

**Theatre versus risk reduction.** The most important finding came from a support agent who used the internal speak-up channel to report that colleagues routinely exported customer files to spreadsheets to troubleshoot billing disputes. The support tooling sat outside the scope of both the ISO/IEC 27001 certificate and the SOC 2 report, so every audit had passed while the riskiest data flow went unexamined. Signing a new policy would have closed the finding on paper. Instead, Pontvert brought the support environment into scope, removed bulk export for most roles, and tracked the volume of exports each month to confirm that the behaviour had actually changed.

What good looked like, a year on, was not a new certificate. It was an executive team that could answer the bank's original question in an afternoon, from a register it maintained anyway, and an audit committee that knew which risks the certificates did not cover.

### Questions managers and boards should ask

The scenario suggests questions that executives and directors can use to test whether compliance is real or merely documented:

- **Register:** Do we maintain a single, current register of compliance obligations, with an owner, mapped controls, and evidence for each?

- **Sources:** Have we identified all the sources of our obligations—laws such as Law 25 and PIPEDA, sector rules, card-industry requirements, and the clauses in our customer and supplier contracts?

- **Inherited obligations:** Which regulatory expectations, such as OSFI B-10 and B-13, reach us indirectly through our clients' contracts, and could we support a client's audit or regulatory examination tomorrow?

- **Contract discipline:** Who approves non-standard security, notification, or audit-rights clauses before they are signed, and how do those commitments reach the register?

- **Assurance types:** For each certificate, attestation report, or assessment we hold, what exactly does it cover, for what period, and what does it leave out?

- **Scope gaps:** Which systems, data flows, or third parties that handle sensitive information sit outside the scope of our certifications and attestations?

- **Exceptions:** How many compliance exceptions are open, how many are past their expiry date, who owns each one, and what compensating controls apply?

- **Theatre test:** For our most important obligations, can we show evidence that the controls actually reduce the risk the obligation addresses, not merely that a policy exists?

- **Independence:** Does the compliance function have direct access to the board or its audit committee, and has it ever used that access to raise an uncomfortable finding?

- **Speaking up:** Can employees raise compliance concerns safely, and what changed the last time one did?

- **Certification decision:** Have we decided deliberately, on cost and stakeholder value, whether to seek certification of our compliance management system, or have we simply drifted?

## Conclusion: compliance as disciplined trust

Cybersecurity compliance is not an administrative add-on to cybersecurity governance; it is the mechanism by which governance becomes credible and defensible. Compliance ensures that obligations are identified and managed systematically, that risks of noncompliance are assessed, that leadership commits resources and integrates compliance into business processes, and that the organization can document and demonstrate its actions. ISO 37301 offers a recognized structure for this work and, where stakeholders value it, a route to certification.

The most resilient organizations do not treat compliance as a fear-driven constraint. They treat it as a form of disciplined trust: trust earned by consistent behaviour, verified by evidence, improved by learning, and protected by governance structures that ensure compliance can speak the truth, especially when it is inconvenient.

## Chapter summary

Chapter 3 presents compliance as the element of the GRC continuum that makes cybersecurity governance credible and defensible. Compliance means following rules from laws, contracts, standards, and internal policies, and being able to prove it with evidence. Drawing on ISO 37301:2021, the chapter treats compliance as a management system grounded in strategy, values, and compliance risks, sustained by culture, leadership commitment, and an independent compliance function. It maps the obligations landscape, explains compliance risk assessment, documentation, speak-up channels, and assessment approaches, and proposes a PDCA-aligned implementation. A Montréal SaaS scenario shows that certificates do not prove compliance on their own and that real compliance reduces risk rather than producing theatre.

### Key points to retain

- Compliance completes the GRC continuum by ensuring the organization can demonstrate that it meets obligations, that controls work as intended, and that deviations are corrected.

- ISO 37301:2021 is a requirements standard for compliance management systems that can be certified, but certification has a cost and should be a deliberate decision.

- Compliance obligations come from laws and regulations, contracts, adopted standards and frameworks, and internal policies, and must be identified, documented, and kept current.

- Compliance is a living capability: continuously knowing what applies, doing what is required, verifying what is done, and correcting what is found.

- A compliance management system starts with a documented scope and must reflect the organization's values, objectives, strategy, and compliance risks.

- Forcing compliance through rules, surveillance, and punishment creates long-term fragility, whereas a culture of compliance and visible leadership commitment make it sustainable.

- Independence, direct access to the governing body, and appropriate authority protect compliance findings from being suppressed, diluted, or ignored.

- A single mapped set of controls lets one well-evidenced control be tested once and reported against several obligations, including PIPEDA, Law 25, OSFI, and PCI DSS requirements.

- Compliance risk assessment asks where the organization could fail to meet obligations, why, and with what consequences, and must cover third parties and material changes.

- Compliance theatre produces polished documents and passed audits while risk is unchanged; real compliance shows that controls actually reduce the risks obligations address.

- Documentation is operational memory, and a safe system for raising concerns surfaces near misses before they become incidents.

- Self-assessments, external audits, technical testing, continuous monitoring, and awareness assessments form an ecosystem of feedback loops organized through Plan-Do-Check-Act.

### Key terms

- **Compliance** — Following the rules that apply to the organization and being able to prove it with verifiable evidence.

- **Compliance management system (CMS)** — An organization-wide management system, as specified by ISO 37301:2021, for establishing, implementing, maintaining, and continually improving compliance.

- **ISO 37301:2021** — The international requirements standard for compliance management systems, which replaced the guidance standard ISO 19600:2014 and supports certification.

- **Management system** — A set of interrelated or interacting elements of an organization to establish policies and objectives, as well as processes to achieve those objectives.

- **Compliance obligations** — Requirements arising from laws and regulations, contracts, adopted standards, and internal policies that the organization must meet.

- **Register of compliance obligations** — Documented information recording each obligation with its source, scope, owner, mapped controls, evidence, and review date.

- **Culture of compliance** — A condition in which conforming to obligations is the normal way of working rather than a periodic disruption.

- **Compliance risk assessment** — The identification, analysis, and evaluation of compliance risks by relating obligations to activities, products, services, and operations.

- **Compliance theatre** — A pattern of polished documentation and passed audits while the underlying risk remains unchanged.

- **Mapped control set** — A single set of controls mapped to several obligations so that each well-evidenced control can be tested once and reported against many requirements.

- **Certification** — A judgment by an accredited certification body that a management system, such as an ISMS, conforms to a standard within a defined scope.

- **Attestation (SOC 2)** — An independent auditor's opinion on whether described controls were suitably designed and, for a Type II report, operated effectively over a period.

- **Exception with expiry** — A formally approved, time-limited deviation from an obligation or policy, with a risk owner, compensating controls, and a plan, that is reviewed rather than renewed automatically.

- **PDCA cycle** — The Plan-Do-Check-Act continuous improvement cycle used to build, embed, and mature a compliance management system.

- **Speak-up system** — A system for raising and addressing concerns, for which ISO 37002:2021 offers guidance, allowing employees to report near misses and noncompliance safely.

## Review questions

### Essay questions

1.  Evaluate the chapter's claim that compliance is “where cybersecurity governance becomes auditable.” Explain how compliance relates to governance and risk management in the GRC continuum, and analyze why compliance is necessary but not sufficient for security.

2.  Analyze why forcing compliance through rules, surveillance, and punishment tends to create long-term fragility. Using ISO 37301's requirements on culture, leadership commitment, and the independence of the compliance function, recommend how an organization can build a sustainable culture of compliance.

3.  Choose an organization you know, such as an employer, a university, or a business that processes card payments. Identify its likely compliance obligations across the four families described in the chapter, and design the structure of an obligations register and a mapped control set for it.

4.  Distinguish real compliance from compliance theatre. Using the Pontvert Systems scenario, analyze how an organization with an ISO/IEC 27001 certificate and a SOC 2 report could still leave its riskiest data flow unexamined, and evaluate the actions Pontvert took to correct it.

5.  Design a PDCA-aligned compliance management system for a mid-sized Canadian organization. Describe what happens in each phase, explain how compliance risk assessment, assessment approaches, exceptions with expiry, and speak-up channels fit into the cycle, and justify whether the organization should seek ISO 37301 certification.

### Multiple-choice questions

Choose the best answer. Answers are listed in the appendix "Answers to multiple-choice questions".

**1.** Which earlier standard did ISO 37301:2021 replace?

> A. ISO/IEC 27001:2013 on information security
>
> B. ISO 37000:2021 on governance of organizations
>
> C. ISO 19600:2014, a guidance standard
>
> D. ISO 37002:2021 on whistleblowing management

**2.** What does the fact that ISO 37301 is a requirements standard, rather than guidance only, make possible?

> A. Certification of the compliance management system
>
> B. Exemption from legal and regulatory obligations
>
> C. Replacement of internal audit by external audit
>
> D. Automatic recognition by all Canadian regulators

**3.** A major customer's contract requires breach notification within a specific timeframe and grants audit rights. Which family of compliance obligations does this belong to?

> A. Legal and regulatory obligations
>
> B. Adopted standards and frameworks
>
> C. Internal policies and procedures
>
> D. Contractual obligations

**4.** Which statement best reflects the chapter's description of compliance as a living capability?

> A. Achieving perfect conformity once and certifying it annually
>
> B. Continuously knowing what applies, doing it, verifying it, and correcting findings
>
> C. Completing a checklist whenever an external audit is scheduled
>
> D. Writing a policy for each obligation and filing it securely

**5.** Which question is part of defining the scope of a cybersecurity compliance management system?

> A. Which third parties, cloud providers, or processors sit inside the compliance boundary?
>
> B. Which security vendor offers the lowest licensing cost?
>
> C. Which employees have the most years of service?
>
> D. Which competitors have obtained ISO certification?

**6.** Why does ISO 37301 require the CMS to reflect organizational values, objectives, strategy, and compliance risks?

> A. To allow the organization to skip compliance risk assessment
>
> B. To make certification less expensive and faster
>
> C. To ensure that compliance is managed by the IT department
>
> D. To avoid controls that are technically sound but misaligned or culturally rejected

**7.** According to the chapter, what does forcing compliance mainly through rules, surveillance, and punishment tend to produce over time?

> A. A durable culture of compliance at all levels
>
> B. Lower audit costs and fewer external findings
>
> C. Workarounds, resentment, quiet noncompliance, and superficial reporting
>
> D. Greater independence for the compliance function

**8.** Why does ISO 37301 call for the compliance function to be independent and to have direct access to the governing body?

> A. So that it can approve all business investments directly
>
> B. So that findings revealing underfunding or risky trade-offs are not suppressed
>
> C. So that it can replace internal audit in the three lines
>
> D. So that it reports only to the chief information officer

**9.** Why do many organizations maintain a single mapped set of controls across their obligations?

> A. Because regulators prohibit separate control sets
>
> B. Because it eliminates the need for evidence
>
> C. Because each obligation then needs its own audit
>
> D. Because one well-evidenced control can be tested once and reported against several requirements

**10.** Since November 2018, what has PIPEDA required organizations to report?

> A. Breaches of security safeguards that create a real risk of significant harm
>
> B. Every cybersecurity incident, whatever the harm, as soon as it is detected
>
> C. Confidentiality incidents to the Commission d'accès à l'information
>
> D. Annual descriptions of board cybersecurity oversight to investors

**11.** What does OSFI Guideline B-10 address for federally regulated financial institutions?

> A. Technology and cyber risk management
>
> B. Payment card data security for merchants
>
> C. Third-party risk management
>
> D. Protection of personal information in Québec

**12.** What is the essential methodological point the chapter makes about the Canadian compliance landscape?

> A. Students should memorize every applicable law and its current version
>
> B. The goal is a repeatable process that identifies what applies and keeps it current
>
> C. Only federal statutes are relevant to cybersecurity compliance
>
> D. International regimes can be ignored by Canadian organizations

**13.** Which situation best illustrates compliance theatre?

> A. A control is tested once and reported against several obligations
>
> B. Polished documents and passed audits while the underlying risk is unchanged
>
> C. An exception is approved with compensating controls and an expiry date
>
> D. An employee reports a near miss through the speak-up channel

**14.** When must compliance risks be reassessed, according to ISO 37301 as described in the chapter?

> A. Only when a regulator announces an examination
>
> B. Only after a confirmed security incident has occurred
>
> C. Once at implementation, then only at recertification
>
> D. Periodically, including outsourced processes, and when material changes occur

**15.** What does the chapter say about penetration testing and vulnerability assessments as compliance assessment approaches?

> A. They can expose gaps against plausible attack paths but do not, by themselves, prove legal compliance
>
> B. They replace the need for internal audit and management review
>
> C. They prove legal compliance whenever no critical findings are reported
>
> D. They are relevant only to organizations seeking ISO 37301 certification

**16.** In the PDCA-aligned approach, what happens in the Check phase?

> A. Scope, context, and obligations are defined and assessed
>
> B. Controls and procedures are implemented with training and resources
>
> C. Performance is monitored, audits conducted, and results evaluated with objective indicators
>
> D. Nonconformities are corrected and documentation is updated

**17.** Why did Pontvert's compliance director recommend deferring formal ISO 37301 certification?

> A. No customer or regulator required it, and the money was better spent making the system work
>
> B. ISO 37301 cannot be certified for software-as-a-service providers
>
> C. The company's ISO/IEC 27001 certificate already covered compliance management
>
> D. Certification would have required abandoning the SOC 2 report

**18.** In the Pontvert scenario, what does a SOC 2 Type II report provide?

> A. A supervisor's examination of a regulated financial institution
>
> B. A certification body's judgment that an ISMS conforms to a standard
>
> C. A self-assessment of compliance with internal policies
>
> D. An auditor's opinion on whether controls were suitably designed and operated effectively over a period

**19.** Pontvert's legacy reporting server could not support multi-factor authentication. How did the organization handle this conflict with its policy and a contract?

> A. It marked the server compliant because few users accessed it
>
> B. It shut down the server immediately, ending service to two clients
>
> C. The risk owner approved a formal exception with compensating controls, a 90-day expiry, and a migration plan
>
> D. It renewed the exception automatically each quarter

**20.** What was the main lesson of the customer-file exports reported through Pontvert's speak-up channel?

> A. Speak-up reports mostly raise low-value complaints that audits already cover
>
> B. Audits passed because support tooling was out of scope; the fix brought it in and tracked behaviour
>
> C. A newly signed policy is the most effective way to close such a finding
>
> D. SOC 2 reports automatically extend to every internal support tool in use
