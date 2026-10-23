<!-- Léger, M.-A. (2026). Introduction to Cybersecurity Governance for Business Technology Management (3rd ed., v2.51). Chapter 2: Cybersecurity Governance. CC BY-NC 4.0 -->

# Chapter 2: Cybersecurity Governance

## Learning outcomes

After completing this chapter, you should be able to:

1.  Explain cybersecurity governance as the structures, decision rights, processes, and accountability mechanisms through which an organization directs and controls cybersecurity.

2.  Distinguish governance (Evaluate-Direct-Monitor) from management (Plan-Build-Run-Monitor) and analyze the failures that follow when the boundary is blurred.

3.  Explain how cybersecurity governance relates to enterprise governance, IT governance, ethics, and the Governance-Risk-Compliance continuum.

4.  Identify the five outcomes cybersecurity governance must achieve and describe how regulators such as NIS2, the SEC, OSFI, and Québec's Law 25 assign accountability to boards and senior management.

5.  Apply the six classic ISO/IEC 38500 principles to assess whether cybersecurity is governed rather than merely managed.

6.  Design a balanced cybersecurity scorecard that combines leading and lagging indicators across the four perspectives.

7.  Evaluate cybersecurity metrics against the SMART criteria and justify why metrics support but do not replace judgment.

8.  Recommend governance decisions on oversight, risk appetite, decision rights, reporting lines, and measurement for an organization, using the chapter's illustrative scenario as a model.

This chapter examines cybersecurity governance: the set of structures, decision rights, processes, and accountability mechanisms through which an organization directs and controls cybersecurity to support performance, resilience, and trust. Chapter 1 introduced core cybersecurity concepts and emphasized that cybersecurity is, first, a governance issue and only then an implementation issue. Building on that foundation, this chapter explains how organizations translate strategic intent into a coherent approach to managing cybersecurity risk in Business Technology Management (BTM) contexts, including how they set priorities, allocate resources, justify controls, measure performance, and demonstrate compliance.

<span id="_Toc243635507" class="anchor"></span>Figure 8: cybersecurity governance

## Governance as direction, control, and accountability

In an organizational setting, governance refers to the structures, mechanisms, and decision-rights through which authority is exercised, resources are allocated, and performance is overseen. In practical terms, governance answers a set of enduring questions: Who decides? On what basis? Who is accountable for outcomes? How are conflicts resolved? How does the organization ensure that decisions remain aligned with its mission, values, stakeholder expectations, and legal obligations?

Good governance rests on key principles—most notably transparency, accountability, participation, and responsiveness. These are not merely abstract ideals; they have direct operational implications. Transparency affects whether stakeholders can clearly understand what is being done, how risks are evaluated, and why decisions are made. Accountability ensures that decisions have clearly identified owners and that results are continuously monitored and verified. Participation dictates whether relevant, cross-functional expertise is included in the decision-making process. Responsiveness determines whether the organization can respond and adapt quickly as circumstances change, such as when cyber threats evolve or incidents occur. Together, these principles form a framework that builds trust and aligns the interests of individuals, the organization, and society.

A fundamental requirement for success is the clear distinction between governance and management. According to frameworks and standards such as COBIT 2019 and ISO/IEC 38500, governance is the responsibility of the board and executive leadership; it sets the strategic direction and exercises oversight. Using the "Evaluate-Direct-Monitor" (EDM) cycle, governance evaluates stakeholder needs and risks, directs policies and strategies, and monitors compliance and performance. Conversely, management executes within that established direction. Following the "Plan-Build-Run-Monitor" (PBRM) model, management is responsible for the day-to-day planning, building, and operating of systems to achieve the enterprise's objectives.

Organizations frequently encounter cybersecurity failures because the boundary between governance and management is blurred. When senior leadership treats cybersecurity as a purely technical management activity to be handled exclusively by IT, critical strategic questions about risk appetite, resource allocation, and trade-offs remain unanswered. Conversely, when cybersecurity is discussed only at a high, abstract level without measurable execution and clear reporting from management, governance becomes symbolic and ineffective. Mature organizations avoid these pitfalls by ensuring that strong governance and competent management are tightly connected, functioning together as a continuous, feedback-driven cycle of directing and controlling.  

<span id="_Toc243635508" class="anchor"></span>Figure 9: governance support structure

## Applied ethics and why governance exists

Cybersecurity governance can be understood as applied ethics under conditions of technological dependence and strategic uncertainty. Ethics, in this context, refers to the shared norms and principles that guide acceptable conduct, responsible decision-making, and fair treatment of stakeholders. Those norms come from many sources—organizational culture, professional standards, laws and regulations, contractual commitments, and broader societal expectations. Ethical norms also evolve over time, which is why governance cannot be static.

The practical relevance of ethics is straightforward. Organizations routinely face situations in which written rules do not cover every possibility, trade-offs are unavoidable, and short-term incentives may conflict with long-term responsibility. Governance provides the organization with a structured way to define acceptable conduct, reduce discretionary ambiguity in high-risk contexts, and ensure that decisions remain defensible, consistent, and aligned with the organization’s stated values. In cybersecurity, this includes decisions that affect privacy, customer trust, employee monitoring, acceptable use, incident disclosure, third-party risk, and resource allocation under constraints.

Ethical norms alone, however, are not sufficient. Societies and organizations also require enforceable constraints: laws, policies, controls, and consequences. This is not because most people are malicious, but because complex systems create opportunities for misuse and because incentives can drift. Cybersecurity governance, therefore, includes both normative guidance (what ought to be done) and formal mechanisms (what must be done, by whom, and how compliance will be verified).

## A hierarchy of governance: enterprise, IT, and cybersecurity

Organizations rarely have only one governance system. Instead, governance exists at multiple levels and for multiple domains. At the highest level, enterprise governance provides the overall framework for direction and oversight across the organization. It defines how power is exercised among the board, executive management, and stakeholders; how strategy is established; how accountability is enforced; and how value is created responsibly.

Within enterprise governance, IT governance applies governance principles to the organization’s information technology and business technologies. Its purpose is to ensure that technology investments and operations support business objectives, create measurable value, and remain aligned with acceptable risk levels. IT governance addresses decision rights for technology strategy, investment prioritization, architecture direction, service performance, vendor management, lifecycle management, and technology-related risk.

<span id="_Toc243635509" class="anchor"></span>Figure 10: governance positioning

Cybersecurity governance is a more specific domain within this hierarchy. It focuses on the governance of cyber risk: ensuring that cybersecurity objectives are defined, unacceptable risks are identified and addressed, responsibilities are assigned to accountable roles, and the organization can demonstrate ongoing control effectiveness and compliance. Cybersecurity governance is not separate from enterprise governance; it is derived from it. It is also not identical to IT governance; cybersecurity governance intersects with IT, but it also intersects with legal and compliance, risk management, human resources, operations, procurement, audit, and executive leadership.

[A useful way to conceptualize cybersecurity governance is through the Governance–Risk–Compliance (GRC) continuum lens. Governance sets direction and accountability; risk management identifies and treats cyber risks; compliance ensures alignment with external and internal obligations and verifies adherence. This orientation aligns strongly with the evolution of major frameworks. Notably, NIST’s Cybersecurity Framework 2.0, released in February 2024, added a sixth function, Govern, to emphasize the centrality of governance in cybersecurity risk management.](https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.29.pdf?utm_source=chatgpt.com)

<span id="_Toc243635510" class="anchor"></span>Figure 11: GRC continuum

## What cybersecurity governance must achieve

Cybersecurity governance exists to ensure that cybersecurity decisions are consistent, defensible, and aligned with organizational priorities. Concretely, this means that cybersecurity governance must enable the organization to reliably do the following:

- First, **governance must set a strategic direction**. The organization needs clearly defined cybersecurity objectives grounded in business realities (for example, service continuity requirements, privacy obligations, critical data and systems, and reputational dependency).

- Second, **governance must establish decision rights and accountability.** Cybersecurity governance clarifies who is responsible for what, who has authority to approve risk decisions, and who must report and be answerable for outcomes.

- Third, **governance must ensure resource allocation is rational.** Cybersecurity competes with other managerial priorities; governance ensures that investments are linked to risk reduction, resilience requirements, and legal obligations rather than to fear, hype, or vendor pressure.

- Fourth, **governance must ensure oversight and measurement.** Governance requires the organization to monitor whether controls are effective in practice, not merely designed on paper, and to continuously improve based on evidence.

- Fifth, **governance must ensure compliance and defensibility**. Organizations must demonstrate that they meet applicable laws, regulations, contractual obligations, and standards in a way that is auditable and credible.

- These expectations are no longer only matters of good practice; regulators increasingly assign cybersecurity accountability to boards and senior management by name. The EU NIS2 Directive (Directive (EU) 2022/2555) requires the management bodies of essential and important entities to approve and oversee cybersecurity risk-management measures, to follow training, and allows them to be held liable for infringements. The US Securities and Exchange Commission’s 2023 disclosure rules require public companies to describe annually, under Regulation S-K Item 106, how the board oversees cybersecurity risk and how management assesses and manages it. In Canada, OSFI Guideline B-13 expects federally regulated financial institutions to establish clear accountability for technology and cyber risk, and Québec’s Law 25 makes the person exercising the highest authority in an enterprise responsible, by default, for the protection of personal information. Chapters 3, 12, and 17 return to these regimes; the governance point here is that decision rights and oversight must be designed so that they can be demonstrated.

A recurring theme in this book is that governance is enabled by four interdependent elements: culture, leadership, processes, and mechanisms. Culture shapes behaviour when no one is watching. Leadership sets priorities and legitimacy. Processes define how work is done and how decisions are made. Mechanisms include the technical and procedural safeguards that reduce risk. A mature cybersecurity program uses mechanisms, but it is governed through culture, leadership, and disciplined processes. When organizations attempt to buy cybersecurity without governance, they often accumulate controls while remaining exposed to unmanaged risk.

## Change management as a governance requirement

Cybersecurity governance requires change. Controls alter how people work. Policies impose constraints. Monitoring changes how people perceive oversight. Cybersecurity measures introduce friction, and friction creates resistance. This is why change management is not optional; it is a core success factor in cybersecurity governance.

Organizations commonly fail in cybersecurity not because they lack knowledge of what to do, but because they cannot implement change sustainably. People revert to old habits; business units bypass cybersecurity to get things done; leadership attention shifts; policies remain unread; controls degrade over time. Governance must therefore include communication, training, reinforcement, and an escalation logic that converts strategy into behavioural reality. Importantly, governance should not rely primarily on threats of discipline. Enforcement matters, but lasting cybersecurity outcomes are achieved when secure behaviours are made easier than insecure behaviours, when leadership models expectations, and when accountability is real.

Resistance to change should also be interpreted intelligently. Some resistance reflects normal human behaviour and operational pressure. Some resistance reflects a legitimate critique of poorly designed controls. Governance must be able to distinguish these situations and respond appropriately, which is precisely why human-centred governance is not a slogan but a design principle.

## Implementing a governance framework

[Organizations can implement cybersecurity governance using a variety of recognized frameworks and standards. In practice, many enterprises combine multiple sources, selecting what fits their context. Common references include COBIT for governance of enterprise IT, ISO/IEC 27001 for an information security management system (ISMS), NIST publications such as SP 800-53 Rev. 5 for security and privacy controls, ITIL 4 for service management, and sector-specific regulatory requirements.](https://www.iso.org/standard/27001?utm_source=chatgpt.com)

<span id="_Toc243635511" class="anchor"></span>Figure 12: implementing a governance framework

In this chapter, a useful anchor is ISO/IEC 38500, which provides guidance for the governance of IT for the organization. The most recent edition, ISO/IEC 38500:2024, emphasizes governance-level direction and oversight of IT use and aligns the standard with ISO 37000:2021, Governance of organizations. ISO/IEC 38500, originally written for IT governance,<https://www.iso.org/standard/81684.html?utm_source=chatgpt.com> maps well to cybersecurity governance because cybersecurity is inseparable from how IT is directed, acquired, operated, and evaluated.

ISO/IEC 38500 is commonly described in terms of six governance principles (often treated as domains): responsibility, strategy, acquisition, performance, conformance (compliance), and human behaviour (human-centred approach). These six principles come from the 2008 and 2015 editions; the 2024 edition adopts the broader set of eleven principles of ISO 37000, such as purpose, value generation, oversight, accountability, stakeholder engagement, and risk governance. The six classic principles remain widely taught and provide a practical framework that senior leadership can use to assess whether cybersecurity is governed rather than merely managed.

Responsibility means that decision rights are clear and that people understand their duties. Accountability means that someone must answer for outcomes and that consequences, positive or negative, are real. Confusing these concepts produces governance failure: organizations assign tasks without assigning ownership of results, or they assign accountability to people without giving them authority and resources.

Strategy means that cybersecurity and technology direction must align with business strategy. Cybersecurity cannot be governed effectively if it is not explicitly linked to what the organization is trying to achieve, what it depends on to succeed, and what failures would be unacceptable.

Acquisition means that technology and cybersecurity investments must be justified, transparent, and evaluated across their lifecycle. In cybersecurity, this includes not only purchasing tools but also ensuring that procurement addresses third-party risk, contractual cybersecurity requirements, operational support, and eventual decommissioning.

Performance means that cybersecurity and IT services must be monitored with evidence. Governance requires measurement: whether services meet requirements, whether controls work, and whether incident response and recovery capabilities perform under stress.

Conformance (compliance) means that the organization meets applicable laws, regulations, contractual obligations, and internal policies, and can demonstrate this through documentation, audits, and control testing. In cybersecurity governance, compliance is necessary but not sufficient: an organization can be compliant and still be insecure if it treats compliance as a checklist rather than as risk management.

Human behaviour means that governance must be designed around how people work. Controls must be usable, training must be continuous, and the organization must create incentives that encourage secure behaviour rather than workarounds. In cybersecurity, human-centred is not a soft concept; it is a risk-control principle because most real-world failures involve human decisions interacting with technical environments.

## The balanced cybersecurity scorecard as a governance tool

Cybersecurity governance requires measurement that leadership can understand and use. One effective approach is to adapt the Balanced Scorecard to cybersecurity. The Balanced Scorecard translates strategy into measurable objectives across multiple perspectives rather than relying exclusively on financial measures. It forces leadership to avoid a narrow interpretation of performance and instead observe how different dimensions interact.

In a cybersecurity context, the financial perspective focuses on the economic consequences of cyber risk and the value of investment decisions. This includes direct incident costs (response, recovery, remediation, legal and regulatory costs) as well as avoided losses and resilience value. It also includes investment discipline: whether cybersecurity spending is proportional to risk and aligned with strategic dependencies.

The customer (or stakeholder) perspective focuses on trust, service reliability, and perceived responsibility. Metrics here examine how cybersecurity affects customer confidence, contractual credibility, and stakeholder expectations. In many sectors, cybersecurity is part of the product: customers evaluate an organization not merely on features but on the reliability and safety of the service environment.

The internal process perspective focuses on operational cybersecurity performance: how well key processes function, such as vulnerability management, identity and access management, incident detection and response, change management, backup and recovery, third-party onboarding, and cybersecurity monitoring. Process measures are essential because failures often occur due to breakdowns in coordination rather than due to the absence of technology.

<span id="_Toc243635512" class="anchor"></span>Figure 13: the balanced cybersecurity scorecard

The learning and growth perspective focuses on capability-building: skills, training, organizational maturity, tooling adoption, and continuous improvement. Cybersecurity must evolve; therefore, governance must ensure that the organization can learn, adapt, and develop its workforce and operating model over time.

A key governance caution is that scorecards can become performative if metrics are chosen poorly. Mature cybersecurity scorecards combine leading indicators (signals that reduce risk before incidents occur, such as patch timeliness, MFA coverage, privileged access reviews completed, backup restoration testing success) with lagging indicators (signals that reflect realized outcomes, such as incident counts, mean time to detect, mean time to recover, and loss magnitude). The goal is not to generate impressive dashboards; it is to support decisions, trade-offs, and accountability. A sample Balanced Cybersecurity Scorecard with Leading and Lagging Indicators is presented in Appendix A.

## SMART governance metrics: making measurement credible

Metrics must be designed so that leadership can rely on them. In cybersecurity, this is difficult because many risk measures are uncertain and because organizations sometimes adopt vanity metrics that appear precise yet capture little about actual risk reduction. Governance must therefore insist on disciplined metric design.

<span id="_Toc243635513" class="anchor"></span>Figure 14: SMART governance metrics

The SMART acronym remains a practical test: metrics should be specific (clearly defined and unambiguous), measurable (based on observable data), achievable (feasible given resources and constraints), relevant (aligned with strategic objectives and material risks), and time-bound (measured within defined cycles to enable tracking). The primary governance value of SMART metrics is that they impose clarity and enforce accountability: if a metric cannot be defined, measured, and tracked within a defined timeframe, it cannot reliably drive improvement.

In cybersecurity governance, SMART metrics support several essential outcomes. They enable evidence-based decision-making about investment and prioritization. They offer a common language between technical teams and management. They support accountability by making performance visible. They enable trend analysis and continuous improvement by revealing gaps over time. They also support compliance by demonstrating ongoing control operations, not merely the existence of policies.

A final caution is necessary: metrics do not replace judgment. They support judgment. Governance means leadership remains responsible for decisions even when measurement is imperfect. This is why governance insists on transparency about assumptions and limitations, and why metrics must be interpreted in context rather than used mechanically.

### Illustrative scenario: a Québec building-products manufacturer formalizes cybersecurity governance

Matériaux Kamouraska (a fictional organization) is a family-controlled Québec manufacturer of engineered wood and building products, with about 1,400 employees, four plants in Bas-Saint-Laurent and Chaudière-Appalaches, and a growing online ordering channel for contractors. Its nine-member board includes four independent directors. Until recently, cybersecurity was managed by an IT security manager who reported to the IT director, who in turn reported to the chief financial officer. The board heard about cybersecurity once a year, as one slide in the IT update, and the slide was always green.

Two events in the same quarter changed that. First, a large retail customer added cybersecurity governance questions to its supplier assessment, asking who at board level oversees cyber risk and how. Second, a fraudster using a compromised supplier mailbox nearly redirected a large supplier payment; an alert accounts payable clerk stopped it only because something "felt wrong". When the chair asked who was accountable for that control, three executives each named someone else. The board then commissioned a governance review, using the Evaluate–Direct–Monitor logic of ISO/IEC 38500:2024 and the COBIT 2019 distinction between governance and management as its reference points.

The review produced five decisions, each with a trade-off the board accepted explicitly.

- **Board oversight:** rather than create a separate technology committee, the board renamed its audit committee the audit and risk committee, wrote cybersecurity oversight into its charter, and recruited an independent director with technology risk experience. Because the committee's agenda was already full, it scheduled two dedicated cybersecurity sessions a year. The full board kept the approval of risk appetite and of any risk accepted above it.

- **Risk appetite statement:** the board approved a short statement expressed in business terms. It set a very low appetite for cyber events that stop a critical production line for more than 48 hours or that compromise employee or customer personal information, and a moderate appetite for risk in the online channel, provided that new features pass a security review before launch. Tolerances were attached to each statement so that management would know when a breach of appetite had to be escalated. The scenarios that informed these thresholds were analyzed with CyberRiskGuardian, which allowed directors to discuss plausible losses rather than abstract ratings.

- **Decision rights:** a one-page decision-rights matrix now states who proposes, who decides, and who must be informed for policy approval, budget, risk acceptance, and exceptions. Plant vice-presidents can accept residual risk for their operations within tolerance; anything beyond tolerance goes to the executive committee, and anything outside appetite goes to the board committee. Every exception carries an owner and an expiry date. The CISO advises on all these decisions but does not accept business risk on anyone's behalf.

- **CISO reporting line:** the IT security manager became the chief information security officer, reporting to the chief executive officer, with direct and unfiltered access to the committee chair and an annual in camera session without management present. The IT director initially objected that this would separate security from operations. The compromise was a joint annual plan and shared operational objectives, while keeping the CISO's assessment of IT controls independent of the people who run them.

- **Personal information accountability:** because Law 25 makes the person exercising the highest authority responsible for the protection of personal information by default, the CEO delegated that function in writing to the general counsel, and the delegation, the role description, and the reporting to the committee were documented.

Measurement changed as well. The old forty-indicator technical dashboard was replaced with a balanced cybersecurity scorecard of twelve metrics across the four perspectives, each tested against the SMART criteria and assigned an owner, a target, and a reporting cycle. Examples included the percentage of supplier banking-detail changes verified by an out-of-band call each month (target: all of them), the percentage of critical plant systems whose restoration had been tested within the past twelve months, the number of risk exceptions past their expiry date, the time taken to answer customer security assessments against contractual deadlines, and the proportion of plant supervisors who had taken part in an incident exercise. Each metric was explicitly linked to a statement in the risk appetite, so that a red result meant something specific.

The first report under the new arrangements was not flattering. Restoration tests had failed for two plant systems, and six exceptions were past expiry. **This was the governance working, not failing.** The committee asked for a funded remediation plan, the executive committee approved it within the month, and the exceptions were either renewed with compensating controls, closed, or escalated. A year later, the retail customer's assessment was answered with documents that already existed, and the chair could name, for each major cyber risk, who owned it and who had decided to accept what remained.

### Questions managers and boards should ask

The scenario suggests questions that executives and directors can use to test whether cybersecurity is governed rather than merely managed:

- **Oversight:** Which board committee oversees cybersecurity, is that oversight written into its charter, and does it receive enough agenda time to challenge management rather than simply receive reports?

- **Appetite:** Has the board approved a cybersecurity risk appetite statement in business terms, with tolerances that tell management when an issue must be escalated?

- **Decision rights:** Who can approve a policy, a budget, a risk acceptance, or an exception, and is this written down in a form that executives would describe the same way?

- **Risk ownership:** For each major cyber risk scenario, which business executive owns the risk, and who decided to accept the residual risk that remains?

- **Independence:** To whom does the CISO report, could that line create a conflict of interest, and does the CISO have direct access to the board or its committee when needed?

- **Accountability by law:** Who is designated as responsible for the protection of personal information under Law 25, and do regulatory regimes that apply to us (such as OSFI B-13, NIS2, or SEC disclosure rules) require us to demonstrate board oversight?

- **Governance and management:** Is the board evaluating, directing, and monitoring, or is it being drawn into operational decisions that management should own?

- **Measurement:** Does each metric on our scorecard pass the SMART test, have an owner and a target, and link to a stated risk or objective?

- **Leading and lagging indicators:** Does the scorecard balance indicators that predict trouble with indicators that report outcomes, or does it only describe the past?

- **Bad news:** When was the last time a cybersecurity report to the board showed a target missed, and what decision followed?

- **Exceptions:** How many risk exceptions are open, how many are past expiry, and who reviews them?

## Closing perspective: governance as the foundation for cybersecurity outcomes

Cybersecurity governance is the structured approach to a strategic problem: how to direct and control cybersecurity in ways that are aligned with business objectives, ethically defensible, operationally realistic, and resilient to evolving threats. It is not the same as buying tools, nor is it the same as writing policies that no one follows. It is the discipline of defining direction, assigning responsibility and accountability, managing risk, demonstrating compliance, credibly measuring performance, and continuously improving.

Chapter 3 turns to compliance, the third element of the GRC continuum. Chapters 4 to 6 then deepen the risk management methods that make governance actionable, and later chapters develop practical approaches to culture, leadership, process design, and control selection—so that cybersecurity becomes governable, measurable, and sustainable rather than reactive and episodic.

## Chapter summary

Chapter 2 explains how organizations direct and control cybersecurity so that it supports performance, resilience, and trust. Governance answers enduring questions: who decides, on what basis, who is accountable, and how decisions stay aligned with mission, values, and obligations. The chapter distinguishes governance from management, frames governance as applied ethics, and positions cybersecurity governance within enterprise and IT governance and the GRC continuum. It sets out what governance must achieve, why change management is essential, and how ISO/IEC 38500 principles, a balanced cybersecurity scorecard, and SMART metrics make governance concrete. A Québec manufacturer's scenario shows that governance means explicit decisions with accepted trade-offs, and that bad news reported honestly is governance working.

### Key points to retain

- Cybersecurity governance is the set of structures, decision rights, processes, and accountability mechanisms through which an organization directs and controls cybersecurity.

- Good governance rests on transparency, accountability, participation, and responsiveness, each with direct operational implications.

- Governance evaluates, directs, and monitors, while management plans, builds, runs, and monitors; failures follow when the boundary is blurred in either direction.

- Cybersecurity governance is applied ethics that combines normative guidance about what ought to be done with formal mechanisms that define what must be done and how it is verified.

- Cybersecurity governance is derived from enterprise governance, intersects with but is not identical to IT governance, and is one element of the Governance-Risk-Compliance continuum.

- Governance must set strategic direction, establish decision rights and accountability, ensure rational resource allocation, ensure oversight and measurement, and ensure compliance and defensibility.

- Regulators such as NIS2, the SEC disclosure rules, OSFI B-13, and Québec's Law 25 increasingly assign cybersecurity accountability to boards and senior management, so oversight must be demonstrable.

- Governance is enabled by culture, leadership, processes, and mechanisms; buying controls without governance leaves risk unmanaged.

- Change management is a core success factor because controls create friction, and governance must distinguish normal resistance from legitimate critique of poorly designed controls.

- The six classic ISO/IEC 38500 principles (responsibility, strategy, acquisition, performance, conformance, and human behaviour) help leaders test whether cybersecurity is governed rather than merely managed.

- A balanced cybersecurity scorecard combines financial, stakeholder, internal process, and learning and growth perspectives, with both leading and lagging indicators.

- SMART metrics impose clarity and accountability, but metrics support judgment rather than replace it, and honest red results show governance working.

### Key terms

- **Cybersecurity governance** — The structures, decision rights, processes, and accountability mechanisms through which an organization directs and controls cybersecurity to support performance, resilience, and trust.

- **Governance** — The structures, mechanisms, and decision rights through which authority is exercised, resources are allocated, and performance is overseen.

- **Evaluate-Direct-Monitor (EDM)** — The governance cycle in which the board and executive leadership evaluate stakeholder needs and risks, direct policies and strategies, and monitor compliance and performance.

- **Plan-Build-Run-Monitor (PBRM)** — The management cycle through which day-to-day planning, building, and operating of systems achieve the enterprise's objectives within the direction set by governance.

- **Enterprise governance** — The overall framework for direction and oversight across the organization, defining how power is exercised among the board, executives, and stakeholders.

- **IT governance** — The application of governance principles to information and business technologies so that investments and operations support objectives, create value, and stay within acceptable risk.

- **GRC continuum** — The view that governance sets direction and accountability, risk management identifies and treats risks, and compliance ensures and verifies adherence to obligations.

- **Govern function** — The sixth function added to the NIST Cybersecurity Framework 2.0 in February 2024 to emphasize the centrality of governance in cybersecurity risk management.

- **Responsibility** — Having clear decision rights and understanding one's duties, as distinct from answering for outcomes.

- **Accountability** — The obligation to answer for outcomes, with real consequences, which must be matched by authority and resources.

- **Balanced cybersecurity scorecard** — An adaptation of the Balanced Scorecard that translates cybersecurity strategy into measurable objectives across financial, stakeholder, internal process, and learning and growth perspectives.

- **Leading indicator** — A signal that reduces risk before incidents occur, such as patch timeliness or backup restoration testing success.

- **Lagging indicator** — A signal that reflects realized outcomes, such as incident counts, mean time to recover, or loss magnitude.

- **SMART metrics** — Metrics that are specific, measurable, achievable, relevant, and time-bound, so that they can reliably drive improvement and accountability.

- **Risk appetite statement** — A board-approved statement, in business terms, of the level of risk the organization is willing to take, with tolerances that signal when issues must be escalated.

## Review questions

### Essay questions

1.  Analyze why cybersecurity failures often occur when the boundary between governance and management is blurred. Using the Evaluate-Direct-Monitor and Plan-Build-Run-Monitor cycles, describe what goes wrong in each direction and recommend how boards and executives can keep the two connected.

2.  Evaluate the claim that cybersecurity governance is a form of applied ethics. Explain why ethical norms are necessary but not sufficient, and illustrate your argument with cybersecurity decisions such as employee monitoring, incident disclosure, or resource allocation under constraints.

3.  Choose an organization you know and assess its cybersecurity governance against the six classic principles of ISO/IEC 38500. For each principle, identify one strength or weakness and recommend a governance improvement.

4.  Design a balanced cybersecurity scorecard of eight metrics for an organization of your choice, two per perspective. Justify each metric against the SMART criteria, classify it as leading or lagging, and explain how it links to a stated risk or objective.

5.  Using the Matériaux Kamouraska scenario, evaluate the five governance decisions the board made. For each, explain the trade-off accepted, the problem it addressed, and whether you would have decided differently, and explain why the first unflattering report was a sign that governance was working.

### Multiple-choice questions

Choose the best answer. Answers are listed in the appendix "Answers to multiple-choice questions".

**1.** According to COBIT 2019 and ISO/IEC 38500, which cycle describes how governance exercises its role?

> A. Evaluate, Direct, Monitor
>
> B. Plan, Do, Check, Act
>
> C. Plan, Build, Run, Monitor
>
> D. Identify, Protect, Detect

**2.** Which activity best reflects management rather than governance in the chapter's distinction?

> A. Approving the organization's cybersecurity risk appetite
>
> B. Directing policies and strategies for cyber risk
>
> C. Planning, building, and operating systems day to day
>
> D. Evaluating stakeholder needs and strategic risks

**3.** What does the chapter say typically happens when senior leadership treats cybersecurity as a purely technical matter for IT?

> A. Technical controls become too expensive to operate
>
> B. Strategic questions about appetite, resources, and trade-offs remain unanswered
>
> C. The board becomes overly involved in operational decisions
>
> D. Compliance reporting becomes more frequent and detailed

**4.** Which statement best describes the governance principle of participation?

> A. Stakeholders can see what is being done and why decisions are made
>
> B. Decisions have identified owners whose results are monitored
>
> C. The organization adapts quickly as threats and incidents evolve
>
> D. Relevant cross-functional expertise is included in decision-making

**5.** Why does the chapter argue that ethical norms alone are not sufficient for cybersecurity governance?

> A. Most employees are assumed to act maliciously when unobserved
>
> B. Complex systems create opportunities for misuse and incentives can drift
>
> C. Ethical norms never change and therefore become outdated
>
> D. Laws already cover every cybersecurity situation that can arise

**6.** How does the chapter position cybersecurity governance relative to enterprise and IT governance?

> A. Derived from enterprise governance and intersecting with, but not identical to, IT governance
>
> B. Independent of enterprise governance and identical to IT governance
>
> C. A subset of IT governance with no links to legal or human resources
>
> D. A replacement for enterprise governance in digital organizations

**7.** Which function did NIST add to the Cybersecurity Framework in version 2.0, released in February 2024?

> A. Recover
>
> B. Respond
>
> C. Identify
>
> D. Govern

**8.** A board requires every cybersecurity investment to be linked to risk reduction, resilience requirements, or legal obligations rather than vendor pressure. Which governance outcome does this serve?

> A. Setting strategic direction
>
> B. Ensuring compliance and defensibility
>
> C. Ensuring rational resource allocation
>
> D. Establishing decision rights

**9.** What does the EU NIS2 Directive require of the management bodies of essential and important entities, as described in the chapter?

> A. Annual disclosure of board oversight under Regulation S-K Item 106
>
> B. Delegating cybersecurity oversight entirely to the CISO
>
> C. Approving and overseeing risk-management measures, following training, and potential liability
>
> D. Reporting technology risk accountability to OSFI each quarter

**10.** Under Québec's Law 25, who is responsible by default for the protection of personal information in an enterprise?

> A. The chief information security officer
>
> B. The person exercising the highest authority
>
> C. The general counsel or legal department
>
> D. The chair of the board's audit committee

**11.** Of the four elements that enable governance, which one does the chapter describe as shaping behaviour when no one is watching?

> A. Culture
>
> B. Leadership
>
> C. Processes
>
> D. Mechanisms

**12.** Business units push back on a new approval step, arguing it delays urgent orders. How does the chapter advise governance to respond?

> A. Enforce the control strictly, since discipline produces lasting outcomes
>
> B. Withdraw the control, since resistance shows it is unnecessary
>
> C. Delegate the decision to IT, since controls are a technical matter
>
> D. Determine whether this is normal resistance or a legitimate critique of the control's design

**13.** With which standard does ISO/IEC 38500:2024 align, adopting its broader set of eleven principles?

> A. ISO 37000:2021, Governance of organizations
>
> B. ISO/IEC 27001:2022, information security management
>
> C. ISO 37301:2021, compliance management systems
>
> D. ITIL 4, the service management framework

**14.** A manager is held answerable for cybersecurity outcomes in her division but receives no authority or budget to change anything. Which governance failure does this illustrate?

> A. Strategy that is not aligned with business objectives
>
> B. Acquisition decisions lacking lifecycle evaluation
>
> C. Accountability assigned without authority and resources
>
> D. Conformance treated as a checklist exercise

**15.** What does the chapter's discussion of the conformance principle imply about compliance and security?

> A. Compliance with standards guarantees an acceptable security posture
>
> B. Compliance is optional when risk management is mature
>
> C. Compliance belongs to legal teams rather than to governance
>
> D. An organization can be compliant and still insecure if compliance is a checklist

**16.** Which of the following is a leading indicator in a balanced cybersecurity scorecard?

> A. Number of incidents in the quarter
>
> B. Backup restoration testing success rate
>
> C. Mean time to recover from incidents
>
> D. Total magnitude of losses from incidents

**17.** In the balanced cybersecurity scorecard, which perspective covers vulnerability management, identity and access management, and incident detection and response?

> A. Financial perspective
>
> B. Customer or stakeholder perspective
>
> C. Learning and growth perspective
>
> D. Internal process perspective

**18.** A proposed board metric reads “Improve security awareness.” Which SMART criteria does it most clearly fail?

> A. Specific, measurable, and time-bound
>
> B. Achievable and relevant only
>
> C. Relevant and time-bound only
>
> D. Achievable and measurable only

**19.** In the Matériaux Kamouraska scenario, what role does the new CISO play in risk acceptance decisions?

> A. Accepts all residual risks on behalf of plant vice-presidents
>
> B. Advises on decisions but does not accept business risk for others
>
> C. Approves only those risks that fall outside board appetite
>
> D. Has no role, since risk decisions belong to the executive committee

**20.** The first report under Kamouraska's new scorecard showed two failed restoration tests and six expired exceptions. How does the chapter interpret this?

> A. As evidence that the scorecard metrics were poorly designed
>
> B. As a reason to return to the previous technical dashboard
>
> C. As governance working, prompting a funded remediation plan
>
> D. As a failure of the CISO that required a change of reporting line
