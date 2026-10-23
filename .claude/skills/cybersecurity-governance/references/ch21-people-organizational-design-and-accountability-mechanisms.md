<!-- Léger, M.-A. (2026). Introduction to Cybersecurity Governance for Business Technology Management (3rd ed., v2.51). Chapter 21: People, Organizational Design, and Accountability Mechanisms. CC BY-NC 4.0 -->

# Chapter 21: People, Organizational Design, and Accountability Mechanisms

## Learning outcomes

After completing this chapter, you should be able to:

1.  Explain why governance is ultimately the deliberate design of accountability, and distinguish governance, enablement, and operations work within a cybersecurity operating model.

2.  Apply the three lines model and board oversight principles to locate cybersecurity roles, committee mandates, and director competence obligations under NIS2, DORA, SEC rules, OSFI B-13, and Law 25.

3.  Design a decision-rights scheme for high-consequence cyber decisions, tiered by impact and linked to risk appetite, with escalation and emergency authority.

4.  Construct and evaluate a RACI that avoids gaps, overlaps, and diffusion of responsibility, with exactly one accountable role per activity.

5.  Compare CISO reporting lines and the roles of risk owners, BISOs, and partner functions, and justify safeguards for smaller organizations that combine roles.

6.  Evaluate the personal liability cases involving security leaders and derive governance measures that align authority with accountability.

7.  Recommend workforce, security champion, human risk management, just culture, and incentive measures that build competence and change behaviour.

8.  Analyze recurring accountability failure patterns and propose a staged implementation blueprint with exceptions, assurance, learning loops, and funding as accountability mechanisms.

Organizations rarely fail at cybersecurity because nobody knows which controls exist. They fail because controls, decisions, and responsibilities are not organized in a way that makes them executable, sustained, and defensible. Policies are approved but never operationalized. Security teams identify risks, but business leaders do not own them. Technical teams implement controls, yet exceptions proliferate without anyone being accountable for them. Training is delivered every year, but behaviour does not change. When an incident finally occurs, everyone asks who was responsible—and discovers that responsibility was ambiguous all along.

For a book written for managers, this is not a secondary or "soft" topic. It is the central governance topic. Chapter 2 defined governance as direction, control, and accountability; this chapter argues that governance is, in the end, the deliberate design of accountability. Someone must be empowered to decide, someone must be obligated to act, someone must fund the work, someone must accept the residual risk, and there must be evidence that each of these obligations is being met. A control framework, a risk assessment, or an incident plan only produces value when these people-related questions have answers that hold under pressure.

The preceding twenty chapters have repeatedly pointed toward this conclusion. Risk management (Chapter 4) depends on named risk owners. Scenario-based risk assessment (Chapters 5 and 6) depends on business leaders who can judge impact and accept residual risk. Control frameworks (Chapter 9) assign responsibilities that someone must actually perform. Identity and access management (Chapter 10), incident management and resilience (Chapter 12), cloud security (Chapter 15), privacy (Chapter 16), investment governance (Chapter 17), AI (Chapter 18), and third-party governance (Chapter 19) all fail in the same way when ownership is unclear: work falls into the gaps between teams, and decisions are made by whoever happens to hold the keyboard.

This chapter therefore treats people, organizational design, and accountability mechanisms as an explicit governance domain, one that the NIST Cybersecurity Framework (CSF) 2.0 recognizes in its **Govern** function through the category on roles, responsibilities, and authorities, and that ISO/IEC 27001:2022 addresses through its requirements on leadership, organizational roles, competence, and awareness. The emphasis is not on human resources administration but on the managerial design choices that determine whether cybersecurity is governable: who decides, who acts, who is answerable, and how the organization knows.

The chapter begins with the operating-model lens and the three lines model, then examines board oversight and director competence. It moves to decision rights, RACI and role clarity, and the roles that a mature operating model should contain, including CISO reporting lines and business information security officers. It considers the debate over the personal liability of security leaders, workforce frameworks and the skills shortage, security champions, human risk management beyond awareness, just culture, incentives, and the mechanisms that make accountability enforceable. It closes with recurring failure patterns, a staged implementation blueprint with an illustrative scenario, questions for boards, and—because this is the final chapter—a short synthesis of the book as a whole.

## Cybersecurity as an operating model: the managerial lens

An **operating model** describes how work actually gets done in an organization: who performs which activities, how decisions are made, how work is funded, how conflicts are resolved, and how outcomes are measured. Every organization has a cybersecurity operating model, whether or not anyone has written it down. The question is whether that model was designed deliberately or emerged by accident from budget lines, reporting relationships, and the personalities of the people involved.

### Two realities the operating model must bridge

In cybersecurity, the operating model must bridge two realities that pull in different directions. The first is that many controls are technical and are therefore implemented by IT, engineering, security operations, and cloud platform teams. The second is that the consequences and trade-offs of cybersecurity decisions are business consequences: service availability, customer trust, regulatory exposure, operational resilience, cost, and strategic risk. If the operating model collapses cybersecurity into technical implementation, business leaders lose visibility into material risk and become spectators to decisions that affect the survival of the organization.

The opposite error is less common but equally damaging. When cybersecurity is treated only as a matter of policy and oversight, without operational capacity, the organization produces well-written documents and dashboards that describe an environment nobody is actually maintaining. A designed operating model avoids both errors by making explicit the link between the people who decide, the people who build and operate, and the people who verify.

### Governance, enablement, and operations

A mature cybersecurity operating model is explicit about the separation of three kinds of work, even when the same team performs more than one of them.

- **Governance** sets policy, defines risk appetite and tolerance, establishes minimum requirements, approves exceptions, assigns ownership, and ensures that evidence exists. Its output is direction and decisions.

- **Enablement** provides architecture, standards, reusable patterns, tooling, training, and advisory capacity so that business and technical teams can build securely without constant reinvention. Its output is capability in other teams.

- **Operations** monitors, detects, responds, maintains controls, manages vulnerabilities, and continuously improves the environment, as described in Chapters 7 and 8. Its output is control performance.

Many organizations perform all three in some form, but they blur responsibilities and decision rights between them. A security architect who also approves her own exceptions, a security operations centre that sets its own risk thresholds, or a governance team that has no enablement capacity and can only say "no" are symptoms of this blurring. The consequence is predictable: security becomes reactive, exceptions multiply, and accountability becomes political rather than structural.

### Centralized, federated, and hybrid designs

Organizations choose, explicitly or implicitly, among three broad structural patterns. In a **centralized** model, a single security function performs governance, enablement, and most operations for the whole enterprise. This model offers consistency, clear standards, and efficient use of scarce specialists, and it suits small and mid-sized organizations or those with a single dominant line of business. Its weakness is distance from the business: a central team can become a bottleneck and may not understand the operational realities of each unit.

In a **federated** model, business units or subsidiaries have their own security staff and considerable autonomy, with a thin central function that sets group-wide policy. This pattern is common in conglomerates, universities, health networks, and organizations built through acquisition. It brings security close to the business but risks inconsistency, duplicated tooling, and uneven maturity. A **hybrid** model, now common in large organizations, centralizes governance, architecture standards, and security operations while embedding business-aligned security roles—often called business information security officers, discussed later in this chapter—in each major unit. The right choice depends on size, regulatory exposure, diversity of business lines, and how much variation in risk appetite the organization is prepared to tolerate across its units.

### The three lines model applied to cybersecurity

Chapter 1 introduced the Institute of Internal Auditors' **Three Lines Model** (the 2020 update of the earlier "three lines of defence") and its extension to five lines of accountability, which names executive management and the board explicitly. This section uses the model to locate cybersecurity roles within enterprise governance. The model distinguishes a governing body that is accountable to stakeholders for oversight; management, which takes actions to achieve objectives through first-line roles that provide products and services and manage risk, and second-line roles that provide expertise, support, monitoring, and challenge on risk matters; and an independent internal audit function, the third line, that provides assurance and advice to the governing body. The 2020 update deliberately softened the "defence" language and emphasized collaboration, but the underlying principle remains: those who own and manage risk, those who challenge and monitor it, and those who independently assure it should be distinguishable.

Applied to cybersecurity, the first line includes business and process owners who own risk, together with IT, engineering, and security operations teams that operate controls. The second line typically includes the enterprise risk management function, compliance, privacy, and—in many organizations—a CISO office that sets standards and provides oversight rather than operating controls. The third line is internal audit, sometimes supplemented by external assurance such as ISO/IEC 27001 certification audits or SOC 2 reports. One recurring design question is where the CISO sits: a CISO who runs the security operations centre is clearly first line (the placement Chapter 1 describes as common), while a CISO who sets policy and challenges IT is acting as second line. Many CISOs do both, which is acceptable only when the organization recognizes the tension and ensures that independent assurance comes from somewhere else. The figure below summarizes how these layers fit together beneath the board.

<span id="_Toc243635583" class="anchor"></span>Figure 84: Three lines and board operating model for cybersecurity

| Every organization has a cybersecurity operating model. **The governance question is whether it was designed deliberately, with decision rights and accountability made explicit, or inherited by accident.** |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## Board oversight and director competence

Boards of directors do not manage cybersecurity, but they are accountable for overseeing it. Over the past decade, regulators, courts, investors, and insurers have converged on the view that cyber risk is an enterprise risk like any other, and that directors who ignore it are failing in their duty of oversight. For business technology managers, understanding what boards need—and what they must not be asked to do—is essential, because managers prepare much of what boards see.

### What the board owns and what it does not

The board's role is to set the tone, approve the organization's risk appetite, ensure that management has a credible program to manage cyber risk within that appetite, and hold management accountable for results. In practice, this means approving the risk appetite statement for cyber and technology risk (Chapter 4), understanding the organization's most material cyber risk scenarios and their potential impacts, satisfying itself that roles and accountabilities are clear, ensuring that investment is commensurate with risk (Chapter 17), and reviewing how the organization would respond to and recover from a severe incident (Chapter 12).

The board does not choose security products, approve individual firewall rules, or review vulnerability scan results. When boards drift into operational detail, two problems follow. Directors lose sight of the strategic questions they are uniquely positioned to ask, and management loses clarity about who is accountable for operational decisions. The healthier pattern is a board that asks probing questions about outcomes, risk trends, and evidence, while leaving the means to management.

### Committee structures and access to expertise

Boards organize cyber oversight in different ways. Some assign it to the audit committee, which already oversees internal controls and financial reporting risk. Others assign it to a risk committee, which is common in financial institutions, or create a dedicated technology or cyber committee. Some keep it at the full board. Each option has merits. The audit committee brings discipline around controls and assurance, but it is often already overloaded. A risk committee places cyber risk alongside other enterprise risks, which supports integrated appetite decisions. A dedicated committee brings focus but can isolate cyber from strategy. Whatever the choice, the mandate should be written into the committee charter, the full board should receive periodic reporting on material cyber risk, and the boundaries between committees should be clear.

Access to expertise is the second structural question. The fifth edition of the *Director's Handbook on Cyber-Risk Oversight*, published in 2026 by the National Association of Corporate Directors (NACD) and the Internet Security Alliance, frames this as one of six principles: boards should establish oversight structures and ensure access to expertise. The other principles call on boards to treat cybersecurity as a strategic risk, monitor legal and disclosure implications, adopt an enterprise framework for managing cyber risk, guide risk measurement and reporting, and encourage systemic resilience and collaboration. Expertise can come from directors with relevant backgrounds, from advisory panels, from independent external advisers who brief the board directly, or from regular deep-dive sessions with the CISO. What matters is that directors can evaluate management's claims rather than simply receive them.

### Director competence: regulators raise the bar

Several regulatory regimes now make board engagement on cybersecurity an explicit obligation rather than a matter of good practice. In the European Union, Article 20 of the NIS2 Directive (Directive (EU) 2022/2555) requires member states to ensure that the management bodies of essential and important entities approve the cybersecurity risk-management measures required under Article 21, oversee their implementation, and can be held liable for infringements. The same article requires members of management bodies to follow training and encourages entities to offer similar training to employees regularly. The Digital Operational Resilience Act (DORA, Regulation (EU) 2022/2554), which has applied to EU financial entities since January 17, 2025, goes further for the financial sector: Article 5 states that the management body "shall define, approve, oversee and be responsible for the implementation of all arrangements related to the ICT risk management framework," must allocate budget for digital operational resilience and ICT security awareness and training, and that its members "shall actively keep up to date with sufficient knowledge and skills to understand and assess ICT risk" and its impact on operations.

In the United States, the Securities and Exchange Commission's cybersecurity disclosure rules adopted in 2023 require public companies to describe annually, under Item 106 of Regulation S-K, the board's oversight of cybersecurity risk (including any responsible committee and how the board is informed) and management's role and expertise in assessing and managing material cybersecurity risks. The final rule did not require companies to disclose whether directors have cybersecurity expertise, but the disclosure obligation itself has pushed many boards to examine and document how they oversee the topic. In Canada, OSFI Guideline B-13 on technology and cyber risk management, in effect since January 1, 2024, expects federally regulated financial institutions to establish clear accountability, leadership, and organizational structures for technology and cyber risk, while Québec's Law 25 makes the person exercising the highest authority in an enterprise the person in charge of the protection of personal information by default, unless that function is delegated in writing. The common thread is that **accountability for cybersecurity is increasingly assigned by law to the top of the organization, and directors are expected to be competent enough to exercise it**.

### Making board reporting decision-useful

Most board cyber reporting fails for a simple reason: it reports activity rather than risk. Slides listing the number of blocked attacks, phishing simulations sent, or patches deployed do not tell directors whether the organization is within its risk appetite. Decision-useful reporting connects the organization's top risk scenarios (Chapters 5 and 6) to trends in key risk indicators, the status of major treatment initiatives, material exceptions and accepted residual risks, significant incidents and lessons learned, and independent assurance results. It distinguishes between what management knows, what it believes, and what has been independently verified.

Good board reporting also makes requests explicit. If management needs the board to approve an appetite change, fund a multi-year program, or accept a specific residual risk, the report should say so clearly and present options with their consequences. A board that is never asked to decide anything about cybersecurity is, in effect, not overseeing it.

## Decision rights: who decides what, and at what level

Cybersecurity governance becomes practical when decision rights are explicit. **Decision rights** define which roles have the authority to decide, approve, or override in specific domains, and at what threshold a decision must be escalated. Without explicit decision rights, organizations rely on informal influence. Informal influence works well enough in calm periods; under pressure—a looming product launch, a ransomware attack, a regulator's deadline—ambiguity produces either paralysis or uncontrolled unilateral action.

### Identifying the decisions that matter

A manager-oriented way to structure decision rights is to identify a small set of recurring cyber decisions that genuinely matter and assign each one explicitly. Trying to assign decision rights for everything produces an unreadable matrix; focusing on a dozen high-consequence decisions produces clarity where it counts. Typical candidates include the following.

- **Risk acceptance** for residual risks above a defined threshold, especially for high-impact systems and data.

- **Exceptions** to minimum security requirements, including their duration and compensating controls.

- **Architecture patterns** with material risk implications, such as internet exposure, cloud connectivity, network segmentation, and identity federation (Chapter 13).

- **Third-party onboarding** for high-risk suppliers and cloud services (Chapter 19).

- **Data classification, retention, and cross-border transfer** rules (Chapter 16).

- **Incident declaration and crisis activation**, including the authority to isolate systems or shut down services (Chapter 12).

- **Security investment priorities** and the trade-offs between them (Chapter 17).

- **Adoption of AI systems** that process sensitive data or make consequential decisions (Chapter 18).

For each decision, the organization should be able to state who proposes, who must be consulted, who decides, who can escalate, and what evidence must accompany the decision. This is the decision-rights counterpart of the evidence discipline emphasized throughout the book: a **defensible decision** is one whose authority, rationale, and supporting analysis can be shown afterwards.

### Tiering decision rights by impact

Decision rights should be tiered by impact. Low-impact decisions should be delegated and standardized through policy, so that teams can proceed without asking permission when they follow approved patterns. Medium-impact decisions can be made by accountable managers with security consultation. High-impact decisions should be elevated to senior leadership, with documented rationale and visibility to the oversight function. The thresholds should be derived from the organization's risk appetite and tolerance statements, so that the same scale used to rate risk scenarios also determines who may accept them.

Tiering serves two purposes at once. It protects senior leaders' attention for decisions that genuinely require it, and it gives delivery teams autonomy within guardrails, which reduces the temptation to bypass governance. A tiering scheme that sends every exception to the executive committee will be ignored; one that lets any team lead accept any risk will be abused.

### The structural error: infrastructure control without outcome ownership

A common and damaging failure pattern is to place strategic security decisions in the hands of teams that control infrastructure but do not own business outcomes. This happens when IT decides what level of risk is acceptable for a customer-facing service without the involvement of the business owner of that service, or when the security team quietly grants an exception because the project manager asked politely. The decision may even be technically sound, but it is not legitimate, because the person who will bear the consequences did not make it.

Governance maturity is demonstrated when business leaders own risk decisions for their services and data, while security and IT provide the expertise, standards, and controls that shape those decisions. This does not mean that business leaders decide technical matters; it means that when a technical choice changes the organization's exposure, the person accountable for the affected outcome must understand and accept that change.

### Escalation, tie-breaking, and emergency authority

Decision rights must also specify what happens when parties disagree. A product owner may wish to launch with a known weakness that the CISO considers unacceptable. A well-designed model does not give either party an absolute veto; it provides an escalation path to a level that owns both outcomes—for example, the executive responsible for the business line, or an executive risk committee—with a defined time frame and a documented outcome. The CISO's essential right is not to veto but to escalate, and to have a dissent recorded when risk is accepted against advice.

Emergency authority deserves separate attention. During a serious incident, there may be minutes, not days, to decide whether to disconnect a plant from the network or take a payment system offline. These authorities should be pre-delegated, documented in incident and crisis plans, and rehearsed in exercises. The figure below summarizes six decision domains, with the typical decider, the escalation level, and the evidence that should accompany each.

<span id="_Toc243635584" class="anchor"></span>Figure 85: Decision rights grid for cybersecurity governance

| The CISO's most important decision right is not a veto. **It is the right to escalate a risk to the level that owns the outcome, and to have dissent recorded when that risk is accepted.** |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## RACI and role clarity: preventing gaps, overlaps, and diffusion of responsibility

RACI models exist because cybersecurity work touches many roles and systems. Building on the distinction between responsibility and accountability drawn in Chapter 2, a **RACI** assigns, for each activity or decision, who is *Responsible* (does the work), who is *Accountable* (answers for the outcome and has final authority), who is *Consulted* (provides input before the decision or action), and who is *Informed* (is notified afterwards). The purpose of a RACI is not to produce a poster for the security team's wall. It is to prevent recurring operational failures that are otherwise almost guaranteed in a complex organization.

### Three failures a RACI prevents

The first failure is the **gap**: no one performs the work because each team believes it belongs to another. Access reviews for a legacy application, log retention for a cloud service, or removal of a former vendor's credentials are classic gaps. The second failure is the **overlap**: several teams perform partial and inconsistent versions of the same work, often with different tools and different standards, which wastes resources and produces contradictory evidence. The third failure is the **diffusion of responsibility**: everyone is nominally involved, so no one feels personally answerable. Diffusion is the most insidious, because it is invisible until an incident review asks who should have noticed.

A sound RACI follows two simple rules. Each activity has exactly one Accountable role, and the Accountable role has the authority and resources to fulfil the obligation. When accountability is placed too low in the organization, it becomes powerless; when it is placed too high without operational ownership beneath it, it becomes symbolic.

### Five recurring domains

A practical RACI for cybersecurity governance should cover at least five recurring domains. The first is risk management and exception handling: who identifies risk, who owns remediation, who accepts residual risk, and who documents decisions. The second is control ownership: who is responsible for implementing and operating key controls such as identity and access management, privileged access, vulnerability management, secure configuration baselines (Chapter 14), logging and monitoring, and backup and recovery. The third is project and change integration: who ensures that security requirements are included at the design stage, who approves security architecture, and who validates readiness before deployment. The fourth is third-party security: who defines minimum requirements, who performs due diligence, who signs contracts, and who monitors ongoing assurance. The fifth is incident management: who declares an incident, who leads operational response, who leads executive crisis management, who manages legal and privacy obligations, and who authorizes external communications.

For each domain, clarity is achieved when accountability is named at the right level. Business owners should be accountable for the risk posture of their products and processes, technical teams are responsible for control implementation, and the security function is consulted on design and responsible for standards and oversight.

### A sample RACI for cybersecurity governance

The table below is a simplified illustration for a mid-sized organization with a CISO reporting outside IT. It is not a template to copy; real RACIs are built from the organization's actual roles and should be tested against recent incidents and projects to see whether they would have prevented the confusion that occurred.

| **Activity**                       | **Accountable**                    | **Responsible**         | **Consulted / Informed**                    |
|------------------------------------|------------------------------------|-------------------------|---------------------------------------------|
| Approve cyber risk appetite        | Board (risk committee)             | CRO and CISO prepare    | C: CEO, CIO; I: executives                  |
| Accept high residual risk          | Business risk owner                | CISO documents analysis | C: CRO, privacy; I: risk committee          |
| Approve security exception         | Business risk owner                | Requesting team         | C: CISO, architect; I: internal audit       |
| Operate IAM and privileged access  | CIO                                | IAM and platform teams  | C: CISO; I: business owners                 |
| Remediate critical vulnerabilities | System owner                       | IT or engineering team  | C: SOC; I: CISO, risk owner                 |
| Onboard high-risk supplier         | Procurement executive              | Vendor manager          | C: CISO, legal, privacy; I: owner           |
| Declare major incident             | Crisis lead (executive)            | Incident commander      | C: CISO, legal, privacy; I: board           |
| Notify regulators and individuals  | General counsel or privacy officer | Privacy team            | C: CISO, communications; I: CEO             |
| Assure control effectiveness       | Chief audit executive              | Internal audit          | C: CISO, control owners; I: audit committee |

Several features of the table deserve comment. The CISO is Accountable for none of the rows, yet is Responsible or Consulted on nearly all of them; this reflects the principle that the CISO owns the program and the expertise, while business and functional leaders own the risks and outcomes. (In practice the CISO is Accountable for the security program itself—its standards, its reporting, and its own services—which a fuller RACI would show.) Risk acceptance and exceptions are accountable to the business risk owner, not to the security team. Notification obligations sit with legal or privacy leadership because they are legal decisions with legal consequences. Assurance sits with internal audit, which preserves independence.

### Keeping a RACI alive

RACIs decay quickly. Reorganizations, cloud migrations, new suppliers, and staff turnover change who actually does the work, while the document remains frozen. A living RACI is reviewed at least annually and whenever a significant organizational or technology change occurs, is tested during exercises and post-incident reviews, and is linked to role descriptions so that new hires inherit their responsibilities explicitly. It is also useful to compare the RACI to the organization's control framework mapping (Chapter 9): every key control should have a named owner who appears in the RACI, and every owner in the RACI should know which controls and evidence they are answerable for.

## Roles in a mature operating model: what executives should expect to exist

Organizations differ in size and complexity, but manager-oriented governance requires that certain roles or responsibilities exist, even when several are combined in one person in a smaller organization. The test is not whether each title appears on the organization chart, but whether each responsibility has a named owner with the authority to fulfil it.

### Risk owners, process owners, and data owners

Product owners, process owners, and data owners are the main business accountability points. A **risk owner** is the person with the accountability and authority to manage a risk: to decide on treatment, to fund it within their budget, and to accept the residual risk within their delegated tolerance. In most organizations, the natural risk owner for a cyber risk is the executive who owns the business service or information that would be harmed. Without their ownership, cybersecurity becomes a technical program rather than a risk-managed business reality.

Risk ownership must be operationalized, not merely declared. Risk owners need to be trained in what ownership means, given decision-useful information about their risks, invited to the decisions that affect them, and held to account in performance reviews. A risk register in which every cyber risk lists "CISO" or "IT" as owner is a clear sign that ownership has not yet moved to the business.

### The CISO mandate and reporting lines

A Chief Information Security Officer (or equivalent) typically owns the security program: policy and standards, architecture oversight, security operations or their oversight, control assurance, incident readiness, workforce development, and risk reporting. The CISO should not, however, be the owner of all cyber risk. The CISO owns the risk management process and the control strategy; business leaders own the risks associated with their products and processes. A written CISO mandate, approved by the executive team and visible to the board, should state this distinction together with the CISO's authority to set standards, to escalate, to report independently to the board or a board committee, and to access the information needed to do the job.

Reporting lines are a recurring and sometimes heated debate. A CISO who reports to the CIO is close to the technology and can influence operations directly, but may face a conflict of interest when security requirements compete with IT budgets, delivery deadlines, or uptime targets. A CISO who reports to the CEO, the chief risk officer, the general counsel, or the chief operating officer gains independence and visibility but may be further from operational levers. There is no universally correct answer. What matters is that the reporting line does not suppress bad news, that the CISO has a direct and regular channel to the board or its committee, and that the CISO's budget and priorities are not decided solely by a function whose own performance the CISO must challenge. In regulated financial institutions, supervisors increasingly expect this kind of independence for the function that oversees technology and cyber risk.

### Business information security officers

In large and federated organizations, a **business information security officer** (BISO) is a security leader embedded in or aligned to a business unit. The BISO translates enterprise security strategy into the unit's context, helps business leaders understand and own their risks, coordinates risk assessments and exceptions, and represents the unit's needs back to the central security function. A good BISO is part risk adviser, part program manager, and part diplomat.

The BISO role succeeds when its reporting line and mandate are clear. A BISO who reports only to the business may be captured by delivery pressure; one who reports only to the CISO may be seen as an outsider. Many organizations use a dual-reporting arrangement, with a solid line to the CISO for standards and professional direction and a dotted line to the business unit leader for priorities. The BISO should not become the unit's risk owner: the role exists to make business ownership effective, not to replace it.

### Technology, privacy, risk, and assurance partners

The CIO or IT leadership owns many operational controls—especially infrastructure, enterprise platforms, and end-user computing—and is therefore a critical partner. In cloud-heavy and product-engineering organizations, engineering leadership and platform teams also become primary control owners, particularly for secure development, cloud configuration (Chapter 15), and the software supply chain. A privacy officer or privacy leadership function participates wherever personal information and regulatory obligations are implicated, which is especially important in incident response, vendor contracting, and AI adoption (Chapters 16 and 18).

Risk management and compliance functions provide second-line oversight, integration with enterprise risk management, and consistency of reporting. Legal counsel interprets obligations and manages privilege during incidents. Human resources owns the people processes on which security depends: background screening, onboarding, role changes, disciplinary procedures, and departures (the joiner–mover–leaver lifecycle of Chapter 10). Internal audit provides independent assessment and helps prevent governance theatre by testing whether controls actually work.

### Smaller organizations: combining roles without losing independence

Small and mid-sized organizations cannot staff all these roles separately. A regional distributor with 150 employees may have one security manager, an IT director, and an external auditor. Combining roles is acceptable, but three safeguards matter. First, the person who operates a control should not be the only person who verifies it; independent assurance can be bought from an external provider. Second, risk acceptance must still be made by an accountable business executive, not by the security manager alone. Third, when security services are outsourced to a managed security service provider or a fractional ("virtual") CISO, the organization must remember that **services can be outsourced, but accountability cannot**. Contracts, reporting, and oversight must preserve the organization's ability to know and decide.

## The CISO's accountability and the personal liability debate

Few topics have changed the conversation about security leadership as much as the question of personal liability. Security leaders have always faced career consequences after major incidents. What changed in the early 2020s was the prospect that individual executives could face criminal or regulatory action in connection with how incidents and security posture were handled and disclosed. The facts of the leading cases are narrower than headlines sometimes suggest, and managers should understand them precisely.

### Cases that changed the conversation

The case most often cited involves Uber's former chief security officer, Joseph Sullivan. In 2022, a United States federal jury convicted him of obstruction of justice and misprision of a felony in connection with the concealment of a 2016 data breach from the Federal Trade Commission, which was then investigating Uber's security practices; in 2023 he was sentenced to probation and a fine, and in March 2025 the U.S. Court of Appeals for the Ninth Circuit upheld the conviction. The lesson is specific: the conviction concerned concealment of a breach during a regulatory investigation, not a failure to prevent the breach.

The second case is the Securities and Exchange Commission's action against SolarWinds and its CISO, Timothy Brown, filed in 2023 after the SUNBURST supply-chain compromise. The SEC alleged that public statements about the company's security practices were misleading. In July 2024, a federal court dismissed most of the claims, including the theory that cybersecurity control weaknesses amounted to failures of internal accounting controls, while allowing claims based on pre-incident public security statements to proceed. In November 2025, the SEC voluntarily dismissed the remaining case. A third example comes from consumer protection: in 2023, the Federal Trade Commission finalized an order against the online alcohol marketplace Drizly that named its chief executive personally and required him to implement information security programs at future companies he leads, signalling that senior executives, not only security staff, can be held answerable for security failures.

### What the debate means for governance design

The direction of travel is not that security leaders will routinely be prosecuted for breaches; the cases above do not support that conclusion, and the SolarWinds dismissal in particular narrowed the risk. The more durable lesson is that accountability follows decisions, statements, and disclosures. Executives who conceal, misrepresent, or make public claims they cannot support expose themselves and their organizations. At the same time, regimes such as NIS2 and DORA explicitly place accountability on management bodies, which shifts the focus from the CISO alone to the leadership team as a whole.

For governance design, several practical conclusions follow.

- **Align authority with accountability.** A CISO who is held responsible for outcomes must have the authority, budget, and access to influence them, or the accountability must be explicitly shared with those who do.

- **Document risk decisions and dissent.** When executives accept a risk against security advice, the acceptance, rationale, and the CISO's view should be recorded; this protects both the organization and the individuals involved.

- **Give the CISO a voice in disclosure.** Public statements about security posture, regulatory filings, and breach notifications should be reviewed by people who know whether they are true, through a defined disclosure process involving legal, finance, and security.

- **Protect the individuals.** Organizations should confirm that directors' and officers' liability insurance and indemnification arrangements cover the CISO and other security leaders, and should consider this when recruiting.

- **Separate investigation from blame.** Post-incident reviews should focus on learning and on systemic causes, with legal counsel engaged where privilege and regulatory exposure are at stake.

## Workforce planning, skills frameworks, and the talent problem

An operating model is only as strong as the people who staff it. Cybersecurity work requires scarce and diverse skills, from governance and risk to cloud engineering, incident response, and AI security. Workforce planning—deciding which capabilities the organization needs, which it will build internally, which it will acquire, and which it will source from partners—is a governance decision, not merely a recruitment task.

### What the data says about the skills shortage

The skills shortage is real, but its character is changing. For several years, ISC2's annual Cybersecurity Workforce Study published a global "workforce gap" estimate of the number of additional professionals needed. Its 2025 study, based on a survey of more than 16,000 professionals and released in December 2025, did not estimate a headcount gap and instead emphasized skills: 59% of respondents reported critical or significant skills shortages in their organizations, up from 44% the previous year, and AI was the most frequently cited skill in short supply. The study also reported that budget pressure and staffing cuts were increasing perceived security risk.

The managerial implication is that the problem is less "we cannot hire enough people" than "the people we have do not yet have the skills our risks now require." This shifts attention toward deliberate skills mapping, internal development, and retention, rather than relying solely on external hiring in a tight market.

### The NICE Framework

The **NICE Framework**, formally NIST Special Publication 800-181 Rev. 1, *Workforce Framework for Cybersecurity* (November 2020), provides a common language for describing cybersecurity work. Its building blocks are task statements and knowledge and skill statements, which are grouped into work roles (such as security control assessor, cyber defence analyst, or cybersecurity policy and planning) and competency areas. Since 2024, NIST has maintained the detailed content separately as the NICE Framework Components, versioned independently of the publication; version 2.0.0 was released in March 2025, and the most recent release at the time of writing, version 2.2.0 (April 2026), added a work role for cybersecurity supply chain risk management. Notably, the Components include an "Oversight and Governance" category of work roles, which recognizes that governance, risk, policy, and executive leadership are cybersecurity work in their own right.

Managers use the NICE Framework to write consistent job descriptions, to identify which work roles their operating model requires, to map current staff to roles and find gaps, to design development paths, and to communicate with educators and training providers. It also supports workforce planning for small organizations, which can identify the handful of roles they truly need and decide which to staff and which to source.

### The European Cybersecurity Skills Framework

In the European Union, ENISA published the **European Cybersecurity Skills Framework** (ECSF) in September 2022. The ECSF defines twelve cybersecurity role profiles: the chief information security officer, cyber legal, policy and compliance officer, cyber threat intelligence specialist, cybersecurity architect, cybersecurity auditor, cybersecurity educator, cybersecurity implementer, cybersecurity researcher, cybersecurity risk manager, digital forensics investigator, incident responder, and penetration tester. Each profile describes a mission, main tasks, key skills, knowledge, and e-competences. The ECSF underpins the European Commission's Cybersecurity Skills Academy initiative, and ENISA has mapped NIS2 obligations to ECSF roles. At the time of writing, ENISA indicates that the framework is being revised, with proficiency levels among the planned additions. Canadian organizations operating in Europe, or hiring from European programs, will increasingly encounter ECSF terminology alongside NICE.

### Build, buy, and borrow

Every organization combines three sourcing strategies. *Building* means developing internal talent through training, rotation, mentoring, and career paths; it is slower but produces people who understand the business and tend to stay. *Buying* means hiring experienced specialists; it is fast but expensive and exposed to market competition. *Borrowing* means using managed security service providers, consultants, or fractional leaders; it provides access to scarce skills and round-the-clock coverage but creates dependence and must be governed as a third-party relationship (Chapter 19).

A defensible workforce plan states which capabilities are core and must be held internally—typically governance, risk ownership support, architecture authority, and incident leadership—and which can be sourced externally under contract. It links staffing to the risk scenarios the organization faces and to the controls it relies on, so that a decision to leave a role unfilled is visible as a risk decision rather than a silent budget outcome.

## The security champions model: scaling security through the organization

One of the most widely effective organizational patterns for modern environments—especially agile development, product, and cloud teams—is the **security champions model**. The core idea is that security capability must exist inside delivery teams, not only as an external review function that appears at the end of a project.

### What a champion is and is not

A security champion is a member of a product, engineering, or operational team who is trained and supported to represent security concerns early, to incorporate security patterns into everyday work, to facilitate secure design and threat-modelling discussions, and to act as a bridge between the security function and the team. A champion is not a substitute for the security team, not an unpaid auditor, and not the person to blame when the team ships a vulnerability. The champion amplifies the team's own ownership; it does not replace it.

The governance rationale is scale. Central security teams cannot review every change, deployment, configuration, third-party integration, and data flow. Champions create localized competence, reduce friction by catching issues earlier, and embed security thinking into daily work. The model also extends beyond software: champions in finance, human resources, or plant operations can play a similar role for fraud-resistant processes, sensitive data handling, or operational technology (Chapter 11).

### Design conditions

For champions to work, governance must define the model explicitly. Several conditions are consistently necessary.

- **Role clarity.** A short written charter states what champions do, what they do not do, and how they relate to the security function and the team lead.

- **Protected time.** Champions need allocated capacity—often expressed as a percentage of their time—agreed with their managers; a title without time produces nothing.

- **Training and community.** Champions receive role-specific training and belong to a community of practice that meets regularly and shares patterns and lessons.

- **Escalation access.** Champions can reach security architects and engineers quickly when an issue exceeds their competence.

- **Recognition.** Contribution is recognized in performance reviews, career progression, and visible acknowledgement, so the role is seen as an asset rather than a burden.

Without these mechanisms, champions programs become symbolic and fade within a year or two, usually when the first enthusiastic cohort moves on.

### Measuring the model

A mature champions model is measured by outcomes, not by the number of champions recruited. Useful indicators include the proportion of significant initiatives with security engagement at the design stage, the reduction in late-stage or post-release findings, adherence to approved secure patterns, the time to remediate recurring classes of issues, and the retention of champions over time. **The objective is not more champions, but better security outcomes through operational integration.**

## Human risk management beyond awareness

Most organizations can say that they "do security awareness." Fewer can show that they manage human risk as a system. The Verizon Data Breach Investigations Report has for years found that most breaches involve a human element; the 2026 edition found that the human element was involved in 62% of the breaches it analyzed. That figure is often quoted to justify more awareness training, but it points to a broader conclusion: human behaviour is shaped by roles, processes, tools, incentives, and culture, and must be governed accordingly.

### From awareness to human risk management

**Human risk management** is the practice of identifying, measuring, and reducing the security risk that arises from human behaviour, using a combination of education, process and technology design, and cultural change. It differs from traditional awareness programs in three ways. It is targeted, focusing effort on the roles and behaviours that create the most risk rather than delivering the same module to everyone. It is design-oriented, recognizing that a well-designed process—phishing-resistant authentication (Chapter 10), a payment approval workflow that requires out-of-band verification, a one-click reporting button—often reduces risk more than any training. And it is measured, using behavioural indicators rather than completion rates.

Standards support this broader view. ISO/IEC 27001:2022 requires that people doing work under the organization's control be competent on the basis of appropriate education, training, or experience, and be aware of the policy and their contribution; ISO/IEC 27002:2022 includes people controls covering screening, terms of employment, awareness and training, disciplinary process, and event reporting. NIST CSF 2.0 includes an Awareness and Training category within Protect, and CIS Controls v8.1 dedicates Control 14 to security awareness and skills training.

### Role-based training governance

Role-based training is the core of competence building. Developers need secure coding patterns, dependency hygiene, secrets management, and threat-modelling practice. Cloud administrators need secure configuration baselines, identity design, and logging competence. Help desk and customer support staff need resistance to social engineering and rigorous identity verification protocols, since help desk impersonation has been a recurring entry point in major incidents. Finance staff need payment fraud and business email compromise controls. Executives need crisis decision-making readiness and an understanding of risk trade-offs. Procurement teams need contract requirements and the ability to interpret assurance evidence.

Training governance should include four mechanisms. A **training standard** defines which training is required for which roles, at what frequency, and with what minimum content. **Quality control** ensures that content is credible and relevant to the organization's technologies and threats, rather than generic modules that breed fatigue. **Measurement** looks beyond completion to effectiveness, such as reduced repeat errors, improved secure design outcomes, and better exercise performance. **Lifecycle integration** aligns training with onboarding, role changes, and major technology transitions such as cloud migration or AI adoption, so that new capabilities arrive with the competence to use them safely.

### Just culture and no-blame reporting

The most valuable security sensor in any organization is a person who reports something suspicious quickly. That behaviour depends on culture. If employees who click on a phishing link are publicly shamed or disciplined, the rational response is to stay silent, and an incident that could have been contained in minutes spreads for days. **Just culture**, a concept developed in aviation and healthcare safety, distinguishes between honest error, at-risk behaviour, and reckless or malicious conduct. Honest errors are treated as learning opportunities and prompts to improve systems; at-risk behaviour is addressed through coaching and design; only reckless or deliberate violations lead to disciplinary consequences.

Applied to cybersecurity, a just culture means that reporting a mistake is always safer than hiding it, that post-incident reviews look for systemic causes before individual blame, and that disciplinary processes are reserved for wilful or repeated disregard of clear rules. It does not mean the absence of accountability. Managers remain accountable for the conditions they create, and individuals remain accountable for reckless choices. **Organizations that punish honest mistakes do not get fewer mistakes; they get fewer reports.**

| In a just culture, reporting a mistake is always safer than hiding it. **The speed at which people report is a better indicator of human risk than the number of people who never click.** |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

### Measuring behaviour responsibly

Measuring human risk requires care. Useful indicators include reporting rates and time-to-report for simulated and real phishing, repeat-susceptibility rates in high-risk roles, the proportion of privileged users on phishing-resistant authentication, and policy violations detected by technical controls. Phishing simulations, however, should be designed to teach rather than to trap, and results should be used primarily at aggregate and role level.

Behavioural monitoring also raises privacy and labour-relations questions. In Québec, Law 25 and the Civil Code's protection of privacy, together with collective agreements in unionized workplaces, constrain how employee data may be collected and used; elsewhere in Canada, PIPEDA and provincial laws apply to varying degrees. Monitoring should be proportionate, transparent, justified by a documented purpose, and reviewed with privacy and human resources leadership. A human risk program that erodes trust will ultimately reduce the reporting behaviour it depends on.

## Incentives and performance: aligning behaviour with risk reality

Accountability mechanisms fail when incentives are misaligned. This is one of the most consistent patterns in cybersecurity governance, and one of the least discussed, because it implicates how organizations reward their most senior people.

### How misaligned incentives externalize risk

If delivery teams are rewarded only for speed, features, and uptime, security becomes a perceived obstacle and will be bypassed. If security teams are rewarded only for policy compliance metrics, they will produce documentation rather than risk reduction. If executives are rewarded only for short-term financial results without long-term risk accountability, investment in resilience will be deferred until after an incident. In each case, the people who create risk do not bear its consequences. Economists call this an externality; in cybersecurity governance it is better described as the **silent externalization of risk**, where teams take decisions that increase the organization's exposure while the cost falls on someone else, often years later.

### Integrating security into performance language

Incentive design does not require elaborate compensation schemes. It requires integrating security outcomes into the performance language of relevant roles. For product leaders, this may mean accountability for risk acceptance decisions and control readiness in major releases. For engineering leaders, it may mean remediation performance for critical vulnerabilities, adherence to secure configuration baselines, and incident readiness. For IT operations, it may mean patching timeliness for high-risk assets, logging coverage for critical systems, and successful recovery tests. For executives, it may mean resilience objectives and evidence-based progress against board-approved targets, and in some organizations a risk-adjustment component in variable compensation.

The objective is not punitive. Linking security outcomes to performance must not undermine the just culture described earlier; individuals should be measured on the conditions and outcomes they control, not punished for honest errors or for incidents caused by adversaries despite reasonable controls. When incentives reflect risk reality, teams internalize security priorities rather than treating them as external demands.

### A balanced cybersecurity scorecard

The balanced cybersecurity scorecard introduced in Chapter 2 is a practical way to make these expectations visible. As explained there, it combines **leading indicators**, which measure preparedness and control execution before harm occurs, with **lagging indicators**, which measure realized outcomes; neither is sufficient alone. What this chapter adds is accountability: each indicator should have an accountable owner and an escalation threshold, so that the scorecard drives decisions rather than decorating reports. The table below gives illustrative examples; targets and thresholds should be derived from the organization's own risk appetite and baseline performance.

| **Indicator**                              | **Type** | **Accountable owner**                 | **Escalate when**                   |
|--------------------------------------------|----------|---------------------------------------|-------------------------------------|
| Role-based training completed on time      | Leading  | HR and CISO                           | High-risk roles remain untrained    |
| Critical vulnerabilities fixed by deadline | Leading  | System owner                          | Internet-facing item is overdue     |
| Privileged access recertified              | Leading  | IAM owner and business owners         | Access remains uncertified          |
| Recovery and crisis exercises completed    | Leading  | Continuity lead and executive sponsor | Actions remain unassigned           |
| Critical findings closed by deadline       | Leading  | Responsible executive                 | Any critical finding is overdue     |
| Time to contain high-severity incidents    | Lagging  | Incident response lead                | Containment exceeds tolerance       |
| Outages of critical services beyond RTO    | Lagging  | Service owner and CIO                 | Recovery-time objective is missed   |
| Recurrence of known control weaknesses     | Lagging  | CISO and control owners               | Repeat incident with overdue action |

The scorecard connects directly to the key risk indicators discussed in Chapter 4 and the investment logic of Chapter 17. Its governance value comes from the owner and escalation columns: an indicator without an owner is a statistic, and an indicator without a threshold is an opinion.

## Accountability mechanisms: how organizations make security enforceable

**Accountability mechanisms** are the routines and tools by which roles and responsibilities become real. Without them, RACIs and operating models remain aspirational documents. Several mechanisms are consistently effective in practice, and most have been introduced in earlier chapters; here they are considered together as the machinery of accountability.

### Exceptions, architecture review, and change governance

Exception governance is fundamental. When teams cannot meet a security requirement, they request an exception. Governance requires that every exception be owned by an accountable risk owner, risk-assessed, supported by compensating controls where possible, approved at the level the decision-rights scheme requires, and granted with an **expiry date**. Exceptions should be tracked in a register, reported in aggregate to senior management, and reviewed before renewal. An exception register that only grows is a sign that requirements are unrealistic, that delivery teams are not funded to meet them, or that nobody is enforcing expiry.

Architecture and change governance is the second mechanism. Major systems and high-risk changes should pass through structured design reviews with security criteria, triggered early by defined thresholds rather than by a team's willingness to ask. The goal is not bureaucracy; it is the early detection of structural design risks that are cheap to fix on a whiteboard and expensive to fix in production (Chapter 13).

### Assurance, evidence, and learning loops

Operational assurance mechanisms—configuration monitoring, continuous control testing, internal audit, red-team and recovery exercises, and periodic independent assessments (Chapter 8)—convert "we think we are compliant" into "we can show that we are effective." They also reduce compliance theatre by forcing evidence, a theme developed in Chapter 3.

Post-incident reviews and learning loops are accountability mechanisms in their own right. A mature organization treats incidents and near-misses as governance signals. Root-cause analysis includes managerial causes—decision delays, unclear authority, inadequate training, weak vendor accountability, underfunded controls, and broken risk-acceptance processes—and not only technical ones. Remediation actions are assigned to named owners, given deadlines, tracked to closure, and reported to executive oversight. When the same root cause appears in two successive incident reviews, the question is no longer technical but managerial: who owned the corrective action, and why was it not completed?

### Funding as an accountability mechanism

Budgeting and investment governance is itself an accountability mechanism. Controls that are required must be funded. If leadership expects security outcomes but funds only minimal staffing and tooling, accountability becomes unfair and ineffective, and the people held responsible will rationally lower their ambitions or leave. Governance maturity aligns expectations with resources, which is why Chapter 17 insists that security investment be tied to risk scenarios and appetite.

| **An unfunded requirement is not a control; it is a deferred risk acceptance that nobody signed.** Aligning expectations with resources is part of accountability, not a separate budgeting exercise. |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

### Common organizational failure patterns and how to fix them

Several accountability failures appear repeatedly across organizations of every size and sector. Naming them explicitly is useful, because they are predictable and therefore preventable.

- **Security owned by IT with no business ownership of risk.** Security becomes a technical negotiation between IT and the security team, while business leaders treat risk as someone else's problem. The fix is explicit risk ownership: business owners are accountable for risk acceptance and take part in exception approvals and prioritization.

- **The CISO as scapegoat.** The CISO is held responsible for enterprise risk without authority over budgets, priorities, or business decisions, and is replaced after each major incident. The fix is to align accountability with authority, document risk acceptance, and give the CISO a direct line to the board.

- **Compliance theatre as a substitute for control effectiveness.** Success is measured by policies approved, training completed, and audit artifacts produced rather than by risk outcomes. The fix is evidence and testing: control effectiveness must be verifiable through monitoring, exercises, and validated coverage metrics.

- **Security review as a late-stage gate.** When security enters at the end of projects, it becomes friction, exceptions rise, and insecure design is locked in. The fix is operating-model design: secure patterns, champions, and early review triggers.

- **Tools without operational capacity.** Organizations purchase platforms that generate alerts and dashboards but lack the staff and workflows to act on them. The fix is to govern security as a capability, not as procurement.

- **Unclear incident authority.** During crises, decision delays amplify harm. The fix is rehearsed crisis governance: who can shut down systems, who can communicate, who can engage counsel and external responders, and who can authorize extraordinary spending.

- **Outsourcing mistaken for transfer of accountability.** Managed services, cloud providers, and AI vendors take over operations, and the organization stops asking questions. The fix is retained internal capability to oversee providers and contracts that preserve visibility and audit rights (Chapter 19).

The common root of these patterns is a mismatch between who decides, who acts, who pays, and who answers. Each fix restores that alignment.

## A practical blueprint: implementing accountable governance in stages

Organizations often attempt to redesign their cybersecurity operating model in one large transformation, usually after an incident or a critical audit. Large transformations tend to stall when attention moves elsewhere. Staged implementation is more effective and easier to sustain, because each stage produces visible value and builds the capacity for the next.

### Five stages

The first stage is **role clarity and decision rights**: define risk owners by business domain, name control owners, write the CISO mandate, establish the decision-rights tiers, and create exception and escalation pathways. This stage creates governance visibility almost immediately. The second stage is **integration into delivery and operations**: launch security champions, embed review triggers in project and change lifecycles, and standardize secure patterns so that teams can comply without constant negotiation. The third stage is **competence and human risk management**: introduce role-based training with measurable objectives, map the workforce against the NICE Framework or ECSF, and build a just-culture reporting environment.

The fourth stage is **assurance and evidence**: expand monitoring and control testing, align internal audit with the top risk scenarios, and build a balanced scorecard with owners and thresholds. This stage reduces compliance theatre and improves defensibility. The fifth stage is **institutionalization**: governance becomes part of routine management reviews, performance systems, and board reporting, and cybersecurity is treated as a normal dimension of running products and processes rather than an episodic IT concern. The figure below summarizes the sequence.

<span id="_Toc243635585" class="anchor"></span>Figure 86: Staged blueprint for accountable cybersecurity governance

### Illustrative scenario: role clarity at a Québec credit union

Consider a fictional credit union in the Laurentians with roughly 400 employees, twenty branches, and a growing digital banking platform operated partly by a Canadian cloud-based core banking provider. The security manager reports to the IT director. A routine internal audit finds that 38 security exceptions are open, 22 of them past their original review dates, and that every cyber risk in the risk register names "IT" as owner. In the same quarter, a fraudster impersonates a branch employee on the phone, persuades the help desk to reset a password and enrol a new multifactor device, and initiates fraudulent transfers that are caught only by a vigilant teller.

The post-incident review, deliberately conducted in a no-blame format, finds that the help desk agent followed the procedure as written; the procedure simply did not require out-of-band verification. It also finds that nobody could say who owned the help desk identity verification process, that the exception allowing legacy password resets had been granted by the security manager three years earlier and never revisited, and that the board's audit committee had never been told about it.

Management responds in stages. The chief executive designates the vice-president of operations as risk owner for member identity and payment fraud scenarios, and the vice-president of digital services as owner for the online platform. The security manager becomes head of information security with a written mandate, a dotted-line reporting relationship to the chief risk officer, and a standing quarterly session with the audit committee. A decision-rights scheme routes high residual risks to the executive risk committee, and every exception now carries an owner and a twelve-month maximum expiry. The help desk procedure is redesigned with callback verification, help desk staff receive targeted training, and a reporting button and a no-blame reporting message are introduced for all staff. Within a year, open exceptions fall to eleven, all within expiry, and the board receives a scorecard tied to the credit union's top five risk scenarios. Because the credit union is provincially regulated, management also aligns the changes with the expectations of its regulator, the Autorité des marchés financiers, including its guideline on information and communications technology risk management, and with Law 25, under which the chief executive remains the person in charge of protecting personal information unless the function is delegated in writing—which the chief executive now does formally, to a newly appointed privacy officer.

The managerial lessons are characteristic. The incident was not caused by a careless employee but by an unowned process and an unexpired exception. The fixes were mostly organizational, not technical, and they were affordable for an institution of modest size.

### Questions boards and executives should ask

- Who is the named risk owner for each of our top cyber risk scenarios, and have they formally accepted the current residual risk?

- Does the CISO have a written mandate, a reporting line that does not suppress bad news, and direct access to the board or a board committee?

- Which committee oversees cybersecurity, is this written in its charter, and how does the full board stay informed?

- Do our directors have enough knowledge to challenge management's cyber reporting, and what training or expertise have we arranged?

- Which cyber decisions require executive or board approval, and when did we last exercise those decision rights under time pressure?

- How many security exceptions are open, how many are past expiry, and who approved the oldest ones?

- What proportion of our critical controls have named owners and recent evidence of effectiveness?

- What skills does our risk profile require, which do we lack, and which have we chosen to source externally?

- Do employees report mistakes and suspicious events quickly, and what happens to people who do?

- Are security outcomes reflected in the objectives and performance reviews of business and technology leaders?

- Which corrective actions from past incidents and audits remain open, and why?

- Are our public and regulatory statements about cybersecurity reviewed by people who know whether they are true?

## Bringing it together: accountability as the mechanism that makes cybersecurity governable

A cybersecurity governance program succeeds when it aligns responsibility with consequence. This chapter has argued that people, organizational design, and accountability mechanisms are not soft topics but the primary mechanism by which security becomes executable. When decision rights are clear, risk ownership sits with the business, and the board is competent enough to challenge what it is told, cybersecurity stops being a perpetual dispute between IT and the business. A CISO mandate that matches the expectations placed on the role, champions inside delivery teams, a just culture, aligned incentives, and accountability mechanisms that produce evidence are what keep it that way. It becomes a governed system of decision-making and execution.

The same conclusion ties together the book as a whole. Chapters 1 through 4 established the foundations: security as the absence of unacceptable risk rather than of all risk, governance as direction, control, and accountability, compliance as a set of obligations that must be met with real controls rather than theatre, and risk management as the discipline of deciding which exposures to reduce, transfer, avoid, or accept within an approved appetite. Each of these foundations presupposes someone who decides and someone who answers for the decision. Risk appetite means little until a named executive accepts a residual risk against it; compliance means little until a control owner can produce evidence.

Chapters 5 and 6 turned risk management into practice through scenario-based assessment and the CyberRiskGuardian method, linking the risk triangle of Chapter 1—threat, vulnerability, and impact—to business consequences that leaders can understand, and producing estimated and tolerated risk for each scenario, risk-to-tolerance verdicts, the treatment portfolios on which investment decisions rest, and the key risk indicators that monitor them. Chapters 7 through 14 examined how risk is reduced in operations: the security operations function; detection, vulnerability management, response, and testing; control frameworks such as ISO/IEC 27001, NIST CSF 2.0, COBIT 2019, and CIS Controls v8.1; identity and access management; operational technology; incident management and resilience; architecture; and configuration baselines. Across all of them, the decisive factor was rarely the choice of technology; it was whether ownership, decision rights, and evidence were clear enough for the controls to be operated and improved over time.

Chapters 15 through 20 extended governance to the domains that most challenge traditional boundaries: cloud services, personal information, investment, artificial intelligence, third parties and supply chains, and cryptography. Each of these domains distributes control across providers, platforms, and partners, and each therefore makes accountability harder to locate and more important to design. Shared responsibility in the cloud, privacy obligations under Law 25, PIPEDA, and the GDPR, the economics of security investment, the governance of AI systems, supplier assurance, and key management all ask the same managerial question: who decides, who acts, who pays, and who answers?

Resilience is where these threads meet. No organization can prevent every incident, and the book has repeatedly emphasized that the goal is not zero risk but risk that is understood, owned, within appetite, and recoverable. An organization becomes resilient when its people know their roles before the crisis, when its leaders have rehearsed the decisions they will need to make, when its evidence shows what actually works, and when it learns from each event rather than searching for someone to blame. These are organizational properties, built deliberately over time.

The final lesson of this book is therefore simple to state and demanding to practise. Organizations do not secure themselves by wishing for better technology or by accumulating frameworks. They secure themselves by designing accountability so that decisions are made by the people who will bear their consequences, in time to matter, and are supported by standards, evidence, and incentives that make secure behaviour the path of least resistance. For business technology managers, that design work is not someone else's job. It is the job.

## Chapter summary

Organizations rarely fail at cybersecurity because nobody knows which controls exist; they fail because controls, decisions, and responsibilities are not organized so that they are executable, sustained, and defensible. This final chapter argues that people, organizational design, and accountability mechanisms are the central governance topic, not a soft one. It examines operating models and the three lines, board oversight and director competence, decision rights, RACI, the roles a mature model needs, the liability debate around security leaders, workforce frameworks, security champions, human risk management, just culture, incentives, and the mechanisms that make accountability enforceable. It closes with failure patterns, a staged blueprint, and a synthesis of the book: the job is designing accountability.

### Key points to retain

- Every organization has a cybersecurity operating model; the question is whether governance, enablement, and operations work, and centralized, federated, or hybrid structures, were designed deliberately or inherited by accident.

- The three lines model distinguishes those who own and manage risk, those who challenge and monitor it, and those who independently assure it, and the CISO's placement within it must be recognized.

- Boards oversee rather than manage cybersecurity, and NIS2, DORA, SEC disclosure rules, OSFI B-13, and Law 25 increasingly assign accountability to the top and expect directors to be competent to exercise it.

- Decision rights should be explicit for a dozen high-consequence decisions, tiered by impact using risk appetite thresholds, with escalation paths, recorded dissent, and pre-delegated emergency authority.

- A RACI prevents gaps, overlaps, and diffusion of responsibility only if each activity has exactly one Accountable role with the authority and resources to fulfil it, and if it is kept alive.

- Business leaders own cyber risks, the CISO owns the program and the expertise under a written mandate, BISOs make business ownership effective, and outsourcing never transfers accountability.

- The liability cases involving security leaders show that accountability follows decisions, statements, and disclosures, so authority must match accountability and risk decisions and dissent must be documented.

- Workforce planning is a governance decision about skills rather than headcount, supported by the NICE Framework and ECSF and by explicit build, buy, and borrow choices.

- Security champions scale security into delivery teams only with role clarity, protected time, training, escalation access, and recognition, and they are measured by outcomes.

- Human risk management is targeted, design-oriented, and measured, and a just culture makes reporting a mistake always safer than hiding it.

- Misaligned incentives externalize risk, so security outcomes belong in performance language, and a balanced scorecard needs an owner and an escalation threshold for each indicator.

- Exceptions, architecture review, assurance, post-incident learning, and funding are the machinery of accountability, implemented in stages from role clarity to institutionalization.

### Key terms

- **Operating model** — How work actually gets done: who performs which activities, how decisions are made, how work is funded, how conflicts are resolved, and how outcomes are measured.

- **Governance, enablement, and operations** — Three kinds of cybersecurity work whose outputs are, respectively, direction and decisions, capability in other teams, and control performance.

- **Hybrid security model** — A structure that centralizes governance, architecture standards, and security operations while embedding business-aligned security roles in each major unit.

- **Three Lines Model** — The IIA's 2020 model distinguishing a governing body, management with first-line and second-line roles, and independent internal audit as the third line.

- **Decision rights** — The defined authority of roles to decide, approve, or override in specific domains, and the thresholds at which decisions must be escalated.

- **RACI** — A matrix assigning, for each activity, who is Responsible, Accountable, Consulted, and Informed.

- **Diffusion of responsibility** — The failure in which everyone is nominally involved, so no one feels personally answerable for an outcome.

- **Risk owner** — The person with the accountability and authority to decide on a risk's treatment, fund it, and accept residual risk within delegated tolerance.

- **Business information security officer (BISO)** — A security leader embedded in or aligned to a business unit who helps its leaders understand and own their risks.

- **NICE Framework** — NIST SP 800-181 Rev. 1, a common language of tasks, knowledge, and skills grouped into cybersecurity work roles and competency areas.

- **Security champion** — A trained and supported member of a delivery or operational team who represents security concerns early and bridges the team and the security function.

- **Human risk management** — Identifying, measuring, and reducing security risk from human behaviour through targeted education, process and technology design, and cultural change.

- **Just culture** — An approach that treats honest error as a learning opportunity, addresses at-risk behaviour through coaching and design, and reserves discipline for reckless or deliberate violations.

- **Externalization of risk** — The situation in which teams make decisions that increase exposure while the cost falls on someone else, often years later.

- **Exception governance** — The discipline of ensuring every exception is owned, risk-assessed, compensated, approved at the right level, tracked, and given an expiry date.

## Review questions

### Essay questions

1.  Analyze the Laurentians credit union scenario. Explain why the fraud incident is better understood as a failure of organizational design than of employee behaviour, and evaluate how well the staged response addressed decision rights, roles, culture, and board oversight.

2.  Choose an organization you know (or a fictional one) and design a decision-rights scheme for five of the high-consequence cyber decisions listed in the chapter. For each, state who proposes, who is consulted, who decides, who can escalate, what evidence is required, and how thresholds link to risk appetite.

3.  Evaluate the debate over CISO reporting lines and personal liability. Using the Uber, SolarWinds, and Drizly cases, assess what the cases do and do not imply for security leaders, and recommend governance measures that align authority with accountability.

4.  Critically assess the claim that more security awareness training is the appropriate response to the human element in breaches. Use the chapter's treatment of human risk management, just culture, measurement, and incentives to propose a better approach.

5.  Using the five-stage blueprint, design an implementation plan for a mid-sized organization whose risk register lists IT as owner of every cyber risk and whose exception register keeps growing. Justify the sequence and identify the indicators that would show each stage is working.

### Multiple-choice questions

Choose the best answer. Answers are listed in the appendix "Answers to multiple-choice questions".

**1.** A security architect approves exceptions to her own designs, and the SOC sets its own risk thresholds. What problem does this illustrate?

> A. Blurred decision rights between governance, enablement, and operations work
>
> B. An overly centralized model that lacks business-aligned security roles
>
> C. A missing third line of independent internal audit assurance
>
> D. A shortage of technical skills within the security operations team

**2.** In 2026, the NICE Framework Components version 2.2.0 added a new work role. Which one?

> A. Business information security officer
>
> B. AI security engineer
>
> C. Cybersecurity supply chain risk management
>
> D. Security awareness program manager

**3.** A CISO directly runs the security operations centre. Where does this place the CISO in the Three Lines Model?

> A. In the third line, providing independent assurance
>
> B. In the first line, operating controls
>
> C. In the governing body, overseeing management
>
> D. Outside the model, as an external provider

**4.** According to the chapter, which of the following should a board leave to management rather than do itself?

> A. Approving the risk appetite statement for cyber and technology risk
>
> B. Understanding the most material cyber risk scenarios and impacts
>
> C. Ensuring investment is commensurate with risk
>
> D. Selecting security products and reviewing vulnerability scan results

**5.** Under DORA Article 5, what is expected of members of a financial entity's management body regarding ICT risk?

> A. They must delegate all ICT risk decisions to the CISO
>
> B. They must hold a recognized cybersecurity certification
>
> C. They must appoint a dedicated board cyber committee
>
> D. They must actively keep up to date with sufficient knowledge and skills to assess ICT risk

**6.** What do the SEC's 2023 cybersecurity disclosure rules require public companies to describe annually under Item 106 of Regulation S-K?

> A. Whether each director has formal cybersecurity expertise
>
> B. The board's oversight of cyber risk and management's role and expertise
>
> C. The number of attacks blocked and patches deployed each quarter
>
> D. The name and compensation of the chief information security officer

**7.** Delivery teams are rewarded only for speed and features, and executives only for short-term financial results. What does the chapter call the resulting dynamic?

> A. Compliance theatre replacing control effectiveness
>
> B. Diffusion of responsibility across delivery teams
>
> C. The silent externalization of risk
>
> D. Concentration of decision rights at the top

**8.** A product owner wants to launch with a weakness the CISO considers unacceptable. What is the CISO's essential decision right?

> A. To escalate to a level owning both outcomes and have dissent recorded
>
> B. To veto the launch until the weakness has been fully remediated
>
> C. To accept the risk on the business's behalf with compensating controls
>
> D. To defer the decision to internal audit for an independent opinion

**9.** An organization's exception register grows every quarter and is rarely reduced. What does the chapter say this most likely indicates?

> A. That the security team is approving exceptions too slowly
>
> B. That requirements are unrealistic, teams are unfunded, or nobody enforces expiry
>
> C. That internal audit is sampling too many exceptions
>
> D. That business owners are refusing to request exceptions

**10.** Which rule does the chapter set for a sound RACI?

> A. Every activity should have at least two Accountable roles for resilience
>
> B. The CISO should be Accountable for every security-related activity
>
> C. Accountable and Responsible roles should always be the same person
>
> D. Each activity has exactly one Accountable role with authority and resources

**11.** What gives a balanced cybersecurity scorecard its governance value, according to the chapter?

> A. Its mix of exactly half leading and half lagging indicators
>
> B. Its comparison of the organization against industry benchmarks
>
> C. Its owner and escalation threshold for each indicator
>
> D. Its alignment with the number of controls in CIS Controls v8.1

**12.** In the chapter's sample RACI, who is Accountable for approving a security exception?

> A. The business risk owner
>
> B. The CISO
>
> C. The requesting team
>
> D. The chief audit executive

**13.** Which activities belong to the first stage of the chapter's staged blueprint for accountable governance?

> A. Launching security champions and embedding review triggers
>
> B. Expanding control testing and aligning internal audit with risk scenarios
>
> C. Defining risk owners, control owners, the CISO mandate, and decision-rights tiers
>
> D. Embedding governance into performance systems and board reporting

**14.** A BISO reports only to the business unit leader and has become the unit's risk owner. How would the chapter assess this arrangement?

> A. As ideal, because it brings security fully inside the business
>
> B. As flawed: the BISO risks capture and should make business ownership effective, not replace it
>
> C. As acceptable, provided the BISO also chairs the risk committee
>
> D. As appropriate only in centralized security models

**15.** What did the conviction of Uber's former chief security officer specifically concern?

> A. Concealment of a 2016 breach during an FTC investigation
>
> B. Failure to prevent the 2016 breach of customer data
>
> C. Misleading statements to investors about security practices
>
> D. Failure to implement an information security program

**16.** ISC2's 2025 study found 59% of respondents reporting critical or significant skills shortages, up from 44% a year earlier. What does the chapter conclude from this?

> A. That the headcount gap has been closed by automation
>
> B. That external hiring alone will resolve the shortage within a year
>
> C. That AI has reduced the need for cybersecurity skills
>
> D. That the problem is shifting from headcount to skills, favouring mapping and development

**17.** A champions program launched with enthusiasm is fading after a year. Which missing design condition does the chapter identify as producing nothing when absent?

> A. Protected time agreed with the champions' managers
>
> B. A larger central security team to review champions' work
>
> C. Mandatory annual awareness modules for all staff
>
> D. Public rankings of each team's vulnerability counts

**18.** Under a just culture, how should the organization respond to an employee who honestly clicks a convincing phishing link and reports it at once?

> A. Apply disciplinary consequences to deter future errors
>
> B. Publicly share the employee's name in awareness training
>
> C. Treat it as a learning opportunity and improve systems
>
> D. Exclude the employee from simulations to avoid skewing data

**19.** The 2026 Verizon DBIR found the human element involved in 62% of analyzed breaches. What broader conclusion does the chapter draw?

> A. That technical controls matter less than awareness training
>
> B. That most breaches are caused by malicious insiders
>
> C. That annual awareness training should become monthly
>
> D. That behaviour is shaped by roles, processes, tools, incentives, and culture and must be governed

**20.** In the Laurentians credit union scenario, what did the no-blame review identify as the underlying cause of the help desk fraud?

> A. A help desk agent who ignored the written procedure
>
> B. An unowned verification process and an exception never revisited
>
> C. A technical flaw in the core banking provider's platform
>
> D. A board that had refused to fund multifactor authentication
