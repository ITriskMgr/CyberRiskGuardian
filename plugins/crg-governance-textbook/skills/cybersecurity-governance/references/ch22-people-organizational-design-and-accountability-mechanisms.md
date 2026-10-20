<!-- Introduction to Cybersecurity Governance, 3rd ed., v2.1g (Aug 2026) - (c) Marc-André Léger. Licensed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/). Third-party standards, frameworks and quotations remain under their owners' terms. -->

# Chapter 22: People, Organizational Design, and Accountability Mechanisms

Organizations do not fail at cybersecurity mainly because they lack knowledge of controls. They fail because the controls, decisions, and responsibilities are not organized in a way that makes them executable, sustained, and defensible. Policies exist, but they are not operationalized. Security teams identify risks, but business leaders do not own them. Technical teams implement controls, but exceptions proliferate without accountability. Training is delivered, but behaviour does not change. When incidents occur, everyone asks who was responsible—and discovers that responsibility was ambiguous.

For a manager-oriented book, this is not a secondary topic. It is a central governance topic. Governance is ultimately the design of accountability: who is empowered to make decisions, who is obligated to act, who funds the work, who accepts residual risk, and what evidence exists that these responsibilities are being met.

This chapter therefore treats people, organizational design, and accountability mechanisms as an explicit governance domain. It provides operating-model patterns, decision rights structures, RACI and role clarity approaches, security champions models, incentive design, training governance beyond awareness, and a clear discussion of common failure modes in real organizations—especially the persistent pattern where security is owned by IT and business ownership of risk disappears.

## Cybersecurity as an operating model: the managerial lens

An operating model describes how work gets done in an organization: who performs which activities, how decisions are made, how work is funded, how conflicts are resolved, and how outcomes are measured.

In cybersecurity, the operating model must bridge two realities. The first is that many controls are technical and therefore implemented by IT, engineering, security operations, and cloud teams. The second is that the consequences and trade-offs are business consequences: service availability, customer trust, regulatory exposure, operational resilience, costs, and strategic risk. If the operating model collapses cybersecurity into technical implementation, then business leaders lose visibility into material risk and become spectators to decisions that affect business survival.

A mature cybersecurity operating model is therefore explicit about the separation between three functions.

The first is governance: setting policy, defining risk appetite, deciding minimum requirements, approving exceptions, and ensuring evidence exists.

The second is enablement: providing architecture, standards, tooling, patterns, and advisory capacity so that business and technical teams can build securely without constant reinvention.

The third is operations: monitoring, responding, maintaining controls, and continuously improving the environment.

Many organizations have these functions in some form, but they often blur responsibilities and decision rights. The consequence is predictable: security becomes reactive, exceptions multiply, and accountability becomes political.

## Decision rights: who decides what, and at what level

Cybersecurity governance becomes practical when decision rights are explicit. Decision rights define which roles have authority to decide, approve, or override decisions in key domains. Without decision rights, organizations rely on informal influence. Informal influence works until there is pressure; under pressure, ambiguity produces delays or uncontrolled unilateral actions.

A manager-oriented way to structure decision rights is to define a small set of cyber decisions that matter and assign them explicitly. Examples include approving risk acceptance for high-impact systems, approving architectural patterns with risk implications (such as cloud connectivity and network segmentation), approving exceptions to minimum security requirements, approving third-party onboarding for high-risk vendors, approving data classification and retention rules, approving incident declaration and crisis activation, and approving security investment priorities.

Decision rights should be tiered. Low-impact decisions should be delegated and standardized through policy. High-impact decisions should be elevated to appropriate leadership levels, with documented rationale and oversight visibility.

A common and damaging failure pattern is to place strategic security decisions in the hands of teams that control infrastructure but do not own business outcomes. This happens when IT decides acceptable risk without explicit business risk ownership. Governance maturity is demonstrated when business leaders own risk decisions for their services and data, while security and IT provide the expertise, standards, and controls that shape those decisions.

## RACI and role clarity: preventing gaps, overlaps, and everyone thought someone else did it

RACI models exist because cybersecurity work touches many roles and systems. In practice, the purpose of a RACI is not to produce a poster. Its purpose is to prevent three operational failures: gaps (no one does the work), overlaps (multiple teams do partial work inconsistently), and diffusion of responsibility (everyone assumes someone else owns it).

A practical RACI for cybersecurity governance should cover at least five recurring domains.

The first is risk management and exception handling: who identifies risk, who owns remediation, who accepts residual risk, and who documents decisions.

The second is control ownership: who is responsible for implementing and operating key controls such as identity and access management, privileged access, vulnerability management, secure configuration baselines, logging and monitoring, and backup and recovery.

The third is project integration: who ensures security requirements are included at the design stage, who approves security architecture, and who validates readiness before deployment.

The fourth is third-party security: who defines minimum requirements, who performs due diligence, who signs contracts, and who monitors ongoing assurance.

The fifth is incident management: who declares an incident, who leads operational response, who leads executive crisis management, who manages legal and privacy obligations, and who authorizes communications.

For each domain, clarity is achieved when accountability is named at the right level. For example, business owners should be accountable for the risk posture of their products and processes, while technical teams are responsible for control implementation. When accountability is placed too low, it becomes powerless; when it is placed too high without operational ownership below it, it becomes symbolic.

## Roles in a mature operating model: what executives should expect to exist

Organizations differ in size and complexity, but manager-oriented governance requires that certain roles or responsibilities exist, even if they are combined in smaller organizations.

A board or senior executive oversight function should have visibility into material cyber risk and ensure alignment with risk appetite. This may be a board committee, an executive risk committee, or another governance mechanism.

A Chief Information Security Officer (or equivalent) typically owns the security program: standards, architecture oversight, control assurance, incident readiness, and risk reporting. However, the CISO should not be the owner of all cyber risk. The CISO owns risk management processes and control strategy; business leaders own the risks associated with their products and processes.

The CIO or IT leadership owns many operational controls—especially infrastructure and enterprise platforms—and is therefore a critical partner. In cloud-heavy and product engineering contexts, engineering leadership and platform teams also become primary security control owners, particularly for secure development and cloud configuration.

A privacy officer or privacy leadership function participates where personal information and regulatory obligations are implicated. This becomes especially important in incident response, vendor contracting, and AI adoption.

Risk management and compliance functions provide oversight, integration with enterprise risk management, and consistency of reporting. Internal audit provides independent assessment and helps prevent governance theatre by testing effectiveness.

Finally, product owners, process owners, and data owners are the main business accountability points. Without their ownership, cybersecurity becomes a technical program rather than a risk-managed business reality.

## The security champion’s model: scaling security through the organization

One of the most widely effective organizational patterns for modern environments—especially agile development and cloud teams—is the security champion’s model. The core idea is that security capability must exist inside delivery teams, not only as an external review function.

A security champion is not a substitute for the security team. A champion is a team member in a product or engineering squad who is trained and supported to represent security concerns early, to incorporate security patterns into everyday work, to facilitate secure design discussions, and to act as a bridge between the security function and delivery teams.

The governance rationale is scale. Central security teams cannot review every change, every deployment, every configuration, every third-party integration, and every data flow. Champions create localized competence and reduce friction by catching issues earlier and embedding security thinking into daily work.

For champions to work, governance must define the model explicitly. Champions require role clarity, training requirements, time allocation (champions must have capacity, not only a title), a community of practice, access to security architects for escalation, and recognition or incentives aligned with their contribution. Without these mechanisms, champions programs become symbolic and fade.

A mature champions model also includes measurement that is meaningful: early security engagement rates, reduction in late-stage findings, improved adherence to secure patterns, and faster remediation of recurring issues. The objective is not number of champions, but improved security outcomes through operational integration.

## Incentives and performance: aligning behaviour with risk reality

Accountability mechanisms fail when incentives are misaligned. This is one of the most consistent patterns in cybersecurity governance.

If delivery teams are rewarded only for speed, features, and uptime, security becomes a perceived obstacle and will be bypassed. If security teams are rewarded only for policy compliance metrics, they will produce documentation rather than risk reduction. If executives are rewarded only for quarterly results without long-term risk accountability, investment in resilience will be deferred.

Incentive design does not require complex compensation schemes. It requires integrating security outcomes into the performance language of relevant roles. For product leaders, this may mean accountability for risk acceptance decisions and control readiness in major releases. For engineering leaders, it may mean vulnerability remediation performance for critical issues, secure configuration compliance, and incident readiness metrics. For IT operations, it may mean patching SLAs for high-risk assets, coverage of logging for critical systems, and successful recovery test outcomes. For executives, it may mean resilience targets and evidence-based risk reporting that demonstrates progress against board-approved objectives.

The objective is not punitive. It is to prevent the silent externalization of risk, where teams create risk for the organization but are not accountable for consequences. When incentives reflect risk reality, teams naturally internalize security priorities rather than treating them as external demands.

## Training governance beyond awareness: building competence, not only compliance

Most organizations can say they do security awareness. Fewer can show they govern training as a capability-building system. Governance beyond awareness involves defining training objectives by role, ensuring training quality, measuring competence outcomes, and integrating training into the operating model.

Role-based training is the central concept. Developers need secure coding patterns, dependency hygiene, secrets management, and threat modelling practices. Cloud administrators need secure configuration baselines, identity design, and logging and monitoring competence. Help desk and customer support teams need social engineering resistance and secure identity verification protocols. Executives need crisis decision-making readiness and understanding of risk trade-offs. Procurement and vendor management teams need contract requirements, assurance evidence interpretation, and escalation pathways.

Training governance should include at least four mechanisms.

First, a training standard: what training is required for which roles, at what frequency, with what minimum content.

Second, quality control: selecting or developing training that is credible and relevant to the organization’s technologies and risks, rather than generic modules that create fatigue.

Third, measurement: not only completion rates, but effectiveness indicators. Examples include reduction in repeated social engineering weaknesses, improved secure design outcomes, reduced recurrence of common configuration errors, and improved incident readiness performance in exercises.

Fourth, lifecycle integration: training is aligned with onboarding, role changes, and major technology transitions. Particularly in the context of cloud migration and AI adoption, training governance becomes a risk management method: it ensures that new capabilities are introduced with appropriate competence and discipline.

## Accountability mechanisms: how organizations make security enforceable

Accountability mechanisms are the tools by which roles and responsibilities become real. Without mechanisms, RACI and operating models remain aspirational.

Several mechanisms are consistently effective in practice.

Exception governance is fundamental. When teams cannot meet a security requirement, they request an exception. Governance requires that exceptions be owned, time-bound, risk-assessed, and approved by the appropriate risk owner. Exceptions should be tracked and reviewed, not allowed to accumulate invisibly.

Architecture and change governance is another mechanism. Major systems and high-risk changes should pass through structured design reviews with security criteria. The goal is not bureaucracy; it is early detection of structural design risks that are expensive to fix later.

Operational assurance mechanisms—configuration monitoring, control testing, internal audit, and periodic assessments—convert we think we are compliant into we can show we are effective. This also reduces compliance theatre by forcing evidence.

Incident post-mortems and learning loops are accountability mechanisms. A mature organization treats incidents and near-misses as governance signals. Root cause analysis includes managerial causes: decision delays, unclear authority, inadequate training, poor vendor accountability, underfunded controls, and broken risk acceptance processes. Remediation actions are assigned and tracked with executive oversight.

Finally, budgeting and investment governance is itself an accountability mechanism. Controls that are required must be funded. If leadership expects security outcomes but funds only minimal staffing and tooling, accountability becomes unfair and ineffective. Governance maturity aligns expectations with resources.

## Common organizational failure patterns (and how to fix them)

Several accountability failures appear repeatedly across organizations. A manager-oriented chapter should name them explicitly because they are predictable.

One common failure pattern is cybersecurity owned by IT with no business ownership of risk. In this pattern, security becomes a technical negotiation between IT and security teams, while business leaders treat risk as external. The fix is explicit risk ownership: business owners are accountable for risk acceptance and must be part of exception approvals and prioritization decisions.

A second failure pattern is the CISO as scapegoat model, where the CISO is held responsible for enterprise risk without authority over budgets, priorities, or business decisions. The fix is to align accountability with control: the CISO owns the security program, but business leaders own their risk and must accept residual risk explicitly.

A third failure pattern is compliance theatre as a substitute for control effectiveness. Here, success is measured by policy completion, training completion, and audit artefacts rather than by risk outcomes. The fix is evidence and testing: control effectiveness must be verifiable through monitoring, exercises, and validated coverage metrics.

A fourth failure pattern is security review as a late-stage gate. When security enters at the end of projects, it becomes friction, exceptions rise, and insecure design becomes locked in. The fix is operating model design: embed security early through standards, architecture patterns, champions, and early-stage review triggers.

A fifth failure pattern is tool-driven security without operational capacity. Organizations purchase tools that generate alerts, dashboards, and reports, but lack the staffing and workflows to act on them. The fix is to govern security as a capability, not as procurement: invest in processes, ownership, training, and measured outcomes.

A sixth failure pattern is unclear incident authority. During crises, decision delays amplify harm. The fix is executive crisis governance and rehearsed decision rights: who can shut down systems, who can communicate, who can engage external counsel, and who can authorize extraordinary measures.

Naming these patterns matters because governance is not only about ideals; it is about preventing predictable organizational failures.

## A practical blueprint: implementing accountable governance in stages

Organizations often attempt to redesign operating models in one large transformation. In practice, staged implementation is more effective and easier to sustain.

A first stage is role clarity and decision rights. Define risk owners by business domain, define security control owners, and define escalation and exception pathways. This stage creates governance visibility immediately.

A second stage is integration into delivery and operations. Implement security champions, embed security review triggers in project and change lifecycles, and standardize secure patterns so that teams can comply without constant negotiation.

A third stage is training governance and competence building. Introduce role-based training with measurable objectives, supported by communities of practice and operational coaching. This stage reduces repeated errors and improves sustainability.

A fourth stage is strengthening assurance and evidence. Expand monitoring, control testing, and internal audit alignment. This stage reduces compliance theatre and improves defensibility.

A fifth stage is institutionalization. Governance becomes part of routine management reviews and performance systems, and cybersecurity is treated as a normal dimension of running products and processes, rather than an episodic IT concern.

The details vary by organization, but a staged approach keeps governance practical: each stage produces visible value, reduces risk, and increases the organization’s capacity to adopt the next stage.

## Bringing it together: accountability is the mechanism that makes cybersecurity governable

A manager-facing cybersecurity governance program succeeds when it makes responsibility and consequence align. This chapter has argued that people, organizational design, and accountability mechanisms are not soft topics but the primary mechanism by which security becomes executable.

When decision rights are clear, risk ownership is anchored in the business, security champions scale capability, incentives align behaviour with risk reality, training builds role-based competence, and accountability mechanisms generate evidence and learning, cybersecurity stops being a perpetual dispute between IT and the business. It becomes a governed system of decision-making and execution.

The core governance lesson is therefore simple. Organizations do not secure themselves by wishing for better technology. They secure themselves by designing accountability so that the right people make the right decisions at the right time, supported by standards, evidence, and incentives that make secure behaviour the path of least resistance.

### Appendix A: Sample Balanced Cybersecurity Scorecard with Indicators

| **Scorecard perspective**    | **Indicator type** | **Objective**                                                   | **Metric**                                                                                 | **Illustrative target**                                                | **Accountable owner**                                | **Primary data source**                                     | **Escalation threshold**                                                                                   |
|------------------------------|--------------------|-----------------------------------------------------------------|--------------------------------------------------------------------------------------------|------------------------------------------------------------------------|------------------------------------------------------|-------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| **Learning and growth**      | **Leading**        | Strengthen cybersecurity awareness and role-specific competence | Percentage of employees completing required cybersecurity training on time                 | At least 95% completion                                                | Human Resources and CISO                             | Learning-management system and HR records                   | Escalate when completion falls below 90% or high-risk personnel remain untrained                           |
| **Internal processes**       | **Leading**        | Reduce vulnerability exposure                                   | Percentage of critical vulnerabilities remediated within the approved deadline             | At least 95% remediated on time                                        | Infrastructure or application owner                  | Vulnerability-management and ticketing systems              | Escalate when an internet-facing critical vulnerability exceeds the deadline or compliance falls below 90% |
| **Internal processes**       | **Leading**        | Maintain appropriate access privileges                          | Percentage of privileged and high-risk access rights reviewed and recertified on schedule  | 100% each review cycle                                                 | IAM owner and business access owners                 | IAM/PAM platform, HR records and access-review reports      | Escalate when privileged access remains uncertified or unauthorized access is identified                   |
| **Learning and growth**      | **Leading**        | Improve incident and recovery readiness                         | Percentage of scheduled tabletop, crisis-management and recovery exercises completed       | 100% of scheduled exercises                                            | CISO, Business Continuity Lead and executive sponsor | Exercise calendar and after-action reports                  | Escalate when a critical exercise is missed or corrective actions remain unassigned                        |
| **Internal processes**       | **Leading**        | Improve control remediation                                     | Percentage of critical audit, incident and exercise findings closed by the agreed deadline | 100% of critical findings and at least 90% overall                     | Responsible executive and control owner              | GRC platform and remediation tracker                        | Escalate any overdue critical finding or repeated high-risk issue                                          |
| **Customer and stakeholder** | **Lagging**        | Protect sensitive information and stakeholder trust             | Number of confirmed material cybersecurity or privacy breaches                             | Zero material breaches                                                 | CISO and Chief Privacy Officer                       | Incident and privacy-breach registers                       | Immediate executive and board escalation for any material breach                                           |
| **Customer and stakeholder** | **Lagging**        | Maintain reliable critical services                             | Number and duration of unplanned outages affecting critical services                       | No outage exceeding the approved recovery-time objective               | Business service owner and CIO                       | Service-monitoring platform and service desk                | Escalate when an outage exceeds the service-level or recovery-time threshold                               |
| **Financial and risk**       | **Lagging**        | Limit financial harm from cyber incidents                       | Total direct and indirect financial losses attributable to cybersecurity incidents         | Within approved risk tolerance, with year-over-year reduction          | Chief Risk Officer and Chief Financial Officer       | Incident-loss records, insurance claims and finance systems | Escalate when losses exceed the approved tolerance or forecast                                             |
| **Internal processes**       | **Lagging**        | Contain serious incidents rapidly                               | Median time to contain high-severity incidents                                             | Target based on the organization’s baseline and risk tolerance         | SOC Manager or Incident Response Lead                | SIEM, SOAR and incident-management platform                 | Immediate escalation when containment exceeds the approved threshold                                       |
| **Internal processes**       | **Lagging**        | Reduce recurring incidents                                      | Percentage of material incidents caused by previously identified control weaknesses        | Downward trend, with zero recurrence of unresolved critical weaknesses | CISO and responsible control owners                  | Incident reviews, root-cause analyses and audit records     | Escalate any repeated material incident linked to an overdue corrective action                             |

**Leading indicators** measure preparedness and control execution before harm occurs. Examples include training completion, patch timeliness, access reviews and exercise completion.

**Lagging indicators** measure realized outcomes after an event or control failure. Examples include breaches, outages, financial losses and incident-containment times.

The scorecard should use both types together. A strong leading indicator does not guarantee that incidents will not occur, while a lagging indicator alone may reveal weaknesses only after damage has already happened.
