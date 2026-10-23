<!-- Léger, M.-A. (2026). Introduction to Cybersecurity Governance for Business Technology Management (3rd ed., v2.51). Chapter 17: Cybersecurity Economics and Investment Governance. CC BY-NC 4.0 -->

# Chapter 17: Cybersecurity Economics and Investment Governance

## Learning outcomes

After completing this chapter, you should be able to:

1.  Explain the four outcomes that cybersecurity investment buys and why cybersecurity is a difficult investment domain to govern.

2.  Calculate single loss expectancy, annualized loss expectancy, and return on security investment for a scenario and a proposed control.

3.  Distinguish point estimates of expected loss from FAIR-based simulation and loss exceedance curves, and explain how each supports risk appetite.

4.  Apply the Gordon–Loeb sanity check and total cost of ownership to evaluate whether a proposed control is economically proportionate.

5.  Compare the monetary ALE view with the CyberRiskGuardian risk index, and interpret risk-to-tolerance ratios after treatment.

6.  Classify investments as mandatory, foundational, or discretionary and design a security investment portfolio with common evaluation criteria.

7.  Evaluate managed services, security platforms, and cyber insurance as outcome instruments rather than substitutes for accountability.

8.  Recommend an appetite-consistent cybersecurity budget and justify the residual risks the organization formally accepts.

Managers rarely struggle to accept that cybersecurity matters. They struggle to decide what to do first, how much to spend, and how to explain those choices in a language that executives, boards, and finance teams recognize as rational. A chief information security officer (CISO) may fail to obtain funding because the request reads as a list of tools rather than a business case; a chief financial officer (CFO) may approve a large budget after a widely reported incident, only to discover later that much of it bought licences nobody operates. Neither problem is technical. Both are failures of **investment governance**: the set of decision rights, criteria, evidence, and oversight mechanisms through which an organization decides how much to spend on cybersecurity, on what, and how it will know whether the spending worked.

Cybersecurity is an unusual investment domain because its primary objective is not revenue generation but loss avoidance, operational reliability, and the protection of trust. Its benefits are counterfactual: when cybersecurity works, the most visible result is that nothing happens. Budgets therefore compete with initiatives that promise visible growth, and they compete at a disadvantage. If cybersecurity leaders cannot translate security work into financial and operational outcomes—reduced expected loss, shorter outages, reduced legal exposure, reduced tail risk, improved recoverability—the organization will oscillate between underinvestment in quiet years and reactive overspending after incidents. That oscillation is itself expensive, because capabilities built in a panic are rarely well designed, well integrated, or sustained.

A second reason an economic lens is essential is that security demand is effectively unlimited. There is always another control, tool, service, or assessment that could be purchased. Without an investment logic, organizations drift toward one of two dysfunctional extremes. In the first, security becomes a purely technical quest for maximum protection, where every gap is treated as unacceptable and every new product category as necessary. In the second, security becomes a compliance exercise focused on checklists and audit artifacts, where the goal is to pass the next review rather than to reduce the chance and severity of harm. Both extremes waste money, and both are forms of poor governance.

This chapter builds directly on earlier material. Chapter 4 introduced risk appetite and tolerance, the risk triangle of threat, vulnerability, and impact, and the idea that cybersecurity spending should be calibrated to appetite. Chapters 5 and 6 showed how to build cybersecurity risk scenarios and how to assess them with the CyberRiskGuardian method, producing an estimated and a tolerated risk for each scenario, a risk-to-tolerance ratio, a costed portfolio of treatment measures, and the key risk indicators (KRIs) that monitor whether those risks are moving. Chapter 9 explained how control frameworks such as ISO/IEC 27001:2022, NIST CSF 2.0, COBIT 2019, and CIS Controls v8.1 structure control selection and assurance. Here, those ideas converge on a single managerial question: given limited resources and an explicit risk appetite, which cybersecurity investments are defensible, and how should they be governed over time?

The chapter frames cybersecurity as an investment portfolio governed by risk appetite, expected loss reduction, tail-risk constraints, and demonstrable outcomes. It begins by clarifying what cybersecurity spending actually buys, then introduces the basic financial model of expected loss, its classic annualized loss expectancy form, and its limits. It explains, at a conceptual level, how the FAIR approach, Monte Carlo simulation, and loss exceedance curves improve on single-number estimates. It then examines control economics, return on security investment and its pitfalls, the Gordon–Loeb model, and total cost of ownership, before working through a numeric example linked to the scenario-based method of Chapters 5 and 6. The second half of the chapter addresses compliance theatre, maturity, prioritization, portfolio governance, board communication, managed services, tooling and platform consolidation, cyber insurance, and budgeting as a governance process.

## What cybersecurity investment is buying

The first step in investment governance is clarifying what the organization is buying when it buys cybersecurity. This sounds obvious, yet many budget discussions begin with products ("we need a new detection platform") rather than outcomes ("we need to reduce the likelihood that a ransomware intrusion goes undetected for more than a few hours"). When the economic object is unclear, every proposal looks equally urgent, and the loudest or most recent argument tends to win.

### Four outcomes that justify spending

At a high level, cybersecurity investment buys four types of outcome. Each can be expressed in business terms, and each connects to the risk triangle introduced in Chapter 4.

- **Loss reduction.** Investment reduces the probability that threats succeed and reduces the impact when they do. In risk terms, it lowers likelihood, impact, or both, for specific scenarios such as fraudulent payments, data exposure, or disruption of critical services.

- **Survivability.** Investment increases the organization's ability to continue operating during disruption and to recover quickly without depending on the goodwill of attackers. With ransomware and destructive attacks, tested recoverability is often the final safeguard that prevents a serious incident from becoming an existential one (see Chapter 12).

- **Compliance capability.** Investment enables the organization to meet legal, regulatory, and contractual obligations, such as those under Québec Law 25, PIPEDA, OSFI Guideline B-13 for federally regulated financial institutions, or customer contracts. The value is concrete: it prevents enforcement actions, penalties, contractual damages, and forced business interruptions. However, compliance capability has value only when it is real in operations, not merely documented.

- **Trust preservation.** Investment reduces the probability of reputational damage, customer attrition, partner hesitation, and intensified regulatory scrutiny after incidents. Trust is difficult to price precisely, but boards regularly see its value when it is lost.

These outcomes share a managerial implication. **Cybersecurity is not best governed as a technology shopping category; it is best governed as an enterprise risk-control portfolio whose value is expressed in avoided and reduced harm, improved operational stability, and the ability to demonstrate diligence.**

### Why cybersecurity is a difficult investment domain

Several features make cybersecurity harder to govern economically than most capital programs. The first is the counterfactual problem already mentioned: benefits are measured against incidents that did not happen, which cannot be observed directly. The second is adversarial adaptation. A new warehouse does not change its behaviour because a competitor has studied it, but attackers do adapt to defences, so the value of a control can decay as threat actors find ways around it. The third is the scarcity of reliable loss data. Many organizations have few major incidents of their own to learn from, and industry data, while useful, describes other organizations with different exposures.

A fourth feature is interdependence. Controls rarely work alone: multifactor authentication depends on identity governance, detection depends on logging, and recovery depends on backups that attackers cannot reach. The benefit of one investment therefore depends on others already in place, which makes simple addition of benefits misleading. Finally, some of the harm from a breach falls on customers, partners, and society, which is one reason regulators impose duties that a narrow profit calculation would not justify. Good investment governance does not wish these difficulties away; it designs decision processes that acknowledge them.

### Decision rights over cybersecurity spending

Investment governance begins with clarity about who decides. In most organizations, the board approves risk appetite and oversees whether management's investment choices are consistent with it; executive management allocates resources across competing priorities; the CISO and technology leaders propose and execute the cybersecurity program; risk owners in the business accept or reject residual risk for the processes they own; and internal audit provides independent assurance that the process is working. This is the three lines model applied to money.

Frameworks reinforce this allocation. NIST CSF 2.0 places resource decisions inside the **Govern** function, expecting that adequate resources be allocated commensurate with the cybersecurity risk strategy, roles, and policies, and that leadership oversee the results of that strategy. COBIT 2019 makes the same point at the enterprise level through its objectives for ensured benefits delivery, ensured resource optimization, managed portfolio, and managed budget and costs. ISO/IEC 27001:2022 requires top management to ensure that the resources needed for the information security management system are available. None of these frameworks prescribes a budget figure. All of them expect that the budget be the result of a governed process that links resources to risk.

## The core financial model: from incident scenarios to expected loss

At the centre of cybersecurity economics is a simple concept: **expected loss**. Organizations do not know which incident will occur next, but they can estimate how often different kinds of loss events are likely to occur and how much they would cost, and they can then ask which investments reduce those losses most effectively. The quality of the estimates matters less than managers fear, provided that the estimates are consistent, transparent, and used to compare options rather than to predict the future.

### Scenarios as the unit of analysis

A practical way to model cyber risk is in scenario terms, as developed in Chapter 5. A scenario is a meaningful business event driven by cyber causes: ransomware disrupting critical operations; credential compromise enabling fraudulent payments; cloud misconfiguration exposing regulated personal information; compromise of a software supplier causing supply-chain disruption; insider misuse of privileged access; and similar events. Each scenario describes a threat, the vulnerabilities it exploits, the assets and processes affected, and the business consequences.

Scenarios are the right unit for investment decisions because they connect technical causes to business harm. A board cannot evaluate a request to "improve endpoint telemetry," but it can evaluate a proposal to reduce the probability that a ransomware intrusion halts order fulfilment for a week. Scenarios also make trade-offs visible. When the same investment reduces risk in several scenarios at once, the scenario view shows its leverage; when an investment affects only a minor scenario, the scenario view shows that too.

The scenario set should be small enough to manage and large enough to be representative. Most organizations find that five to fifteen well-constructed scenarios capture their material exposures. The CyberRiskGuardian approach of Chapters 5 and 6, which screens twenty candidate scenarios down to the ten most material, is a reasonable starting point.

### Single loss expectancy, annual rate of occurrence, and annualized loss expectancy

The oldest and simplest quantitative model of cyber risk uses three terms. The **single loss expectancy (SLE)** is the expected monetary loss if a scenario occurs once. In its classic form, it is calculated as the value of the affected asset multiplied by an exposure factor representing the share of that value lost in the event. In practice, managers more often build SLE directly by summing cost categories: response and forensics, restoration, business interruption, legal and notification costs, regulatory consequences, and customer impact. The **annual rate of occurrence (ARO)** is the expected number of times the scenario occurs per year. An ARO of 0.2 means the scenario is expected about once every five years; an ARO of 3 means about three times per year.

The **annualized loss expectancy (ALE)** is the product of the two:

- **ALE = SLE × ARO.** If a fraudulent-payment scenario is expected to cost \$160,000 per occurrence and to occur about once every two years (ARO of 0.5), its ALE is \$80,000.

Summed across scenarios, ALE gives a rough estimate of the organization's expected annual cyber loss. The model has real virtues. It is easy to explain to a finance committee, it forces analysts to separate frequency from magnitude, and it produces a number in the same unit as the budget. It also supports a simple first test of any proposed control: does the reduction in ALE it produces exceed its annual cost?

### The limits of a single expected value

ALE is a useful starting point, but managers must understand its limits before using it to justify investment.

- **False precision.** Multiplying two uncertain estimates produces a result that looks precise but is not. An SLE that could plausibly be anywhere between \$500,000 and \$5 million, multiplied by an ARO that could be 0.05 or 0.3, yields an ALE whose true range spans more than an order of magnitude.

- **Hidden tails.** An average conceals the shape of the distribution. Two scenarios with identical ALE may differ completely: one produces frequent small losses, the other a rare loss large enough to threaten the organization. Expected value treats them as equivalent; a prudent board does not.

- **Rare events.** For low-frequency scenarios, the ALE is a sum that will almost never be experienced. A scenario with an ARO of 0.05 and an SLE of \$10 million has an ALE of \$500,000, yet in nineteen years out of twenty the loss is zero and in the twentieth it is \$10 million.

- **Static assumptions.** ALE estimates are snapshots; threat activity, exposure, and asset values change, so estimates must be refreshed.

- **Additivity problems.** Summing scenario ALEs can double-count losses when scenarios overlap, and summing the benefits of several controls can overstate their combined effect when they protect against the same failure.

These limits do not make ALE useless. They mean it should be presented as a range with stated assumptions, complemented by an explicit view of worst plausible outcomes, and used to compare options rather than to promise results.

### Hard constraints and tail risk

Two clarifications prevent expected loss from becoming the only decision criterion. First, **some risks are intolerable regardless of their expected value**. Scenarios involving threats to safety, severe harm to individuals, catastrophic regulatory exposure, or existential business disruption are hard constraints. Investment decisions for these scenarios are driven by risk appetite, legal obligation, and ethics, not by expected-value arithmetic. A hospital does not decide whether to protect infusion pumps by comparing ALE to cost; it protects them because patient harm is outside its appetite (see Chapter 11).

Second, **expected loss is often dominated by tail events**. A year may pass with no major incident, followed by one ransomware event that overwhelms several years of average losses. Investment governance must therefore consider not only the mean but also high-impact, low-frequency outcomes—the tail risk that threatens mission and solvency. This is why the CyberRiskGuardian method of Chapter 6 asks for both an expected damage (δe) and a maximum damage (δm) for every scenario and carries their average into the estimated risk, and why boards should ask to see the plausible worst case alongside the most likely one.

The figure below summarizes the path from scenarios to an expected-loss view that can support investment decisions.

<span id="_Toc243635571" class="anchor"></span>Figure 72: From incident scenarios to expected loss

## Beyond point estimates: FAIR, simulation, and loss exceedance curves

As organizations mature, many move from single-number estimates toward methods that represent uncertainty explicitly. This field is often called **cyber risk quantification**. It is not necessary for sound governance—a disciplined qualitative or semi-quantitative approach, consistently applied, is far better than an elaborate model nobody trusts—but it can improve transparency about uncertainty and reduce emotionally driven spending. ISO/IEC 27005:2022 explicitly accommodates qualitative, semi-quantitative, and quantitative approaches, so the choice is one of proportionality rather than conformity.

### The FAIR taxonomy in managerial terms

The most widely used quantitative framework is **FAIR** (Factor Analysis of Information Risk). Its open standard form, maintained by The Open Group as **Open FAIR**, consists of two documents: the Risk Taxonomy (O-RT) Standard, Version 3.0.1, and the Risk Analysis (O-RA) Standard, Version 2.0.1. The taxonomy defines risk as "the probable frequency and probable magnitude of future loss," and it decomposes each into factors that can be estimated separately.

On the frequency side, **loss event frequency** depends on **threat event frequency** (how often threat agents act against an asset in a way that could cause loss) and **vulnerability**, defined in FAIR as the probability that a threat event becomes a loss event. Threat event frequency itself depends on how often threat agents come into contact with the asset and how likely they are to act. On the magnitude side, FAIR separates **primary loss**, borne directly by the organization as a result of the event, from **secondary loss**, which arises when external stakeholders such as customers, regulators, or litigants react. It also classifies losses into six forms: productivity, response, replacement, fines and judgments, competitive advantage, and reputation.

For managers, the value of this decomposition is discipline rather than mathematics. It forces analysts to state what they are estimating, prevents the common confusion between a threat, a vulnerability, and a risk, and makes it possible to explain why a control matters: it either reduces how often loss events occur or reduces how much they cost. Note that FAIR's use of the word "vulnerability" as a probability differs from the everyday technical meaning of a software flaw scored with CVSS v4.0; analysts should be explicit about which meaning they are using.

### Monte Carlo simulation in plain terms

FAIR analyses, and many other quantification approaches, express each input as a range rather than a single value. An analyst might estimate that a ransomware scenario's loss event frequency is at least once in twenty years, at most once in three years, and most likely once in eight, and that its loss magnitude lies between \$400,000 and \$8 million, most likely around \$2 million. These ranges are typically elicited from calibrated experts and supported by internal incident data, insurance claims information, and industry studies.

**Monte Carlo simulation** combines these ranges. A computer simulates thousands of possible years. In each simulated year, it draws a frequency and, for each simulated event, a magnitude from the specified distributions, then records the total annual loss. After, say, ten thousand simulated years, the results form a distribution: how often the annual loss was zero, how often it exceeded \$1 million, how often it exceeded \$10 million, and so on. The average of this distribution corresponds to the expected loss, but the distribution also shows what the average hides.

Managers do not need to run simulations themselves; they need to understand what the output means and how sensitive it is to its assumptions. A useful discipline is to ask which two or three inputs, if wrong, would change the decision; those inputs deserve the most scrutiny and the best evidence.

### Loss exceedance curves and risk appetite

The most useful output of a simulation for governance is the **loss exceedance curve**. It plots, for each possible loss amount, the probability that annual losses will exceed it. A curve might show, for example, a 40% chance of losses above \$100,000 in a year, a 10% chance above \$1 million, and a 1% chance above \$15 million. Reading the curve answers the questions boards actually ask: How bad could a bad year be? How likely is a loss that would breach our covenants or exhaust our insurance?

Loss exceedance curves also give risk appetite a concrete form. A board can express its tolerance as a curve of its own—for instance, "no more than a 5% chance per year of cyber losses above \$3 million, and no more than a 1% chance above \$10 million." Where the organization's current loss exceedance curve lies above the tolerance curve, investment or risk transfer is needed. Proposed investments can be modelled by shifting the curve: a recovery program lowers the right-hand tail, a fraud-prevention control lowers the frequent small losses on the left. Insurance can be represented by showing how the net-of-insurance curve differs from the gross curve, which makes the value of a policy limit visible.

| A single expected-loss number answers "what is the average?"; a loss exceedance curve answers "how bad could it get, and how likely is that?"—the question boards are actually accountable for. |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

### FAIR-CAM and the economics of control effectiveness

Traditional quantification often struggled with a key question: how much does a given control actually reduce risk? The **FAIR Controls Analytics Model (FAIR-CAM)**, introduced by the FAIR Institute in 2021 and published as a formal standard (Version 1.0) in January 2025, addresses this by describing how controls affect loss. It groups control functions into three domains. **Loss event control functions** act directly on loss events, for example by preventing, detecting, or limiting them through authentication or access privileges. **Variance management control functions** act indirectly by keeping other controls performing as intended—patching, configuration management, and auditing, for instance, prevent and correct the drift that degrades protection. **Decision support control functions** act indirectly by improving the quality of management decisions, through asset inventories, threat intelligence, and risk reporting.

FAIR-CAM carries three lessons that matter to investment governance even for organizations that never adopt FAIR formally. First, a control's risk-reduction value depends on its **operational efficacy**—whether it is working as intended now—not on whether it was purchased or documented. Second, controls depend on one another: an excellent patching tool is worth little without a reliable asset inventory, which is why foundational capabilities often deliver more value than their direct effect suggests. Third, governance and decision-support capabilities have measurable economic value, because poor decisions are a source of loss in their own right.

### Connecting quantification to the CyberRiskGuardian approach

The CyberRiskGuardian method introduced in Chapters 5 and 6 deliberately uses a dimensionless calculated risk index rather than dollars. For each scenario, the **estimated risk (Re)** multiplies the probability that the threat is present, Pb(A), the probability that it exploits the weakness given existing controls, Pb(ψ,A), the CVSS v4.0 score of the weakness, the average of expected damage δe and maximum damage δm, and the organizational criticality or expected utility μ(E) of the affected service, divides the product by organizational resilience θ, and scales it by a multiplication factor of 1,000: Re = Pb(A) × Pb(ψ,A) × CVSS × ((δe + δm) ÷ 2) × μ(E) ÷ θ × 1,000. The **tolerated risk (Rt)** uses the same terms with the organization's appetite in place of the average damage. Treatment is expressed as a probability reduction and an impact reduction, each the share removed; the mitigated risk is Re × probability reduction × impact reduction, and the residual risk is Re minus the mitigated risk. The residual is then compared with the tolerated risk: a risk-to-tolerance ratio below 0.90 is read as below tolerance, 0.90 to 1.10 as approximately at tolerance, and above 1.10 as above tolerance. The index is a relative indicator for comparison and prioritization, not a dollar amount, and it should not be confused with the KRIs that monitor risk drivers over time. This design has a pedagogical and practical advantage: it reduces the emotional reactions that monetary figures can trigger and keeps attention on relative priorities and on whether residual risk is within tolerance.

Monetary quantification and the risk index are complementary rather than competing. The risk-to-tolerance ratio tells management whether each scenario's residual risk is within the tolerance implied by its appetite; a monetary estimate tells the CFO whether the cost of getting there is proportionate to the loss avoided. Because both rest on the same scenario structure—a threat, a vulnerability, a likelihood, an impact, and the effect of resilience—the same workshop inputs can feed both. They do not, however, measure risk reduction in the same way. In an ALE calculation, a treatment that cuts frequency and a treatment that cuts severity each shrink expected loss on their own; in the CyberRiskGuardian convention, the two reductions are multiplied, so a package is credited only to the extent that it acts on both likelihood and consequence. As the worked example later in this chapter shows, the two views can therefore point in different directions, and the divergence is itself informative.

For organizations that wish to connect scenario-level analysis to enterprise risk management, the NIST IR 8286 series on integrating cybersecurity and enterprise risk management offers guidance on recording cyber risks in risk registers, estimating them, and staging them for executive and board oversight; NIST IR 8286C Rev. 1, published in December 2025, specifically addresses aggregating cybersecurity risks for governance oversight.

## Control economics: cost of control versus risk reduction

Once risks are expressed as scenarios with plausible loss distributions, investments can be evaluated as risk-reducing controls. Controls have costs, and they produce benefits by reducing expected loss and, ideally, by reducing tail risk. Control economics is the discipline of comparing the two honestly.

### Marginal risk reduction per dollar

The simplest justification compares the annualized cost of a control with the annualized reduction in expected loss it provides. If a control costs \$500,000 per year and is expected to reduce ALE by \$1.5 million, it is economically attractive, subject to uncertainty and operational feasibility. If it costs \$2 million and reduces ALE by \$300,000, it may still be required for legal reasons or to address an intolerable scenario, but it should not be defended as a good return on investment. Being explicit about the reason a control is funded—economic, mandatory, or tail-risk-driven—is itself an act of good governance, because it prevents the organization from using inflated benefit estimates to justify decisions that were really made for other reasons.

This framing leads to a disciplined managerial question: **what is the marginal risk reduction achieved per dollar spent?** The word "marginal" matters. The first investment in a domain often produces large benefits; each additional dollar typically produces less. A mature program asks not whether a control is good in general, but whether the next increment of spending on it produces more risk reduction than the same money would produce elsewhere in the portfolio.

### Return on security investment and its caveats

A common way to express control economics is **return on security investment (ROSI)**:

- **ROSI = (reduction in ALE − annual cost of the control) ÷ annual cost of the control.** A control costing \$200,000 per year that reduces ALE by \$420,000 has a ROSI of (\$420,000 − \$200,000) ÷ \$200,000, or 110%.

ROSI is attractive because it resembles the return metrics finance teams use elsewhere. Used carelessly, however, it can mislead. Managers should keep several caveats in view.

- **The benefit is an estimate of something unobservable.** Unlike revenue from a new product, avoided loss cannot be measured after the fact. ROSI is therefore only as credible as the scenario estimates behind it, and should be presented with a range rather than a single percentage.

- **Benefits do not simply add up.** If two controls each reduce the same scenario's likelihood by half, their combined effect is a reduction to one quarter, not to zero. Summing individual ROSI figures overstates the value of a portfolio.

- **ROSI ignores tail risk.** A control can have modest ROSI yet be essential because it lowers the worst plausible outcome below the board's tolerance. Conversely, a high-ROSI control that only addresses frequent small losses may leave the existential scenario untouched.

- **Vendor ROSI claims are not evidence.** Return figures offered in sales material are built on assumptions about someone else's environment. They may be useful as hypotheses, but the organization must re-estimate them using its own scenarios.

- **Time matters.** Multi-year investments should be evaluated with the organization's normal capital-budgeting tools, such as net present value, rather than a single-year ratio, and the benefit should start only when the control is actually operating, not when it is purchased.

The managerial conclusion is that ROSI is a useful comparative indicator within a portfolio, not a stand-alone justification. **A low or negative ROSI does not mean a control should be rejected, and a high ROSI does not mean it should be approved; it means the reasoning must be explicit.**

### The Gordon–Loeb model

Economists have also asked how much an organization should invest to protect a given information set. The best-known answer is the **Gordon–Loeb model**, published by Lawrence Gordon and Martin Loeb in "The Economics of Information Security Investment" in *ACM Transactions on Information and System Security* in 2002. The model considers an information set with a potential loss, a probability that it will be breached, and a function describing how investment reduces that probability. Under the classes of breach-probability functions the authors examined, they showed that the optimal investment should not exceed about 37% (one divided by the mathematical constant *e*) of the expected loss from a breach.

The model's managerial value is less the 37% figure than the reasoning behind it. Because each additional dollar reduces breach probability by less than the previous one, it is rarely optimal to spend anything close to the full expected loss on protection; and the optimal investment does not always rise with vulnerability, because some extremely vulnerable assets are too costly to protect effectively. The model also reinforces the idea that investment decisions should be made per information set or scenario rather than for "security" as a whole.

Its limits are equally important. The model assumes a risk-neutral decision-maker, a single period, and a well-behaved relationship between spending and breach probability. Later research has shown that other plausible functions can justify higher investment, and real organizations face hard constraints, regulatory duties, and tail risks that a risk-neutral expected-value model does not capture. The Gordon–Loeb result is best used as a **sanity check**: if a proposed investment to protect a scenario is much larger than a third of that scenario's expected loss, the proposal should explain clearly why—usually because of tail risk, legal obligation, or benefits that extend to other scenarios.

### Total cost of ownership

Control costs must be measured properly. Many organizations undercount security costs because they focus only on licensing or procurement. The **total cost of ownership (TCO)** of a control includes acquisition and implementation, integration with existing systems, internal labour, tuning, false-positive handling, training, ongoing operations, renewal increases, eventual replacement or exit, and the opportunity cost of the people who run it. A control that is cheap to buy but expensive to operate may be economically inferior to one that is expensive to buy but inexpensive to run.

A detection platform illustrates the point. Its true cost includes endpoint agent management, data ingestion and storage, alert triage labour, investigation overhead, integration with identity and cloud telemetry, content development, and the cost of retaining skilled analysts who can interpret its outputs. In some organizations, the people and data costs exceed the licence several times over. In investment governance, this is not a technical complaint; it is a financial reality that belongs in the business case.

For comparison across options, TCO should be annualized. Capital expenditures are spread over the expected useful life of the capability, recurring operating expenditures are added, and internal staff time is costed at a loaded rate. The CyberRiskGuardian treatment records of Chapter 6 already separate one-time cost, recurring cost, and additional staff time, expressed over the same 12-month horizon as the scenarios, for this reason. Annualized TCO is the cost figure that belongs in ALE and ROSI comparisons.

## Worked example: from scenarios to an investment decision

Numbers make these concepts concrete. The following example is deliberately simple and entirely fictional. Its purpose is to show the reasoning, not to suggest realistic values for any particular industry.

### Illustrative scenario: a Québec distributor prepares its cybersecurity budget

Boréal Distribution is a fictional wholesale distributor headquartered in Québec, with about 600 employees, three warehouses, an enterprise resource planning (ERP) system that runs order management and logistics, and a growing customer portal hosted in a public cloud. Its annual revenue is about \$180 million. After completing a scenario-based assessment using the method of Chapters 5 and 6, the risk committee has retained ten scenarios. Three of them dominate the discussion, and management has commissioned rough monetary estimates for each, expressed as most likely values with ranges recorded in the supporting notes.

- **S1: Ransomware disrupts the ERP and warehouse systems.** The most likely loss per occurrence (SLE) is estimated at \$3 million, combining about ten days of disrupted shipments, recovery costs, forensic and legal fees, and customer credits. The ARO is estimated at 0.2, about once in five years given sector threat activity. ALE is \$3,000,000 × 0.2 = \$600,000. A plausible worst case, involving data theft and prolonged outage, is estimated at \$12 million.

- **S2: Business email compromise leads to a fraudulent supplier payment.** SLE is \$160,000; ARO is 0.5. ALE is \$160,000 × 0.5 = \$80,000.

- **S3: A cloud storage misconfiguration exposes customer personal information.** SLE, including notification, Law 25 obligations, legal costs, and customer impact, is \$1.5 million; ARO is 0.1. ALE is \$1,500,000 × 0.1 = \$150,000.

The combined ALE for the three scenarios is \$600,000 + \$80,000 + \$150,000 = \$830,000. Management evaluates four candidate investments, each costed on an annualized TCO basis.

- **Option A: Privileged access management, immutable backups, and a tested recovery program (S1).** Annualized TCO is \$200,000. The team estimates that it reduces S1's ARO from 0.2 to 0.12, because privileged pathways become harder to abuse, and reduces SLE from \$3 million to \$1.5 million, because recovery time falls from about ten days to about three. New ALE is \$1,500,000 × 0.12 = \$180,000. The reduction is \$600,000 − \$180,000 = \$420,000, and ROSI is (\$420,000 − \$200,000) ÷ \$200,000 = 110%.

- **Option B: Phishing-resistant multifactor authentication for email and a mandatory call-back procedure for changes to supplier bank details (S2).** Annualized TCO is \$25,000. ARO falls from 0.5 to 0.15; SLE is unchanged. New ALE is \$160,000 × 0.15 = \$24,000. The reduction is \$80,000 − \$24,000 = \$56,000, and ROSI is (\$56,000 − \$25,000) ÷ \$25,000 = 124%.

- **Option C1: A commercial cloud security posture management platform (S3).** Annualized TCO, including integration and analyst time, is \$140,000. ARO falls from 0.1 to 0.04. New ALE is \$1,500,000 × 0.04 = \$60,000. The reduction is \$150,000 − \$60,000 = \$90,000, and ROSI is (\$90,000 − \$140,000) ÷ \$140,000, or about −36%.

- **Option C2: A secure configuration baseline using the cloud provider's native policy tools, with quarterly review (S3).** Annualized TCO is \$45,000. ARO falls from 0.1 to 0.05. New ALE is \$1,500,000 × 0.05 = \$75,000. The reduction is \$150,000 − \$75,000 = \$75,000, and ROSI is (\$75,000 − \$45,000) ÷ \$45,000, or about 67%.

The table below summarizes the recommended combination.

| **Scenario**      | **ALE before** | **Recommended option (annual cost)** | **ALE after** |
|-------------------|----------------|--------------------------------------|---------------|
| S1 Ransomware     | \$600,000      | A (\$200,000)                        | \$180,000     |
| S2 Payment fraud  | \$80,000       | B (\$25,000)                         | \$24,000      |
| S3 Cloud exposure | \$150,000      | C2 (\$45,000)                        | \$75,000      |
| **Total**         | **\$830,000**  | **\$270,000**                        | **\$279,000** |

With options A, B, and C2, total ALE falls from \$830,000 to \$279,000, a reduction of \$551,000, for an annual cost of \$270,000. Portfolio ROSI is (\$551,000 − \$270,000) ÷ \$270,000, or about 104%. Because each option addresses a different scenario in this simplified example, the reductions can be added; in reality, phishing-resistant authentication would also make the ransomware scenario less likely, and the analyst would have to avoid counting that shared benefit twice.

### Reading the example as a manager

Several governance lessons emerge. First, the expensive platform in Option C1 reduces S3's expected loss more than the native baseline in Option C2, but the additional \$95,000 per year buys only \$15,000 of further reduction (\$90,000 versus \$75,000). The marginal question—what does the next dollar buy?—favours C2. The Gordon–Loeb sanity check points the same way: the 1/e (about 37%) bound on S3's \$150,000 expected loss is about \$55,000, which C2's \$45,000 respects and C1's \$140,000 far exceeds. This does not mean posture-management platforms are poor investments in general; for an organization with hundreds of cloud accounts, the analysis might reverse. It means the decision must be made with the organization's own numbers.

Second, Option A is justified on two grounds, not one. Its ROSI is attractive, but its more important effect is on the tail. The board of Boréal has stated that no single cyber event should plausibly cost more than \$6 million, roughly the level at which bank covenants would come under pressure. The worst case for S1, at \$12 million, breaches that tolerance. The recovery program is estimated to reduce the worst case to about \$5 million by limiting outage duration and data loss. Even if its ROSI were negative, the investment would be needed to bring the scenario within appetite, or the residual exposure would have to be transferred or formally accepted at board level.

Third, the monetary view and the CyberRiskGuardian view do not move in proportion, and the reason matters. To run the cross-check, the risk committee entered the three scenarios in CyberRiskGuardian with Boréal's approved appetite of 0.40, a moderately risk-averse position, and the standard multiplication factor of 1,000. The parameters are workshop judgments on the 0–1 scales of Chapter 6. For the ransomware scenario, for example, the team used Pb(A) 0.80, Pb(ψ,A) 0.25, δe 0.60, δm 0.90, μ(E) 0.90, and θ 0.40, with a CVSS v4.0 base score of 8.7 for the exploited remote-access weakness; every value carries the marker *Analytical estimate — validation required*. The payment-fraud scenario rests on a process weakness, the absence of an independent call-back on changes to supplier bank details, which was scored by equivalency (CVSS 6.9) and labelled as of limited applicability. Each option's effect is entered as the **share removed**: Option A has a probability reduction of 0.40 (ARO from 0.2 to 0.12) and an impact reduction of 0.50 (SLE from \$3 million to \$1.5 million), while Options B and C2 have probability reductions of 0.70 and 0.50 and, because they leave SLE unchanged, impact reductions of zero. The table below shows the results calculated with the CyberRiskGuardian engine.

| **Scenario and option**      | **Estimated / tolerated (ratio before)** | **Reductions entered (P × Q)** | **Residual, ratio after, verdict** |
|------------------------------|------------------------------------------|--------------------------------|------------------------------------|
| S1 Ransomware, Option A      | 2,936 / 1,566 (1.88)                     | 0.40 × 0.50 = 0.20             | 2,349; 1.50; above tolerance       |
| S2 Payment fraud, Option B   | 939 / 1,366 (0.69)                       | 0.70 × 0.00 = 0                | 939; 0.69; below tolerance         |
| S3 Cloud exposure, Option C2 | 600 / 418 (1.44)                         | 0.50 × 0.00 = 0                | 600; 1.44; above tolerance         |

Read against the ALE results, the contrast is stark. Option A removes 70% of S1's expected loss but only 20% of its estimated risk, because the method multiplies the two reductions (0.40 × 0.50 = 0.20) rather than compounding what remains after each (ALE falls to 0.6 × 0.5 = 0.30 of its former value). Options B and C2 cut expected loss by 70% and 50% yet mitigate nothing in the index, because under this convention a package that reduces only probability, or only impact, receives no credit. Across the three scenarios, estimated risk falls from 4,476 to 3,889 risk units, a 13% reduction, while ALE falls by 66%. Neither figure is wrong. They answer different questions under different conventions, and the risk units are relative indicators for comparison, not dollars.

What should a manager take from the divergence? First, the tolerance verdicts reinforce the tail argument. S1 remains above tolerance after Option A; to bring its residual down to the tolerated level, the product of the two reductions would have to reach at least 1 − appetite ÷ average damage, or 1 − 0.40 ÷ 0.75, about 0.47, more than twice what Option A delivers. S1 therefore needs further treatment, transfer, or formal acceptance whatever its ROSI says, and the report flags it *Management decision required*. Second, S2 sits below tolerance even before treatment, because its average damage (0.275) is lower than the appetite: Option B is justified on economic grounds, by its 124% ROSI, not as a condition of staying within appetite. Third, the zero credit for Options B and C2 is a signal about treatment design rather than a verdict on their value. CyberRiskGuardian's guidance is that a package must also limit the consequence, for example through detection, containment, or tested recovery. For S3, pairing the configuration baseline with access logging and alerting on the storage service, and with encryption that limits what an exposure reveals, might plausibly justify an impact reduction of about 0.60; with the same probability reduction of 0.50, P × Q becomes 0.30 and the ratio falls to about 1.01, approximately at tolerance, at an additional cost that would have to be estimated and added to the business case. **The cross-check is not a test of whether the two methods agree; it is a test of whether each treatment addresses both sides of the causal chain and whether the scenarios that matter to the board end within its appetite.** When the money says "worth it" and the index says "still above tolerance", the honest message to the board contains both.

Finally, the numbers are inputs to judgment, not substitutes for it. Each estimate carries uncertainty, and the committee should ask which assumptions matter most. In this example, the ransomware ARO and the recovery-time assumption drive most of the result; if recovery testing showed that restoring the ERP actually takes seven days rather than three, Option A's benefit would shrink and its design would need revisiting. That is why the business case should specify the evidence—recovery test results, coverage metrics for privileged accounts—that will confirm whether the expected benefit is being realized.

## Avoiding compliance theatre and the maturity trap

Two recurring dysfunctions undermine cybersecurity economics even when budgets are adequate. The first is spending that produces evidence of security without producing security. The second is treating maturity scores as if they were the outcome rather than a proxy for it. Both cause organizations to spend money without purchasing the intended reduction in harm.

### How compliance theatre emerges

**Compliance theatre** occurs when an organization invests primarily in artifacts that satisfy governance appearances—policies, checklists, audit binders, controls implemented in name—without producing meaningful reduction in real risk. This is as much an economic failure as a governance failure: the organization pays for an outcome it does not receive, and it may also acquire false confidence that discourages genuine investment. Compliance theatre typically emerges through predictable patterns.

- **Late-stage security reviews.** When security is introduced at the end of projects, the organization produces paperwork, exceptions, and last-minute controls that are minimally integrated and rarely sustainable. Security-by-design, discussed in Chapter 13, is usually cheaper as well as more effective.

- **Metric distortion.** When leaders measure activity rather than outcome, teams optimize for the metric rather than for risk reduction. Counting the number of vulnerabilities scanned does not ensure that the vulnerabilities that matter are remediated within required timeframes.

- **Policy without enforcement.** Policies that cannot be operationalized are narratives, not controls. A data retention policy that is not implemented in systems and backups does not reduce breach impact; an access policy without identity engineering and privileged access governance does not reduce compromise probability.

- **Exceptions without expiry.** Risk acceptances that are granted once and never revisited accumulate silently until the documented control environment bears little resemblance to the real one.

- **Framework accumulation.** Adopting additional frameworks or certifications without integrating them into a coherent architecture, as Chapter 9 warns, multiplies assurance cost without multiplying protection.

### From "we have it" to "we can show it works"

The antidote is to make operational validation part of investment governance. A control should be treated as effective only when it can be demonstrated to work. In mature programs, this is done through several kinds of evidence: **coverage metrics**, such as the percentage of privileged accounts protected by phishing-resistant authentication; **outcome metrics**, such as time to detect and time to contain; **testing evidence**, such as successful restore tests and tabletop exercises; and **independent verification**, through internal audit, penetration tests, configuration compliance reporting, and third-party assessments.

The economic logic is direct. If the benefit of an investment depends on operational efficacy, as FAIR-CAM emphasizes, then funding should include the cost of verifying that efficacy, and benefit claims should be updated when verification reveals gaps. **The governance standard shifts from "we bought it" or "we have it" to "we can show it works, on the assets that matter, today."**

### Maturity as a proxy, not the economic object

Many managers assume that higher maturity automatically reduces loss. There is a relationship, but it is neither linear nor guaranteed. Maturity and capability models, such as COBIT 2019's capability levels, describe process completeness and repeatability: whether practices exist, are documented, are standardized, and are consistently performed and improved. The tiers of NIST CSF 2.0 play a related role, characterizing how rigorously cybersecurity risk governance and management practices are applied. This is valuable because inconsistent execution is a common cause of failure. However, maturity is a proxy. The economic object is the loss distribution: incidents avoided, impact reduced, downtime shortened, legal exposure contained.

Maturity improvements typically follow a pattern of diminishing returns that resembles an S-curve. The first tranche of investment—multifactor authentication, basic patching discipline, an accurate asset inventory, centralized logging of critical systems, and tested backups—closes foundational gaps and yields disproportionate benefit. This is the logic behind the implementation groups of CIS Controls v8.1, whose first group (IG1) defines the essential cyber hygiene that most organizations should reach before pursuing refinements. Later investments, such as moving a well-functioning process from "defined" to "quantitatively managed," often produce smaller incremental benefit and should be justified on their own merits.

### Directing maturity where it reduces loss

Two further implications follow. First, maturity must be directed toward high-leverage domains. A highly mature process applied to low-impact systems produces little risk reduction, whereas moderate maturity applied to identity, privileged access, and recoverability may produce large loss reduction. Second, maturity must be coupled to threat reality. A control can be mature and still irrelevant if it does not address the threats that matter in the organization's environment.

A sound governance practice is to treat maturity metrics as indicators of execution quality while keeping risk and loss metrics as the primary decision language. Maturity targets should be set per domain, based on the scenarios each domain affects, rather than as a uniform aspiration across the program. **Maturity helps manage how well the organization executes; loss reduction remains the reason it invests.**

## Investment prioritization: deciding what to fund first

When leaders ask what to invest in first, they are implicitly asking for a prioritization logic that does not depend on fear, recent headlines, or whichever team has the loudest voice. The prioritization criteria used with the CyberRiskGuardian method in Chapter 6—estimated and residual risk, the risk-to-tolerance ratio, maximum consequence, business criticality, legal, privacy, and safety obligations, risk velocity and threat activity, cost-effectiveness (risk units removed per \$1,000 spent), shared benefits, dependencies, feasibility, and time to benefit—provide the detailed toolkit, applied as judgment rather than mathematical certainty. This section distils them into an investment logic suitable for executive decisions.

### Three anchoring principles

A credible prioritization logic rests on three principles. The first is **business criticality**. Controls that protect or enable critical services, regulated data, and safety-relevant operations are naturally prioritized because their failure yields disproportionate harm. Criticality should be established through business impact analysis (see Chapter 12), not by technical teams alone.

The second is **leverage**. Some controls reduce many risks at once because they address structural failure modes. Identity and privileged access governance, secure configuration baselines, segmentation, logging, and recoverability tend to be high-leverage because they reduce an attacker's ability to gain entry, move, persist, and cause catastrophic harm across many scenarios. In the scenario view, leverage appears as an investment whose reduction factors apply to several scenarios at once.

The third is **feasibility**. Some improvements can be made quickly with modest disruption; others require deep organizational change. Feasibility affects sequencing, not importance. As Chapter 6 noted for individual mitigation plans, a program that the available people, procurement lead times, and change capacity cannot deliver reduces no loss, however well it is designed.

### Mandatory, foundational, and discretionary investments

These principles support a simple classification that helps executive committees allocate money.

- **Mandatory investments** are required by law, regulation, contract, or insurance conditions, or by audit findings with committed remediation dates. They must be funded regardless of their apparent ROSI, but they should still be designed to deliver real risk reduction rather than mere compliance artifacts.

- **Foundational investments** close structural gaps that undermine many scenarios at once, such as identity, privileged access, asset inventory, baseline hardening on critical assets, high-value logging, segmentation, and tested recovery. They usually offer the best marginal risk reduction early in a program.

- **Discretionary investments** address specific scenarios or refine existing capabilities. They should compete on marginal risk reduction per dollar, tail-risk impact, and strategic fit.

Viewed through this lens, many organizations gain the earliest benefit by strengthening identity and privileged access pathways, enforcing baseline hardening and patching discipline on critical assets, establishing logging and monitoring coverage where it matters most, reducing blast radius through segmentation, and validating recoverability through tested backups and recovery exercises. This is not a universal sequence, but it reflects a governance truth: **foundational capabilities often reduce broad classes of risk more efficiently than niche controls.**

## Portfolio governance: a security investment portfolio rather than a tooling list

A common dysfunction in cybersecurity budgeting is treating the budget as a collection of unrelated purchases. Under that model, investment decisions become debates about products, not about risk reduction, and the budget grows by accretion as each year's new tools are added to last year's renewals. Mature governance treats cybersecurity spending as a portfolio of capabilities designed to protect business objectives, with explicit categories, common evaluation criteria, and periodic rebalancing.

### Categories of a security investment portfolio

A portfolio approach divides investments into categories that map to outcomes and time horizons. **Run investments** maintain ongoing control operation: staffing, monitoring, vulnerability management cycles, incident readiness, licence renewals, and compliance reporting. **Change investments** build new capabilities or materially improve existing ones, such as implementing privileged access management, restructuring identity governance, deploying configuration baselines with drift enforcement, modernizing logging for cloud control planes, or redesigning backup and recovery for ransomware resilience. **Risk transfer and preparedness investments** include cyber insurance where appropriate, incident response retainers, legal and communications preparedness, and crisis exercises; they do not prevent events, but they reduce tail impact and improve survivability.

Two further categories are often neglected. **Mandatory obligations** deserve their own line so that the board can see how much of the budget is driven by law, regulation, and contracts rather than by discretionary risk choices. **Retirement and consolidation** recognizes that removing redundant tools, closing unused accounts, and decommissioning legacy systems are investments too: they free money, reduce attack surface, and reduce operational burden. A portfolio that only ever adds is not being governed. The figure below presents these six categories.

<span id="_Toc243635572" class="anchor"></span>Figure 73: The security investment portfolio

### Common criteria for every investment

Portfolio governance requires that investments be evaluated against common criteria, so that a request for a new detection service and a request for an identity program can be compared on the same basis. In practice, leaders benefit from a small, stable set of decision questions applied to every proposal above a materiality threshold.

- **Risk addressed.** Which specific scenarios does this investment reduce, and does it bring any of them within tolerance?

- **Size of benefit.** How large is the expected reduction in loss or disruption, as a range, and how does it affect the worst plausible outcome?

- **Time to benefit.** How quickly does the benefit materialize, and what milestones mark it?

- **Total cost of ownership.** What is the annualized cost over the capability's life, including people, integration, and exit?

- **Dependencies.** What other capabilities must exist for this one to work, and what new dependencies does it create?

- **Evidence.** What metrics, tests, or assurance activities will demonstrate that it is working?

- **Consequence of not funding.** What happens if we do not do it, and who would have to accept the resulting residual risk?

When investments must compete within a structured process, vendor-driven spending and duplication decline, and the organization is less likely to buy security without improving outcomes. The criteria also produce a natural record of why decisions were made, which is the core of a defensible decision.

### Benefits realization and post-investment review

Governance does not end at approval. A portfolio needs periodic review to confirm that funded capabilities are delivering the expected outcomes. Six to twelve months after a major initiative is declared complete, the organization should compare actual coverage, performance, and cost with the business case. Did privileged access management reach all critical systems, or only the easy ones? Did the recovery program achieve its target recovery time in a realistic test? Is the annual operating cost close to the estimate?

Post-investment reviews serve three purposes. They hold sponsors accountable for benefits, not just for delivery. They improve future estimates, because the organization learns how optimistic its assumptions tend to be. And they identify capabilities that should be fixed, scaled, or retired. Internal audit, as the third line, can add value by periodically reviewing whether the portfolio process itself is operating as designed.

## Communicating cyber risk in financial terms

Many cybersecurity leaders communicate fluently in technical language but struggle to produce decision-quality risk narratives. Others present dramatic narratives without defensible quantification. Investment governance requires a middle path: scenario narratives supported by financial estimates, with uncertainty and assumptions explicitly acknowledged. The goal is not to impress the board with precision but to give it what it needs to exercise oversight.

### Decision-quality narratives

An effective board-level cyber risk narrative includes a clear statement of the scenario, its business impacts, and its financial consequences. The scenario should be described in operational terms ("ransomware disables the order management platform and customer service environment for five days") rather than technical ones ("domain controller compromise"). The impacts should cover operational downtime, data exposure, legal and regulatory consequences, customer and partner response, and remediation effort. The financial consequences should distinguish direct costs (forensics, restoration, external services), indirect costs (lost revenue, overtime, business interruption), and contingent costs (litigation, regulatory action, contractual penalties). Where precision is impossible, ranges should be given and the assumptions stated.

External benchmarks can help calibrate these estimates, provided they are used carefully. The IBM *Cost of a Data Breach Report 2026*, produced with the Ponemon Institute, reported a global average breach cost of about US\$4.99 million, roughly 12% higher than the previous year, with a considerably higher average for breaches in the United States. Such figures are useful for context and for testing whether internal estimates are wildly off, but they describe a sample of other organizations' breaches with different sizes, sectors, and jurisdictions. **Industry averages should inform scenario estimates, never replace them.**

A useful practice is to frame the conversation around a small number of prioritized scenarios—often five to ten—that represent the organization's most material exposures. This keeps the discussion coherent and lets leadership see how different investments change outcomes across scenarios. Reporting should be consistent over time, so that directors can see whether exposure is rising or falling relative to appetite.

### Disclosure and regulatory expectations

Financial framing of cyber risk is no longer only an internal management choice. In the United States, the Securities and Exchange Commission's 2023 rules on cybersecurity risk management, strategy, governance, and incident disclosure require registrants to disclose material cybersecurity incidents on Form 8-K (Item 1.05) and to describe their risk management processes and board oversight annually under Regulation S-K Item 106. Determining materiality requires exactly the kind of business-impact reasoning described in this chapter. In Canada, OSFI Guideline B-13 expects federally regulated financial institutions to govern technology and cyber risk within their enterprise risk frameworks, and European rules such as NIS2 and DORA place explicit accountability for cyber risk management on management bodies. Across these regimes, the common expectation is that leaders understand cyber risk in business terms and can show how their decisions follow from that understanding.

### Illustrative board budget discussion

Returning to Boréal Distribution, consider how the budget conversation might unfold at the board's audit and risk committee. The CISO presents the three dominant scenarios, the loss ranges behind them, and the recommended options A, B, and C2, totalling \$270,000 per year in addition to the existing run budget. She shows each scenario's estimated, tolerated, and residual risk and its risk-to-tolerance ratio before and after treatment, points out that the ransomware scenario remains above tolerance even with Option A, and shows that the worst-case ransomware loss of \$12 million exceeds the board's stated \$6 million limit.

The CFO asks why the organization should not simply buy more insurance instead of funding the recovery program. The CISO answers that insurance would reimburse some costs but would not shorten a ten-day outage, would not prevent customer defections, and might not respond at all if underwriting conditions on backups and privileged access were not met; insurers increasingly price or condition cover on precisely the controls in Option A. One director asks why the committee should trust the ARO of 0.2 for ransomware. The CISO explains that the estimate draws on sector threat intelligence, insurer claims observations shared by the broker, and Boréal's own near-miss history, and that the recommendation would not change if the true ARO were half as high, because the tail-risk argument stands on its own. Another director asks what will tell the board, a year from now, whether the money was well spent. The CISO commits to three indicators: privileged-account coverage, the measured recovery time of the ERP in a full restoration test, and the number of supplier bank-detail changes verified by call-back.

The committee approves options A and B, approves C2 with a request to add an impact-limiting measure, since the cloud-exposure scenario remains above tolerance with a probability-only treatment, and to revisit the posture-management platform if the cloud footprint doubles, and asks management to return with an insurance recommendation sized to the residual worst case. It also records that, until the recovery program is complete, the residual ransomware tail risk is formally accepted by the chief executive officer as risk owner, with an expiry date aligned to the program's completion milestone. Every director leaves knowing what was decided, why, and how it will be checked.

| A good cybersecurity budget request answers four questions: which scenarios it changes, by how much, at what total cost, and what evidence will show it worked. |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|

## Evaluating managed services: buying outcomes rather than outsourcing responsibility

Managed services—managed detection and response, outsourced security operations centres, vulnerability management services, managed cloud security, and others—are attractive because they offer expertise and round-the-clock operations that many organizations cannot staff internally. For small and mid-sized organizations in particular, a well-chosen service can provide capability at a fraction of the cost of building it. However, managed services become economically inefficient when they are purchased as generic packages rather than governed as outcome contracts. Chapter 19 addresses third-party governance in depth; this section focuses on the investment logic.

### Five dimensions of evaluation

A managed service should be evaluated on at least five dimensions.

- **Scope realism.** Many services promise broad coverage but exclude critical domains such as cloud control planes, identity telemetry, operational technology, or bespoke applications. Governance requires mapping the service's actual scope to the organization's critical assets and scenarios.

- **Measurable outcomes.** Leaders should ask how the service will change detection time, response time, coverage of critical logs, remediation outcomes, containment success, and investigation quality. If a provider cannot express outcomes beyond "we monitor alerts," the service is likely to become expensive theatre.

- **Integration cost.** Managed services depend on data. If integration with identity, endpoint, cloud, and network telemetry is weak, the service will perform poorly while the organization still bears the risk. Integration effort belongs in the TCO.

- **Governance and accountability.** Outsourcing monitoring does not outsource responsibility. The organization must define escalation thresholds, decision rights during incidents, communication pathways, evidence preservation, and legal and regulatory coordination, including breach notification duties under laws such as Québec Law 25.

- **Exit and resilience.** In a systemic incident affecting many clients, providers can become overloaded. Contracts should address surge capacity, incident prioritization, what happens if the provider itself is compromised, and how data and knowledge are returned at exit.

### Build, buy, or blend

The economic comparison is rarely a simple choice between building an internal team and buying a service. A credible internal security operations capability requires enough analysts to cover every hour of the week, allowing for leave, training, and turnover, plus engineering and management support. For many organizations, that cost far exceeds a managed service fee. Yet a fully outsourced model can leave the organization without the internal knowledge needed to judge the provider's work, make containment decisions quickly, or understand its own environment.

Many organizations therefore adopt a **blended model**: the provider delivers continuous monitoring and first-level response, while a small internal team owns the architecture, the risk decisions, the relationship with the business, and the oversight of the provider. The managerial conclusion is that **managed services can be excellent investments when they deliver measurable outcomes aligned with risk reduction, and poor investments when they are purchased as a substitute for governance and internal ownership.**

## Evaluating security tooling: total cost, consolidation, and platformization

Security tooling decisions often fail because they are made by feature comparison rather than by operational economics. Tools have value only when they are deployed, adopted, integrated, tuned, and sustained. An unused feature is not a capability, and an undeployed agent protects nothing.

### Start from the control objective

A disciplined evaluation begins with the control objective rather than the product category. If the objective is to reduce ransomware impact, recoverability and privileged access controls may have higher leverage than a new detection dashboard. If the objective is to reduce data exposure, identity governance, access evidence, and data lifecycle controls (see Chapter 16) may outrank a data loss prevention tool that cannot realistically be implemented. Once the objective is fixed, tools should be evaluated on TCO, on coverage—what assets and data the tool actually reaches in this environment—and on effectiveness evidence from pilots, validation exercises, and references from comparable organizations.

A particularly important principle is that **tools cannot substitute for people and process**. A tool that requires expertise the organization does not possess becomes shelfware. In economic terms, it is a sunk cost that may also create false confidence, because its existence appears in control inventories and audit responses even though it reduces no risk.

### The economics of tool sprawl

Tool sprawl is an economic problem before it is a technical one. Organizations that have bought point solutions over many years often operate dozens of security products, many with overlapping functions, separate consoles, separate data stores, and separate renewal cycles. Each additional tool adds licence cost, integration effort, training, and configuration maintenance. Overlaps create confusion about which tool is authoritative, gaps hide between tools, and teams cannot maintain consistent configurations across so many products, so sprawl often reduces actual security.

Investment governance should therefore require **architecture coherence**: tools should fit into a designed reference architecture in which data flows, responsibilities, and control points are clear (see Chapter 13). A periodic tool rationalization review—listing each product, its owner, the control objectives it serves, its coverage, its utilization, and its annual TCO—often reveals redundant licences that can fund higher-priority work.

### Consolidation and platformization: benefits and concentration risk

In response to sprawl, many vendors now offer integrated platforms that combine endpoint, identity, cloud, email, network, and analytics functions under a shared data model and console, and many organizations are consolidating toward a smaller number of such platforms. This trend, often called **platformization**, can bring real economic benefits: fewer integrations to build and maintain, shared telemetry that improves detection, simplified training, stronger negotiating positions, and lower aggregate TCO.

Consolidation also has costs that belong in the business case. Dependence on a single vendor creates **concentration risk**: a defect, outage, or compromise in one platform can affect many controls at once. The worldwide disruption caused in July 2024 by a faulty content update to a widely deployed endpoint security product, which halted airlines, hospitals, and banks, illustrated how a single security supplier can become a single point of failure. Consolidation can also weaken negotiating power at renewal, make exit expensive, and bundle modules of uneven quality, so that the organization pays for components it would never choose on their merits.

The governance response is not to avoid platforms but to choose them deliberately. Managers should ask which control objectives the platform will serve, which modules will actually be deployed, how the organization would operate if the platform failed for several days, what contractual protections exist for renewal pricing and exit, and whether critical recovery capabilities—especially backups and identity recovery—remain independent of the platform they would be needed to restore. **Consolidation is an investment decision with its own risks, not a free simplification.**

## Cyber insurance: risk transfer with constraints

Cyber insurance is often treated as a financial solution to a technical problem. In governance terms, it is better understood as a **partial risk transfer mechanism** for certain cost categories, subject to conditions, limits, and exclusions. It belongs in the portfolio as one treatment option among several, alongside reduction, avoidance, and acceptance, and it should be evaluated in the same economic language as other controls.

### What insurance can and cannot transfer

Cyber policies typically combine **first-party coverage**, for the insured's own costs—incident response, forensic investigation, data restoration, business interruption, extortion-related costs where lawful, and notification—with **third-party coverage**, for claims brought by others, such as privacy liability, network security liability, and in some jurisdictions the defence of regulatory proceedings. Many policies also provide access to a panel of breach coaches, lawyers, and forensic firms.

Many consequences, however, cannot be transferred cleanly. Reputational damage, loss of customer trust, operational disruption beyond the policy's waiting period or indemnity period, management distraction, and the long-term cost of rebuilding are only partly insurable or not insurable at all. Whether regulatory fines and penalties can be insured varies by jurisdiction and is often restricted. Policy limits may be small relative to a severe scenario, and retentions mean the organization bears the first portion of every loss. In the Boréal example, insurance could help with the financial tail of the ransomware scenario but could not substitute for the recovery program that shortens the outage.

### Exclusions and systemic risk

Insurers worry about **systemic** or **accumulation risk**: a single event, such as a widespread malware outbreak or the failure of a major cloud provider, that triggers claims from many policyholders at once. Several common exclusions and limitations reflect this concern, and managers must read them carefully.

- **War and state-backed cyber attacks.** Lloyd's Market Bulletin Y5381, issued in August 2022, required managing agents to include in standalone cyber-attack policies, from March 31, 2023, a clause excluding losses arising from war and from state-backed cyber attacks that significantly impair a state's ability to function or its security capabilities, and setting out a robust basis for attributing an attack. Other insurers have adopted their own war and hostile-act wording. The litigation over Merck's NotPetya losses, in which a New Jersey appellate court held that a traditional war exclusion did not apply and the parties settled in early 2024, shows how much depends on exact policy language.

- **Infrastructure failures.** Many policies exclude or limit losses arising from failures of power, telecommunications, or other core infrastructure not under the insured's control.

- **Prior knowledge and prior acts.** Incidents that began, or were known, before the policy period are commonly excluded.

- **Control conditions and misrepresentation.** Coverage may be reduced or denied if the insured failed to maintain controls it represented in the application, such as multifactor authentication or offline backups. Some policies also restrict cover for losses arising from long-unpatched known vulnerabilities.

- **Betterment.** Policies generally pay to restore systems to their prior state, not to upgrade them.

Because wording varies widely between insurers and changes over time, the organization should rely on qualified brokers and legal counsel to interpret its specific policy, and should map exclusions explicitly against its top scenarios. A scenario that sits squarely within an exclusion is, for planning purposes, uninsured.

### Market conditions and underwriting

Cyber insurance markets are cyclical and can change quickly. After a period of sharp price increases and tightened terms in 2021 and 2022, driven largely by ransomware losses, broker reports indicate that pricing softened in subsequent years; Marsh's Global Insurance Market Index, for instance, reported that global cyber rates fell in the second quarter of 2026, the twelfth consecutive quarter of decreases. Managers should treat such conditions as temporary. Major loss events or new forms of systemic risk could tighten the market again, and favourable pricing today is not a reason to defer control investment.

Regardless of the cycle, underwriting has become more evidence-driven. Insurers commonly ask about multifactor authentication, endpoint detection and response, privileged access management, backup design and testing, patching practices, incident response planning, and security awareness. Some offer better terms to organizations that can demonstrate strong controls. This creates a useful alignment, because the controls insurers value are largely the foundational controls this chapter recommends. It also creates a risk: organizations should not allow insurance questionnaires to become their security strategy, nor answer them more optimistically than the evidence supports, since inaccurate representations can jeopardize coverage when it is most needed.

### Insurance as an investment decision

Insurance interacts with incident response. Policies may impose notification deadlines, require the use of approved vendors, require the insurer's consent before certain expenses are incurred, and demand evidence of control operation. Buying insurance therefore requires preparedness: incident procedures that include the insurer, legal and communications readiness, evidence of control operation, and clarity about who has decision rights during a crisis. Decisions about ransom payment, which raise legal, sanctions, and ethical questions, should be addressed in advance as a governance matter rather than improvised during an incident.

The insurance decision itself should be framed economically. What loss categories does the policy reduce, under what conditions, with what residual exposure? How do the limit and retention compare with the organization's loss exceedance curve or worst-case scenario estimates? What is the total cost, including premiums, retentions, compliance obligations, and operational restrictions? Insurance is most rational when it addresses tail losses that would otherwise threaten solvency or strategic stability, and when the organization can meet underwriting requirements without distorting its priorities. In the CyberRiskGuardian method, insurance is recorded as **risk transfer**, not as a mitigation measure: a policy reimburses part of the financial consequence but does nothing to the likelihood of the event, and under the multiplicative mitigation convention a single-lever treatment of that kind mitigates nothing in the index. Transfer is therefore documented, with avoidance and formal acceptance, as a treatment decision for a scenario that remains above tolerance, together with the share of loss it covers and the decision-maker. The figure below summarizes the four questions a cyber insurance decision should answer.

<span id="_Toc243635573" class="anchor"></span>Figure 74: Governing the cyber insurance decision

| Cyber insurance can pay for parts of a bad day; it cannot shorten the outage, restore trust, or compensate for controls the organization claimed but did not operate. |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## Budgeting as governance: from annual cycles to multi-year capability building

Cybersecurity investment governance is most effective when it is treated as a multi-year capability program rather than an annual procurement negotiation. Annual budget cycles encourage short-term optimization, whereas cybersecurity risk reduction often requires sustained investment across people, process, and architecture. Identity modernization, recovery redesign, or segmentation of an operational environment may take two to four years, and funding interruptions midway can leave the organization with costs incurred and benefits unrealized.

### Run and change funding

Budget governance benefits from separating baseline operational funding from transformational funding. Security operations, monitoring, vulnerability management, incident readiness, licence renewals, and compliance reporting require stable run funding that grows predictably with the organization. Major capability upgrades—identity and privileged access programs, foundational logging architecture, segmentation, recovery redesign—need dedicated multi-year change funding with project governance, milestones, and benefits tracking. Mixing the two makes it hard to see whether the program is improving or simply running to stand still. Because every new capability adds to the run budget, a multi-year plan should show each initiative's steady-state operating cost as well as its build cost, so that the board does not approve a program whose future run costs it cannot sustain.

### Benchmarks and their limits

Executives often ask how much an organization like theirs should spend on cybersecurity. Chapter 4 introduced the best-practice guideline this book uses, and Chapter 6 applies it to test whether a treatment portfolio is realistic. Cybersecurity spending is expressed as a percentage of the **total IT budget, including salaries**, and calibrated to risk appetite: about **12%** for a risk-averse organization (appetite 0.30 or below), about **7.8%** for a risk-neutral organization (appetite 0.50), and about **4%** for a risk-seeking organization (appetite 0.70 or above), with linear interpolation in between. An organization with a total IT budget of \$10 million would therefore target roughly \$1.2 million at an appetite of 0.30, about \$990,000 at 0.40 (9.9%), \$780,000 at 0.50, and \$400,000 at 0.70. Such a guideline is useful as a reasonableness test: a highly regulated, risk-averse organization spending far below its appetite-consistent target should ask whether it is underinvesting, and an organization spending far above it should ask whether it is buying outcomes or accumulating tools.

The guideline should not, however, set the budget on its own. Spending ratios vary with sector, size, regulatory regime, technology footprint, and how organizations classify costs, and a percentage says nothing about whether the money is well spent. **The defensible budget is the one that brings scenario risks within appetite at a justified cost, not the one that matches a percentage.** The appetite-consistent target is a check on that reasoning, not a replacement for it; a large gap between the risk-driven portfolio and the target is a finding to explain to management, not an error to hide.

### Residual risk acceptance as a budget output

Not every risk can be treated within budget constraints. Governance requires that the organization knowingly accept some residual risks, with documented rationale, a named risk owner with the authority to accept them, and appropriate oversight. When a funding request is declined or deferred, the corresponding residual risk should be recorded and accepted explicitly, ideally with an expiry date that forces reconsideration. This turns the budget from a list of what will be bought into a statement of what risk the organization has chosen to carry.

This practice protects everyone involved. It ensures that the board understands the consequences of its funding choices, that the CISO is not held responsible for risks the organization decided not to treat, and that the organization can later demonstrate that its decisions were deliberate and proportionate. **The governance deliverable is therefore not merely a budget number; it is an investment story with decision logic**: which risks are targeted, why these initiatives are prioritized, what outcomes are expected, what evidence will confirm progress, and what residual risk remains and who owns it.

### Questions boards should ask about cybersecurity investment

Boards and executive committees can strengthen investment governance by asking a consistent set of questions whenever cybersecurity budgets or major initiatives are presented.

- **Which scenarios matter most?** What are our five to ten most material cyber risk scenarios, and how has our exposure to each changed since the last review?

- **Where are we outside appetite?** Which scenarios currently exceed our risk tolerance, by how much, and what is the plan and timeline to bring them within it?

- **What does this investment change?** For each major request, which scenarios does it affect, by how much, and how does it change the worst plausible outcome?

- **What is the real cost?** What is the annualized total cost of ownership, including people, integration, run costs after implementation, and exit?

- **How confident are the estimates?** Which assumptions drive the result, what evidence supports them, and would the decision change if they were wrong?

- **What is mandatory?** How much of the budget is driven by legal, regulatory, contractual, or insurance obligations, and are those obligations being met in substance rather than on paper?

- **What evidence will show it works?** Which coverage, outcome, and testing metrics will we see, and when, to confirm that the investment is delivering?

- **What are we choosing not to do?** Which risks are we accepting because we are not funding treatment, who has accepted them, and when will we revisit that decision?

- **How dependent are we on key suppliers?** What happens if our main security platform or managed service provider fails or is compromised?

- **What does insurance really cover?** How do our policy limits, retentions, and exclusions compare with our worst-case scenarios, and are we meeting the conditions we represented to insurers?

- **Did last year's money work?** What did post-investment reviews show about the initiatives we funded previously?

## Bringing it together: defensible choices in a domain that will never be finished

The objective of cybersecurity economics is not to reduce security to finance. It is to govern cybersecurity as a disciplined investment domain so that managers can make defensible choices under uncertainty and constraint. A defensible choice is not one that turns out to be right in hindsight—no organization can guarantee that—but one that a reasonable board, regulator, or court would recognize as informed, proportionate, consistent with stated risk appetite, and supported by evidence.

A mature organization uses scenario-based risk framing, as developed in Chapters 5 and 6, to connect cyber threats to business harm. It estimates expected loss with tools ranging from simple ALE calculations to FAIR-based simulation and loss exceedance curves, choosing the level of sophistication that its decisions require and always presenting uncertainty honestly. It evaluates controls through marginal risk reduction against total cost of ownership, uses ROSI and the Gordon–Loeb model as comparative aids and sanity checks rather than as verdicts, and respects hard constraints where safety, legal obligations, or intolerable tail risks dominate. It cross-checks monetary estimates against its calculated risk index and risk-to-tolerance ratios, treats any divergence between the two as a prompt to examine treatment design, and monitors progress through KRIs, so that the question "is it worth it?" and the question "is it within appetite?" are both answered.

Such an organization also resists compliance theatre by demanding operational validation, and treats maturity as a measure of execution quality rather than a substitute for loss reduction. It governs spending as a coherent portfolio, reviews whether funded capabilities delivered their benefits, and evaluates managed services, platforms, and insurance as outcome instruments rather than ways to outsource accountability. Finally, it communicates risk in decision-quality language and records the residual risks it chooses to accept.

Cybersecurity will never be finished. Threats adapt, technology changes, and the organization's own exposure shifts with every new service, supplier, and acquisition. This is precisely why investment governance matters. The organization does not need perfection; it needs coherence, discipline, and evidence that its spending is producing real reductions in likelihood, impact, and recovery time, aligned with its risk appetite and its business objectives. Chapter 18 extends this reasoning to one of the fastest-moving areas of investment, artificial intelligence, both as a tool for cybersecurity and as a new source of risk that must itself be governed.

## Chapter summary

Chapter 17 argues that cybersecurity should be governed as an investment portfolio rather than a technology shopping list. Because its benefits are counterfactual and demand for security is effectively unlimited, organizations need explicit decision rights, criteria, evidence, and oversight to decide how much to spend, on what, and how they will know it worked. The chapter moves from expected loss and its limits, through FAIR, simulation, and loss exceedance curves, to control economics, ROSI, the Gordon–Loeb model, and total cost of ownership. A worked example cross-checks monetary results against the CyberRiskGuardian index. It closes with portfolio governance, board communication, managed services, platforms, insurance, and budgeting as governance.

### Key points to retain

- Cybersecurity investment buys four outcomes—loss reduction, survivability, compliance capability, and trust preservation—and is best governed as an enterprise risk-control portfolio.

- Counterfactual benefits, adversarial adaptation, scarce loss data, interdependent controls, and externalities make cybersecurity harder to govern economically than most capital programs.

- ALE equals SLE multiplied by ARO; it is useful for comparing options but hides tails, rare events, and uncertainty, so it should be presented as a range.

- Some scenarios are hard constraints: safety, severe harm to individuals, or existential disruption are governed by appetite, law, and ethics rather than expected-value arithmetic.

- FAIR decomposes risk into loss event frequency and loss magnitude, and Monte Carlo simulation produces loss exceedance curves that give risk appetite a concrete form.

- FAIR-CAM shows that a control's value depends on its operational efficacy and on other controls, and that decision-support capabilities have economic value.

- ROSI is a comparative indicator, not a verdict; benefits do not simply add, tail risk is ignored, vendor claims are not evidence, and timing matters.

- The Gordon–Loeb model suggests investment should not exceed about 37% (1/e) of the expected loss, which makes it a useful sanity check rather than a rule.

- Annualized total cost of ownership, including people, integration, operations, and exit, is the cost figure that belongs in ALE and ROSI comparisons.

- In CyberRiskGuardian, mitigated risk is Re × probability reduction × impact reduction, so a single-lever treatment earns no credit and can diverge sharply from ALE results.

- Compliance theatre and maturity scores can absorb money without reducing harm; controls count only when the organization can show they work on the assets that matter.

- Budgets should separate run and change funding, be tested against the appetite-consistent benchmark (12%, 7.8%, 4% of the IT budget), and record accepted residual risk with an owner and an expiry date.

### Key terms

- **Investment governance** — The decision rights, criteria, evidence, and oversight mechanisms through which an organization decides how much to spend on cybersecurity, on what, and how it will know the spending worked.

- **Single loss expectancy (SLE)** — The expected monetary loss if a scenario occurs once, often built by summing response, restoration, interruption, legal, regulatory, and customer costs.

- **Annual rate of occurrence (ARO)** — The expected number of times a scenario occurs per year; an ARO of 0.2 means about once every five years.

- **Annualized loss expectancy (ALE)** — The product of SLE and ARO, giving an expected annual loss in the same unit as the budget.

- **Tail risk** — The high-impact, low-frequency outcomes that an average conceals and that can threaten an organization's mission or solvency.

- **FAIR (Factor Analysis of Information Risk)** — A quantitative framework, maintained as Open FAIR by The Open Group, that defines risk as the probable frequency and probable magnitude of future loss.

- **Loss exceedance curve** — A plot showing, for each loss amount, the probability that annual losses will exceed it, typically produced by Monte Carlo simulation.

- **FAIR-CAM** — The FAIR Controls Analytics Model, which groups control functions into loss event, variance management, and decision support domains.

- **Return on security investment (ROSI)** — The reduction in ALE minus the annual cost of a control, divided by the annual cost of the control.

- **Gordon–Loeb model** — An economic model showing that, for the breach-probability functions examined, optimal investment should not exceed about 37% (1/e) of the expected loss.

- **Total cost of ownership (TCO)** — The full cost of a control over its life, including acquisition, integration, labour, tuning, training, operations, renewals, exit, and opportunity cost.

- **Risk-to-tolerance ratio** — The CyberRiskGuardian comparison of residual risk with tolerated risk: below 0.90 is below tolerance, 0.90 to 1.10 approximately at, above 1.10 above.

- **Compliance theatre** — Investment in artifacts that satisfy governance appearances, such as policies and audit binders, without meaningful reduction in real risk.

- **Platformization** — Consolidation toward integrated security platforms with a shared data model and console, bringing economic benefits but also concentration risk.

- **Residual risk acceptance** — A documented decision by a named risk owner with authority to carry a risk that is not treated, ideally with an expiry date that forces reconsideration.

## Review questions

### Essay questions

1.  Using the Boréal Distribution example, explain why Option A is justified on two grounds and evaluate whether the board should rely on ROSI, the tail-risk argument, or the CyberRiskGuardian tolerance verdict when approving it.

2.  The ALE view shows a 66% reduction in expected loss for Boréal's recommended portfolio, while the CyberRiskGuardian index shows only a 13% reduction. Analyze why the two views diverge and what a manager should tell the board as a result.

3.  Critically evaluate ROSI and the Gordon–Loeb model as tools for justifying cybersecurity spending. Under what circumstances should a control with a low or negative ROSI still be funded?

4.  Choose an organization you know (or Boréal Distribution). Design a security investment portfolio using the six categories in the chapter and the mandatory, foundational, and discretionary classification, and explain the criteria you would apply to every proposal.

5.  A CFO proposes buying more cyber insurance instead of funding a recovery program. Evaluate this proposal using the chapter's treatment of insurance coverage, exclusions, underwriting, and the CyberRiskGuardian approach to risk transfer.

### Multiple-choice questions

Choose the best answer. Answers are listed in the appendix "Answers to multiple-choice questions".

**1.** Which of the following is one of the four outcomes that, according to the chapter, cybersecurity investment buys?

> A. Revenue growth from new security products
>
> B. Higher maturity scores in audit reports
>
> C. Survivability during and after disruption
>
> D. Lower licence costs through vendor discounts

**2.** An organization signs a managed detection and response contract. Which statement best reflects the chapter's guidance on accountability?

> A. The organization still owns escalation thresholds, incident decision rights, and notification duties
>
> B. Responsibility for detection and breach notification moves to the provider under contract
>
> C. Accountability is shared equally between the provider and the organization's board
>
> D. The provider's certifications make internal oversight of the service unnecessary

**3.** Under NIST CSF 2.0, which function contains the expectation that adequate resources be allocated commensurate with the cybersecurity risk strategy?

> A. Identify
>
> B. Protect
>
> C. Respond
>
> D. Govern

**4.** A scenario has a single loss expectancy of \$400,000 and is expected to occur about once every four years. What is its annualized loss expectancy?

> A. \$40,000
>
> B. \$100,000
>
> C. \$160,000
>
> D. \$1,600,000

**5.** Which risk does the chapter associate specifically with consolidating security tools onto a single vendor platform?

> A. Higher integration costs from maintaining many separate consoles
>
> B. Concentration risk, where one defect or compromise affects many controls
>
> C. Loss of shared telemetry that would otherwise improve detection
>
> D. Weaker training because analysts must learn more products

**6.** What question does a loss exceedance curve answer that a single expected-loss number cannot?

> A. What is the average annual loss across all scenarios
>
> B. Which control has the highest return on security investment
>
> C. Which threat actor is most likely to target the organization
>
> D. How bad a bad year could be, and how likely that is

**7.** In the FAIR taxonomy, how is the term vulnerability defined?

> A. The probability that a threat event becomes a loss event
>
> B. A software flaw scored with the CVSS v4.0 calculator
>
> C. The frequency with which threat agents contact an asset
>
> D. The share of an asset's value lost in an incident

**8.** Under FAIR-CAM, patching and configuration management that prevent control drift belong to which control domain?

> A. Loss event control functions
>
> B. Decision support control functions
>
> C. Variance management control functions
>
> D. Risk transfer control functions

**9.** Boréal's Option B costs \$25,000 per year and reduces S2's ALE from \$80,000 to \$24,000. What is its ROSI?

> A. 69%
>
> B. 96%
>
> C. 224%
>
> D. 124%

**10.** According to the chapter, how should a ransomware scenario be described in a board-level cyber risk narrative?

> A. As a technical event, such as domain controller compromise
>
> B. As a maturity gap against the NIST CSF 2.0 tiers
>
> C. In operational terms, such as order management disabled for five days
>
> D. As an industry average breach cost adjusted for company size

**11.** Boréal's S3 cloud-exposure scenario has an expected loss of \$150,000. Using the Gordon–Loeb sanity check, roughly what is the maximum investment that would not require special justification?

> A. About \$55,000
>
> B. About \$37,000
>
> C. About \$75,000
>
> D. About \$150,000

**12.** For Boréal's S1, Pb(A) = 0.80, Pb(ψ,A) = 0.25, CVSS = 8.7, δe = 0.60, δm = 0.90, μ(E) = 0.90, θ = 0.40, factor 1,000. What is the estimated risk Re?

> A. About 1,566
>
> B. About 2,936
>
> C. About 1,174
>
> D. About 3,524

**13.** S1 has Re = 2,936 and Rt = 1,566. Option A has a probability reduction of 0.40 and an impact reduction of 0.50. What are the residual risk, ratio, and verdict?

> A. 2,349; 1.50; above tolerance
>
> B. 881; 0.56; below tolerance
>
> C. 1,468; 0.94; approximately at tolerance
>
> D. 2,055; 1.31; above tolerance

**14.** In the CyberRiskGuardian cross-check, Options B and C2 cut expected loss by 70% and 50% but mitigate nothing in the index. Why?

> A. Their CVSS scores were scored by equivalency and therefore excluded
>
> B. Their ROSI was below the portfolio average of about 104%
>
> C. Their scenarios were already below tolerance before treatment
>
> D. Their impact reduction is zero, and the method multiplies the two reductions

**15.** S1's average damage is 0.75 and Boréal's appetite is 0.40. What minimum product of probability and impact reductions brings S1's residual risk down to the tolerated level?

> A. About 0.20
>
> B. About 0.40
>
> C. About 0.47
>
> D. About 0.53

**16.** A risk acceptance for an unpatched server was granted three years ago and never reviewed. Which compliance theatre pattern does this illustrate?

> A. Late-stage security reviews
>
> B. Exceptions without expiry
>
> C. Metric distortion
>
> D. Framework accumulation

**17.** Why does the chapter compare maturity improvement to an S-curve of diminishing returns?

> A. Because maturity scores rise only after a major incident forces investment
>
> B. Because each maturity level costs the same but delivers less protection
>
> C. Because foundational hygiene, such as CIS Controls IG1, yields disproportionate early benefit
>
> D. Because quantitative management is always required before basic patching

**18.** Under the chapter's classification, an audit finding with a committed remediation date that must be funded regardless of ROSI is which type of investment?

> A. Foundational
>
> B. Mandatory
>
> C. Discretionary
>
> D. Run

**19.** How does the CyberRiskGuardian method record a cyber insurance policy for a scenario that remains above tolerance?

> A. As a mitigation measure with an impact reduction equal to the policy limit
>
> B. As a probability reduction, because insurers condition cover on strong controls
>
> C. As an increase in organizational resilience θ for the affected scenario
>
> D. As risk transfer, documented with the share of loss covered and the decision-maker

**20.** An organization has a total IT budget of \$10 million and an approved risk appetite of 0.60. Using the chapter's best-practice benchmark with linear interpolation, what is its appetite-consistent cybersecurity budget?

> A. About \$590,000
>
> B. About \$780,000
>
> C. About \$400,000
>
> D. About \$990,000
