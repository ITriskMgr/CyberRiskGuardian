<!-- Introduction to Cybersecurity Governance, 3rd ed., v2.1g (Aug 2026) - (c) Marc-André Léger. Licensed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/). Third-party standards, frameworks and quotations remain under their owners' terms. -->

# Chapter 5: Creating Cybersecurity Risk Scenarios

This chapter presents a scenario-based cybersecurity risk assessment methodology designed to help students learn to assess cybersecurity risks through a simplified, teachable approach. It is intentionally lighter than the full methods used in large organizations because formal risk methods require time, training, and organizational data that are rarely available in a course setting. They also require expensive licences. The underlying hypothesis is that students who understand this simplified approach will have the knowledge and skills needed to learn more robust risk assessment methodologies used by the organizations that will hire them after graduation.

At the same time, the approach in this chapter is not anti-standard. It is deliberately structured so students can learn risk thinking in a way compatible with mainstream governance and risk standards. Concretely, the scenario work in this chapter can be viewed as the practical implementation of the **cybersecurity risk assessment process** that ISO/IEC 27001 requires organizations to define and apply, including setting risk criteria and producing consistent, valid, and comparable results.

The scenario lifecycle also aligns with ISO/IEC 27005’s emphasis on risk scenarios as a bridge among threats, vulnerabilities, consequences, and controls, including both event-based and asset-based approaches. In NIST terms, risk scenarios serve as a practical driver for selecting and tailoring safeguards (controls) and for planning how those safeguards will be assessed using recognized assessment methods and evidence types.

The chapter also addresses a core point: risk assessment is vulnerable to bias and subjectivity. Scenarios effectively reduce these biases because they require participants to describe concrete event sequences, trigger conditions, and business consequences, rather than debating vague risks in the abstract.

## Standards compatibility

Because this is an educational method, compatibility means the following:

- First, the chapter produces artifacts that map cleanly to ISO/IEC 27001 requirements for risk assessment and risk treatment documentation (including risk treatment planning and the Statement of Applicability logic).

- Second, the process steps align with ISO/IEC 27005’s risk management flow and scenario concepts (risk components, scenario construction, and monitoring indicators).

- Third, the method supports a NIST-aligned control strategy: scenarios can map to control baselines and families (800-53B), and assessments can be planned using NIST assessment objectives and methods (800-53A).

## Using cybersecurity risk scenarios

Scenarios have long been used to help decision-makers reflect on plausible futures, stress-test assumptions, and consider alternatives. In cybersecurity, scenarios are especially useful because threats evolve quickly, organizations have complex dependencies, and purely historical loss data can be misleading (it often fails to capture recent architectural changes, new vendors, cloud migrations, emerging attack patterns, or new regulatory exposure). This aligns with standards and best practices, such as ISO/IEC 27005, which frames scenarios as a structured way to connect threats, vulnerabilities, consequences, and desired end states and treats risk as a combination of past evidence and forward-looking factors. 

In this book, scenarios are treated as the central unit of analysis for learning cybersecurity risk assessment. **A scenario is a story.** The risk triangle illustrates how a **threat agent** exploits one or more **vulnerabilities**, resulting in an undesirable outcome in which **exposure to risk** manifests as an adverse outcome. Hence, the outcome negatively affects the organization’s mission, strategy, legal and regulatory obligations, and confidentiality, integrity, and availability requirements.

Scenarios become even more useful when they are explicit about event sequences, triggers, stakeholders, and business consequences—precisely where hidden assumptions and optimism bias tend to live.

This book uses four strategies for creating cybersecurity risk scenarios, and the chapter maintains the same four-part structure: a top-down approach, a bottom-up approach, generic scenarios, and generative AI.

## The proposed approach

The proposal is to create and regularly update a manageable set of risk scenarios, including those describing threats, vulnerabilities, existing controls, detection capabilities, response capabilities, and recovery measures. This aligns closely with ISO/IEC 27001’s expectation that an organization define criteria for risk assessment and risk acceptance, and plan risk treatment choices, with necessary controls identified and documented in a way that supports ongoing operation and reassessment. 

There is no universally correct number of scenarios needed to properly assess cybersecurity risks. In an educational setting, ten scenarios are typically sufficient for students to learn the mechanics and reasoning. In real life, a reasonable maturity progression for organizations is to start with about ten scenarios and gradually increase to twenty or more over a few assessment cycles, as stakeholders become more comfortable and the organization becomes more data driven. In the long term, the ideal number of scenarios would depend on the capabilities the organization has built.

From a best-practices perspective, this ten-to-twenty scenario portfolio serves as the working set that supports scenario analysis, indicator monitoring, periodic risk assessment expectations, and control selection and assessment planning.

## Scenario fundamentals

A cybersecurity risk scenario should explicitly document the practical information needed to ensure it is understandable, testable, and comparable over time. At a minimum, the scenario should identify the affected business process; the primary business asset and supporting technical assets; the information involved; the relevant threat actor or threat event; the vulnerability or control weakness; the conditions that trigger the scenario; and the sequence of events leading to the undesirable outcome.

The scenario should also identify the business owner, risk owner, and other affected stakeholders; describe the potential operational, financial, legal, regulatory, privacy, safety, and reputational consequences; and document the preventive, detective, response, and recovery controls already in place. Proposed treatment measures should explain how they are expected to reduce the likelihood of the scenario, mitigate its impact, or improve organizational resilience.

To support consistent assessment and governance, each scenario should also reference the organization’s risk criteria and acceptance thresholds, identify relevant monitoring indicators, and map proposed controls to applicable control frameworks, such as ISO/IEC 27001 or NIST SP 800-53. To improve alignment with ISO/IEC 27001 and ISO/IEC 27005, additional elements should be included in cybersecurity risk scenarios:

- First, **include risk criteria and risk acceptance references**: the scenario should reference the risk criteria used (for example, the organization’s defined impact and likelihood scales and risk acceptance criteria), because ISO/IEC 27001 requires these criteria to exist before risk assessment results are meaningful and comparable. ISO/IEC 27005’s guidance similarly emphasizes risk acceptance criteria and shows how an organization can operationalize acceptance thresholds (e.g., through a matrix and management approval levels).

- Second, **asset linkage**: ISO/IEC 27005 explicitly states that scenario development should link threats, vulnerabilities, events, and consequences to assets, and that controls apply to subsets of assets. Therefore, each scenario should identify at least one business asset (primary asset) and one supporting asset (system, application, or infrastructure element).

- Third, **control mapping**: each scenario should include a control mapping section that maps proposed mitigation measures to ISO/IEC 27001 Annex A control themes (where relevant) and to one or more NIST SP 800-53 control families, enabling disciplined control selection and subsequent assessment planning. NIST SP 800-53B supports this by organizing controls into baselines and families, including the Risk Assessment (RA) family and other families commonly used in scenario-driven risk reduction.

## Scenario creation strategy

Scenario creation should be treated as a project, typically led by a cybersecurity analyst, and should use interviews, brainstorming, and consensus-building techniques. There are many ways to perform this task. The top-down and bottom-up strategies presented below can be understood as two complementary perspectives found in many standards and frameworks. One begins with strategic objectives and undesirable end states (event-based thinking), and the other begins with critical assets and operational dependencies (asset-based thinking). Risk scenarios can be built using an event-based approach, an asset-based approach, or both.

### Top-down approach

The top-down approach begins with organizational strategy, governance requirements, and the cybersecurity governance framework, then asks how deficiencies, nonconformities, or control breakdowns could realistically occur and cause business harm. This naturally aligns with risk-based planning expectations because it requires the scenario to link business objectives, obligations, and risk criteria before discussing technical fixes. 

### Bottom-up approach

The bottom-up approach starts with operational value creation: critical systems, critical processes, critical data flows, and critical third parties. In a risk assessment process, scenario development should link events, consequences, threats, and vulnerabilities to assets, and controls should then be applied to subsets of those assets. This is precisely what the bottom-up method trains students to do.

<img src="media/image30.png" style="width:6in;height:4.5in" />

<span id="_Toc235166912" class="anchor"></span>Figure 30: scenario creation strategy

### Using generic scenarios

Generic scenarios are useful as scaffolding, especially early in the maturity cycle. When using generic scenarios, the standards-consistent improvement is to require that each be localized in a disciplined way: anchor the scenario to real or plausible assets and events, and to an explicit undesirable outcome that aligns with the organization’s risk criteria and risk appetite/tolerance. As the organization matures in cybersecurity, it should abandon this strategy in favour of the others. 

### Using generative AI

Generative AI is the fourth strategy. The core hypothesis is that in low-maturity environments or course projects with limited stakeholder access, a well-prompted LLM can produce plausible initial scenario drafts comparable to early-stage human brainstorming, provided the outputs are critically reviewed and aligned with risk criteria. This approach is used in a classroom setting and is presented in more detail in this chapter. In this setting, students receive a specific guidance document to help them apply this strategy.

## Scenario validation

Scenario validation is an area where standards compatibility becomes more tangible. This helps ensure the risk assessment process yields results that are consistent, valid, and comparable, and it supports risk treatment planning and documentation.

<img src="media/image31.png" style="width:6in;height:4.5in" />

<span id="_Toc235166913" class="anchor"></span>Figure 31: scenario validation

Validation helps ensure that the scenario correctly links sources, vulnerabilities, consequences, and controls, and supports the subsequent definition of key risk indicators for monitoring. Validation also serves as the bridge to assessing ability. Even when risk scenarios are used in a classroom, or when a full control assessment is not performed, scenarios should be written so that a real organization could later assess the existence and effectiveness of controls using recognized methods and evidence. Assessment objectives should be supported by assessment methods and objects, most commonly by examining documents and artifacts, interviewing personnel, and testing mechanisms or processes. 

In other words, scenario documentation should be written so that, later, someone could reasonably say:

- What would we examine,

- Who would we interview, and

- What would we test to validate this control claim?

## Who to include in scenario creation

Cybersecurity risk scenarios should be developed collaboratively, not by a single technical specialist. Involving a representative cross-section of stakeholders reduces individual bias, improves the scenario's completeness and realism, and increases organizational acceptance of the resulting risk assessment and treatment decisions.

At a minimum, the scenario-development group should include the business owner of the affected asset or process, who can explain its strategic and operational importance and may ultimately serve as the risk owner. IT and cybersecurity personnel with knowledge of the relevant systems, dependencies, vulnerabilities, and existing controls should also participate. In addition, a representative from risk management, compliance, legal, or governance should contribute expertise on the organization’s risk criteria, regulatory obligations, risk appetite, and formal acceptance practices.

Depending on the scenario, other participants may include privacy specialists, records-management personnel, human resources, finance, procurement, business continuity and incident-response teams, physical-security personnel, system architects, internal audit, and representatives responsible for key suppliers or outsourced services. When safety, operational technology, or critical infrastructure is involved, include engineering and operational personnel as well.

This multidisciplinary participation is not meant to achieve unanimous agreement on every estimate. Rather, it ensures the scenario reflects business realities, technical conditions, legal and contractual obligations, control capabilities, and the full range of potential consequences. Document and resolve differences in professional judgment through the organization’s established risk-governance and decision-making processes.

This approach emphasizes defined risk criteria, accountable risk ownership, and explicit risk acceptance. It also ensures that risk treatment decisions and residual risk acceptance are managed intentionally, documented, and assigned to appropriate authorities.

## Automated scenario creation with Generative AI

This chapter continues by describing how students can use generative AI to draft scenarios when real stakeholder engagement is impractical, while emphasizing the importance of review and academic integrity. With recent advancements and the widespread availability of generative AI, automated scenario creation has also become a viable option in real-world organizational settings, provided security concerns are addressed or local LLMs are used. In addition to accelerating scenario creation, it enables the development of many scenarios with fewer human or financial resources.

### Step 1: Provide context

While many Generative AI offerings are available, students should use the current ChatGPT interface and select a model that, at a minimum, supports long-context reasoning and file-based analysis. This is because scenario quality improves when the model can refer to a stable description of the target organization and any provided architectural notes. ChatGPT’s data analysis workflow supports uploading and working with common file types, such as spreadsheets and PDFs, which is directly useful for drafting scenarios from a business case or a simplified architecture description. If the course uses a standardized business case, students can simply paste the case text online. A more robust approach is to provide the business case as a file and instruct the model to treat it as the authoritative reference. Where possible, students should also define the risk criteria they will use (even if simplified) and explicitly state that the output must be compatible with ISO/IEC 27005-style scenario components (threats, vulnerabilities, consequences, assets, controls). 

**In ChatGPT,** upload the assigned business case only when the tool and course rules permit. Tell the model that the case is the authoritative source. Define the organization, scope, assessment period, risk appetite, scoring scales, required frameworks, and prohibited assumptions. Ask the model to identify uncertainty rather than invent missing facts. Then use the following prompt, replacing \[ORGANIZATION\] with your value:

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th>I’m in the role of a cybersecurity risk analyst, working for an IT and cybersecurity consulting firm. I was hired by a customer to assist them with their cybersecurity governance. You are assisting me with performing a cybersecurity risk assessment for [ORGANIZATION].<br />
I’ve attached a business case and documentation describing the target organization that has hired me. Treat the attached case and information as the authoritative organizational source.<br />
Do not invent systems, incidents, legal duties, or control capabilities that are not supported by the case, by evidence or by legitimate reliable sources.<br />
Use a 12-month assessment horizon and 0-to-1 scales.<br />
Distinguish case facts, external evidence, assumptions, and analytical inferences.<br />
Align scenario structure with COBIT, ISO/IEC 27001, ISO/IEC 27002, ISO/IEC 27005, NIST CSF concepts and NIST SP 800-30 risk factors.<br />
When useful search the Internet to improve accuracy.<br />
Keep track of the sources you are using and list them in your answers.</th>
</tr>
</thead>
<tbody>
</tbody>
</table>

### Step 2: Create high-level scenarios

To help ChatGPT learn the process, ask it to generate 50 candidate cybersecurity risk scenarios. Think of it as brainstorming. In this step, use the model to produce scenarios explicitly tagged as primarily event-based, primarily asset-based, or hybrid. While you can skip it, research shows that including it improves the quality of the scenarios produced in subsequent steps. Then use the following prompt:

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th>I need your help with some initial brainstorming.<br />
Help me, by create a list of at least 50 concise cybersecurity risk scenarios for [ORGANIZATION].<br />
For each scenario provide: name; one-sentence scenario statement; primary business asset; supporting technical asset; threat source or event; enabling vulnerability or control weakness; main consequence; and generation type (top-down, bottom-up, generic-localized, or hybrid).<br />
Avoid duplicates and ensure coverage of governance, people, third parties, cloud, identity, data, operations, detection, response, and recovery.<br />
When useful search the Internet to gather inspiration for likely scenarios for similar organizations, organizations in the same industry or geographic region but do not limit yourself to these sources. These scenarios must be reasonably credible and realistic.<br />
Remember these concise cybersecurity risk scenarios, which will be used in the next prompts to create detailed risk scenarios.<br />
Create a Word document with the results.</th>
</tr>
</thead>
<tbody>
</tbody>
</table>

For the next step, use 10 of the 50 scenarios. To do this, it is suggested that students select 10 that seem most relevant to them and document or justify their choices. Because this is usually done in a group assignment, consensus-building is proposed as part of the scenario selection process, as it should occur in a real-world situation.

### Step 3: Create detailed scenarios

The final step is to create the detailed scenarios. Repeat this step for each required scenario. Individual scenarios are created using the following prompt:

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th>Expand the cybersecurity risk scenario [NAME] created previously into a detailed scenario including: scenario name; stakeholders and risk owner; background and scoping assumptions; detailed incident description and event sequence; consequences (business, operational, financial, privacy, legal/regulatory, safety, and reputational as applicable); implicated assets (primary and supporting); existing controls and detection/response measures; proposed treatment options with mapping to ISO/IEC 27001 Annex A themes and NIST SP 800-53 control families; and monitoring indicators.<br />
Provide estimates on a 0-to-1 scale for hazard or threat presence, exploitation likelihood, estimated damage, maximum damage, resilience, and expected utility or organizational criticality.<br />
Document the rationale, evidence, assumptions, and confidence level for each estimate.<br />
Compute a CVSS Base score for the primary technical vulnerability and provide the complete vector. If CVSS is not applicable, explain why rather than inventing a score and provide a best-guess estimate on a 0-to-10 scale for the vulnerability level.<br />
Provide a rough treatment budget estimate in Canadian dollars and estimate impact reduction and likelihood reduction if the proposed treatment options were implemented on a 0-to-1 scale. Identify dependencies, shared controls, implementation constraints, and residual uncertainties.<br />
Follow the structure and evidence conventions of the supplied template as the authoritative model to create a Word document with the detailed scenario. Make sure the file has a unique file name to facilitate it’s use.</th>
</tr>
</thead>
<tbody>
</tbody>
</table>

Once all the required scenarios are created, they can then be used to perform the cybersecurity risk assessment, which is the subject of the next chapter.
