<!-- Léger, M.-A. (2026). Introduction to Cybersecurity Governance for Business Technology Management (3rd ed., v2.51). Chapter 18: AI for Cybersecurity and the Cybersecurity of AI. CC BY-NC 4.0 -->

# Chapter 18: AI for Cybersecurity and the Cybersecurity of AI

## Learning outcomes

After completing this chapter, you should be able to:

1.  Distinguish AI for cybersecurity from the cybersecurity of AI and explain why each requires different managerial decisions and owners.

2.  Evaluate AI security tools using operational outcome measures and explain how hallucination and automation bias create failure modes.

3.  Explain how AI changes attacker economics and recommend structural controls that hold even when users are deceived.

4.  Classify AI threats by confidentiality, integrity, availability, autonomy, and human reliance using recognized taxonomies such as OWASP, MITRE ATLAS, and NIST AI 100-2.

5.  Design governance for AI agents as non-human identities, including least privilege, approval gates, connector controls, and a working off switch.

6.  Apply secure AI lifecycle controls, from planning and data governance through red teaming, deployment, monitoring, and retirement.

7.  Compare current and scheduled AI obligations under Law 25, Canadian federal guidance, OSFI E-23, and the EU AI Act, and map them to one governance system.

8.  Recommend an AI governance program for an organization, covering inventory, acceptable use, incident response, and build, buy, or prohibit decisions.

Artificial intelligence has become a dual force in cybersecurity. It strengthens defenders by accelerating detection, triage, investigation, and the routine work of security operations. It also strengthens adversaries by lowering the cost of reconnaissance, social engineering, malware development, and exploitation. At the same time, AI systems have themselves become high-value assets and, in many organizations, high-risk systems. This creates a governance reality that did not exist a decade ago: organizations must manage cybersecurity in an AI-enabled threat environment while also managing the cybersecurity risks of the AI systems they build, buy, and allow their people to use.

For managers, the governance shift is not primarily about the mathematics of machine learning. It is about decision rights, accountability, risk appetite, and evidence. AI introduces failure modes that traditional security control catalogues do not always capture. Some are technical, such as model theft, prompt injection, and leakage of training data. Some are operational, such as an AI agent that takes an unintended action in a production system because it was given broad permissions and misread its instructions. Others are legal and reputational: privacy violations, discriminatory outcomes, misleading outputs, and questions of duty of care when a customer or student relies on what a chatbot said. In addition, AI capabilities are usually acquired as services—models, application programming interfaces, plug-ins, connectors, and managed platforms—so the organization's exposure expands through dependencies it does not control.

The governance problem is compounded by speed and diffusion. Cloud computing (Chapter 15) arrived largely through procurement and architecture decisions the security function could see; generative AI arrived through browsers, productivity suites, and personal accounts, often before any policy existed. Many organizations discovered they were already "using AI" before deciding whether, where, and how they wanted to. The first managerial task is to recover that ground: to establish what AI is in use, who owns it, what it can see and do, and which uses fall inside the organization's risk appetite.

This chapter builds directly on earlier material. The risk management and scenario methods of Chapters 4 to 6 apply to AI just as they apply to any other system, but the threat events and vulnerabilities change. The operations and detection practices of Chapters 7 and 8 are where AI for cybersecurity delivers, or fails to deliver, its value. The control frameworks of Chapter 9 provide the management-system logic into which AI governance should be integrated rather than bolted on. Identity and access management (Chapter 10), data protection and privacy (Chapter 16), and investment governance (Chapter 17) all reappear here in new forms, and the AI supply chain is one important case of the third-party governance discussed in Chapter 19.

The chapter integrates two perspectives. The first is **AI for cybersecurity**: how organizations can use AI responsibly to improve detection, response, and resilience, and how adversaries use the same capabilities against them. The second is **cybersecurity of AI**: how organizations govern, secure, monitor, and assure AI systems and AI-enabled business processes, including generative AI assistants and AI agents that can take actions through tools. Along the way, the chapter addresses shadow AI and acceptable use, the AI supply chain, human oversight, the state of regulation and standards in late 2026, incident management for AI, and the investment decisions of what to build, what to buy, and what to prohibit. The guiding theme is that **AI must be treated as a governed capability and, whenever it touches sensitive data, influences decisions, or triggers actions, as a critical system** that requires lifecycle controls from design to decommissioning, aligned with risk appetite and regulatory obligations.

The figure below contrasts the two perspectives. They share a vocabulary and many controls, but they answer different managerial questions and are often owned by different people.

<span id="_Toc243635574" class="anchor"></span>Figure 75: AI for cybersecurity and the cybersecurity of AI

## AI for cybersecurity: what it can do—and what it cannot do

AI can materially improve cybersecurity operations when applied to problems where scale, speed, and pattern recognition matter. In most security operations centres, the persistent constraint is less a shortage of tools than the mismatch between alert volume and human capacity. Analysts face thousands of events, fragmented telemetry, and incomplete context, and they must decide quickly which signals deserve attention. AI can help by reducing noise, accelerating triage, and assisting analysts in forming coherent hypotheses about what is happening. Used properly, it can shorten time to detect and time to respond, which are often more valuable than marginal improvements in prevention. Used poorly, it can add a layer of confident-sounding opacity on top of an already overloaded process.

### High-value applications

The applications that tend to deliver value share a pattern: they assist people with tasks that are repetitive, data-heavy, and time-sensitive, while leaving consequential decisions with accountable humans. The following uses are now common in mature security programs.

- **Alert enrichment and correlation.** Models can group related alerts across endpoint, identity, cloud, network, and application logs and attach context such as asset criticality and recent changes, so analysts see incidents rather than isolated events.

- **Identity and behaviour analytics.** Anomaly detection on sign-ins, privilege use, and data access can surface account takeover and insider misuse earlier than static rules (Chapter 10).

- **Investigation assistance.** Generative AI can summarize incident timelines, translate questions into query languages, explain unfamiliar scripts, and draft case notes, shortening the learning curve for junior analysts.

- **Malware and phishing triage.** Classifiers and language models can sort suspicious files, messages, and URLs at volume and extract indicators of compromise.

- **Vulnerability and exposure management.** AI can correlate vulnerability data with exploitability and asset context, supporting the prioritization discipline of Chapter 8.

- **Governance and compliance support.** Generative AI can help draft policies, control descriptions, risk entries, and audit narratives, provided confidentiality is protected and an accountable person reviews every output.

These uses fit the Detect, Respond, and Identify functions of NIST CSF 2.0. NIST's draft Cyber AI Profile (NIST IR 8596, first released as a preliminary draft in December 2025 and still in draft at the time of writing) frames them as "Defend," alongside "Secure" (securing AI system components) and "Thwart" (countering AI-enabled attacks)—a three-part structure managers can use to frame their own AI-related objectives.

The risk assessment method of this book already contains one such use. Chapter 5 showed how a generative AI model can draft candidate and detailed risk scenarios from a business case, and Chapter 6 how it can propose first estimates of the scenario parameters and CVSS v4.0 vectors that analysts then critique, mark as analytical estimates requiring validation, and recompute with a verified calculation engine. The cautions given there—protect the confidentiality of what is uploaded, validate every scenario and number, expect hallucination and bias, and keep humans accountable for the risk decision—are the same controls this chapter develops at enterprise scale. In a real organization, that work would be done only with an approved enterprise or self-hosted tool, under the acceptable-use rules discussed later in this chapter.

### Limits and failure modes

AI does not eliminate the need for security engineering judgment, and organizations should be cautious about treating it as an oracle. Generative models can produce plausible but incorrect explanations, a behaviour often called hallucination or, in NIST's terminology, confabulation. A summary that confidently attributes an incident to the wrong root cause can send responders in the wrong direction at the worst possible moment. Machine learning detection can be misled by poor data quality, drift in the environment, or training examples that do not reflect current attacker behaviour. Attackers can also deliberately manipulate what AI systems see, through adversarial inputs, poisoned data, or text crafted to redirect a language model, so AI-driven detection should not be considered inherently more trustworthy than well-designed deterministic controls.

There is also a subtler organizational risk: **automation bias**, the tendency to over-trust automated outputs, especially under time pressure. When an AI assistant labels an alert benign, a tired analyst may close it without scrutiny, and skills no longer exercised can atrophy, leaving the team weaker when the AI is wrong or unavailable. These are reasons not to avoid AI but to design its use deliberately, measure its effect, and keep humans genuinely in control of consequential decisions.

### Governing AI in security operations

A mature governance stance treats AI as a force multiplier for competent teams, not as a replacement for them. The objective is to improve operational outcomes—faster detection, fewer false positives, better containment discipline, stronger recoverability—while retaining clear human accountability for decisions that carry material risk. In practice, this means defining in advance which actions an AI-enabled security tool may recommend, which it may execute automatically, and which always require an analyst's approval. Isolating a laptop with known ransomware indicators may be an acceptable automated action; disabling an executive's account or blocking a payment gateway usually is not.

Managers should require the same evidence of value from AI security tools as from any other investment (Chapter 17). The relevant measures are operational: mean time to detect and respond, analyst hours per incident, false-positive and false-negative rates on a representative sample, and the proportion of AI recommendations that analysts override. A rising override rate may indicate drift; a near-zero rate may indicate automation bias rather than perfection. Either way, these metrics become key risk indicators for the AI capability itself.

| AI in the security operations centre should be measured by outcomes, not demonstrations: if it does not reduce time to detect, time to respond, or analyst workload on real incidents, it is cost and complexity, not capability. |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## AI for attackers: why the threat landscape accelerates

The most important change AI brings to the threat landscape is economic. It reduces the cost and skill required to perform tasks that previously demanded expertise, time, or language fluency. A capability that was once limited to well-resourced groups becomes available to many more actors, and the volume and polish of attacks rise accordingly. Cybersecurity agencies, including the Canadian Centre for Cyber Security (CCCS) in its national cyber threat assessments, have described AI as an accelerant of existing threats rather than a source of entirely new ones. For governance, this distinction matters: the right response is usually to strengthen the structural controls the organization already knows it needs, not to search for a magic AI countermeasure.

### Cheaper persuasion and synthetic media

Social engineering becomes cheaper and more convincing. AI-assisted phishing and business email compromise messages can be generated at scale, in fluent and context-appropriate language, and tailored to individual targets using public information. Synthetic voice and video—deepfakes—increase the credibility of fraud attempts against finance teams, executives, help desks, and customer service. Well-publicized cases have involved employees authorizing large transfers after video calls in which every other participant was synthetic. The traditional advice to "look for spelling mistakes" is obsolete.

The governance implication is that identity assurance and transaction controls become more important. When persuasion improves, organizations must depend less on users noticing deception and more on process controls that hold even when the user is fooled: phishing-resistant authentication (Chapter 10), out-of-band verification of payment changes using contact details already on file, dual authorization, help-desk identity proofing that does not rely on researchable facts, and explicit permission to pause and verify an urgent request from a senior leader. Awareness training should shift from "spot the fake" to "follow the verification procedure every time."

### Faster tooling, reconnaissance, and exploitation

Malware development and exploit adaptation can accelerate. AI can assist in code generation, obfuscation, the creation of variants that evade signature-based detection, and rapid refinement of malicious tooling. Even when AI does not produce novel sophisticated exploits, it compresses attacker iteration cycles and increases the volume of plausible attempts. The window between public disclosure of a vulnerability and its exploitation, already short, is under further pressure, which reinforces the importance of exposure management and disciplined patching of internet-facing systems.

Reconnaissance also becomes more systematic, as adversaries automate the synthesis of public information about structure, personnel, vendors, and exposed services. AI helps attackers "live off the land" by quickly understanding enterprise tooling and administrative workflows, so they blend into normal operations. And attackers increasingly target AI systems themselves—chatbots, agents, and model endpoints—as new entry points.

### The governance response: structural controls

The conclusion for governance is not that AI renders security impossible. It is that **fragile controls—those that depend on user perfection, obscure detection rules, or untested recovery—will fail more often**. The organization must deliberately invest in structural controls: strong identity and least privilege, segmentation, secure configuration baselines (Chapter 14), attack surface management, logging of administrative actions, tested backup and recovery, and transaction controls that assume persuasion will sometimes succeed.

Risk scenarios should be updated accordingly. When revisiting the scenario library (Chapter 5), managers should ask how AI changes the likelihood and scale of each threat event. Executive impersonation leading to fraudulent payment, for example, is now more likely than a few years ago, and its treatment options should be reassessed against risk appetite. This translates a general concern about "AI threats" into defensible, prioritized decisions.

## Cybersecurity of AI: assets, attack surfaces, and threat categories

AI systems introduce risks that straddle security, privacy, safety, compliance, and reputation. Confidentiality, integrity, and availability remain essential, but additional properties become operationally important: the integrity of training data, the reliability of model behaviour, the confidentiality of model internals and hidden instructions, the control of outputs and actions, and the governance of human reliance on the system.

### An AI system is an ecosystem, not a model

An AI system is rarely just a model. It is typically an ecosystem of data pipelines, feature stores or embedding databases, training and evaluation environments, model registries, deployment infrastructure, APIs, user interfaces, tool integrations, and operational monitoring. Generative AI applications add further components. **Retrieval-augmented generation (RAG)** connects a language model to the organization's document repositories so it can answer questions grounded in internal content. System prompts and prompt templates define how the model should behave. Tool-calling and agent frameworks allow the model to query databases, send messages, or change configurations. External plug-ins and connectors extend its reach into software-as-a-service platforms. Every one of these components introduces potential attack paths and every one needs an owner.

For managers, a useful mental model is that an AI system adds three kinds of asset to the familiar ones: the **model** itself, whose weights may embody intellectual property or memorized fragments of training data; the **context** it receives at run time, such as system prompts, retrieved documents, and tool results; and the **capabilities** it has been granted, meaning the tools it can call, the data it can reach, and the actions it can take. Each must be inventoried, classified, and protected, and each enters the risk triangle as an asset with its own vulnerabilities.

### Confidentiality, integrity, and availability in AI terms

Threats to confidentiality, integrity, and availability still apply, but the mechanisms differ. The figure below groups the main threat categories in a form managers can use when building risk scenarios.

<span id="_Toc243635575" class="anchor"></span>Figure 76: Threat categories for AI systems

**Confidentiality threats** include leakage of sensitive information through prompts and outputs; exfiltration of proprietary data reachable through a RAG pipeline; inference of training data through model inversion or membership inference techniques; exposure of hidden system prompts that contain business logic or, worse, credentials; and theft of the model itself through compromised repositories, insecure endpoints, or systematic querying designed to replicate its behaviour. In regulated contexts, confidentiality also includes privacy breaches: personal information appearing in outputs, being used improperly in training, or being transmitted to third-party model providers without appropriate governance.

**Integrity threats** include data poisoning (malicious manipulation of training or fine-tuning data), backdoored models obtained through the supply chain, manipulation of context to force incorrect behaviour, and prompt injection. **Prompt injection** occurs when text supplied by a user or embedded in content the model processes is interpreted as instructions, causing the model to ignore its constraints, reveal secrets, or take unsafe actions. *Direct* injection comes from the user typing into the interface. *Indirect* injection is more dangerous for enterprises: instructions hidden in an email, web page, shared document, or calendar invitation are read by an AI assistant processing that content on behalf of a legitimate user. Because language models do not reliably distinguish between data and instructions, indirect prompt injection is best understood as a structural weakness to be contained by design, not a bug that a vendor will soon patch away.

**Availability threats** include denial of service against model APIs, rate-limit abuse, and dependency on external AI providers whose outages or policy changes interrupt business processes. Generative AI adds a financial dimension: because usage is often billed per token or per call, attackers or careless integrations can drive costs sharply upward, a risk OWASP calls *unbounded consumption*. In practice, AI availability is not only technical uptime; it is also the availability of trustworthy behaviour. **A model that is online but producing systematically wrong or unsafe outputs is, from a governance standpoint, unavailable.**

### Autonomy, action, and human reliance

A distinct category exists for autonomy and action. When AI systems are connected to tools—sending emails, generating configuration changes, initiating transactions, querying internal systems—misbehaviour becomes an operational risk, not merely a content risk. The next section treats this in depth because agentic AI is where many organizations' exposure is now growing fastest.

A final category concerns human reliance. Misinformation and subtle errors become harmful when people or downstream systems act on them without verification: a customer who follows incorrect instructions about a financial product has suffered harm even though nothing was "hacked." These risks belong in the same register because they share owners, controls, and incident processes with security risks, and because attackers can deliberately induce them.

### Using threat taxonomies

Managers do not need to invent their own taxonomy of AI threats. Several public references provide shared vocabulary and can be used to structure scenario workshops, testing plans, and supplier questionnaires.

- **NIST AI 100-2 E2025**, *Adversarial Machine Learning: A Taxonomy and Terminology of Attacks and Mitigations* (March 2025), distinguishes attacks on predictive AI (evasion, poisoning, and privacy attacks) from attacks on generative AI (including prompt injection and misuse), and describes known mitigations and their limits. NIST has stated that it intends to update the report regularly, so the edition in use should be recorded.

- **The OWASP Top 10 for LLM Applications 2025** lists the most significant application-level risks for systems built on large language models: prompt injection, sensitive information disclosure, supply chain, data and model poisoning, improper output handling, excessive agency, system prompt leakage, vector and embedding weaknesses, misinformation, and unbounded consumption. It complements the general OWASP Top 10:2025 for web applications, and OWASP revises it periodically.

- **The OWASP Top 10 for Agentic Applications**, published by the OWASP GenAI Security Project in December 2025 for the 2026 cycle, addresses risks specific to autonomous agents, such as agent goal hijacking, tool misuse, identity and privilege abuse, and cascading failures across multi-agent systems.

- **MITRE ATLAS** (Adversarial Threat Landscape for Artificial-Intelligence Systems) is a knowledge base of adversary tactics, techniques, and real-world case studies against AI systems, modelled on MITRE ATT&CK. It is useful for threat-informed testing and for linking AI threats to detection engineering.

Their managerial value is consistency: when scenarios, red-team plans, vendor questionnaires, and incident categories share one vocabulary, evidence accumulates coherently and gaps become visible.

## Agentic AI: when models can act

An **AI agent** is a system in which a model does not merely answer a question but plans and executes a sequence of steps toward a goal, calling tools, reading results, and deciding what to do next with limited or no human intervention at each step. Agents can book meetings, reconcile invoices, open and update tickets, write and run code, query databases, and coordinate with other agents. Their appeal is obvious: they promise to automate work that previously required a person to move between several systems. Their risk is equally clear: they combine the unpredictability of a language model with the permissions of a privileged user.

### What makes an agent different

Three features distinguish agents from conversational assistants. The first is **autonomy**: the agent chooses its own intermediate steps, so the designer cannot fully enumerate in advance what it will do. The second is **tool access**: the agent acts through APIs, scripts, browsers, or connectors, so its outputs are actions in real systems rather than text a person reads. The third is **persistence**: many agents maintain memory across sessions or run continuously in the background, so a compromise or a bad instruction can persist and compound. When several agents are chained together, an error or manipulation in one can propagate through the others, producing cascading effects that are difficult to trace.

These features shift the governance question from "Is the output accurate?" to "What can this system do, on whose authority, and how would we know if it did something wrong?" Those are familiar questions from privileged access and change management; the novelty is an actor whose behaviour is shaped at run time by whatever content it reads.

### Tool integrations and connector protocols

Agents reach tools through integrations. A growing share of these use standardized connector protocols, of which the **Model Context Protocol (MCP)**, introduced in late 2024 and since adopted widely across the industry, is the best-known example. Such protocols let an AI application discover and call "servers" that expose tools and data—a file system, a code repository, a customer relationship management platform, an email account—through a common interface. Standardization is valuable because it makes integrations cheaper and more consistent. It also means that adding a new capability to an agent can be as simple as installing a connector, often without any review by security or architecture teams.

Connector ecosystems raise several concerns. A third-party connector is supply-chain software that may be malicious, abandoned, or poorly secured; its tool descriptions are text the model reads, so a compromised connector can inject instructions; it often holds long-lived, broadly scoped tokens; and when one agent can call many connectors, a single injected instruction can chain actions across systems never meant to be connected. The control response is to treat connectors as privileged integrations: approved from an allow-list, scoped to minimum permissions, authenticated with short-lived credentials tied to the requesting user where possible, logged, and reviewed when they change.

### Excessive agency and the confused deputy

OWASP's term **excessive agency** captures the core risk: an AI system is granted more functionality, permissions, or autonomy than its purpose requires, so that when it is manipulated or simply mistaken, the damage is larger than necessary. An email-summarizing assistant that can also send and delete messages, a support agent that can issue refunds without limit, or a coding agent with write access to production all exhibit excessive agency. The classic security concept of the **confused deputy**—a privileged program tricked into misusing its authority on behalf of a less privileged party—describes what happens when indirect prompt injection meets excessive agency. The attacker never needs the agent's credentials; they only need the agent to read their text.

The most effective mitigations are architectural rather than model-based: limit tools to those the task requires and each tool to minimum (preferably read-only) permissions; separate components that read untrusted content from those that take sensitive actions; require human approval for irreversible, high-value, or externally visible actions such as payments, deletions, permission changes, or external messages; set hard limits on spending, volume, and duration; and provide a reliable way to stop an agent immediately. The joint guidance *Careful Adoption of Agentic AI Services*, released at the end of April 2026 by the Australian Signals Directorate's Australian Cyber Security Centre, CISA, the NSA, the CCCS, and the British and New Zealand national cyber security centres, makes the same points: adopt agents incrementally, starting with clearly defined low-risk tasks; prioritize resilience, reversibility, and risk containment over efficiency gains; apply least privilege; test agent behaviour in sandboxed environments before production; require human approval for high-impact actions; and maintain a trusted registry of authorized agents and their identities.

### Governing agents as non-human identities

The most practical governance move is to **treat every agent as a non-human identity with an owner, a purpose, and a permission set**, subject to the same joiner–mover–leaver discipline as any service account (Chapter 10). Each agent should be registered in the AI inventory and the identity system, with a named business owner who approves its scope and accepts its residual risk. Its credentials should be unique, rotated, and revocable. Its actions should be attributable—logs should show which agent did what, on whose behalf, using which tool, and in response to which input. Access reviews should include agents, and an agent whose owner leaves should be suspended until reassigned, just as an orphaned account would be.

This framing also clarifies decision rights. A business unit may propose an agent, but granting it access to sensitive data or the ability to act in financial or customer-facing systems is a risk decision that belongs with the relevant risk owner, informed by security and privacy review. Where the organization's risk appetite does not yet support autonomous action in a domain, the agent can still be useful in a recommend-only mode, preparing actions that a person reviews and executes.

| An AI agent is a privileged user that takes instructions from whatever it reads; **govern it as a non-human identity with least privilege, an accountable owner, approval gates for consequential actions, and a working off switch**. |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## Governance foundations: inventory, classification, and accountability

A recurring governance failure in AI adoption is that AI systems proliferate without clear ownership. Security leaders discover them after deployment, privacy leaders discover them after a complaint, and operational leaders discover them after an incident. Meanwhile, AI features are switched on inside platforms the organization already uses, sometimes by default and sometimes through a vendor update. The first controls are therefore managerial: know what AI is in use, classify it by risk, and assign accountability.

### The AI inventory

An **AI inventory** is the register of AI systems and AI-enabled services that the organization develops, procures, or permits. It should include internally developed models and applications, AI features embedded in purchased software, externally consumed AI services and APIs, AI agents and connectors, and, as far as can be established, shadow AI usage. For each entry, the inventory should record the purpose, the business owner, the technical owner, the provider and model where relevant, the data categories processed, the users or populations affected, the tools and permissions granted, the deployment location and data residency, the risk classification, and the date of the last review.

The inventory is not a one-time project. It needs feeding mechanisms: procurement review that asks whether a product includes AI functionality, architecture review of new integrations, telemetry on AI service usage, periodic attestation by business units, and a low-friction way for staff to register a new use. Integrating it with the existing application inventory, with AI-specific attributes added, avoids parallel registers that drift apart and links AI systems to the risk register and the ISMS scope.

### Classification by risk

The classification scheme should reflect criticality and exposure so that controls and oversight can be proportionate. A small number of tiers is usually enough. High-risk categories typically include systems that process sensitive personal information; systems that make or materially influence decisions about individuals, such as hiring, credit, admissions, or eligibility; systems embedded in financial, safety-related, or operational technology workflows; systems with broad access to internal data; customer-facing systems whose outputs could mislead; and systems that can trigger actions without human review. Low-risk categories might include internal drafting assistance with no sensitive data and no tool access.

Classification should consider both internal risk criteria and external regimes. The EU AI Act prohibits some uses, designates others high-risk, and attaches transparency duties to still others; OSFI's Guideline E-23 on model risk management, finalized in 2025 and taking effect in May 2027, expects federally regulated financial institutions to inventory and risk-rate models, explicitly including AI and machine learning. Aligning internal tiers with these categories avoids duplicate assessments.

### Accountability and decision rights

Accountability must be explicit. Each AI system should have a **business owner** accountable for its intended use, its benefits, and the acceptance of residual risk; a **technical owner** accountable for engineering, operation, and change control; and defined roles for security, privacy, legal, compliance, and risk management. Where an organization has a model risk management function, as is common in financial services, AI governance should be integrated with that discipline rather than treated as an isolated innovation project.

In the three lines model, first-line teams own AI risks and operate controls; second-line risk, privacy, and security functions set policy, challenge assessments, and monitor key risk indicators; and internal audit provides independent assurance. Many organizations add an AI governance committee to approve high-risk uses and maintain policy. It is useful only if its decision rights are clear and its decisions timely; a committee that meets quarterly and delays every request will simply accelerate shadow AI.

Every AI system in a medium or high tier should have a documented purpose, boundaries of use, approved data sources and users, permitted tools and actions, output constraints, oversight arrangements, escalation pathways, and monitoring requirements. This is not bureaucracy; it is the minimum structure for defensible management of systems that can influence decisions and leak sensitive data at scale.

### Human oversight as a designed control

**Human oversight** is often invoked as the answer to AI risk, but it is effective only when it is designed, resourced, and tested like any other control. Three broad patterns exist. In a *human-in-the-loop* design, a person must approve each consequential output or action before it takes effect. In a *human-on-the-loop* design, the system acts autonomously while a person monitors and can intervene. In a *human-in-command* design, people set the goals, boundaries, and conditions under which the system may operate and retain the authority to suspend it. The appropriate pattern depends on the consequence of error, the reversibility of actions, and the speed at which decisions must be made.

Oversight fails in predictable ways. Reviewers who approve hundreds of AI recommendations a day without the time, information, or authority to disagree provide the appearance of control without its substance—a form of compliance theatre. Managers should monitor approval rates, time per review, and override frequency, and periodically test oversight by inserting known errors to see whether they are caught. The EU AI Act's human oversight requirements for high-risk systems, and the opportunity Québec's Law 25 gives individuals to submit observations to a person who can review a decision based exclusively on automated processing, both assume that oversight is real.

## Shadow AI and acceptable use

**Shadow AI** is the use of AI tools and services without the organization's knowledge, approval, or controls. It is the AI-era version of shadow IT, but it spreads faster because the tools are free or cheap, require no installation, and offer immediate productivity gains. Employees paste customer emails into public chatbots to draft replies, upload spreadsheets to analysis tools, install browser extensions that read every page they visit, or connect personal AI assistants to corporate calendars and mailboxes. Most do so with good intentions. The resulting exposures—confidential data processed under consumer terms, personal information transferred outside Canada without assessment, intellectual property used to train third-party models, and unvetted connectors holding corporate tokens—are nonetheless real.

### Why shadow AI happens

Shadow AI is primarily a signal of unmet demand. People turn to unsanctioned tools when sanctioned ones do not exist, are harder to use, or are slow to approve. Prohibition alone tends to drive usage further out of sight, onto personal devices and accounts where the organization has no visibility at all. Effective governance therefore combines three elements: a clear acceptable-use policy, sanctioned tools that meet real needs, and monitoring that detects significant deviations without treating every employee as a suspect.

Discovery is the starting point. Web gateway and cloud access security broker logs, identity sign-in records, browser extension inventories, and expense reports reveal which AI services are in use, and a short, non-punitive survey surfaces what telemetry misses. The aim is not to punish but to understand demand and decide which tools to sanction, replace, or block.

### Designing an acceptable-use policy

An **AI acceptable-use policy** translates the organization's risk appetite into rules that employees can follow. To be effective, it should be short, concrete, and tied to the organization's data classification scheme (Chapter 16). A useful policy typically addresses the following points.

- **Approved tools and accounts.** The policy names the AI tools that are sanctioned, distinguishes enterprise accounts (with contractual protections such as no training on customer data, defined retention, and data residency options) from consumer accounts, and states that work data may be used only with approved tools and accounts.

- **Data rules by classification.** The policy states which classes of information may be entered into which tools: public information broadly, internal information only in enterprise tools, and confidential, personal, or regulated information only in tools specifically approved for that class, if at all.

- **Verification and accountability for outputs.** Users remain responsible for the accuracy, appropriateness, and originality of any work product they create with AI assistance, and must verify facts, citations, code, and calculations before relying on them.

- **Prohibited uses.** The policy lists uses that are not permitted regardless of tool, such as making decisions about individuals without required review, generating deceptive content, attempting to circumvent safeguards, or connecting agents to corporate systems without approval.

- **Disclosure.** The policy sets expectations for disclosing AI assistance to customers, students, colleagues, or regulators where it matters, including transparency obligations that apply under law.

- **Agents, extensions, and connectors.** The policy makes clear that installing AI browser extensions, connecting AI tools to corporate accounts, or deploying agents requires approval, because these grant ongoing access rather than one-time use.

- **Reporting and help.** The policy explains how to request a new tool, how to report a suspected AI-related incident such as accidental disclosure, and whom to ask when in doubt.

### Enablement rather than prohibition

The policy is only as effective as the alternatives the organization provides. **The most effective control against shadow AI is a sanctioned option that is genuinely useful**, backed by contractual protections, integrated with corporate identity, and configured with appropriate logging and data boundaries. Bans without alternatives usually drive usage underground.

Enforcement should be proportionate: blocking high-risk services, data loss prevention on uploads, and extension controls for high-risk data flows; education and nudges for the rest. Exceptions should follow the normal exception process, with a business justification, compensating controls, an accountable approver, and an expiry date.

## The AI supply chain: models, data, and components

Few organizations build foundation models. Most consume them—through commercial APIs, cloud platforms, embedded software features, or open-weight models downloaded from public repositories—and then add their own data, prompts, retrieval layers, fine-tuning, and integrations. The AI supply chain therefore includes model providers, hosting platforms, dataset sources, open-source libraries and frameworks, model hubs, connector and plug-in authors, and the subcontractors behind each of them. It is a specific and fast-moving instance of the third-party and supply-chain governance discussed in Chapter 19, with several distinctive features.

### What is different about AI components

AI components are harder to inspect than conventional software. A model's weights do not reveal what data it was trained on, whether it contains a backdoor that activates on a particular trigger, or how it will behave on unfamiliar inputs. Some serialization formats used to distribute models allow arbitrary code to run when the file is loaded, so downloading a model from an untrusted source can be equivalent to running an untrusted program. Datasets can be poisoned at the source, and providers update hosted models frequently, sometimes without notice.

These features mean that **provenance and change control matter more than inspection**. Managers should require that models come from approved sources, that their integrity is verified (for example, through cryptographic signatures or checksums), that safer serialization formats are used where available, that the specific model version in production is pinned and recorded, and that provider updates trigger re-evaluation proportionate to the system's risk tier.

### Model cards, data documentation, and the AI bill of materials

Transparency artifacts help make the supply chain governable. A **model card** is a structured document, published by a model developer, that describes a model's intended uses, limitations, training data at a high level, evaluation results, and known risks. Comparable documentation for datasets—often called datasheets or data cards—describes how data was collected, what it contains, and what restrictions apply. These documents vary widely in quality, but their presence, completeness, and candour are themselves useful due-diligence signals.

An **AI bill of materials (AI-BOM)** extends the software bill of materials concept to AI systems, listing the models, datasets, libraries, prompts, connectors, and services that make up a system, with versions and sources. CycloneDX includes machine-learning BOM capabilities, and SPDX 3.0 introduced AI and dataset profiles. For managers, the point is not the format but the capability: when a flaw is disclosed in a popular model-serving library, or a provider reports a compromise, the organization should be able to determine within hours which of its systems are affected.

### Contracting with AI providers

Contracts are the primary control over AI providers, and standard terms are often inadequate for enterprise use. Due diligence should ask whether the provider will use the organization's inputs, outputs, or fine-tuning data to train its models; how long prompts, outputs, and logs are retained, where, and by which subprocessors; which certifications or attestations (ISO/IEC 27001, ISO/IEC 42001, SOC 2) the provider holds and with what scope; how it will notify incidents, material model changes, and subprocessor changes; and what rights the organization has to evidence, audit, and exit with its data.

These questions belong in the existing third-party risk process (Chapter 19), not in a separate AI questionnaire. They also connect to concentration risk: if many critical processes depend on a single model provider, an outage, pricing change, or regulatory action affecting that provider becomes a business continuity scenario (Chapter 12).

## Secure AI lifecycle: controls from design to decommissioning

AI security becomes manageable when it is governed across the system lifecycle. The joint *Guidelines for Secure AI System Development* (2023), led by the UK National Cyber Security Centre and the US Cybersecurity and Infrastructure Security Agency and co-sealed by the CCCS and many other agencies, organize their recommendations around secure design, development, deployment, and operation and maintenance. ISO/IEC 42001:2023 and the NIST AI Risk Management Framework similarly expect controls to be applied at each stage. The figure below presents the lifecycle as a continuous loop, because changes to models, data, prompts, and tools send the system back through earlier stages.

<span id="_Toc243635576" class="anchor"></span>Figure 77: The secure AI lifecycle

### Plan and assess

During concept and design, the organization should define the purpose, expected benefits, and boundaries of the system, classify it in the AI inventory, and perform a risk assessment proportionate to its tier. The assessment should use the organization's standard scenario method (Chapters 5 and 6), enriched with AI-specific threat events drawn from the taxonomies described earlier. For systems that affect individuals, an **AI impact assessment** considers effects on people and society—fairness, accessibility, autonomy, and the consequences of errors—beyond the organization's own risk. ISO/IEC 42005:2025 provides guidance for conducting such assessments.

For Québec and Canadian organizations, this is also where privacy impact assessment becomes operationally relevant. Law 25 requires a privacy impact assessment for any project to acquire, develop, or overhaul an information system or electronic service delivery system involving personal information, and before communicating personal information outside Québec. Many AI projects meet one or both triggers. At the same stage, cybersecurity requirements should be specified: identity and access controls, segmentation, logging, abuse monitoring, data boundaries, constraints on tool integration, and the human oversight model.

### Govern the data

During data preparation, the organization must govern provenance, minimization, sensitive data handling, labelling quality, and retention. Training data can embed privacy exposure and bias; data indexed for retrieval becomes a leakage vector if access controls are not carried through. The joint *AI Data Security* guidance published in May 2025 by the NSA, CISA, the FBI, and Australian, New Zealand, and UK partners emphasizes data provenance tracking, integrity verification, protection of data in storage and transit, and monitoring for data drift and poisoning throughout the lifecycle.

A frequent and avoidable error is to index an entire repository for a RAG assistant on the assumption that existing permissions are correct. They rarely are: shared drives accumulate overly broad access over years, and an assistant that can search everything a user can technically reach will surface documents the user never knew existed. Managers should require a permissions clean-up of the content in scope, or restrict retrieval to curated collections.

### Build and evaluate

During training, fine-tuning, and integration, the organization should control access to development environments, protect datasets and model artifacts, apply secure development practices to the surrounding application code, and manage prompts and configuration as code under version control. Evaluation must include not only accuracy and quality metrics but also security and safety testing appropriate to the system's tier: attempts to induce leakage of sensitive data or hidden instructions, prompt injection through every channel the system reads, attempts to misuse tools or exceed permissions, generation of harmful or policy-violating outputs, and abuse of cost and rate limits.

**AI red teaming**—structured adversarial testing by people who try to make the system fail—is now an expected practice for high-risk and customer-facing generative AI systems. It differs from traditional penetration testing (Chapter 8) because the system's behaviour is probabilistic and the attack surface includes natural language, so testing must be repeated across many variations and after significant changes. Automated testing tools can provide coverage at scale, while human testers find creative failure modes that tools miss. Results, residual issues, and the decision to proceed should be documented and approved by the business owner, because **launching a system with known, unremediated weaknesses is a risk acceptance decision that someone must own**.

### Deploy with guardrails

During deployment, controls must address API security, authentication and authorization, secrets management, segmentation, rate limiting, logging, input and output filtering, and the human oversight arrangements defined at design. For generative systems, prompt and context management is central. System prompts and templates should be treated as sensitive configuration items with change control and access restrictions, because they define the system's behaviour—but they should never contain credentials or information whose disclosure would be harmful, because determined users can often extract them. Tools and connectors should be enabled only from an approved list, with the minimum permissions needed. A staged release—a pilot with limited users, conservative permissions, and enhanced monitoring, against go-live criteria agreed in advance—lets the organization observe real behaviour before broadening access.

### Operate, monitor, and change

During operations, monitoring must cover behaviour as well as uptime and security: abnormal access patterns, unusual token usage, repeated policy violations, anomalous tool calls, filter triggers, user reports of harmful answers, and signs of prompt injection. These signals should feed the security operations centre and the key risk indicators reported to the system's owner.

AI systems also require disciplined change management, because changes to the model version, system prompt, retrieval corpus, or tools can alter behaviour materially—and hosted providers may make such changes on their own schedule. A governance-ready organization can explain what changed, why it changed, how it was tested, and who approved it. Changes that alter the system's risk profile—new data categories, new tools, new user populations, autonomous rather than assisted operation—should send it back to the assessment stage.

### Retire safely

During decommissioning, the organization must ensure proper deletion and revocation: removing access paths, retiring API keys and connector tokens, deleting or archiving models, embeddings, prompts, and logs according to retention rules, and confirming that vendors no longer retain data where contractual or legal obligations require deletion. Law 25 requires that personal information be destroyed or anonymized once the purposes for which it was collected are achieved, subject to retention obligations, and derived artifacts such as embeddings and fine-tuned models can still contain or reveal personal information. Retirement should be recorded in the AI inventory so that the register reflects reality.

## Control patterns that matter in practice

Across the lifecycle, several control patterns have emerged as consistently important. None of them is exotic; most are established security disciplines applied to new components. Their consistent application, and the evidence that they operate, distinguish governed AI from AI experiments that drifted into production.

### Identity, permissions, and data boundaries

Identity and access management is foundational. AI systems should not be accessible through shared keys or uncontrolled endpoints. Users should authenticate through the corporate identity provider, AI services should use managed, rotated credentials, and development, test, and production environments should be separated so that data and model artifacts do not become broadly accessible. Agents and connectors should be governed as non-human identities, as discussed earlier.

Data boundary controls are equally central. For RAG systems, the retrieval layer determines what the model can see. If retrieval can access confidential repositories without enforcing the requesting user's permissions, the system becomes a data exfiltration channel available to anyone who can type a question. **The essential pattern is permission-aware retrieval: the model sees only content the requesting user is already authorized to access**, enforced at the retrieval layer rather than through instructions in the prompt. Prompts that tell a model "do not reveal confidential information" are guidance, not access control.

### Untrusted content and output handling

Prompt injection requires dedicated handling. The governance framing is simple: treat any content the model reads that does not come from a trusted, controlled source as untrusted input, even when it looks like ordinary documentation or email. Mitigations include separating trusted instructions from untrusted data, filtering retrieved content, and constraining which tools can be invoked after untrusted content has been processed. None is complete on its own, which is why the architectural limits on agency described earlier, including auditable approval for sensitive actions, remain essential.

Output handling deserves equal attention. Outputs passed to other systems—rendered in a web page, executed as code, inserted into a query, or used as tool parameters—must be validated and encoded like any untrusted input, because whoever influences the model influences the output. This is the AI-era version of the injection flaws application security has addressed for decades.

### Logging designed as evidence

Logging must be designed as evidence. AI systems should log user identity, prompts and key context elements, retrieval sources, tool calls and their parameters, output summaries, filter and policy decisions, and approvals, with privacy-conscious handling of sensitive content. Logs should be tamper-protected and retained according to investigation needs and privacy rules; without them, the organization cannot say after an incident what the system saw, what it did, and on whose behalf.

Logging produces both security value and privacy risk, because prompts and outputs may contain exactly the sensitive information the organization is trying to protect. Retention, access to logs, and any secondary use must therefore be explicit and reflected in privacy notices and impact assessments.

### Cost, abuse, and resilience

Cost and abuse protections matter because generative AI introduces variable-cost computing. Rate limits, quotas per user and per application, budget alerts, anomaly detection for usage, and protections against automated scraping or model extraction attempts become part of security design. A misconfigured agent caught in a loop can generate a substantial bill overnight.

Resilience completes the picture. Because "turning off the AI" is one of the most important containment actions, the business must be able to tolerate it: for each AI-dependent process, managers should know whether a fallback model, a manual procedure, or a graceful degradation path exists.

## Privacy, compliance, and the cross-border reality

AI systems frequently trigger privacy issues because they depend on data at scale and because their outputs can unintentionally reveal personal information. For Québec and Canadian managers, the governance discipline developed in Chapter 16 applies directly. AI projects should be treated as privacy-relevant by default when they involve personal information, even if their intent is benign. At the same time, the regulatory landscape for AI specifically is uneven across jurisdictions and has been changing quickly, so managers must distinguish clearly between obligations that apply now, obligations that are scheduled, and proposals that may never take effect.

### Purpose, minimization, retention, and transfer

Several recurring privacy issues must be governed explicitly. Data used for training, fine-tuning, or retrieval must respect purpose limitation and minimization. If an organization repurposes customer communications, call transcripts, or employee records for model training without proper authority and transparency, it may create compliance exposure and lasting trust harm. Cross-border transfer is common when using external AI providers, and Law 25 requires an assessment before personal information is communicated outside Québec, including consideration of the legal framework in the destination jurisdiction. Managers must understand where data is processed, what is retained, and which provider policies apply.

Retention and deletion are often neglected. Prompts and logs may contain sensitive information; embeddings and vector stores contain derived representations that can still reveal it; and fine-tuned models may memorize fragments of their training data. Governance must define retention and enforce deletion, recognizing that removing information from a trained model is technically difficult and that the simplest protection is not to include unnecessary personal information in the first place.

### Québec Law 25 and automated decisions

Québec's private-sector privacy law, as amended by Law 25, contains one of the clearest provisions in Canada on automated decision-making. Since September 2023, section 12.1 has required an enterprise that uses personal information to render a decision based exclusively on automated processing to inform the person concerned no later than when it informs them of the decision. On request, the person must also be informed of the personal information used to render the decision, the reasons and the principal factors and parameters that led to it, and their right to have the personal information corrected. The person must be given the opportunity to submit observations to a member of the personnel who is in a position to review the decision.

For managers, section 12.1 means the AI inventory must record each system's decision role, the principal factors and parameters must be documentable (which may constrain model choice at design time), and a genuine review function must be staffed. Law 25 also requires that individuals be informed when technology is used that allows them to be identified, located, or profiled, and that such functions be deactivated by default, which is relevant to many AI-enabled analytics and personalization features.

### The Canadian federal landscape

At the federal level, Canada has no comprehensive AI statute in force. The proposed Artificial Intelligence and Data Act (AIDA), Part 3 of Bill C-27, died on the Order Paper when Parliament was prorogued in January 2025. Since then, the federal government has created a ministerial portfolio for artificial intelligence and digital innovation, consulted on a national AI strategy and on AI transparency, and introduced or signalled targeted legislation on specific harms and on privacy reform, but at the time of writing no general framework comparable to AIDA had been enacted. Organizations should monitor developments without assuming that AIDA's former structure will return.

In the meantime, existing law applies. PIPEDA governs private-sector personal information outside Québec and in federally regulated sectors, and Canada's privacy commissioners have issued joint principles for responsible generative AI. The federal Voluntary Code of Conduct on the Responsible Development and Management of Advanced Generative AI Systems (2023) sets expectations for its signatories. OSFI's Guideline E-23 brings AI and machine learning models within model risk management expectations for federally regulated financial institutions from May 2027, alongside Guideline B-13 on technology and cyber risk, and federal institutions are subject to the Treasury Board Directive on Automated Decision-Making. Human rights, consumer protection, and employment law apply to AI-assisted decisions in any case.

### The EU AI Act and the international picture

The EU Artificial Intelligence Act (Regulation (EU) 2024/1689) is the most comprehensive AI law in force and applies to providers and deployers outside the EU when their AI systems are placed on the EU market or their outputs are used in the EU. It entered into force in August 2024 and applies in stages. Prohibitions on certain AI practices and the AI literacy provisions have applied since February 2, 2025. Obligations for providers of general-purpose AI models, the governance structure, and penalties have applied since August 2, 2025. Most remaining provisions, including transparency obligations for certain AI systems such as chatbots and generated content, have applied since August 2, 2026.

The schedule for high-risk systems has changed. A targeted amendment adopted in 2026 as part of the EU's digital simplification agenda—the Digital Omnibus on AI (often called the AI Omnibus), Regulation (EU) 2026/1744, published in the Official Journal on July 24, 2026—postponed the main obligations for stand-alone high-risk systems listed in Annex III (such as employment, education, and creditworthiness uses) to December 2, 2027, and for high-risk AI embedded in products covered by EU product legislation (Annex I) to August 2, 2028. It also softened the AI literacy provision, which now requires providers and deployers to take measures supporting AI literacy without prescribing a specific level for any individual, set December 2, 2026 as a transitional date for content-marking obligations on systems already on the market, and added new prohibitions, applicable from December 2, 2026, on AI systems that generate non-consensual intimate imagery of identifiable people or child sexual abuse material. Managers should verify these dates before relying on them, and should not treat postponement as a reason to defer the inventory, classification, and documentation work any credible program requires.

Other jurisdictions are taking different paths, from sector-specific rules to voluntary frameworks. The practical approach is one internal AI governance system that can be mapped to multiple regimes, rather than a separate compliance project for each.

| Regulation of AI is uneven and still moving, but the core evidence every regime expects is the same: **know your AI systems, classify their risk, document their purpose and controls, keep humans able to intervene, and be ready to explain decisions**. |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## Assurance frameworks: integrating AI risk management into cyber governance

Organizations benefit from structured frameworks that prevent them from reinventing governance. Several standards and frameworks are particularly relevant, and they should be integrated with the existing ISMS, privacy management program, enterprise risk management, and secure development practices rather than operated as parallel systems. The managerial challenge, as with the control frameworks of Chapter 9, is to build a coherent architecture from multiple references.

### The ISO/IEC 42001 family

**ISO/IEC 42001:2023** specifies requirements for an **AI management system (AIMS)**. It follows the same harmonized management-system structure as ISO/IEC 27001—context, leadership, planning, support, operation, performance evaluation, and improvement—so organizations with an ISMS can extend existing processes for policy, risk assessment, internal audit, management review, and corrective action. Its annex provides AI-specific controls covering topics such as AI policy, roles and responsibilities, resources, impact assessment, the AI system lifecycle, data for AI systems, information for interested parties, use of AI systems, and third-party relationships. It is certifiable, and ISO/IEC 42006:2025, published in July 2025, sets requirements for the bodies that audit and certify AI management systems, which strengthens the credibility of certification claims.

Supporting standards fill in method. **ISO/IEC 23894:2023** provides guidance on AI risk management, adapting the principles and process of ISO 31000 to AI. **ISO/IEC 42005:2025** provides guidance on AI system impact assessment. Together with ISO/IEC 27001:2022 and ISO/IEC 27701:2025 for privacy information management, these allow an organization to run an integrated management system that covers information security, privacy, and AI with shared governance, shared risk methods, and a single audit program.

### NIST: AI RMF, the Generative AI Profile, and cyber-specific work

The **NIST AI Risk Management Framework (AI RMF 1.0, NIST AI 100-1, January 2023)** organizes AI risk management into four functions: Govern, Map, Measure, and Manage. Govern establishes culture, policies, roles, and accountability; Map establishes context and identifies risks; Measure analyzes and tracks them; and Manage prioritizes and treats them. The framework is voluntary and outcome-based, and it frames trustworthiness through characteristics such as validity and reliability, safety, security and resilience, accountability and transparency, explainability, privacy enhancement, and fairness. **NIST AI 600-1, the Generative AI Profile (July 2024)**, extends the framework with twelve risks that are unique to or exacerbated by generative AI—including confabulation, information security, data privacy, information integrity, and value chain and component integration—and with suggested actions for each.

NIST has also been developing cybersecurity-specific guidance. The draft Cyber AI Profile (NIST IR 8596) maps AI considerations onto NIST CSF 2.0, so organizations can express AI security objectives in the structure they already use, and NIST has been drafting SP 800-53 control overlays for AI use cases such as generative assistants, predictive AI, and agent systems, which at the time of writing remained in development. Managers need not wait for final versions: the AI RMF, the Generative AI Profile, and CSF 2.0 already provide a usable structure.

### Government guidance and practitioner references

Government cybersecurity agencies have produced a series of practical guides. The Five Eyes series—*Guidelines for Secure AI System Development* (2023), *Deploying AI Systems Securely* (2024), and *Careful Adoption of Agentic AI Services* (2026), each co-sealed by the CCCS—provides lifecycle and deployment recommendations that map well to the controls in this chapter. The CCCS also publishes guidance for Canadian organizations on generative AI risks and safe use (ITSAP.00.041). Practitioner references such as the OWASP GenAI Security Project resources and MITRE ATLAS translate these principles into testable weaknesses and attack techniques.

The table below summarizes how these references can be combined. The purpose is not to adopt every document in full but to assign each a clear role in the governance architecture.

| **Reference**                     | **Primary role**       | **Typical use**                                     |
|-----------------------------------|------------------------|-----------------------------------------------------|
| ISO/IEC 42001:2023                | Management system      | Certifiable AI governance integrated with the ISMS  |
| ISO/IEC 23894 and 42005           | Risk and impact method | AI risk assessment and impact assessment practice   |
| NIST AI RMF and AI 600-1          | Risk outcomes          | Structuring AI risk work and GenAI-specific actions |
| NIST CSF 2.0 and Cyber AI Profile | Cyber outcomes         | Reporting AI security within the cyber program      |
| OWASP, MITRE ATLAS, NIST AI 100-2 | Threat vocabulary      | Scenarios, testing, and supplier questions          |
| Five Eyes and CCCS guidance       | Practical controls     | Secure development, deployment, and agent adoption  |

### Building an integrated evidence model

AI assurance should be repeatable and auditable. The organization should be able to show its inventory, classification logic, assessment records, control designs, testing evidence, monitoring and incident processes, change records, and oversight arrangements. Internal audit should include AI governance in its plan, starting with design and progressing to operating effectiveness for the highest-risk systems.

Certification adds external credibility but does not substitute for effective control. An ISO/IEC 42001 certificate shows that a management system meets the standard within a defined scope; it does not show that every AI system is secure. Managers should ask what is in scope and what residual risks were accepted, as they would for ISO/IEC 27001 (Chapter 9). Otherwise AI governance becomes compliance theatre: policies and minutes that do not change how systems are built and run.

## Incident management for AI: from harmful outputs to model compromise

AI incidents can look different from traditional cyber incidents. Some are classic security incidents—credential compromise, data leakage, malware—in which the AI system becomes the path or the amplifier. Others are behavioural incidents: the system produces harmful or false outputs, reveals confidential information, treats individuals unfairly, or takes unintended actions. Governance requires an incident model that can handle both, integrated with the organization's existing incident response process (Chapter 12) and aligned with NIST SP 800-61 Rev. 3.

### Defining AI incident categories and triggers

A useful managerial stance is to define AI incident categories with clear triggers so that staff know what to report and responders know how to classify it. Typical categories include the following.

- **Sensitive information disclosure.** The system revealed personal, confidential, or regulated information to someone not authorized to see it, through outputs, retrieval, or logs.

- **Manipulation and injection.** Evidence that prompt injection or other manipulation caused the system to bypass controls, access data, or take unauthorized actions.

- **Unintended or unauthorized actions.** An agent or tool-connected system performed an action outside its intended scope, whether through manipulation, error, or misconfiguration.

- **Model or data integrity compromise.** Indicators of poisoning, backdoors, unauthorized model modification, or tampering with prompts, retrieval content, or configuration.

- **Harmful or materially incorrect outputs.** Sustained or high-impact outputs that are unsafe, discriminatory, defamatory, or misleading, particularly in customer-facing or decision-support uses.

- **Model theft or extraction.** Suspected theft of model weights, or systematic querying aimed at replicating the model or extracting training data.

- **Provider incidents.** Security incidents, outages, or unannounced material changes at an AI provider that affect confidentiality, integrity, or availability.

Each category should have a defined severity scale linked to the organization's impact criteria, and the incident process should specify when privacy, legal, communications, and the system's business owner must be engaged. Under Law 25 and PIPEDA, an AI-related disclosure of personal information is assessed for notification in the same way as any other confidentiality incident.

### Containment: kill switches and rollback

Response procedures must include kill switches and rollback strategies. For many AI systems, the fastest way to reduce harm is to disable a feature, cut tool access, restrict retrieval scope, revoke connector tokens, revert to a known-safe model version or prompt, or switch to a manual process. These actions should be designed in advance, tested, and assigned to named roles with the authority to execute them quickly. Logs designed as evidence, as described earlier, are what allow responders to reconstruct what the system saw and did, determine whose data was exposed, and assess whether the incident is ongoing.

Technical response must be coordinated with legal, privacy, and communications functions when individuals, regulators, or customers may need notification. Behavioural incidents can become public quickly through shared screenshots, and where a third party provides the AI service, its incident notification and cooperation terms become critical.

### Learning from AI incidents

Post-incident review must address governance causes, not only technical ones. In AI contexts, causes often include poor data boundary controls, lack of permission-aware retrieval, excessive agency, uncontrolled logging and retention, insufficient adversarial testing, weak change management for prompts and models, unclear accountability for system behaviour, and oversight arrangements that existed on paper but not in practice. A governance-ready organization treats these as control failures to be remediated systematically, updates its scenario library and test cases, and shares lessons across AI systems that use similar patterns.

## Investment governance: deciding what to build, what to buy, and what to prohibit

AI often arrives through pressure to move fast. Competitors are announcing AI features, vendors are bundling them into renewals, and staff are asking for tools. Investment governance prevents speed from becoming negligence. It applies the economic discipline of Chapter 17 to two distinct decisions: whether AI will improve the organization's cybersecurity, and how much it will cost to deploy AI safely elsewhere in the business.

### The economics of AI for security

For AI used in cybersecurity operations, the economic test is the outcome test described earlier in this chapter, applied before purchase: a structured pilot on the organization's own data and incidents, against a baseline measured beforehand. Total cost should include licensing, usage-based charges, integration, tuning, training, and the overhead of governing the tool itself as an AI system with broad data access. A tool that moves work from one queue to another, or creates dependence on unreliable summaries, may increase risk despite an impressive demonstration.

### Pricing controls into AI projects

For AI systems deployed in business functions, governance should price security and privacy controls into the project from the start rather than treating them as afterthoughts. This includes funding for data clean-up and permission reviews before retrieval is enabled, logging architecture, evaluation and red teaming, monitoring, human oversight staffing, privacy and impact assessments, and ongoing change management. A business case that counts the productivity benefits of an AI assistant but omits these costs is incomplete. **A system that cannot be monitored, explained, and switched off is not production-ready, regardless of how well it performs in a demonstration.**

The build, buy, or configure decision also has security consequences. Building on open-weight models gives control over data location and model version but places supply-chain integrity, hosting security, and evaluation on the organization. Buying an enterprise service shifts that work to the provider but increases dependency. Enabling AI features in existing platforms is fastest but can bring broad default data access. None is inherently safer; each should be assessed against the use case and the organization's capabilities.

### Prohibiting until conditions are met

Finally, governance must include the discipline to prohibit certain uses until conditions are met. Not every organization should deploy autonomous agents that act across internal systems, not every dataset should be used for training, and not every external AI service should receive sensitive information. These are governance decisions grounded in risk appetite, not anti-innovation ideology. Stating them explicitly—for example, "no autonomous agents with write access to financial systems until permission-scoped connectors, approval gates, and logging have been implemented and tested"—gives teams a clear path to approval rather than an indefinite refusal, and turns a prohibition into a roadmap.

### Illustrative scenario: a university deploys a generative AI assistant

Consider a mid-sized Québec university with about 30,000 students and 4,000 employees. Faculty and staff are already using a variety of public chatbots, some students have built unofficial study tools that scrape course sites, and a vendor has offered to enable an AI assistant across the university's productivity suite at a discounted price. The senior administration asks the chief information officer and the chief information security officer to propose a plan for a university-wide generative AI assistant within one academic term, with a later phase in which the assistant could answer students' administrative questions and help staff process routine requests.

The first step is discovery and classification. A two-week review of gateway logs and sign-ins, plus a short survey, reveals more than forty AI services in regular use, several browser extensions with access to institutional email, and a research group that has connected an AI agent to a shared drive of interview transcripts with identifiable participants. An AI governance working group—chaired by the provost's office and including the CIO, the CISO, the person in charge of the protection of personal information under Law 25, legal counsel, research ethics, and student and faculty representatives—classifies the proposed uses into three tiers. Drafting and summarization with non-sensitive content is low risk. Use with student records, research data, or human resources files is high risk. Any assistant that answers students about admissions, grades, or financial aid, or that can take actions in the student information system, is high risk and potentially subject to Law 25's provisions on decisions based exclusively on automated processing.

The working group then makes a series of decisions, each with an owner. It approves the enterprise assistant for the low-risk tier, with institutional sign-in, contract terms excluding training on university data, and a privacy impact assessment covering transfers outside Québec. Before enabling retrieval over collaboration sites, it funds a permissions clean-up, after a pilot shows the assistant surfacing draft budgets and hiring files to any employee who asks. It publishes a two-page acceptable-use policy tied to the data classification scheme, blocks the riskiest browser extensions, and suspends the research group's agent pending ethics and security review. For the student-facing phase, it requires retrieval restricted to curated, published content; a recommend-only design that routes anything affecting a student's record to a staff member; logging with a defined retention period; red-team testing that includes indirect prompt injection through submitted documents; and clear notice that students are interacting with an AI system, with a route to a person.

The CISO adds AI-specific threat events to the university's scenario library, including executive impersonation using synthetic voice against the finance office and indirect prompt injection leading to disclosure of student information, and defines key risk indicators for the board's audit committee: the number of AI systems in the inventory by tier, the proportion of high-tier systems with completed assessments and tests, sensitive-data policy violations detected, AI-related incidents, and agent permissions reviewed on schedule. Six months later, the university can show its board what AI is in use, who owns each system, what risks were accepted and by whom, and what evidence supports those decisions. It has not eliminated risk, but it has turned an uncontrolled situation into a governed one, and staff and students have access to useful tools under conditions the institution can defend.

### Questions boards and senior managers should ask

Boards and executive committees do not need to understand model architectures, but they should be able to test whether AI is governed. The following questions are a practical starting point.

- **Inventory.** Do we have a complete, current inventory of the AI systems, embedded AI features, and agents we use, including those adopted by business units, and how do we know it is complete?

- **Risk appetite.** Have we stated our risk appetite for AI, including uses we will not permit, uses that require senior approval, and conditions for autonomous action?

- **Accountability.** Does every high-risk AI system have a named business owner who has formally accepted its residual risk, and are AI risks reported through our normal risk governance?

- **Data boundaries.** Can our AI systems see only the information their users are authorized to see, and have we reviewed the permissions on repositories connected to them?

- **Agency.** Which AI systems can take actions in our systems or send information externally, with what permissions, and with what human approval for consequential actions?

- **Shadow AI.** How much unsanctioned AI use do we detect, what sanctioned alternatives have we provided, and is our acceptable-use policy understood and followed?

- **Supply chain.** What contractual protections do we have with AI providers on data use, retention, location, incident notification, and model changes, and how concentrated is our dependency?

- **Testing.** What adversarial testing have our customer-facing and high-risk AI systems undergone, what issues remain open, and who accepted them?

- **Oversight.** Where we rely on human review of AI outputs, is that review real—adequately resourced, informed, and tested—or a formality?

- **Regulation.** Which of our AI uses fall under Law 25's automated-decision provisions, OSFI expectations, the EU AI Act, or other regimes, and are we prepared to meet current and scheduled obligations?

- **Incidents.** Could we detect, contain, and explain an AI incident—including switching off an AI capability without halting the business—and have we rehearsed it?

- **Value.** What measurable improvements have our AI security investments delivered, and what does it cost to operate our business AI systems safely?

## Bringing it together: AI as a governed capability, not a convenience

AI is now a permanent feature of organizational life and of the threat environment. The governance question is not whether to use it, but how to use it without degrading security, privacy, reliability, and trust—and how to defend against adversaries who are using it too.

This chapter has presented two intertwined truths. First, AI can improve cybersecurity operations when used as a disciplined accelerator for competent teams, with measurable outcomes and human oversight that is real rather than nominal, while attackers' use of AI rewards structural controls over defences that depend on people never being fooled. Second, AI systems introduce new assets, attack surfaces, and failure modes that require lifecycle governance: inventory and classification, clear accountability, acceptable-use rules that channel demand toward sanctioned tools, supply-chain provenance, permission-aware data boundaries, adversarial testing, logging designed as evidence, change management, and incident readiness for both technical compromise and behavioural harm.

Agentic AI sharpens every one of these requirements, because an error or a manipulated instruction becomes an operational event rather than a misleading paragraph. Treating agents as non-human identities is the most practical way to bring them within existing governance, and it illustrates a broader point: most AI governance is established governance applied consistently to new components, not a separate discipline.

Standards and regulation are still moving, but together they already provide more than enough structure for a defensible program. What its many references share is a demand for evidence: that the organization knows its AI systems, understands their risks, has made explicit decisions about them, and can show that its controls operate. Organizations that integrate AI into their existing ISMS, privacy program, risk management, and three-lines assurance will adapt as individual requirements change; those that run AI governance as a separate initiative, or not at all, will find each change disruptive.

In the broader arc of this book, the correct organizational mindset is to treat AI as a critical system whenever it touches sensitive data, influences decisions about people, or triggers actions. When AI is governed with the same seriousness as identity systems, payment systems, and operational continuity capabilities, organizations can benefit from its speed and scale without turning innovation into uncontrolled risk. The next chapter extends one of the themes introduced here—dependence on providers, models, and components the organization does not control—into a full treatment of third-party and supply-chain cybersecurity governance.

## Chapter summary

Chapter 18 argues that AI is a dual force in cybersecurity: it strengthens defenders and adversaries, and AI systems themselves have become high-value, high-risk assets. For managers, the shift is about decision rights, accountability, risk appetite, and evidence rather than machine-learning mathematics. The chapter integrates two perspectives—AI for cybersecurity and the cybersecurity of AI—and covers attacker economics, AI threat categories, agentic AI, inventory and accountability, shadow AI, the AI supply chain, the secure AI lifecycle, regulation in late 2026, assurance frameworks, incident management, and investment decisions. Its guiding theme is that AI must be governed as a critical system whenever it touches sensitive data, influences decisions, or triggers actions.

### Key points to retain

- AI for cybersecurity delivers value when it assists people with repetitive, data-heavy, time-sensitive tasks while accountable humans keep consequential decisions.

- Hallucination, data drift, adversarial manipulation, and automation bias mean AI security tools must be measured by outcomes such as time to detect, time to respond, and override rates.

- AI accelerates existing threats by lowering their cost and skill requirements, so the right response is structural controls rather than a search for an AI countermeasure.

- An AI system is an ecosystem that adds three kinds of asset: the model, the context it receives at run time, and the capabilities it has been granted.

- Indirect prompt injection is a structural weakness of language models that must be contained by design, not a bug a vendor will soon patch away.

- AI agents combine model unpredictability with privileged permissions, so excessive agency must be limited by architecture, approval gates, hard limits, and a reliable off switch.

- Every agent should be governed as a non-human identity with an owner, a purpose, unique revocable credentials, attributable logs, and access reviews.

- An AI inventory, risk-based classification, explicit accountability, and designed human oversight are the managerial foundations of AI governance.

- Shadow AI signals unmet demand; acceptable-use rules tied to data classification work only when sanctioned tools genuinely meet people's needs.

- AI supply-chain governance relies on provenance, integrity verification, version pinning, model cards, AI bills of materials, and contracts covering data use, retention, and change notice.

- Law 25 section 12.1, OSFI Guideline E-23, and the EU AI Act as amended by the 2026 AI Omnibus require inventory, classification, documentation, and real human review.

- AI incident response must cover behavioural as well as technical incidents, with pre-tested kill switches and rollback, and investment cases must price security and privacy controls from the start.

### Key terms

- **AI for cybersecurity** — The responsible use of AI to improve detection, response, and resilience, and the understanding of how adversaries use the same capabilities.

- **Cybersecurity of AI** — The governance, security, monitoring, and assurance of AI systems and AI-enabled business processes, including assistants and agents.

- **Hallucination (confabulation)** — The production by a generative model of plausible but incorrect content, which NIST terms confabulation.

- **Automation bias** — The tendency to over-trust automated outputs, especially under time pressure, which can let AI errors pass unchallenged.

- **Retrieval-augmented generation (RAG)** — A design that connects a language model to an organization's document repositories so it can answer questions grounded in internal content.

- **Prompt injection** — An attack in which text supplied by a user or embedded in processed content is interpreted as instructions, causing the model to ignore its constraints.

- **Indirect prompt injection** — Prompt injection through instructions hidden in emails, web pages, documents, or invitations that an AI assistant reads on a legitimate user's behalf.

- **AI agent** — A system in which a model plans and executes a sequence of steps toward a goal, calling tools and deciding next actions with limited human intervention.

- **Excessive agency** — OWASP's term for granting an AI system more functionality, permissions, or autonomy than its purpose requires.

- **Confused deputy** — A privileged program tricked into misusing its authority on behalf of a less privileged party, as when injection meets excessive agency.

- **Shadow AI** — The use of AI tools and services without the organization's knowledge, approval, or controls.

- **AI inventory** — The register of AI systems, embedded features, services, agents, and connectors the organization develops, procures, or permits, with owners and risk tiers.

- **AI bill of materials (AI-BOM)** — A list of the models, datasets, libraries, prompts, connectors, and services that make up an AI system, with versions and sources.

- **Permission-aware retrieval** — Retrieval that shows the model only content the requesting user is already authorized to access, enforced at the retrieval layer rather than in the prompt.

- **AI red teaming** — Structured adversarial testing by people who try to make an AI system fail, repeated across many variations and after significant changes.

## Review questions

### Essay questions

1.  Using the university scenario, evaluate the AI governance working group's decisions. Which decisions most reduced risk, which risks remained, and what evidence could the university show its board six months later?

2.  An AI agent with broad permissions reads an email containing hidden instructions and sends confidential files externally. Analyze this incident using the concepts of indirect prompt injection, excessive agency, and the confused deputy, and recommend the architectural and governance controls that would have contained it.

3.  Choose an organization you know. Design its approach to shadow AI, including discovery, an acceptable-use policy tied to data classification, sanctioned alternatives, and proportionate enforcement, and justify why prohibition alone would fail.

4.  Evaluate the claim that AI is an accelerant of existing threats rather than a source of entirely new ones. What does this imply for how managers should update risk scenarios and invest in controls?

5.  A vendor offers to enable an AI assistant across your organization's productivity suite at a discount. Recommend how management should decide whether to build, buy, configure, or prohibit, and which costs and controls the business case must include.

### Multiple-choice questions

Choose the best answer. Answers are listed in the appendix "Answers to multiple-choice questions".

**1.** According to the chapter, what is the most important change AI brings to the threat landscape?

> A. It creates entirely new classes of attack that existing controls cannot address
>
> B. It makes attacks easier to detect because AI-generated content is uniform
>
> C. It mainly benefits nation-states, which already had advanced capabilities
>
> D. It reduces the cost and skill required for attacks, so their volume and polish rise

**2.** An AI assistant connected to company systems starts disclosing confidential data. What is often the fastest way to reduce harm, according to the chapter?

> A. Retrain the model on corrected data before restoring service
>
> B. Execute pre-tested actions such as disabling the feature or revoking connector tokens
>
> C. Ask the provider to patch the prompt injection vulnerability
>
> D. Add instructions to the system prompt forbidding further disclosure

**3.** Analysts in a security operations centre almost never override their AI assistant's recommendations. How does the chapter suggest interpreting this?

> A. It may indicate automation bias rather than perfect AI performance
>
> B. It proves the AI tool is accurate and needs no further evaluation
>
> C. It shows that analysts lack the authority to disagree with the tool
>
> D. It means the tool should be allowed to execute all actions automatically

**4.** A finance clerk receives a convincing deepfake video call from the CFO requesting an urgent transfer. Which control best reflects the chapter's guidance?

> A. Training staff to spot visual artifacts in synthetic video
>
> B. Deploying an AI tool that detects deepfakes in real time
>
> C. Out-of-band verification using contact details already on file
>
> D. Restricting video calls to internal participants only

**5.** Which situation is an example of indirect prompt injection?

> A. A calendar invitation contains hidden text instructing an assistant to forward emails
>
> B. A user types a request asking the chatbot to ignore its rules
>
> C. An attacker floods a model API with requests to raise costs
>
> D. A developer downloads a model file that runs code when loaded

**6.** A policy states: no autonomous agents with write access to financial systems until permission-scoped connectors, approval gates, and logging are implemented and tested. What is the main governance value of this wording?

> A. It permanently excludes agents from financial processes
>
> B. It transfers the decision to the AI governance committee
>
> C. It turns a prohibition into a roadmap with a clear path to approval
>
> D. It satisfies the EU AI Act's requirements for high-risk systems

**7.** Which of the following is one of the three kinds of asset that an AI system adds to the familiar ones?

> A. The user interface through which staff access the system
>
> B. The capabilities granted: tools it can call, data it can reach, actions it can take
>
> C. The cloud region in which the application servers are hosted
>
> D. The service-level agreement signed with the hosting provider

**8.** Which design best illustrates what OWASP calls excessive agency?

> A. A chatbot whose answers are filtered before display
>
> B. A coding agent limited to a sandbox with read-only repository access
>
> C. A support agent that drafts refunds for a person to approve
>
> D. An email-summarizing assistant that can also send and delete messages

**9.** Why does the chapter say that, when indirect prompt injection meets excessive agency, the attacker never needs the agent's credentials?

> A. Because agents store their credentials in plain-text system prompts
>
> B. Because connector protocols disable authentication for internal tools
>
> C. Because the agent is a confused deputy that acts on the attacker's text
>
> D. Because model providers share credentials across all customers

**10.** The business owner of a production AI agent leaves the organization. What does the chapter recommend?

> A. Suspend the agent until a new owner is assigned, as with an orphaned account
>
> B. Let the agent continue, since its permissions were already approved
>
> C. Transfer ownership automatically to the CISO for continuity
>
> D. Delete the agent and its logs to prevent misuse

**11.** How should an organization control third-party connectors, such as Model Context Protocol servers, that agents use to reach tools and data?

> A. Allow any connector listed in a public marketplace
>
> B. Rely on the model to refuse unsafe instructions in tool descriptions
>
> C. Grant broad long-lived tokens so integrations do not fail
>
> D. Treat them as privileged integrations: allow-listed, minimally scoped, short-lived credentials, logged

**12.** What does the chapter identify as the primary cause of shadow AI?

> A. Employees deliberately trying to circumvent security controls
>
> B. Unmet demand, when sanctioned tools are missing, harder to use, or slow to approve
>
> C. Vendors enabling AI features without informing their customers
>
> D. Lack of technical controls to block consumer AI websites

**13.** In which human oversight pattern does an AI system act autonomously while a person monitors and can intervene?

> A. Human-in-the-loop
>
> B. Human-on-the-loop
>
> C. Human-in-command
>
> D. Human-after-the-fact

**14.** A RAG assistant's system prompt says do not reveal confidential information, but it can search every repository. What is the chapter's recommended control?

> A. Permission-aware retrieval enforced at the retrieval layer for each requesting user
>
> B. A longer system prompt listing every confidential document category
>
> C. Output filtering that blocks any answer mentioning confidential topics
>
> D. Training users not to ask questions about confidential matters

**15.** Why does the chapter say provenance and change control matter more than inspection for AI models?

> A. Because open-weight models are always safer than hosted services
>
> B. Because model cards are legally binding on providers
>
> C. Because providers are required to disclose all training data
>
> D. Because weights reveal little and some model files can run code when loaded

**16.** Under section 12.1 of Québec's private-sector law, what must an enterprise provide when it renders a decision based exclusively on automated processing?

> A. Prior approval from the CAI before the system is deployed
>
> B. A guarantee that the model contains no personal information
>
> C. Notice, on request the principal factors, and a chance to submit observations for review
>
> D. An annual third-party audit of the algorithm's fairness

**17.** Under the EU AI Act as amended by the 2026 AI Omnibus, when do the main obligations for stand-alone high-risk systems listed in Annex III apply?

> A. February 2, 2025
>
> B. August 2, 2026
>
> C. August 2, 2028
>
> D. December 2, 2027

**18.** What makes ISO/IEC 42001:2023 easier to adopt for an organization that already operates an ISO/IEC 27001 ISMS?

> A. It is a technical testing standard for model accuracy
>
> B. It replaces ISO/IEC 27001 controls for any system using AI
>
> C. It follows the same harmonized management-system structure
>
> D. It is a voluntary profile with no certification option

**19.** In the university scenario, a pilot showed the assistant surfacing draft budgets and hiring files to any employee. What did the working group decide?

> A. Fund a permissions clean-up before enabling retrieval over collaboration sites
>
> B. Cancel the enterprise assistant and block all AI services
>
> C. Accept the exposure, since employees already had technical access
>
> D. Move the assistant to a recommend-only mode for all users

**20.** Which list correctly names the four functions of the NIST AI Risk Management Framework (AI RMF 1.0)?

> A. Identify, Protect, Detect, Respond
>
> B. Govern, Map, Measure, Manage
>
> C. Defend, Secure, Thwart, Recover
>
> D. Plan, Build, Deploy, Retire
