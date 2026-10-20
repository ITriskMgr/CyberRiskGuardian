<!-- Introduction to Cybersecurity Governance, 3rd ed., v2.1g (Aug 2026) - (c) Marc-André Léger. Licensed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/). Third-party standards, frameworks and quotations remain under their owners' terms. -->

# Chapter 19: AI for Cybersecurity and the Cybersecurity of AI

Artificial intelligence has become a dual-force in cybersecurity. It strengthens defenders by accelerating detection, triage, and automation, yet it also strengthens adversaries by lowering the cost of reconnaissance, social engineering, malware development, and exploitation. At the same time, AI systems themselves have become high-value assets and high-risk systems. This creates a governance reality that did not exist a decade ago: organizations must manage cybersecurity in an AI-enabled threat environment while also managing the cybersecurity risks of AI systems they deploy.

For managers, the governance shift is not primarily about the mathematics of machine learning. It is about decision rights, accountability, risk appetite, and evidence. AI introduces new failure modes that are not always captured by traditional security control catalogues. Some are technical (model theft, prompt injection, training data leakage), some are operational (uncontrolled autonomous actions by agentic systems), and some are legal and reputational (privacy violations, discrimination, misleading outputs, duty-of-care concerns). In addition, AI systems are often acquired as services—models, APIs, plug-ins, and managed platforms—meaning that the organization’s exposure expands through third-party dependencies.

This chapter integrates two perspectives. The first is AI for cybersecurity: how organizations can use AI responsibly to improve detection, response, and resilience. The second is cybersecurity of AI: how organizations govern, secure, monitor, and assure AI systems and AI-enabled business processes. The guiding theme is that AI must be treated as a governed capability and, in many cases, as a critical system—one that requires lifecycle controls from design to decommissioning, aligned with risk appetite and regulatory obligations.

## AI for cybersecurity: what it can do—and what it cannot do

AI can materially improve cybersecurity operations when applied to problems where scale, speed, and pattern recognition matter. In modern security operations centers, the most persistent constraint is not the absence of tools; it is the mismatch between alert volume and human capacity. AI can help by reducing noise, accelerating triage, and assisting analysts in producing coherent hypotheses about what is happening. Used properly, it can shorten time-to-detect and time-to-respond, which are often more important than marginal improvements in prevention.

Common high-value applications include alert enrichment and correlation across fragmented telemetry (endpoint, identity, cloud control plane, network, and application logs), anomaly detection in identity and access behaviour, natural-language summarization of incident timelines, malware triage assistance, automated extraction of indicators of compromise, and acceleration of threat hunting queries. Generative AI can also support governance work by helping security teams draft policies, risk registers, control descriptions, training content, and audit narratives, if confidentiality and accuracy are managed explicitly.

However, AI does not eliminate the need for security engineering judgement, and organizations should be cautious about treating AI as an oracle. AI systems can hallucinate plausible but incorrect explanations; they can be misled by poor data quality; and they can amplify bias in the underlying training examples. Moreover, attackers can deliberately manipulate what AI systems see—through adversarial inputs or data poisoning—so AI-driven detection should not be considered inherently more trustworthy than well-designed deterministic controls.

A mature governance stance is therefore to treat AI as a force multiplier for competent teams, not as a replacement for them. The objective is to improve operational outcomes—faster detection, fewer false positives, better containment discipline, stronger recoverability—while retaining clear human accountability for decisions that carry material risk.

## AI for attackers: why the threat landscape accelerates

The most important change AI brings to the threat landscape is economic. It reduces the cost and skill required to perform tasks that previously demanded expertise, time, or language fluency. This has several direct consequences.

First, social engineering becomes cheaper and more convincing. AI-assisted phishing, business email compromise narratives, and multilingual persuasion can be produced at scale. Deepfake voice and video can increase the credibility of fraud attempts against finance teams, executives, and customer service. The governance implication is that identity assurance and transaction controls become even more important: when persuasion improves, organizations must depend less on users noticing deception and more on strong authentication, verification steps for sensitive actions, and fraud controls.

Second, malware development and exploit adaptation can accelerate. AI can assist in code generation, obfuscation, creation of variations, and rapid refinement of malicious tooling. Even when AI does not produce novel sophisticated exploits, it compresses attacker iteration cycles and increases the volume of plausible attempts.

Third, reconnaissance and target selection become more systematic. Adversaries can automate collection and synthesis of public information, identifying organizational structure, vendors, exposed services, and likely weak points. This reinforces the importance of attack surface management, secure default configurations, and disciplined third-party risk management.

Finally, AI enables more effective living off the land behaviour by helping attackers blend into normal operations. When adversaries can quickly understand enterprise tooling, cloud platforms, and administrative workflows, they can move laterally with less obvious signatures. This increases the value of privileged access governance, rigorous logging of administrative actions, and continuous detection of identity misuse.

For governance, the conclusion is not that AI renders security impossible. It means that fragile controls—those that depend on user perfection, obscure detection rules, or untested recovery—will fail more often. The organization must deliberately invest in structural controls: identity, least privilege, segmentation, secure configuration baselines, and recoverability.

## Cybersecurity of AI: why AI systems require dedicated protection

AI systems introduce risks that straddle security, privacy, safety, compliance, and reputation. In traditional enterprise systems, the primary concern is usually confidentiality, integrity, and availability of data and services. In AI systems, those remain essential, but additional dimensions become operationally important: the integrity of training data, the reliability of model behaviour, the confidentiality of model internals, the control of outputs, and the governance of human reliance on the system.

An AI system is rarely just a model. It is typically an ecosystem consisting of data pipelines, feature stores or embeddings, training and evaluation environments, model registries, deployment infrastructure, APIs, user interfaces, tool integrations, and operational monitoring. In generative AI applications, the system may include retrieval-augmented generation (RAG), prompt templates, tool-calling agents, external plug-ins, and workflow automation that can trigger business actions. Every one of these components introduces potential attack paths.

A governance-driven organization therefore treats AI as a lifecycle-managed system. It maintains an inventory of AI use cases, classifies them by criticality, defines acceptable risk and constraints for each, and ensures the system is built and operated with evidence of due diligence comparable to other high-risk systems.

## Core threat categories for AI systems (including generative AI)

AI threats can be organized in a way that is useful for managers: threats to confidentiality, integrity, and availability still apply, but the mechanisms differ.

Confidentiality threats include leakage of sensitive information through prompts and outputs, exfiltration of proprietary data used in RAG pipelines, inference of training data through model inversion or membership inference techniques, and theft of the model itself or its weights via compromised repositories, insecure endpoints, or API abuse. In regulated contexts, confidentiality also includes privacy breaches: personal information appearing in outputs, being used improperly in training, or being transmitted to third-party model providers without appropriate governance.

Integrity threats include data poisoning (malicious manipulation of training or fine-tuning data), backdoored models in the supply chain, manipulation of prompts and context to force incorrect behaviour, and prompt injection attacks where untrusted content instructs the model to ignore constraints and reveal secrets or perform unsafe actions. For RAG systems, integrity threats often sit at the boundary between content and control: if untrusted documents can influence the model’s instructions, the system’s behaviour can be redirected.

Availability threats include denial-of-service on model APIs, cost exhaustion attacks (driving up token usage), rate-limit abuse, and operational failures created by dependency on external AI providers. In practice, AI availability is not only technical uptime; it is also availability of trustworthy behaviour. A model that is online but producing systematically wrong or unsafe outputs is operationally unavailable from a governance standpoint.

Finally, a distinct category exists for autonomy and action. When AI systems are connected to tools—sending emails, generating configuration changes, initiating transactions, querying internal systems—misbehaviour becomes an operational risk, not merely a content risk. The governance response is to treat tool-connected AI as a privileged actor that must be constrained, logged, and controlled through least privilege and robust approval gates.

## Governance foundations: inventory, classification, and accountability

A recurring governance failure in AI adoption is that AI systems proliferate without clear ownership. Security leaders discover them after deployment, privacy leaders discover them after a complaint, and operational leaders discover them after an incident. The first control is therefore managerial: maintain an AI system inventory and a classification scheme.

The inventory should include internal systems and externally consumed AI services, including shadow AI usage where employees rely on public tools. The classification scheme should reflect criticality and exposure. High-risk categories typically include systems that process sensitive personal information; systems that make or influence decisions affecting individuals; systems that interact with financial workflows; systems that operate with broad internal data access; and systems that can trigger actions without human review.

Accountability must be explicit. Each AI system should have a business owner accountable for intended use and risk acceptance, a technical owner accountable for engineering and operations, and defined roles for security, privacy, legal, and risk management. Where organizations have model risk management functions (common in financial services), AI governance should be integrated with that discipline, rather than treated as an isolated innovation project.

A practical governance posture is to require that every AI system has a documented purpose statement, boundaries of use, approved data sources, approved users, output constraints, escalation pathways, and monitoring requirements. This requirement is not bureaucracy; it is the minimum structure needed for defensible management of systems that can influence decisions and leak sensitive data at scale.

## Secure AI lifecycle: controls from design to decommissioning

AI security becomes manageable when it is governed across the system lifecycle.

During concept and design, the organization should perform a risk assessment proportionate to the use case. For Québec and Canadian organizations, this is where privacy impact thinking (PIA / DPIA) becomes operationally relevant: what personal information is used, what consent basis applies, what cross-border transfers occur, what retention rules apply, and what safeguards will be demonstrable. At the same stage, cybersecurity requirements should specify identity and access controls, segmentation, logging, abuse monitoring, and constraints for tool integration.

During data preparation, the organization must govern data provenance, data minimization, sensitive data handling, and retention and deletion. Data used in training or fine-tuning can embed privacy exposure and bias; data used in RAG can become a leakage vector. The discipline here is not only security; it is data governance.

During training and evaluation, the organization should control access to training environments, protect datasets and model artefacts, and document evaluation results. Evaluation must include not only accuracy metrics but also security and safety testing appropriate to the system. This includes attempts to trigger leakage, manipulation, unsafe outputs, and policy violations. For some systems, this resembles red teaming with structured adversarial testing.

During deployment, controls must address API security, authentication and authorization, secrets management, segmentation, rate limiting, logging, and output filtering. For generative systems, prompt and context management is central: templates and system prompts should be treated as sensitive configuration items with change control and access restrictions, because they define the system’s behaviour and can be used to extract secrets.

During operations, continuous monitoring is required. This includes traditional uptime and security monitoring, but also behaviour monitoring: abnormal access patterns, excessive token usage, repeated policy violations, anomalous tool calls, or signs of prompt injection. AI systems also require change management discipline, because model updates can change behaviour materially. A governance-ready organization can explain what changed, why it changed, and how it was tested.

During decommissioning, the organization must ensure proper deletion and revocation: removing access paths, retiring keys, deleting sensitive prompts and logs according to retention rules, and ensuring that vendor services no longer retain data where contractual or legal obligations require it.

## Control patterns that matter in practice

In operational terms, several control patterns have emerged as consistently important.

Identity and access management is foundational. AI systems should not be accessible through shared keys or uncontrolled endpoints. Role-based access, strong authentication, and clear separation between development and production environments reduce the likelihood that data and model artefacts become broadly accessible.

Data boundary controls are equally central. For RAG systems, the retrieval layer determines what the model can see. If retrieval can access confidential repositories without strong authorization checks, the system becomes a data exfiltration channel. A mature pattern is to ensure that retrieval results respect the user’s permissions, so the model can only see documents the user is already authorized to access.

Prompt injection and untrusted content require dedicated handling. The governance framing is simple: treat external or user-provided content as untrusted input, even if it is text that looks like documentation. Technical mitigations include instruction hierarchy discipline, sanitization or partitioning of content, strict tool-calling constraints, and refusal patterns for sensitive requests. Where the system has tool access, the safest pattern is to require explicit, auditable approval for sensitive actions, rather than allowing the model to execute them freely.

Logging must be designed as evidence. AI systems should log prompts, key context elements, tool calls, output summaries, user identity, and access decisions, with privacy-conscious handling of sensitive content. Logs should be protected against tampering and retained according to incident investigation needs and privacy retention rules. This is one of the most important governance intersections: logging produces both security value and privacy risk, so retention and access controls for logs must be explicit.

Cost and abuse protections matter, because AI introduces variable-cost computing. Rate limiting, quotas, anomaly detection for token usage, and protections against automated scraping or model extraction attempts become part of security design. In procurement terms, this is also a financial governance consideration.

Finally, supply chain assurance becomes critical. Many organizations consume third-party models, libraries, and hosted services. Governance requires a disciplined approach to vendor due diligence, contract clauses addressing data use, retention, breach notification, and subcontractors, and technical measures such as compartmentalization and minimal exposure.

## Privacy, compliance, and cross-border reality in AI deployments

AI systems frequently trigger privacy issues because they depend on data scale and because outputs can unintentionally reveal personal information. For Québec and Canadian managers, the governance discipline developed in privacy engineering chapters applies directly here. AI projects should be treated as privacy-relevant by default when they involve personal information, even if their intent is benign.

Several recurring issues must be governed explicitly.

Data used for training, fine-tuning, or analytics must respect purpose limitation and minimization. If an organization repurposes customer communications, call transcripts, or employee records for model training without proper governance, it may create compliance exposure and trust harm.

Cross-border transfer is common when using external AI providers. Managers must understand where data is processed, what data is retained, and what provider policies apply. Contractual terms, technical configurations, and privacy impact assessments must reflect this reality.

Retention and deletion are often neglected. AI logs and prompts may contain sensitive information; embeddings and vector stores may contain derived representations that still create confidentiality risk. Governance must define retention and enforce deletion rules, especially where obligations require destruction or anonymization once purposes are achieved.

Transparency and accountability also matter. For some uses, organizations must be prepared to explain how AI influences decisions and what safeguards exist to prevent harm. Even when the law does not demand algorithmic explanations, stakeholders frequently do.

The practical approach is to treat AI systems as part of the organization’s privacy management program: privacy impact assessments (including PIA in Québec contexts where appropriate), documented lawful basis for processing, clear vendor governance, and operational safeguards that can be demonstrated.

## Assurance frameworks: integrating AI risk management into cyber governance

Organizations benefit from using structured frameworks to avoid reinventing governance. Several standards and frameworks are particularly relevant as scaffolding, and they should be integrated with existing ISMS and enterprise risk management rather than treated as parallel systems.

NIST’s AI Risk Management Framework (AI RMF 1.0) provides a practical structure for governing, mapping, measuring, and managing AI risk. NIST AI 600-1, the Generative AI Profile, extends that framework with risks and actions specific to generative AI. ISO/IEC 42001:2023 provides a management-system approach to AI, ISO/IEC 23894:2023 provides AI risk-management guidance, and the OWASP Top 10 for LLM Applications 2025 provides an application-security perspective. These references should be integrated with the organization’s ISMS, privacy management, enterprise risk management, and secure-development practices rather than operated as parallel governance systems.

The governance point is that AI assurance should become repeatable and auditable. The organization should be able to show its AI inventory, its risk classification logic, its assessment records, its control designs, its testing evidence (including adversarial testing where appropriate), its monitoring and incident processes, and its change management discipline. Without this, AI innovation becomes operational and legal fragility.

## Incident management for AI: from harmful outputs to model compromise

AI incidents can look different from traditional cyber incidents. Some are classic security incidents—credential compromise, data leakage, malware—but the AI system becomes the path or amplifier. Others are behavioural incidents: the system produces harmful outputs, reveals confidential information, or takes unintended actions. Governance requires an incident model that can handle both.

A useful managerial stance is to define AI incident categories with clear triggers: suspected leakage of sensitive information; evidence of prompt injection causing unauthorized access; model or data poisoning indicators; abnormal tool-calling behaviour; sustained unsafe outputs; suspected model theft or extraction; and third-party provider incidents affecting confidentiality or availability.

Response procedures must include kill switches and rollback strategies. For many AI systems, the fastest way to reduce harm is to disable a feature, cut tool access, restrict retrieval scope, or revert to a known-safe configuration. Technical response must be coordinated with legal, privacy, and communications functions when affected individuals, regulators, or customers may need notification.

Post-incident review must include governance causes. In AI contexts, causes often include poor data boundary controls, lack of permission-aware retrieval, uncontrolled logging and retention, insufficient adversarial testing, weak change management, and unclear accountability for system behaviour. A governance-ready organization treats these as control failures that must be remediated systematically.

## Investment governance: deciding what to build, what to buy, and what to prohibit

AI often arrives through pressure to move fast. Investment governance prevents speed from becoming negligence.

For AI used in cybersecurity operations, the economic test is whether the tool improves measurable outcomes: reduced time-to-detect, reduced time-to-triage, reduced false positives, improved coverage of critical telemetry, faster containment, and better recoverability. If an AI SOC tool increases complexity and creates analyst dependence on unreliable summaries, it may increase risk despite impressive demonstrations.

For AI systems deployed in business functions, governance should price security and privacy controls into the project from the start, rather than treating them as afterthoughts. This includes funding for data governance, logging architecture, testing, red-teaming where appropriate, and ongoing monitoring. A system that cannot be monitored and controlled is not production-ready.

Finally, governance must include the discipline to prohibit certain uses until conditions are met. Not every organization should deploy tool-connected autonomous agents that can act across internal systems. Not every dataset should be used for training. Not every external AI service should be permitted to receive sensitive information. These are governance decisions grounded in risk appetite, not anti-innovation ideology.

## Bringing it together: AI as a governed capability, not a convenience

AI is now a permanent feature of organizational life. The governance question is not whether to use it, but how to use it without degrading security, privacy, reliability, and trust.

This chapter has presented two intertwined truths. AI can improve cybersecurity operations when used as a disciplined accelerator for competent teams, with measurable outcomes and strong oversight. At the same time, AI systems introduce new security and privacy exposures that require lifecycle governance: inventory and classification, secure design, permission-aware data boundaries, adversarial testing, rigorous logging and monitoring, change control, and incident readiness for both technical compromise and behavioural harm.

In the broader arc of cybersecurity governance, the correct organizational mindset is to treat AI as a critical system whenever it touches sensitive data, influences decisions, or triggers actions. When AI is governed with the same seriousness as identity systems, payment systems, and operational continuity capabilities, organizations can benefit from its speed and scale without turning innovation into uncontrolled risk.
