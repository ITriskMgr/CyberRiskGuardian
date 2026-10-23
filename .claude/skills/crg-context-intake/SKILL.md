---
name: crg-context-intake
description: >
  This skill should be used when the user asks to "start a risk assessment", "build the organizational context", "identify crown
  jewels", "create an evidence register", "interview me about my organization for a cyber risk assessment", "map critical assets and
  dependencies", or "define the assessment scope", or when the cyber-risk-assessment workflow reaches steps 1 to 3.
metadata:
  version: "2.1.0"
  author: "Marc-André Léger (methodology) - packaged with Claude"
  edition: "Community Edition (Core plugin, free, CC BY-NC 4.0)"
---

# Context intake (CyberRiskGuardian steps 1–3)

Build the factual foundation of the assessment: what evidence exists, what the organization does, what it cannot afford to lose, and what the assessment covers. Everything later in the assessment traces back to this step.

## 1. Evidence register

Read every document supplied before writing. List each source as a row: ID (E1, E2...), title, type (documented, exercise or simulation evidence, SME input, external), and what it supports. For each important assertion later in the assessment, cite the evidence ID or label the assertion as an assumption.

- Exercise or simulation material (tabletop injects, sample logs) is indicative evidence to validate, never a confirmed incident.
- Instructor, template or benchmark values are calibration aids, not organizational evidence.

## 2. Organizational context

Cover, where evidence exists: mission and objectives; sector; products and services; customers or beneficiaries; critical business processes; information assets and their sensitivity; critical systems and applications; infrastructure and cloud; identity and remote access; OT, IoT or medical devices; third parties and suppliers; geography and jurisdictions; legal, regulatory and contractual obligations; availability and continuity requirements; existing controls (evidence-backed only); incident history and audit findings; transformation initiatives; cybersecurity maturity.

Where a topic has no evidence, write **Information gap — organizational validation required** rather than filling it in.

## 3. Crown jewels and dependencies

For each crown-jewel asset or service record: business purpose, owner, confidentiality / integrity / availability requirement, technical dependencies and third-party dependencies. Map Objective → Process → Primary asset → Supporting assets → Dependencies → Consequences.

Then analyse single points of failure and concentration: identity, DNS and network, cloud, virtualization, backup, telecom, privileged administration, SaaS, managed service providers, software supply chain, payment providers. Flag peripheral assets that could still give access to crown jewels.

## 4. Scope

State the assessment scope, the period (12 months by default), the risk perspectives considered (confidentiality, integrity, availability, privacy, operations, finance, compliance, reputation, safety, strategy), exclusions, assumptions and material information gaps.

## Interview mode

When the user has no documents, interview them in short rounds of **at most five questions** (never more, even to save time), in the user's language (French or English). Start with mission, critical services and data; then systems, identity and cloud; then suppliers and obligations; then incidents and existing controls. Record each answer as SME input (E-number). Never guess the answer to a question the user skipped.

## Output

An evidence register table, a context summary, the crown-jewel table and dependency map, the scope statement, and a list of gaps to validate. Hand off to `crg-risk-appetite` next.

## Language

Reply in the user's language. In French, write the markers as **Estimation analytique — validation requise**, **Lacune d'information — validation organisationnelle requise** and **Décision de la direction requise**.
