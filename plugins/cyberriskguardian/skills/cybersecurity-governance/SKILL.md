---
name: cybersecurity-governance
description: >
  This skill should be used when the user asks about cybersecurity governance, risk, compliance or resilience concepts as taught in
  "Introduction to Cybersecurity Governance" (Marc-André Léger) - e.g. "explain the CIA triad", "fraud triangle", "risk triangle",
  "three lines model", "risk appetite vs tolerance", "IPM risk cycle", "balanced cybersecurity scorecard", "SMART metrics",
  "compliance management system", "cybersecurity maturity", "investment governance", "third-party governance" - or when a risk
  assessment needs the book's definitions, frameworks or chapter guidance as the reference text.
metadata:
  version: "1.0.0"
  author: "Marc-André Léger"
  source: "Introduction to Cybersecurity Governance for Business Technology Management, 3rd edition, v2.1g (August 2026)"
  license: "CC BY-NC 4.0"
---

# Cybersecurity Governance reference (Léger, 3rd ed., v2.1g)

Use the author's textbook as the reference for governance, risk, compliance and resilience concepts behind the CyberRiskGuardian method. It covers business technology management and uses this standards baseline:
- NIST CSF 2.0, SP 800-61 Rev. 3 and SP 800-63-4;
- CIS Controls v8.1;
- CVSS v4.0;
- ISO/IEC 27001:2022/Amd 1:2024.

## How to use

1. Identify the concept or chapter the question needs from the index below. Read only that file from `references/`. Chapters are long, so use Grep on the `##` section headings to jump to the relevant part.
2. Quote or paraphrase the book faithfully and cite it as: *Léger, M.-A. (2026). Introduction to Cybersecurity Governance (3rd ed., v2.1g), Chapter N.*
3. Keep the book's own terminology, e.g. "security is the absence of unacceptable risk", the IPM cycle, risk appetite as the managerial core.
4. Where the book and a newer official standard differ, say so and defer to the official publication for operational decisions. This follows the book's editorial note.
5. For assessments, pair the book with the `cyber-risk-assessment` skill:
   - Chapter 5 underpins scenario construction.
   - Chapter 6 describes the CyberRiskGuardian Excel model, its formulas and the step-by-step procedure.
   - Chapter 4 covers treatment strategies, appetite and tolerance, KRIs, threats and vulnerabilities.
   - Chapter 17 covers the economics of controls and investment.
6. Figures are not included in this text version. The figure list is in `00-figures.md`; refer to figures by number when needed.

## Chapter index (`references/`)

| File | Content |
|---|---|
| `00-front-matter.md` | Title, revision note, standards baseline, notice on AI use |
| `00-acronyms.md` · `00-figures.md` | Acronyms; list of figures |
| `00-introduction.md` | Purpose and structure of the book |
| `ch01-basic-concepts-of-cybersecurity.md` | Covers: <ul><li>domain definition; security as the absence of unacceptable risk</li><li>CIA triad; privacy vs confidentiality; data classification</li><li>fraud triangle; risk triangle; risk treatment</li><li>human factor and zero trust; maturity</li><li>three lines model and assurance; risk appetite as the managerial core</li></ul> |
| `ch02-cybersecurity-governance.md` | Covers: <ul><li>direction, control and accountability; ethics</li><li>enterprise, IT and cyber governance hierarchy; governance outcomes</li><li>change management; implementing a governance framework</li><li>balanced cybersecurity scorecard; SMART metrics</li></ul> |
| `ch03-cybersecurity-compliance.md` | Covers: <ul><li>compliance as a management system; why compliance fails</li><li>obligations landscape; compliance risk assessment</li><li>evidence and reporting; assessment approaches; PDCA</li></ul> |
| `ch04-cybersecurity-risk-management.md` | Covers: <ul><li>risk fundamentals; the IPM cycle; ISO alignment</li><li>treatment strategies (accept, avoid, transfer, mitigate); appetite and tolerance</li><li>biases; metrics and KRIs</li><li>threat agents; threat intelligence</li><li>vulnerability identification and management</li><li>exploits: kill chain, MITRE ATT&CK, CAPEC</li></ul> |
| `ch05-creating-cybersecurity-risk-scenarios.md` | Scenario-based assessment method for teaching; scenario construction |
| `ch06-performing-a-cybersecurity-risk-assessment.md` | Covers: <ul><li>the CyberRiskGuardian Excel model: purpose, inputs, structure</li><li>the risk index and formulas; step-by-step procedure</li><li>IM, BCP and DRP for residual risk</li><li>ISO 27001/27005 and NIST compatibility; responsible use of generative AI</li></ul> |
| `ch07-cybersecurity-operations.md` | Security operations, tooling and maturity, SOC |
| `ch08-cybersecurity-detection-orchestration-vulnerabilities-response-an.md` | Detection, SIEM/SOAR, vulnerabilities, response and testing |
| `ch09-cybersecurity-control-frameworks.md` | Control frameworks (NIST CSF, CIS, ISO/IEC 27001/27002 and others) |
| `ch10-identity-and-access-management.md` | Identity and access management (NIST SP 800-63-4, MFA, PAM, lifecycle) |
| `ch11-ot-iot-and-cyber-physical-security.md` | OT, IoT and cyber-physical security (incl. medical devices) |
| `ch12-incident-management-business-continuity-and-resilience.md` | Incident management (SP 800-61r3), BCP, DRP, resilience |
| `ch13-security-architecture-and-security-engineering.md` | Security architecture and engineering |
| `ch14-security-baselines-and-configuration-governance.md` | Baselines and configuration governance |
| `ch15-cloud-cybersecurity.md` | Cloud security and shared responsibility |
| `ch16-data-protection-and-privacy.md` | Data protection, privacy engineering |
| `ch17-cybersecurity-economics-and-investment-governance.md` | Control costs, run vs change investments, TCO, investment governance |
| `ch19-ai-for-cybersecurity-and-the-cybersecurity-of-ai.md` | AI for security and security of AI |
| `ch20-third-party-and-supply-chain-cybersecurity-governance.md` | Third-party and supply-chain governance |
| `ch21-cryptography-and-key-management-in-managerial-terms.md` | Cryptography and key management for managers |
| `ch22-people-organizational-design-and-accountability-mechanisms.md` | People, organizational design, accountability |

The book has no Chapter 18; the numbering jumps from 17 to 19 in v2.1g.

## Licence and attribution

The book text is © Marc-André Léger and is licensed under **CC BY-NC 4.0** (https://creativecommons.org/licenses/by-nc/4.0/).
- **Attribution:** credit the author, edition and licence, and indicate any changes made.
- **Non-commercial:** do not use it for commercial purposes.
- **Third-party material:** quoted standards, frameworks and names (NIST, ISO/IEC, CIS, FIRST, MITRE, OWASP) remain under their owners' terms.
