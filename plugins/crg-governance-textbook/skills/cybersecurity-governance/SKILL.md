---
name: cybersecurity-governance
description: >
  This skill should be used when the user asks about cybersecurity governance, risk, compliance or resilience concepts as taught in
  "Introduction to Cybersecurity Governance" (Marc-André Léger) - e.g. "explain the CIA triad", "fraud triangle", "risk triangle",
  "three lines model", "risk appetite vs tolerance", "IPM risk cycle", "balanced cybersecurity scorecard", "SMART metrics",
  "compliance management system", "cybersecurity maturity", "CVSS v4.0 by equivalency", "risk-to-tolerance bands",
  "investment governance", "third-party governance", "AI governance", "post-quantum cryptography" - or when a risk
  assessment needs the book's definitions, frameworks or chapter guidance as the reference text.
metadata:
  version: "2.1.0"
  author: "Marc-André Léger"
  source: "Introduction to Cybersecurity Governance for Business Technology Management, 3rd edition, v2.51 (October 2026)"
  license: "CC BY-NC 4.0"
---

# Cybersecurity Governance reference (Léger, 3rd ed., v2.51)

Use the author's textbook as the reference for governance, risk, compliance and resilience concepts behind the CyberRiskGuardian method. It covers business technology management in 21 chapters and uses this standards baseline (October 2026):
- NIST CSF 2.0, SP 800-61 Rev. 3 (incident response) and SP 800-63-4 (digital identity);
- CIS Critical Security Controls v8.1;
- FIRST CVSS v4.0;
- ISO/IEC 27001:2022/Amd 1:2024;
- OWASP Top 10:2025 and OWASP GenAI Security guidance;
- ISO/IEC 27701:2025 (privacy) and ISO/IEC 42001:2023 (AI management systems);
- ISA/IEC 62443 and NIST SP 800-82 Rev. 3 (OT);
- NIST FIPS 203, 204 and 205 (post-quantum cryptography);
- NIST SP 800-161 Rev. 1 and SP 800-218 (supply chain, secure development);
- Québec Law 25.

## How to use

1. Identify the concept or chapter the question needs from the index below. Read only that file from `references/`. Chapters are long, so use Grep on the `##` section headings to jump to the relevant part.
2. Quote or paraphrase the book faithfully and cite it as: *Léger, M.-A. (2026). Introduction to Cybersecurity Governance (3rd ed., v2.51), Chapter N.* For the appendices, cite "Appendix: Using CyberRiskGuardian with this book" or "Appendix: Answers to multiple-choice questions".
3. Keep the book's own terminology, e.g. "security is the absence of unacceptable risk", the IPM cycle, risk appetite as the managerial core, **risk indices** (computed) versus **key risk indicators** (measured), the risk-to-tolerance ratio and its bands (below 0.90, 0.90–1.10, above 1.10).
4. Where the book and a newer official standard differ, say so and defer to the official publication for operational decisions. This follows the book's editorial note.
5. For assessments, pair the book with the `cyber-risk-assessment` skill of the CyberRiskGuardian Core plugin:
   - Chapter 4: risk appetite on a 0–1 scale, appetite versus tolerance, the tolerance bands, treatment strategies (including transfer recorded as a decision), KRIs and the budget benchmark.
   - Chapter 5: CRG's scenario process — evidence register, context and crown jewels, about 25–40 candidates (never fewer than 20), five generation strategies, twelve screening criteria, escalation flags, families, weighting schemes, coverage and the final ten.
   - Chapter 6: the CRG workflow (19 steps), the risk index and its formulas, valid ranges, the risk-to-tolerance ratio and what cancels, CVSS v4.0 and scoring by equivalency, the workbook procedure (steps 1–8), the MediBec worked example, prioritization, portfolio and budget calibration, KRIs, the management report and quality control.
   - Chapter 17: cybersecurity economics — ALE, FAIR and Monte Carlo, ROSI, Gordon–Loeb, insurance as transfer, budgeting, and the CRG cross-check.
   - The CRG appendix: what CRG is, the editions, three ways to work, the chapter-by-chapter guide and the Excel teaching template.
6. For review-question answers, use `zz-appendix-mcq-answer-key.md` (answer letters only; rationales and essay guidance are in the instructor resource guide, which is not part of this plugin).
7. Figures are not included in this text version. The figure list is in `00-figures.md`; refer to figures by number when needed.

## Chapter index (`references/`)

| File | Content |
|---|---|
| `00-front-matter.md` | Title, revision note (v2.51 changes), standards baseline, notice on AI use |
| `00-acronyms.md` · `00-figures.md` | Acronyms; list of figures |
| `00-introduction.md` | Purpose and structure of the book |
| `ch01-basic-concepts-of-cybersecurity.md` | Domain definition; security as the absence of unacceptable risk; CIA triad; privacy vs confidentiality; data classification; fraud and risk triangles; human factor and zero trust; maturity; three lines model and lines of accountability; risk appetite as the managerial core |
| `ch02-cybersecurity-governance.md` | Direction, control and accountability; ethics; governance hierarchy and outcomes; change management; implementing a framework; balanced cybersecurity scorecard; SMART metrics |
| `ch03-cybersecurity-compliance.md` | Compliance as a management system; obligations landscape; compliance risk assessment; evidence and reporting; assessment approaches; PDCA |
| `ch04-cybersecurity-risk-management.md` | Risk fundamentals; IPM cycle; ISO alignment; treatment strategies; risk appetite and tolerance (0–1 scale, bands, budget benchmark); biases; metrics and KRIs; threat agents and intelligence; vulnerability management; exploits (kill chain, ATT&CK, CAPEC); exposure; scenario-based assessment |
| `ch05-creating-cybersecurity-risk-scenarios.md` | Scenarios as the unit of analysis; standards compatibility; CRG approach; causal chains; context first; candidate generation (25–40, five strategies); twelve screening criteria, escalation flags, families, weighting schemes, coverage; MediBec portfolio; validation; AI-assisted drafting; preparing quantification |
| `ch06-performing-a-cybersecurity-risk-assessment.md` | Standards alignment; inputs and the 19-step CRG workflow; editions; evidence discipline and the three markers; the risk index, formulas and valid ranges; risk-to-tolerance ratio and bands; cost-effectiveness; shared controls, transfer, avoidance; CVSS v4.0 and the NVD calculator; non-technical vulnerabilities by equivalency; workbook procedure; MediBec worked example; prioritization, portfolio, budget calibration; KRIs; management report and QC; IM/BCP/DRP; generative AI |
| `ch07-cybersecurity-operations.md` | Security operations, tooling and maturity, SOC |
| `ch08-cybersecurity-detection-orchestration-vulnerabilities-response-and-tes.md` | Detection, SIEM/SOAR, vulnerabilities (EPSS, KEV, SSVC), response and testing |
| `ch09-cybersecurity-control-frameworks.md` | Control frameworks (NIST CSF 2.0, CIS v8.1, ISO/IEC 27001/27002 and others) |
| `ch10-identity-and-access-management.md` | Identity and access management (NIST SP 800-63-4, MFA, PAM, lifecycle) |
| `ch11-ot-iot-and-cyber-physical-security.md` | OT, IoT and cyber-physical security (incl. medical devices) |
| `ch12-incident-management-business-continuity-and-resilience.md` | Incident management (SP 800-61r3), BCP, DRP, resilience |
| `ch13-security-architecture-and-security-engineering.md` | Security architecture and engineering |
| `ch14-security-baselines-and-configuration-governance.md` | Baselines and configuration governance |
| `ch15-cloud-cybersecurity.md` | Cloud security and shared responsibility |
| `ch16-data-protection-and-privacy.md` | Data protection, privacy engineering |
| `ch17-cybersecurity-economics-and-investment-governance.md` | Investment economics; ALE; FAIR, simulation and loss exceedance; control economics; worked example and CRG cross-check; prioritization; portfolio governance; managed services and tooling; cyber insurance; budgeting |
| `ch18-ai-for-cybersecurity-and-the-cybersecurity-of-ai.md` | AI for defenders and attackers; AI assets and attack surfaces; agentic AI; inventory and classification; shadow AI; AI supply chain; secure AI lifecycle; privacy and compliance; assurance (ISO/IEC 42001); AI incidents; investment |
| `ch19-third-party-and-supply-chain-cybersecurity-governance.md` | Third-party lifecycle; risk tiering; due diligence and assurance; procurement and contracts; monitoring; concentration risk; fourth parties and software supply chain; incident coordination; exit planning; operating model |
| `ch20-cryptography-and-key-management-in-managerial-terms.md` | What encryption does; classification-driven encryption; key management; key ownership; BYOK/HYOK; escrow and recovery; PKI; post-quantum cryptography and crypto-agility; assurance |
| `ch21-people-organizational-design-and-accountability-mechanisms.md` | Operating model; board oversight; decision rights; RACI; roles; CISO accountability; workforce and skills; security champions; human risk management; incentives; accountability mechanisms |
| `zz-appendix-using-cyberriskguardian.md` | Appendix: what CRG is; editions (Community, Educational, Professional); three ways to work; chapter-by-chapter guide; the Excel teaching template; using AI responsibly; where to get it |
| `zz-appendix-mcq-answer-key.md` | Appendix: answer letters for the twenty multiple-choice questions of every chapter |
| `zz-references.md` | References (APA 7th edition): standards, laws, guidance, reports and publications cited |
| `zz-bibliography.md` | Bibliography and further reading (not cited in the text) |

Every chapter ends with learning outcomes, a summary, key terms, five essay questions and twenty multiple-choice questions.

## Licence and attribution

The book text is © Marc-André Léger and is licensed under **CC BY-NC 4.0** (https://creativecommons.org/licenses/by-nc/4.0/).
- **Attribution:** credit the author, edition and licence, and indicate any changes made.
- **Non-commercial:** do not use it for commercial purposes.
- **Third-party material:** quoted standards, frameworks and names (NIST, ISO/IEC, CIS, FIRST, MITRE, OWASP) remain under their owners' terms.
