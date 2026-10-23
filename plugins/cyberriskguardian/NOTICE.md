# NOTICE

**CyberRiskGuardian Agent** — © 2026 Marc-André Léger.

## Licensing at a glance
CyberRiskGuardian is offered under an open-core model with three editions.

| Edition | Licence | Who | Price |
|---|---|---|---|
| **Community Edition** (this repository) | CC BY-NC 4.0 | Students, for their own coursework including graded work; non-profit organisations, for their own risk; instructors, who may adapt the free materials non-commercially with attribution; anyone using it for non-commercial purposes | Free |
| **Educational edition** | CyberRiskGuardian Educational Licence | Colleges, CEGEPs and universities, as institutions | Reduced educational price; includes installation and customization, the current desktop application, CRG Pro Quantitative and the Educator Edition of the teaching materials |
| **Professional editions** | CyberRiskGuardian Commercial Licence | Companies, consultants, MSSPs, government bodies and other organizations, and any commercial use of the Community Edition | Professional, Team, Enterprise or MSSP tiers |

Everything in this repository is the **Community Edition**. It is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0). The full text is in `LICENSE` and at https://creativecommons.org/licenses/by-nc/4.0/legalcode.

The content of the **Educational and Professional editions** is proprietary and is **not** in this repository. It includes the CRG Pro Quantitative, Policy Suite, SecOps Bridge, Third-Party Risk and Audit & Assurance plugins, the Compliance Packs and Sector Packs, the Educator Edition of the teaching materials, the hosted CRGO ontology service, and CyberRiskGuardian Desktop from release 1.8.0 onward (Desktop 1.7.0 Beta and earlier remain available under CC BY-NC 4.0; later releases are licensed under the desktop licence shipped with the application). It is distributed through a private marketplace, licence-key activation or authenticated hosted services.

Details, tiers and licence enquiries: https://cyberriskguardian.ca · sales@cyberriskguardian.ca.

## When you need a paid licence
- **Community Edition, free (CC BY-NC 4.0):** a student using it for their own coursework, including graded work; a non-profit organisation assessing its own risks; an instructor using or adapting the free materials in a course, non-commercially and with attribution; anyone sharing or adapting it for non-commercial purposes with attribution.
- **Professional editions (Commercial Licence):** a consultant, firm or MSSP using CyberRiskGuardian to deliver paid work; a company or government body using it for its own risk management; any use of the Educational or Professional edition content outside a college or university.
- **Educational edition (Educational Licence):** a college, CEGEP or university, as an institution, that wants CRG Pro Quantitative, the desktop application, the Educator Edition, hosted services, installation, customization or support for its programs.

Definitions used by the licences:
- *Student*: an individual enrolled in a program of study, using CyberRiskGuardian for personal learning and their own coursework, including graded work.
- *Non-profit organisation*: a registered charity or not-for-profit corporation, excluding colleges, universities and government bodies.
- *Commercial use*: any use intended for, or directed toward, commercial advantage or monetary compensation, including delivering paid services to third parties.

Versions already released under CC BY-NC 4.0 remain available under that licence. Version 1.1.1, which bundled the budget guideline, workbook builder and sensitivity tools, stays available on the `release/1.x` branch.

## What the Community Edition covers
- The CyberRiskGuardian methodology (core method specifications), including:
  - the master task specification (19 steps);
  - the scenario-selection instructions;
  - the formula conventions.
- The plugins in this repository (CyberRiskGuardian Core and CRG Governance Textbook): skills, commands, subagents, output styles, hooks, evaluation suites and configuration.
- The Core calculator script and the Core MCP calculator server.
- The textbook *Introduction to Cybersecurity Governance for Business Technology Management*, 3rd edition, version 2.51 (October 2026), packaged as a reference skill in `plugins/crg-governance-textbook/skills/cybersecurity-governance/references/`.
- The fictional MediBec teaching case and the example deliverables in `examples/medibec/`.
- The education package in `education/` (instructor guide, lab workbook, assessment kit, case library, introductory slides) and the blank CyberRiskGuardian Excel model (`education/templates/`). The 21 chapter slide decks are published as release assets of this repository.

## Suggested attribution
> *CyberRiskGuardian* and *Introduction to Cybersecurity Governance* (3rd ed., v2.51) by Marc-André Léger, licensed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/). Changes: <describe your changes>.

## Third-party material
The following remain the property of their respective owners and are used here by reference or quotation:
- names, standards and frameworks, including NIST CSF 2.0, NIST SP 800-61/800-63, ISO/IEC 27001/27002/27005/31000, CIS Controls, FIRST CVSS, MITRE ATT&CK and CAPEC, OWASP, ITIL and COBIT;
- the Creative Commons legal code, © Creative Commons.

No CyberRiskGuardian licence grants rights in that third-party material.

## Software dependencies
The dependencies keep their own licences:
- `cvss`, `openpyxl`, `mcp`;
- `docx` (npm);
- LibreOffice.

## Notes
- Creative Commons does not recommend its licences for software. The scripts in this repository are licensed under CC BY-NC 4.0 at the author's choice; commercial use of the code is available under the CyberRiskGuardian Commercial Licence.
- SPDX expression for this repository: `CC-BY-NC-4.0 OR LicenseRef-CyberRiskGuardian-Commercial`.
- The book and the generated assessments were produced with the assistance of generative AI. The estimates they contain are illustrative and must be validated before any organizational use.
