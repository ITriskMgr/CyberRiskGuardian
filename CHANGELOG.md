# Changelog

All notable changes to the **CyberRiskGuardian Community Edition** (Core plugin `cyberriskguardian`, textbook plugin
`crg-governance-textbook`, examples and education package). Versions follow semver per plugin; the marketplace carries
the Core version. Earlier releases are described in the commit history (Core 2.1.0 and 2.1.1, 23 October 2026).

## Education Package 2.1.2 — 2026-11-08

Teaching materials (`education/`) refreshed for **CyberRiskGuardian Desktop 1.9.2** (findings NV-E01 to NV-E07 of the test
campaign of 8 November 2026, decisions FR-1 to FR-3 and D3). No plugin, formula or example changed. The version number stays
aligned with the textbook plugin's 2.1.x line; release asset `CyberRiskGuardian_Education_Package_2.1.2.zip` (release
`education-v2.1.2`).

- **Lab 12, optional step 4 [B] — Security Policy Writer** (EN/FR, NV-E07): students draft the IAM policy with Governance →
  Security Policy Writer from the case only (Unknown / Decide later, confirmed facts, a recorded legal decision with a named
  reviewer, their own decisions instead of *Accept all suggestions*), then compare the generated draft with their own five
  statements and explain the `[DECISION: …]` placeholders; the deliverable and the facilitation notes say the draft never
  replaces the students' statements.
- **Instructor guide, troubleshooting** (NV-E05, NV-E01): the duplicate-MediBec row says *fixed in Desktop 1.9.2*; a new
  *Fixed in Desktop 1.9.2* table lists what students still on 1.9.1 will meet and what to do — `[object Object]` from the
  rules-only extraction (#18, Lab 1), promotion losing parameters (#20, Lab 3), CVSS score vs vector (#21, Lab 5), QC-02 and
  empty approvers (#22, Labs 10–11), two windows (#2, all labs), duplicate workspaces (#4, #5), and the Lab 0 path. Lab 1
  carries the same known-issue note.
- **Lab 0 [A]** (NV-E02, EN/FR, and the set-up checklist): `crg_calc.py ../../../evals/_shared/medibec30.json` from the
  plugin's `crg-quantify/scripts` folder (the example is not beside the script); `examples/medibec/source/medibec30.json` in
  a clone.
- **Lab 8** (NV-E03, EN/FR): *Drivers* is the button on each row of the Monte Carlo *By scenario* table, not a Forecasts
  screen.
- **French terminology of Desktop 1.9.2** (FR-1 to FR-3): the formula sheet reads « Ratio (risque résiduel / risque toléré) »
  and the terminology sentences « ratio risque résiduel / risque toléré »; Lab 4 says « probabilité », not « vraisemblance »;
  Lab 11 says « Consigner l'élément probant » and names the card *Éléments probants des opérations de sécurité*. Interface
  labels and the marker « Lacune d’information — validation organisationnelle requise » use the typographic apostrophe
  (NV-E04).
- **Blank Excel model** (`templates/CyberRiskGuardian_Excel_Model_Blank_v2-6.xlsx`, NV-P12 / decision D3): the CIS Controls
  v8.1 notice (bilingual, CC BY-NC-ND 4.0, identifiers and titles only) written on each M sheet (U36, U37, U245) and on
  Analyse!C53:C54 with `scripts/add_cis_notice.py` from Core 2.2.1; no formula, value or layout changed, 0 formula errors
  after recalculation.
- **`scripts/build_education.py`** (NV-E06): LibreOffice's UNO port is free by default, or fixed with `--port` /
  `CRG_LO_PORT`; the dependencies (pandoc, python-docx, LibreOffice with its Python UNO bridge; the Node `docx` package is not
  used) are documented and checked before the build.
- Version 2.1.2 and Desktop 1.9.2 in every subtitle, README/LISEZMOI and lab prerequisite; all 16 Word and PDF files
  rebuilt. The introductory decks keep their 1.9.1 screenshots (the 1.9.2 screens differ in wording only; see the notes).

## Education Package 2.1.1 — 2026-11-05

Teaching materials (`education/`) updated for **CyberRiskGuardian Desktop 1.9.1**, Core 2.2.0 and the textbook **v2.51b**
(English and French). No plugin, formula or example changed. Release asset: `CyberRiskGuardian_Education_Package_2.1.1.zip`
(release `education-v2.1.1`).

- **Lab 13 — Identity, access and configuration posture** (EN/FR): the desktop application's optional Identity & access
  management (IAM/PAM) and Configuration management modules on MediBec's sample registers; self-check answers with the
  reference indicator values; learning outcome LO12; Week 11 of the full-term model; facilitation notes.
- **CanaRetail practice case**: its Module D solution ships in the desktop application (27,526 / 16,049), so it becomes a
  self-study case for Labs 4–7 and is no longer recommended for graded A1 work; graded cases are Chez Pierre, ACRB and CNIC.
- **1.9 interface**: menu paths, next-step card, detail levels, top-bar Help, chapter-linked help, footer *✓ Engine verified*;
  Lab 0 and the set-up checklist point to the first-time installation and setup guides.
- **Virtual consultant**: decided per assignment (A1–A3 briefs, AI policy, AI acknowledgement).
- **A2**: the brief allows the IAM/PAM registers as evidence; rubric unchanged.
- **Textbook v2.51b** in English and French; the French documents cite the French edition and use its terms.
- **Introductory decks** (EN/FR): corrected text and three new slides with 1.9.1 screenshots; 13 labs.
- All 16 Word and PDF files rebuilt with `scripts/build_education.py`.

## Core 2.2.0 · Textbook 2.1.1 — 2026-11-02

Aligned with the textbook *Introduction to Cybersecurity Governance for Business Technology Management*, 3rd ed.,
**v2.51b** (the corrected v2.51 of October 2026). No risk formula changed. **MediBec regression unchanged: 84,305 /
41,023** (S2 and S8 scored AV:L; 12 above, 11 approximately at, 7 below tolerance).

### Added
- **`/label-check`** command: runs `hooks/check_labels.py` on deliverables (Markdown, text, HTML, CSV, JSON, Word) and
  tabulates missing markers and outdated wording; read-only and offline (book ch05 "keep the markers", CRG appendix).
- Evals: `quick-scan-table` and `label-check-cli` (Core, with the fixture `draft_summary.md`); `product-rule` and
  `candidate-population` (textbook). Core 12 cases, textbook 7.
- `plugins/crg-governance-textbook/skills/cybersecurity-governance/references-fr/`: placeholder for the French
  reference set (see *Pending*).

### Changed
- **Label check** (`check_labels.py`, ported from the Threat Live-feed plugin): the three standard markers, EN and FR —
  **Analytical estimate — validation required** / *Estimation analytique — validation requise*, **Information gap —
  organizational validation required** / *Lacune d’information — validation organisationnelle requise*, **Management
  decision required** / *Décision de la direction requise* — singular or plural; quick scans count as reports; retired
  method wording is reported as outdated (computed values called KRIs, a "4–12%" budget band, an ABS() residual, a
  10,000 factor, the 1.25 threshold) unless negated in the same sentence; outdated wording alone no longer blocks a
  reply (warning only); retired markers still never count as the current marker.
- **`/crg-quick-scan`**: steps 1–11 in compressed form — at least twenty candidates, typically 25–40 (`--candidates
  N`), 12-criterion screening and escalation flags, 10 selected, CVSS v4.0, risk indices with the calculator, and the
  pre-treatment verdicts in **one table**; no treatment, costs, residual risk or budget.
- **Candidate population** worded as in the book everywhere: "at least twenty candidates, typically 25–40"
  (scenario builder, selection instructions, master specification, agent, commands, plugin and marketplace
  descriptions, README, quick starts).
- French information-gap marker written with the typographic apostrophe used by the desktop application
  (*Lacune d’information*); the check accepts both apostrophes.
- **Textbook skill 2.1.1**: reference set replaced by the corrected v2.51b text — MediBec totals 84,305 / 41,023;
  treatment packages combine by the **product rule 1 − ∏(1 − rᵢ)** (ch06 incl. multiple-choice question 14, answer A
  unchanged; its wording is reconstructed from the erratum record until the published v2.51b text is re-ingested);
  "at least twenty candidates, typically 25 to 40" (appendix, chapter table, ch17); risk triangle = threat,
  vulnerability and exposure (ch01, ch13, ch15, ch17, ch20, ch21); ch17 question 19 names the approver; appendix
  editions and chapter table. Citation and source metadata name v2.51b; chapter numbering 1–21 unchanged.
- Book citations v2.51b in AGENTS.md, the session context, the workbook rule, the formulas reference and
  `crg_calc.py`; education README/LISEZMOI note that the chapter decks (v2.51) defer to the v2.51b text.
- Versions: Core plugin, its skills, `package.json` and the marketplace 2.2.0; textbook plugin 2.1.1.

### Pending
- **French textbook reference set**: awaits the French v2.51b source from the author. It will not be produced by
  machine translation; until then the textbook skill answers French questions from the English text and says the
  quotation is translated.
