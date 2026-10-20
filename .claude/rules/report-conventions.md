---
paths:
  - "**/*.docx"
  - "**/build_docx*.js"
  - "**/*report*"
---

# Report conventions

- Generate Word reports with docx-js from the same JSON data model as the workbook, so numbers cannot diverge.
- Follow the outline in `skills/crg-summary-report/references/deliverables.md` (Core summary) or the CRG Pro `crg-board-report` skill (full report).
- Tables: explicit DXA column widths that sum to the content width; `ShadingType.CLEAR`; no literal bullet characters.
- Every detailed scenario shows its parameter table (value, reading, rationale, evidence, confidence), CVSS vector and metric rationale, and the calculation substitution.
- Render to PDF and inspect page images before delivery; fix overflow, empty tables and placeholder text.
- Any statement of the form "only X remains above tolerance" or "N scenarios change" must be computed from the data, not typed by hand.
