---
paths:
  - "**/*.xlsx"
  - "**/build_*.py"
  - "**/*workbook*"
---

# Workbook conventions

- Build with openpyxl using live formulas; never paste computed results where a formula belongs.
- Arial font; blue = input, black = formula, green = cross-sheet link, yellow fill = key assumption.
- Implement the CRG formulas exactly (Excel Guide v1.0c s.11); keep the factor in one Parameters cell; keep reductions within 0-1 and add a validity check column.
- Prefer Excel-2007 functions; post-2007 functions need the `_xlfn.` prefix (e.g. `_xlfn.MAXIFS`); never use XLOOKUP, FILTER, SORT, UNIQUE, SEQUENCE.
- Recalculate with LibreOffice headless after every build and deliver only with zero formula errors; then cross-check key cells against `crg_calc.py`.
- Document every hard-coded assumption in a comment or adjacent note, with its evidence reference.
