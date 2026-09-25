# Worked example — MediBec (fictional teaching case)

MediBec is a fictional private medical centre (Montreal head office, 10 clinics) from *Integrated Cybersecurity Teaching Case v2.0b*, M.-A. Léger. This example is the first full application of the CyberRiskGuardian agent (September 2026).

## Settings used
| Setting | Value |
|---|---|
| Horizon | 12 months |
| Risk appetite | 0.30 (analyst estimate; the case states 0.40) |
| Factor | 1,000 |
| CVSS | v4.0 |
| Classification band | ±10% around a ratio of 1.0 |

## Deliverables (`deliverables/`)
| File | Content |
|---|---|
| `MediBec_Cybersecurity_Risk_Assessment.docx` + `..._Workbook.xlsx` | 10-scenario assessment (20 candidates) |
| `..._Risk_Assessment_30.docx` + `..._Workbook_30.xlsx` | 30-scenario assessment (45 candidates, 23 initiatives) |
| `..._Risk_Assessment_30_Budget.docx` + `..._Workbook_30_Budget.xlsx` | The 30-scenario assessment plus budget calibration (Section 11; Budget_* sheets) |
| `MediBec_Scenario_Selection.xlsx` | Re-selection under the scenario-selection instructions: Sections A–E, 45 candidates → P01–P10 |

## Key results

**30 scenarios**
- After treatment: 12 above tolerance, 11 approximately at, 7 below.
- Total estimated risk 84,491 against residual 41,104, a 51% reduction.

**Budget**
- Current spend is about 0.9% of the IT budget, below the 4% floor.
- The risk-driven budget is 5.1–5.5%, which implies an appetite of about 0.64.
- The recommended ramp is 7% / 9.5% / 12%.

**Selection**
- P05 and P10 were chosen by judgement.
- Admin sabotage (C31) is the strongest unselected candidate and sits on the watch list.
- Fraud/BEC is the only coverage gap.

## Rebuild (`source/`)
Run from `source/`; outputs go to `../output/`.
```
python3 data.py && python3 data30.py && python3 budget.py     # data models + JSON
python3 build_xlsx.py; python3 build_xlsx30.py; python3 build_xlsx_budget.py; python3 build_sel.py
node build_docx.js; node build_docx30.js; node build_docx30b.js
```

Before running:
- `build_xlsx_budget.py` extends `../output/MediBec_CyberRiskGuardian_Workbook_30.xlsx`. Build and recalculate that workbook first.
- The `build_docx30*.js` files are assembled from `part_*.js`.

**Regression check:**
```
python3 ../../../plugins/cyberriskguardian/skills/cyber-risk-assessment/scripts/crg_calc.py medibec30.json
```
It must print total estimated 84,491 and residual 41,104.

The input case document is in `input/`. All scores are illustrative analyst estimates for teaching, not empirical benchmarks.
