#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Marc-André Léger
"""Cybersecurity budget calibration against the 4% / 7.8% / 12% guideline.

Examples:
    python3 budget_calibrator.py --it-budget 100000000 --appetite 0.30
    python3 budget_calibrator.py --it-budget 100000000 --spend 5.05e6 5.55e6 4.87e6

Guideline: cybersecurity budget = 4%-12% of total IT budget including salaries.
4% = risk seeking (appetite 0.70), 7.8% = median / risk neutral (0.50), 12% = risk averse (0.30).
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crg_calc import budget_target, implied_appetite, band_status, BUDGET_BAND

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--it-budget", type=float, required=True, help="Total IT budget including salaries")
ap.add_argument("--appetite", type=float, help="Approved or estimated risk appetite (0-1)")
ap.add_argument("--spend", type=float, nargs="*", help="Planned cybersecurity spend per year")
a = ap.parse_args()

print(f"IT budget: {a.it_budget:,.0f}")
print(f"Guideline band: {BUDGET_BAND['min']:.1%} (risk seeking) - {BUDGET_BAND['median']:.1%} (neutral) - {BUDGET_BAND['max']:.1%} (risk averse)")
print(f"  = {BUDGET_BAND['min'] * a.it_budget:,.0f} - {BUDGET_BAND['median'] * a.it_budget:,.0f} - {BUDGET_BAND['max'] * a.it_budget:,.0f}")
if a.appetite is not None:
    t = budget_target(a.appetite)
    print(f"Appetite {a.appetite:.2f} -> target {t:.1%} = {t * a.it_budget:,.0f} per year")
for i, s in enumerate(a.spend or [], 1):
    pct = s / a.it_budget
    ia = implied_appetite(pct)
    print(f"Year {i}: spend {s:,.0f} = {pct:.1%}  {band_status(pct)}  implied appetite {'n/a' if ia is None else f'{ia:.2f}'}")
