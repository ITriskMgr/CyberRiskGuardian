#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Marc-André Léger
"""Add the CIS Controls notice to the blank Excel teaching template (NV-P12, author decision D3).

The Community and Educational editions keep the eighteen CIS Controls v8.1 titles on the M1-M10 mitigation sheets
(CC BY-NC-ND 4.0 permits non-commercial redistribution with attribution); the template carried the titles without
a notice. This script writes, on every M sheet, a bilingual notice beside the catalogue summary (U36-U38) and above
the CIS block (C245 keeps its heading; the notice goes in the free cell U245), and a one-line notice on the Analyse
sheet below the last used row. Formulas, values and layout are untouched. Idempotent: an existing notice is replaced.

    python3 scripts/add_cis_notice.py education/templates/CyberRiskGuardian_Excel_Model_Blank_v2-6.xlsx
    python3 scripts/add_cis_notice.py <template.xlsx> --out /tmp/check.xlsx    # write elsewhere, leave the source alone

The commercial desktop build carries CIS identifiers only until CIS approves (desktop edition switch, ws2); the
template is an education asset and keeps titles + notice. Run it from the education workstream's branch, where the
template lives; this script is kept here so the notice text is the same everywhere.
"""
import argparse, sys
from openpyxl import load_workbook
from openpyxl.styles import Font

NOTICE_EN = ("CIS Controls® v8.1 titles © Center for Internet Security, Inc. (CIS), licensed CC BY-NC-ND 4.0 "
             "(https://creativecommons.org/licenses/by-nc-nd/4.0/). Identifiers and titles only; no CIS text is reproduced. "
             "Cite CIS by identifier: https://www.cisecurity.org/controls/v8-1. CIS Controls is a registered trademark of CIS. "
             "Non-commercial, educational use of this template only.")
NOTICE_FR = ("Titres des CIS Controls® v8.1 © Center for Internet Security, Inc. (CIS), sous licence CC BY-NC-ND 4.0. "
             "Identifiants et titres seulement; aucun texte CIS n'est reproduit. Citer CIS par identifiant : "
             "https://www.cisecurity.org/controls/v8-1. CIS Controls est une marque déposée de CIS. "
             "Usage non commercial et pédagogique de ce modèle seulement.")
MARK = "CIS Controls®"


def add_notice(path, out=None):
    wb = load_workbook(path)
    small = Font(name="Arial", size=9, italic=True, color="595959")
    touched = []
    for ws in wb.worksheets:
        if ws.title.startswith("M") and ws.title[1:].isdigit():
            # catalogue summary block (U32 names the catalogues); U36-U37 are free in v2-6
            for row, text in ((36, NOTICE_EN), (37, NOTICE_FR)):
                c = ws.cell(row=row, column=21)
                if c.value and MARK not in str(c.value):
                    raise SystemExit(f"{ws.title}!U{row} is not free ({str(c.value)[:40]!r}); adjust the target cell")
                c.value = text
                c.font = small
            # beside the CIS block heading (C245)
            head = ws["C245"].value or ""
            if "CIS Controls v8.1" not in str(head):
                raise SystemExit(f"{ws.title}!C245 is not the CIS heading ({str(head)[:40]!r}); the template layout changed")
            c = ws["U245"]
            c.value = NOTICE_EN
            c.font = small
            touched.append(ws.title)
    a = wb["Analyse"]
    row = a.max_row + 2
    for r in range(1, a.max_row + 1):          # idempotent: reuse an existing notice row
        if a.cell(row=r, column=3).value and MARK in str(a.cell(row=r, column=3).value):
            row = r
            break
    a.cell(row=row, column=3, value=NOTICE_EN).font = small
    a.cell(row=row + 1, column=3, value=NOTICE_FR).font = small
    wb.save(out or path)
    return touched, row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("template")
    ap.add_argument("--out", help="write the result here instead of overwriting the template")
    a = ap.parse_args()
    touched, row = add_notice(a.template, a.out)
    print(f"CIS notice written on {len(touched)} M sheets (U36, U37, U245) and on Analyse!C{row}:C{row + 1} -> {a.out or a.template}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
