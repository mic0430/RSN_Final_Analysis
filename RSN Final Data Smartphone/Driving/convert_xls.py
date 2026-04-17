#!/usr/bin/env python3
"""
convert_xls.py – Batch convert all .xls files to .xlsx (pure Python)
=====================================================================
Walks the directory tree from the script's location, finds every
RSN Final *.xls file, reads it with xlrd, and writes an .xlsx copy
alongside it using openpyxl.  No LibreOffice required.

Run once:
    pip install xlrd openpyxl
    python convert_xls.py

After this, analysis.py reads .xlsx directly and never touches .xls.
"""

import os
import glob
import sys

import xlrd
from openpyxl import Workbook


def convert_xls_to_xlsx(xls_path):
    """Convert a single .xls to .xlsx using xlrd → openpyxl."""
    xlsx_path = os.path.splitext(xls_path)[0] + ".xlsx"

    xls_book = xlrd.open_workbook(xls_path)
    wb = Workbook()
    # Remove the default empty sheet created by openpyxl
    wb.remove(wb.active)

    for sheet_name in xls_book.sheet_names():
        xls_sheet = xls_book.sheet_by_name(sheet_name)
        ws = wb.create_sheet(title=sheet_name)

        for row_idx in range(xls_sheet.nrows):
            for col_idx in range(xls_sheet.ncols):
                cell = xls_sheet.cell(row_idx, col_idx)
                value = cell.value
                # xlrd returns dates as floats; convert if needed
                if cell.ctype == xlrd.XL_CELL_DATE:
                    value = xlrd.xldate_as_datetime(value, xls_book.datemode)
                ws.cell(row=row_idx + 1, column=col_idx + 1, value=value)

    wb.save(xlsx_path)
    return xlsx_path


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    xls_files = glob.glob(os.path.join(base_dir, "**", "*.xls"), recursive=True)

    if not xls_files:
        print("No .xls files found under", base_dir)
        sys.exit(0)

    print(f"Found {len(xls_files)} .xls file(s) to convert:\n")

    converted = 0
    skipped = 0
    failed = 0

    for xls_path in sorted(xls_files):
        rel = os.path.relpath(xls_path, base_dir)
        xlsx_path = os.path.splitext(xls_path)[0] + ".xlsx"

        if os.path.isfile(xlsx_path):
            print(f"  SKIP     {rel}  (xlsx already exists)")
            skipped += 1
            continue

        print(f"  CONVERT  {rel} ... ", end="", flush=True)
        try:
            convert_xls_to_xlsx(xls_path)
            print("OK")
            converted += 1
        except Exception as e:
            print(f"FAILED ({e})")
            failed += 1

    print(f"\nDone.  converted={converted}  skipped={skipped}  failed={failed}")


if __name__ == "__main__":
    main()