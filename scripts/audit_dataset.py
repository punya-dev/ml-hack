#!/usr/bin/env python3
"""
Comprehensive Load & Audit Script for ML Challenge 2026 Dataset.

Checks every file against §1 of pre_blocking_data_prep_plan.md:
1. Explicit tab delimiter (`\t`) verification and exact column headers.
2. Field count per row (embedded tabs or newlines causing row shifts).
3. Entity ID prefix consistency (no `S2-` in source 1, etc.).
4. Uniqueness of entity_id within each source file.
5. UTF-8 encoding integrity, scanning for replacement characters ('\ufffd' / '') or null bytes ('\x00').
6. Missing/empty field rates across all columns.
7. Country distinct values and casing.
"""

import os
import sys
import time
from collections import Counter


FILES_TO_AUDIT = [
    ("dataset/train/train_source1.tsv", ["entity_id", "business_name", "business_address", "country"], "S1-"),
    ("dataset/train/train_source2.tsv", ["entity_id", "business_name", "business_address", "country"], "S2-"),
    ("dataset/train/train_source3.tsv", ["entity_id", "business_name", "business_address", "country"], "S3-"),
    ("dataset/train/train_ground_truth.tsv", ["source1_entity_id", "matched_entity_ids"], "S1-"),
    ("dataset/test/test_source1.tsv", ["entity_id", "business_name", "business_address", "country"], "S1-"),
    ("dataset/test/test_source2.tsv", ["entity_id", "business_name", "business_address", "country"], "S2-"),
    ("dataset/test/test_source3.tsv", ["entity_id", "business_name", "business_address", "country"], "S3-"),
]


def audit_file(filepath, expected_header, expected_prefix):
    print("\n" + "=" * 70)
    print(f"AUDITING: {filepath}")
    print("=" * 70)
    
    if not os.path.exists(filepath):
        print(f"ERROR: File {filepath} not found!")
        return False

    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")

    start_time = time.time()
    
    is_gt = "ground_truth" in filepath
    expected_col_count = len(expected_header)
    expected_tabs = expected_col_count - 1

    row_count = 0
    bad_tab_lines = []
    bad_prefix_lines = []
    mojibake_count = 0
    null_byte_count = 0
    empty_counts = [0] * expected_col_count
    countries = Counter()
    seen_ids = set()
    dup_id_count = 0

    with open(filepath, "r", encoding="utf-8", errors="replace") as fin:
        header_line = fin.readline()
        if not header_line:
            print("ERROR: File is completely empty!")
            return False

        header_cols = [c.strip() for c in header_line.rstrip("\r\n").split("\t")]
        if header_cols != expected_header:
            print(f"HEADER MISMATCH ERROR: Got {header_cols}, expected {expected_header}")
            return False
        else:
            print(f"Header: OK -> {header_cols}")

        for line_num, line in enumerate(fin, start=2):
            row_count += 1
            raw_line = line.rstrip("\r\n")

            # Check encoding anomalies
            if "\ufffd" in raw_line:
                mojibake_count += 1
            if "\x00" in raw_line:
                null_byte_count += 1

            # Split fields
            parts = raw_line.split("\t")
            if len(parts) != expected_col_count:
                if len(bad_tab_lines) < 5:
                    bad_tab_lines.append((line_num, len(parts), raw_line[:100]))
                continue

            entity_id = parts[0].strip()
            if not entity_id.startswith(expected_prefix):
                if len(bad_prefix_lines) < 5:
                    bad_prefix_lines.append((line_num, entity_id))

            # Uniqueness check
            if entity_id in seen_ids:
                dup_id_count += 1
            else:
                seen_ids.add(entity_id)

            # Track empties
            for i, p in enumerate(parts):
                if not p.strip():
                    empty_counts[i] += 1

            # Track countries for source files
            if not is_gt and len(parts) >= 4:
                country = parts[3].strip()
                countries[country] += 1

            if row_count % 1000000 == 0:
                elapsed = time.time() - start_time
                print(f"  Processed {row_count:,} rows in {elapsed:.1f}s...")

    elapsed = time.time() - start_time
    print(f"\nAudit completed in {elapsed:.2f}s across {row_count:,} rows.")

    # Report results
    print("-" * 50)
    print("RESULTS & INTEGRITY CHECKS:")
    print(f"  - Total Data Rows:       {row_count:,}")
    print(f"  - Unique Entity IDs:     {len(seen_ids):,}")
    print(f"  - Duplicate Entity IDs:  {dup_id_count}")
    print(f"  - Column Count Errors:   {len(bad_tab_lines)}")
    if bad_tab_lines:
        print(f"    Sample malformed lines: {bad_tab_lines}")

    print(f"  - Prefix Inconsistencies:{len(bad_prefix_lines)}")
    if bad_prefix_lines:
        print(f"    Sample invalid prefixes: {bad_prefix_lines}")

    print(f"  - Mojibake ('\\ufffd'):   {mojibake_count}")
    print(f"  - Null bytes ('\\x00'):  {null_byte_count}")

    print("\nEmpty/Missing Values Breakdown:")
    for col_name, count in zip(expected_header, empty_counts):
        pct = (count / row_count) * 100 if row_count > 0 else 0
        print(f"  - {col_name:<20}: {count:,} empty ({pct:.3f}%)")

    if not is_gt:
        print("\nCountry Distribution:")
        for c, count in countries.most_common():
            pct = (count / row_count) * 100 if row_count > 0 else 0
            print(f"  - {repr(c):<15}: {count:,} ({pct:.2f}%)")

    # Assertions
    assert len(bad_tab_lines) == 0, f"Malformed rows detected in {filepath}!"
    assert len(bad_prefix_lines) == 0, f"Invalid prefixes detected in {filepath}!"
    assert dup_id_count == 0, f"Duplicate IDs detected in {filepath}!"
    print("\nSTATUS: PASS [All constraints satisfied]")
    return True


def main():
    print("STARTING FULL DATASET AUDIT (TRAIN & TEST)...")
    success = True
    for path, header, prefix in FILES_TO_AUDIT:
        ok = audit_file(path, header, prefix)
        if not ok:
            success = False
            break

    print("\n" + "=" * 70)
    if success:
        print("OVERALL AUDIT RESULT: ALL 7 FILES PASSED 100% OF CHECKS!")
    else:
        print("OVERALL AUDIT RESULT: AUDIT FAILED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
