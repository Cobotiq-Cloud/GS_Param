#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import re
from pathlib import Path

import pandas as pd

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Clean CSV files (or all CSVs in a directory)'
    )
    parser.add_argument('input_path', help='Input CSV file or directory')
    parser.add_argument('-o', '--output', help=(
        'Output cleaned CSV file path (only for single-file mode; '
        'otherwise files are overwritten in-place)'
    ), default=None)
    parser.add_argument('--encoding', help='File encoding', default='utf-8')
    parser.add_argument('--no-header', help='Force no-header mode', action='store_true')
    parser.add_argument('--force-header', help='Force header mode', action='store_true')
    parser.add_argument('--preserve-linebreaks', help='Preserve line breaks in data', action='store_true')
    return parser.parse_args()

def create_generic_headers(n: int) -> list[str]:
    return [f'Column_{i+1}' for i in range(n)]

def dedupe_columns(cols: list[str]) -> list[str]:
    """
    If a name appears multiple times, append _1, _2, ... to make them unique.
    """
    seen = {}
    out = []
    for c in cols:
        if c in seen:
            seen[c] += 1
            out.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            out.append(c)
    return out

def clean_text_content(text: str, preserve_linebreaks: bool = False) -> str:
    """
    Clean text content by removing or replacing problematic characters.
    """
    if pd.isna(text) or text == '':
        return ''
    
    # Convert to string if not already
    text = str(text)
    
    if preserve_linebreaks:
        # Replace different line break types with a single space to preserve readability
        # but prevent CSV structure issues
        text = re.sub(r'\r\n|\r|\n', ' ', text)
    else:
        # Remove all line breaks completely
        text = re.sub(r'\r\n|\r|\n', '', text)
    
    # Remove or replace other problematic characters
    text = re.sub(r'\x00', '', text)  # Remove null characters
    text = re.sub(r'[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)  # Remove other control characters
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def clean_csv_file(input_file: str,
                   output_file: str,
                   encoding: str = 'utf-8',
                   force_no_header: bool = False,
                   force_header: bool = False,
                   preserve_linebreaks: bool = False):
    
    print(f"🔧 Processing: {input_file}")
    
    # 1) Sniff dialect + header
    with open(input_file, 'r', encoding=encoding, newline='') as f:
        sample = f.read(10_000)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.get_dialect('excel')
        has_header = csv.Sniffer().has_header(sample)

    if force_no_header:
        has_header = False
    elif force_header:
        has_header = True

    # 2) Read rows
    with open(input_file, 'r', encoding=encoding, newline='') as f:
        reader = csv.reader(f, dialect)
        all_rows = list(reader)

    if not all_rows:
        print(f"⚠️  {input_file}: no data, skipping.")
        return

    # 3) Max columns
    max_cols = max(len(r) for r in all_rows)

    # 4) Header or generic
    if has_header:
        raw = all_rows[0] + [''] * (max_cols - len(all_rows[0]))
        header = raw[:max_cols]
        data = all_rows[1:]
    else:
        header = create_generic_headers(max_cols)
        data = all_rows

    # 5) Normalize rows
    norm = []
    for r in data:
        r2 = r + [''] * (max_cols - len(r))
        norm.append(r2[:max_cols])

    # 6) Build DF
    df = pd.DataFrame(norm, columns=header, dtype=str)

    # ** Deduplicate any repeated columns so df[col] is always a Series **
    df.columns = dedupe_columns(df.columns.tolist())

    # 7) Clean up - IMPROVED VERSION
    # Handle various representations of missing values
    df.replace({
        'nan': '', 'NaN': '', 'NAN': '',
        'None': '', 'NONE': '', 'none': '',
        'null': '', 'NULL': '', 'Null': '',
        '#N/A': '', '#NA': '', 'N/A': '', 'NA': ''
    }, inplace=True)
    
    # Clean each column's text content
    for col in df.columns:
        df[col] = df[col].apply(lambda x: clean_text_content(x, preserve_linebreaks))
    
    # Remove completely empty rows
    df = df.dropna(how='all')
    df = df[~(df == '').all(axis=1)]
    
    # Reset index after dropping rows
    df = df.reset_index(drop=True)

    # 8) Save (with UTF-8 BOM for Excel)
    df.to_csv(output_file, index=False, encoding='utf-8-sig', lineterminator='\n')
    print(f"✅ Cleaned: {input_file} → {output_file} ({len(df)}×{len(df.columns)})")

def main():
    args = parse_arguments()
    inp = args.input_path

    if os.path.isdir(inp):
        # recursive mode
        csv_count = 0
        for root, _, files in os.walk(inp):
            for fn in files:
                if fn.lower().endswith('.csv'):
                    csv_count += 1
                    path = os.path.join(root, fn)
                    try:
                        clean_csv_file(
                            input_file=path,
                            output_file=path,
                            encoding=args.encoding,
                            force_no_header=args.no_header,
                            force_header=args.force_header,
                            preserve_linebreaks=args.preserve_linebreaks
                        )
                    except Exception as e:
                        print(f"❌ Error processing {path}: {e}")
        print(f"\n🎉 Processed {csv_count} CSV files in directory.")
    else:
        # single-file
        if not os.path.exists(inp):
            print(f"❌ Error: {inp} not found")
            sys.exit(1)
        out = args.output or inp
        try:
            clean_csv_file(
                input_file=inp,
                output_file=out,
                encoding=args.encoding,
                force_no_header=args.no_header,
                force_header=args.force_header,
                preserve_linebreaks=args.preserve_linebreaks
            )
            print("\n🎉 CSV cleaning completed!")
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()