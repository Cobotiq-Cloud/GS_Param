#!/usr/bin/env bash
# Example automated workflow for parameter processing
set -euo pipefail

RAW_DIR="Param_CSV"
UPDATE_DIR="Update"
TRANSLATED_DIR="Translated"
OUTPUT_DIR="Results"
YAML_CONFIG="Yaml/A3-00-8p-20241025_user_config.yaml"
CACHE_FILE="translation_cache.json"
GLOSSARY_FILE="glossary_zh2en.txt"

# Step 1: clean all CSV files
python3 src/Clean.py "$RAW_DIR"
python3 src/Clean.py "$UPDATE_DIR"

# Step 2: translate cleaned CSVs to English
mkdir -p "$TRANSLATED_DIR"
for csv in "$RAW_DIR"/*.csv "$UPDATE_DIR"/*.csv; do
    [ -f "$csv" ] || continue
    name=$(basename "$csv")
    out="$TRANSLATED_DIR/${name%.csv}_en.csv"
    python3 src/Translate.py "$csv" -o "$out" \
        -g "$GLOSSARY_FILE" -c "$CACHE_FILE" --skip-existing
done

# Step 3: merge with translations and YAML configuration
python3 src/DataProcess_Pipeline.py \
    --base-csv "$TRANSLATED_DIR/Full_Param_en.csv" \
    --yaml-config "$YAML_CONFIG" \
    --translated-dir "$TRANSLATED_DIR" \
    --output-dir "$OUTPUT_DIR"
