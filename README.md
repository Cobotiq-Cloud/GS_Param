# Parameter Processing Pipeline

This project cleans and translates parameter CSV files and merges them with YAML configuration files to generate finalized datasets.

## Directory Overview

- **Param_CSV/** – Raw Chinese parameter lists. Contains large files such as `Full_Param.csv`.
- **Update/** – Small incremental updates that are not specific to any machine or software version.
- **Translated/** – English translations produced by the pipeline.
- **Results/** – Output from the merge process (`DataProcess_Pipeline.py`).
- **Yaml/** – YAML configuration files describing machines/models.

## Installation

Install dependencies with `pip` (Python 3.8+ recommended):

```bash
pip install pandas googletrans==4.0.0rc1 tqdm PyYAML
```

## Workflow

1. **Clean raw CSVs**
   ```bash
   python3 src/Clean.py Param_CSV
   python3 src/Clean.py Update
   ```
   Each directory is processed recursively and files are overwritten with cleaned versions.

2. **Translate to English**
   ```bash
   mkdir -p Translated
   for file in Param_CSV/*.csv Update/*.csv; do
       name=$(basename "$file")
       python3 src/Translate.py "$file" \
           -o "Translated/${name%.csv}_en.csv" \
           -g glossary_zh2en.txt -c translation_cache.json --skip-existing
   done
   ```
   Translated CSVs are saved under `Translated/` with `_en.csv` suffix.

3. **Merge and match with YAML**
   ```bash
   python3 src/DataProcess_Pipeline.py \
       --base-csv Translated/Full_Param_en.csv \
       --yaml-config Yaml/<model_config.yaml> \
       --translated-dir Translated \
       --output-dir Results
   ```
   This updates `Full_Param_en.csv` with any translated update files and then matches parameters from the selected YAML configuration.

## Automated Pipeline

The repository includes `run_pipeline.sh` which performs all steps above. Edit the variables at the top of the script to choose the YAML file and directories, then execute:

```bash
bash run_pipeline.sh
```

All intermediate and final outputs will be stored under `Translated/` and `Results/`.

