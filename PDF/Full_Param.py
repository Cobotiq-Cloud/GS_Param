"""
extract_40_A4_params.py
=======================

Generate a standalone CSV that lists every parameter used by **Machine 40** running **Software A4**, together with its
current value (from the YAML), Chinese name, English explanation, and any extra metadata that lives in the master
parameter definition sheet.

**Source files**
---------------
* YAML config (values)  : `/Users/chenzaowen/Desktop/GS_Param/Param_CSV/A4-00-14p-20250417 _user_config.yaml`
* Master definitions CSV: `/Users/chenzaowen/Desktop/GS_Param/Param_CSV/GS Param - Full_Param.csv`

If the master‐definition file sits somewhere else, just tweak the constant `FULL_PARAM_CSV` below.  
(We _don’t_ need the machine‑specific CSV `GS Param - 40_A4_10.csv` for this merge because the YAML already holds the
live A4 values.)

The script writes its merged output to:
`/Users/chenzaowen/Desktop/GS_Param/Param_CSV/40_A4_params_full.csv`

Run it with:  
```bash
python extract_40_A4_params.py
```
"""

from __future__ import annotations

import os
import yaml
import pandas as pd
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# ⇢⇢⇢  CONFIGURE PATHS  ⇠⇠⇠
# ---------------------------------------------------------------------------
YAML_PATH = "/Users/chenzaowen/Desktop/GS_Param/Param_CSV/A4-00-14p-20250417 _user_config.yaml"
FULL_PARAM_CSV = "/Users/chenzaowen/Desktop/GS_Param/Param_CSV/Full_Param.csv"
OUTPUT_PATH = "/Users/chenzaowen/Desktop/GS_Param/Param_CSV/40_A4_params_full.csv"

# ---------------------------------------------------------------------------
# ⇢⇢⇢  HELPERS  ⇠⇠⇠
# ---------------------------------------------------------------------------

def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = "/") -> List[Dict[str, Any]]:
    """Recursively flattens a nested dictionary.

    Each leaf is returned as a dict with keys ``path`` and ``value``.  Paths are **slash‑delimited** and
    always start with a leading ``/``.
    """
    items: List[Dict[str, Any]] = []
    for key, value in d.items():
        full_key = f"{parent_key}{sep if parent_key else '/'}{key}"
        if isinstance(value, dict):
            items.extend(flatten_dict(value, full_key, sep=sep))
        else:
            items.append({"path": full_key, "value": value})
    return items


def normalise_path(p: str) -> str:
    """Ensure every parameter path is **slash‑delimited** and starts with a leading slash."""
    p = str(p).strip()
    return p if p.startswith("/") else f"/{p}"

# ---------------------------------------------------------------------------
# ⇢⇢⇢  MAIN  ⇠⇠⇠
# ---------------------------------------------------------------------------

def main() -> None:
    # 1️⃣  Load YAML and flatten
    if not os.path.exists(YAML_PATH):
        raise FileNotFoundError(f"YAML config not found: {YAML_PATH}")

    with open(YAML_PATH, "r", encoding="utf-8") as fh:
        yaml_cfg = yaml.safe_load(fh)

    flat_rows = flatten_dict(yaml_cfg)
    df_flat = pd.DataFrame(flat_rows)

    # 2️⃣  Load master definitions CSV
    if not os.path.exists(FULL_PARAM_CSV):
        raise FileNotFoundError(f"Master definition CSV not found: {FULL_PARAM_CSV}")

    df_full = pd.read_csv(FULL_PARAM_CSV, dtype=str, encoding="utf-8")

    # Identify which column holds the parameter path.  It’s often called something like
    # "1参数位置和名称" but we’ll search heuristically just in case.
    path_col_candidates = [
        col for col in df_full.columns if "参数" in col or col.lower().startswith("path")
    ]
    if not path_col_candidates:
        raise RuntimeError(
            "Could not identify the column containing parameter paths in the master CSV."
        )
    path_col = path_col_candidates[0]

    # Normalise and rename
    df_full[path_col] = df_full[path_col].apply(normalise_path)
    df_full = df_full.rename(columns={path_col: "path"})

    # 3️⃣  Merge values + definitions (left join keeps every YAML param even if definition is missing)
    df_merged = (
        df_flat.merge(df_full, on="path", how="left", suffixes=("", "_def"))
        .assign(machine="40", software_version="A4")
    )

    # 4️⃣  Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_merged.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"✔  Merged parameter sheet written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
