import os
import glob
import numpy as np
import pandas as pd

KEEP_LOGS = {"errorsGD", "rankGD", "nuclearrankGD"}
KEEP_MODELS = {"DeepNMF_2W", "DeepNMF_2W_Deepness", "DeepNMF_Article"}


def detect_model_family(fpath: str, meta: dict | None = None) -> str | None:
    """
    Attempt to identify the model family using:
      1) metadata (if a key resembles a model name)
      2) the file path (model_tag, run_name, etc.)

    Returns one of KEEP_MODELS or None.
    """
    # 1) Try using metadata (if such a field exists)
    if meta:
        # Common keys typically found in experiment metadata
        for key in ("model", "model_name", "model_tag", "algo", "method", "function_name"):
            if key in meta:
                val = str(meta[key])
                for m in KEEP_MODELS:
                    if m in val:
                        return m

    # 2) Try using the full file path
    haystack = fpath  # contains model_tag, run_name, and the rest of the path
    for m in KEEP_MODELS:
        if m in haystack:
            return m

    return None


def collect_experiments(root_dir: str) -> pd.DataFrame:
    rows = []

    files = glob.glob(
        os.path.join(root_dir, "**", "factors_and_logs*.npz"),
        recursive=True
    )

    print(f"Found {len(files)} runs (raw)")

    kept = 0
    for fpath in files:
        data = np.load(fpath, allow_pickle=True)

        if "meta" not in data.files:
            # If a file is incomplete, skip it safely
            continue

        meta = data["meta"].item()

        model_family = detect_model_family(fpath, meta)
        if model_family is None:
            continue  # discard this run

        kept += 1
        row = {"model_family": model_family, "run_path": fpath}

        # ---------- METADATA ----------
        # Add all metadata entries as columns
        for k, v in meta.items():
            row[k] = v

        # ---------- LOGS (final values only) ----------
        for log_name in KEEP_LOGS:
            col = f"{log_name}_final"
            if log_name not in data.files:
                row[col] = np.nan
                continue

            arr = np.asarray(data[log_name])

            # Enforce expected shape (T,)
            if arr.ndim != 1 or len(arr) == 0:
                raise ValueError(
                    f"{log_name} in {fpath} has shape {arr.shape}, expected (T,)"
                )

            row[col] = float(arr[-1])

        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Kept {kept} runs after filtering by model family")
    return df


df = collect_experiments("./sweeps")

print(df.head())
print(df.columns)

df.to_csv("summary_runs.csv", index=False)
df.to_latex("summary_runs.tex", index=False)
