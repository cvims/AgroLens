#!/usr/bin/env python
# Merge weather data (columns starting with "OW_") from PARTIAL_WEATHER_PATH
# into INPUT_PATH, keyed by POINTID. Any missing or partial values become 0.

import csv
import os
from pathlib import Path

# Adjust these paths as needed:
PARTIAL_WEATHER_PATH = (
    Path(os.environ["DATASET_PATH"])
    / "Model_A+_Soil+Sentinel_v3_with_weather+yield.csv"
)
INPUT_PATH = Path(os.environ["DATASET_PATH"]) / "Model_A+_Soil+Sentinel_v4.csv"
OUTPUT_PATH = (
    Path(os.environ["DATASET_PATH"]) / "Model_A+_Soil+Sentinel_v4_with_weather.csv"
)


def main():
    # ---------------------------------------------------------
    # 1) Read the "partial" file into a dictionary keyed by POINTID
    # ---------------------------------------------------------
    with open(PARTIAL_WEATHER_PATH, "r", encoding="utf-8", newline="") as partial_f:
        partial_reader = csv.DictReader(partial_f)
        partial_rows = list(partial_reader)

        # Identify all columns that start with "OW_"
        ow_cols = [col for col in partial_reader.fieldnames if col.startswith("OW_")]

        # Build a lookup dict: {POINTID: row_data_dict}
        # (We assume "POINTID" is present in the partial file)
        partial_lookup = {}
        for row in partial_rows:
            pid = row.get("POINTID", "").strip()
            if not pid:
                continue  # skip rows without a valid POINTID
            partial_lookup[pid] = row

    # ---------------------------------------------------------
    # 2) Read the "new" file (v3) and prepare for writing
    # ---------------------------------------------------------
    with open(INPUT_PATH, "r", encoding="utf-8", newline="") as v3_f:
        v3_reader = csv.DictReader(v3_f)
        v3_rows = list(v3_reader)
        v3_columns = v3_reader.fieldnames

        # Merge original columns with the OW_ columns
        # (while preserving original order and appending new ones at the end)
        combined_columns = list(v3_columns)
        for col in ow_cols:
            if col not in combined_columns:
                combined_columns.append(col)

    # ---------------------------------------------------------
    # 3) Write out the merged CSV
    # ---------------------------------------------------------
    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=combined_columns)
        writer.writeheader()

        for row in v3_rows:
            pid = row.get("POINTID", "").strip()

            # If we find that PID in partial lookup, pull the weather columns from there
            if pid in partial_lookup:
                partial_row = partial_lookup[pid]
                # For each OW_ column, copy from partial_row or fill in 0 if missing
                for col in ow_cols:
                    value = partial_row.get(col, "0")
                    # If the partial file has an empty string, also treat that as 0
                    if value in ("", None):
                        value = "0"
                    row[col] = value
            else:
                # If the PID is not in the partial file, set all OW_ columns to 0
                for col in ow_cols:
                    row[col] = "0"

            writer.writerow(row)

    print(f"Done. Created {OUTPUT_PATH} with weather columns merged.")


if __name__ == "__main__":
    main()
