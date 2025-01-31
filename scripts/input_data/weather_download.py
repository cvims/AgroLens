#!/usr/bin/env python
# Queries and adds Open Weather Map data to a given soil data CSV
import csv
import os
from datetime import datetime

import requests

# Find key in https://home.openweathermap.org/api_keys
API_KEY = input("Open Weather Map API Key: ")
CSV_PATH = "/media/data/Datasets/Model_A+_Soil+Sentinel_v4_with_weather.csv"
FULL_DAY_SECONDS = 86400  # 24*60*60s

# List of all OW_ columns we wish to populate
OW_FIELDS = [
    "OW_lat",
    "OW_lon",
    "OW_timezone",
    "OW_timezone_offset",
    "OW_clouds",
    "OW_dew_point",
    "OW_feels_like",
    "OW_humidity",
    "OW_pressure",
    "OW_sunrise",
    "OW_sunset",
    "OW_temp",
    "OW_wind_deg",
    "OW_wind_speed",
    "OW_day_length",
]


def all_ow_fields_empty_or_zero(row, ow_fields):
    """
    Returns True if *every* OW_ field in 'row' is either empty ("") or "0".
    Otherwise returns False.
    """
    for field in ow_fields:
        val = str(row.get(field, "")).strip()
        if val != "" and val != "0":
            return False
    return True


def main():
    """
    This script reads a CSV file, makes an API call to OpenWeather for each row
    ONLY IF all existing OW_ fields are empty or "0", appends weather data,
    and writes a new CSV with the combined data.
    """

    # -----------------------------
    # Read the CSV file into memory
    # -----------------------------
    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    row_count = len(rows)
    print(f"Found {row_count} rows in {CSV_PATH}.")

    # ----------------------------------------------------------
    # Prepare output CSV path (e.g. "whatever_with_weather.csv")
    # ----------------------------------------------------------
    base, ext = os.path.splitext(CSV_PATH)
    output_csv_path = f"{base}_with_weather{ext}"

    # The new CSV will have the original columns plus the OW_ ones.
    # Collect the original fieldnames:
    new_fieldnames = list(rows[0].keys())

    # Add any missing OW_ fields to the fieldnames
    for ow_col in OW_FIELDS:
        if ow_col not in new_fieldnames:
            new_fieldnames.append(ow_col)

    # -------------------------------
    # Open the output CSV for writing
    # -------------------------------
    with open(output_csv_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=new_fieldnames)
        writer.writeheader()

        # ---------------------------------------
        # Process each row and make the API calls
        # ---------------------------------------
        for index, row in enumerate(rows, start=1):

            date_str = row.get("SENTINEL_DATE", "").strip()
            pid = row.get("POINTID", "")

            # 1) Check we have a valid date string
            if not date_str:
                print(f"[{index}] POINTID {pid} Missing SENTINEL_DATE, skipping.")
                writer.writerow(row)  # Write the row without weather data
                continue

            # Convert "YYYY-MM-DD" to a datetime object
            try:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                print(
                    f"[{index}] POINTID {pid} Invalid date format for SENTINEL_DATE={date_str}, skipping."
                )
                writer.writerow(row)  # Write the row as-is
                continue

            # 2) Decide whether to call the API or not
            #    We only call if *all* OW_ fields are empty or "0"
            if not all_ow_fields_empty_or_zero(row, OW_FIELDS):
                print(
                    f"[{index}] POINTID {pid}: OW fields already filled. Skipping API call."
                )
                writer.writerow(row)
                continue

            # If we get here, all OW fields are empty or "0" => we call the API

            # Convert to Unix timestamp (seconds since 1970-01-01 UTC); use noon for robustness
            timestamp_unix = int(date_obj.timestamp()) + int(FULL_DAY_SECONDS / 2)

            # 3) Get lat/lon from the row
            try:
                lat = float(row["TH_LAT"])
                lon = float(row["TH_LONG"])
            except (ValueError, KeyError) as e:
                print(f"[{index}] POINTID {pid} Missing or invalid TH_LAT/TH_LONG: {e}")
                writer.writerow(row)  # Write the row as-is
                continue

            # 4) Construct the OpenWeather "timemachine" URL
            url = (
                "https://api.openweathermap.org/data/3.0/onecall/timemachine"
                f"?lat={lat:.5f}"
                f"&lon={lon:.5f}"
                f"&dt={timestamp_unix}"
                f"&appid={API_KEY}"
            )

            print(f"[{index}] POINTID {pid} Request => {url}")

            # 5) Send the GET request
            try:
                response = requests.get(url, timeout=30)
                if not response.ok:
                    print(
                        f"[{index}] POINTID {pid} ERROR {response.status_code}: {response.text}"
                    )
                    # We can still write the original row or skip
                    writer.writerow(row)
                    continue

                # ------------------------------------------------
                # If successful, parse the JSON and extract fields
                # ------------------------------------------------
                data_json = response.json()

                # lat/lon/timezone/timezone_offset
                row["OW_lat"] = data_json.get("lat", "0")
                row["OW_lon"] = data_json.get("lon", "0")
                row["OW_timezone"] = data_json.get("timezone", "0")
                row["OW_timezone_offset"] = data_json.get("timezone_offset", "0")

                data_list = data_json.get("data", [])
                if len(data_list) > 0:
                    first_block = data_list[0]
                    row["OW_clouds"] = first_block.get("clouds", "0")
                    row["OW_dew_point"] = first_block.get("dew_point", "0")
                    row["OW_feels_like"] = first_block.get("feels_like", "0")
                    row["OW_humidity"] = first_block.get("humidity", "0")
                    row["OW_pressure"] = first_block.get("pressure", "0")
                    row["OW_sunrise"] = first_block.get("sunrise", "0")
                    row["OW_sunset"] = first_block.get("sunset", "0")
                    row["OW_temp"] = first_block.get("temp", "0")
                    row["OW_wind_deg"] = first_block.get("wind_deg", "0")
                    row["OW_wind_speed"] = first_block.get("wind_speed", "0")

                    sunrise = first_block.get("sunrise")
                    sunset = first_block.get("sunset")

                    # Compute day length if both sunrise and sunset are present
                    if sunrise and sunset:
                        try:
                            row["OW_day_length"] = str(int(sunset) - int(sunrise))
                        except (ValueError, TypeError):
                            row["OW_day_length"] = "0"
                    else:
                        row["OW_day_length"] = "0"

                else:
                    print(f"[{index}] POINTID {pid} Warning: 'data' array is empty.")
                    # Optionally set fields to 0
                    for field in OW_FIELDS:
                        row[field] = "0"

            except requests.exceptions.RequestException as e:
                print(f"[{index}] POINTID {pid} REQUEST ERROR: {e}")
                # Decide whether to skip or write the row without weather data
                writer.writerow(row)
                continue

            # 6) Write out the updated row with appended data
            writer.writerow(row)

    print(f"Done. Output written to {output_csv_path}.")


if __name__ == "__main__":
    main()
