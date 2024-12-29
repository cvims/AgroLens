#!/usr/bin/env python3

import csv
import os
from datetime import datetime

import requests

API_KEY = input("API Key: ")
CSV_PATH = "/media/data/Datasets/Model_A_Dataset_v2_2024-12-28.csv"
FULL_DAY_SECONDS = 86400  # 24*60*60s

def main():
    """
    This script reads a CSV file, makes an API call to OpenWeather for each row,
    appends weather data (including the delta time from sunrise to sunset),
    and writes a new CSV with the combined data.
    """
    # ----------------------------
    # Read the CSV file into memory
    # ----------------------------
    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    row_count = len(rows)
    print(f"Found {row_count} rows in {CSV_PATH}.")

    # -----------------------------------------------------------------
    # By default, let's consider ONLY the first row to save on API calls.
    # Comment out this line if you want to process ALL rows.
    # -----------------------------------------------------------------
    # rows = [rows[0]]  # <-- comment or remove this line to iterate over every row
    # -----------------------------------------------------------------

    # --------------------------------------------
    # Prepare output CSV path (e.g. "whatever_with_weather.csv")
    # --------------------------------------------
    base, ext = os.path.splitext(CSV_PATH)
    output_csv_path = f"{base}_with_weather{ext}"

    # We’ll define extra field names that we plan to append to each row.
    # For example, we'll store:
    #   - OpenWeather lat/lon/timezone/timezone_offset
    #   - (first) block from the "data" array: clouds, temp, sunrise, sunset, wind_speed, ...
    #   - day_length (sunset - sunrise)
    #
    # If you want to store more fields from the API, just add them here or dynamically inside the loop.
    extra_fields = [
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
        "OW_day_length"  # This is the delta time from sunrise to sunset
    ]

    # The new CSV will have the original columns plus the new ones:
    new_fieldnames = rows[0].keys()  # original columns
    # Convert to a list so we can append easily
    new_fieldnames = list(new_fieldnames)
    new_fieldnames.extend(extra_fields)

    # ----------------------------
    # Open the output CSV for writing
    # ----------------------------
    with open(output_csv_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=new_fieldnames)
        writer.writeheader()

        # ----------------------------------------
        # Process each row and make the API calls
        # ----------------------------------------
        for index, row in enumerate(rows, start=1):
            date_str = row.get("SENTINEL_DATE", "").strip()

            # 1) Check we have a valid date string
            if not date_str:
                print(f"[{index}] Missing SENTINEL_DATE, skipping.")
                writer.writerow(row)  # Write the row without weather data
                continue

            # Convert "YYYY-MM-DD" to a datetime object
            try:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                print(f"[{index}] Invalid date format for SENTINEL_DATE={date_str}, skipping.")
                writer.writerow(row)  # Write the row as-is
                continue

            # Convert to Unix timestamp (seconds since 1970-01-01 UTC); use noon for robustness
            timestamp_unix = int(date_obj.timestamp()) + int(FULL_DAY_SECONDS / 2)

            # 2) Get lat/lon from the row
            try:
                lat = float(row["TH_LAT"])
                lon = float(row["TH_LONG"])
            except (ValueError, KeyError) as e:
                print(f"[{index}] Missing or invalid TH_LAT/TH_LONG: {e}")
                writer.writerow(row)  # Write the row as-is
                continue

            # 3) Construct the OpenWeather "timemachine" URL
            url = (
                "https://api.openweathermap.org/data/3.0/onecall/timemachine"
                f"?lat={lat:.5f}"
                f"&lon={lon:.5f}"
                f"&dt={timestamp_unix}"
                f"&appid={API_KEY}"
            )

            print(f"[{index}] POINTID {row.get('POINTID','')} Request => {url}")

            # 4) Send the GET request
            try:
                response = requests.get(url, timeout=30)
                if not response.ok:
                    print(f"[{index}] POINTID {row.get('POINTID','')} ERROR {response.status_code}: {response.text}")
                    # We can still write the original row or skip
                    writer.writerow(row)
                    continue

                # ------------------------------------------------
                # If successful, parse the JSON and extract fields
                # ------------------------------------------------
                data_json = response.json()

                # lat/lon/timezone/timezone_offset
                row["OW_lat"] = data_json.get("lat")
                row["OW_lon"] = data_json.get("lon")
                row["OW_timezone"] = data_json.get("timezone")
                row["OW_timezone_offset"] = data_json.get("timezone_offset")

                # We'll assume there's at least one item in data.
                data_list = data_json.get("data", [])
                if len(data_list) > 0:
                    first_block = data_list[0]
                    row["OW_clouds"]      = first_block.get("clouds")
                    row["OW_dew_point"]   = first_block.get("dew_point")
                    row["OW_feels_like"]  = first_block.get("feels_like")
                    row["OW_humidity"]    = first_block.get("humidity")
                    row["OW_pressure"]    = first_block.get("pressure")
                    row["OW_sunrise"]     = first_block.get("sunrise")
                    row["OW_sunset"]      = first_block.get("sunset")
                    row["OW_temp"]        = first_block.get("temp")
                    row["OW_wind_deg"]    = first_block.get("wind_deg")
                    row["OW_wind_speed"]  = first_block.get("wind_speed")

                    sunrise = first_block.get("sunrise")
                    sunset  = first_block.get("sunset")

                    # Compute day length if both sunrise and sunset are present
                    if sunrise and sunset:
                        row["OW_day_length"] = sunset - sunrise
                    else:
                        row["OW_day_length"] = ""

                else:
                    print(f"[{index}] POINTID {row.get('POINTID','')} Warning: 'data' array is empty.")
                    # Optionally set fields to empty or None
                    for field in extra_fields:
                        row[field] = ""

            except requests.exceptions.RequestException as e:
                print(f"[{index}] POINTID {row.get('POINTID','')} REQUEST ERROR: {e}")
                # Decide whether to skip or write the row without weather data
                writer.writerow(row)
                continue

            # 5) Write out the updated row with appended data
            writer.writerow(row)

    print(f"Done. Output written to {output_csv_path}.")

if __name__ == "__main__":
    main()
