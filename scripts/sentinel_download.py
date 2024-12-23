#!/usr/bin/env python
import argparse
import os
import shutil
import sys
import traceback
from pathlib import Path

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.sentinel_api import SentinelApi


def setup_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Downloads and crops all Sentinel 2 images in the given CSV table. \n\n"
        "Required environment variables: \n"
        "SENTINEL_DIR: Target directory for the cropped Sentinel 2 files \n"
        "TMP_DIR: Temporary working directory (ramdisk recommended) \n"
        "COPERNICUS_USER: Copernicus API user \n"
        "COPERNICUS_PASSWORD: Copernicus API password \n"
        "AWS_ACCESS_KEY_ID: Copernicus S3 key ID \n"
        "AWS_SECRET_ACCESS_KEY: Copernicus S3 secret \n",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--api-threads",
        "-a",
        help="Number of Copernicus API threads",
        nargs="?",
        default="4",
    )
    parser.add_argument(
        "--download-threads",
        "-d",
        help="Number of download threads",
        nargs="?",
        default="4",
    )
    parser.add_argument(
        "--worker-threads",
        "-w",
        help="Number of CPU worker threads",
        nargs="?",
        default="4",
    )
    parser.add_argument(
        "--start", help="First CSV row to download", nargs="?", default="1"
    )
    parser.add_argument(
        "--end", help="Last CSV row to download", nargs="?", default="0"
    )
    parser.add_argument(
        "--input", "-f", help="Path to input LUCAS SOIL CSV file", required=True
    )
    return parser.parse_args()


def download_row(row: pd.Series, target_dir: Path) -> None:
    date = row["SURVEY_DATE"].split("-")
    date = f"20{date[2]}-{date[1]}-{date[0]}"
    latitude = float(row["TH_LAT"])
    longitude = float(row["TH_LONG"])
    print(f"Date: {date}, Lat: {latitude}, Long: {longitude}")

    data = SentinelApi.get_data(date, latitude, longitude)
    if not data:
        raise ValueError("No Sentinel data found!")

    data = data[0]
    sentinel_date = data["properties"]["startDate"].split("T")[0]
    print(f"Best dataset: {data["id"]} ({sentinel_date})")

    result_path = (
        Path(os.environ["SENTINEL_DIR"]) / sentinel_date / f"{latitude} {longitude}"
    )
    if result_path.is_dir():
        print("Dataset already downloaded")
        return

    SentinelApi.download_data(data["id"], str(row["POINTID"]), os.environ["TMP_DIR"])
    SentinelApi.crop_images(
        str(row["POINTID"]), latitude, longitude, os.environ["TMP_DIR"]
    )

    os.makedirs(Path(os.environ["SENTINEL_DIR"]) / sentinel_date, exist_ok=True)
    shutil.move(Path(os.environ["TMP_DIR"]) / str(row["POINTID"]), result_path)


def main():
    args = setup_parser()

    data = pd.read_csv(args.input, sep=",", header=0)

    target_dir = Path(os.environ["SENTINEL_DIR"])
    os.makedirs(os.environ["TMP_DIR"], exist_ok=True)
    os.chdir(os.environ["TMP_DIR"])

    start = int(args.start) - 1
    end = int(args.end)

    if end > 0:
        data = data.iloc[start:end]
    else:
        data = data.iloc[start:]

    print(f"Loaded {len(data)} rows (lines {start+1}-{end or "∞"}).")

    errors = 0

    for index, row in data.iterrows():
        try:
            print(f"--- Downloading point {row["POINTID"]} (row #{index})...")
            download_row(row, target_dir)
        except KeyboardInterrupt:
            sys.exit(0)
        except:
            traceback.print_exc()
            errors += 1

    print(f"--- Finished downloading {len(data)} Sentinel 2 datasets.")
    print(f"Failed: {errors}")


if __name__ == "__main__":
    main()
