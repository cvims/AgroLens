#!/usr/bin/env python
import argparse
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.sentinel_api import SentinelApi


class ThreadPrefixStd:
    """
    Used to prefix '[Thread Name]' to every output message of a thread.
    """

    def __init__(self, stdout):
        self.prefixes = {}
        self.stdout = stdout
        self.lock = threading.Lock()
        self.line_break = True

    def add_prefix(self, prefix):
        self.prefixes[threading.get_ident()] = prefix

    def write(self, message):
        with self.lock:
            if self.line_break and threading.get_ident() in self.prefixes:
                self.stdout.write(f"{self.prefixes[threading.get_ident()]} {message}")
            else:
                self.stdout.write(message)
            self.line_break = message[-1] == "\n"

    def flush(self):
        self.stdout.flush()

    def isatty(self):
        return self.stdout.isatty()


api_counter = 0
errors = 0
counter_lock = threading.Lock()

download_jobs = []
crop_jobs = []

api_threads = 0
download_threads = 0
crop_threads = 0

dataframe: pd.DataFrame
data_length: 0

api_finished = False
download_finished = False


def setup_parser() -> argparse.Namespace:
    global api_threads, download_threads, crop_threads

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
    args = parser.parse_args()

    api_threads = int(args.api_threads)
    download_threads = int(args.download_threads)
    crop_threads = int(args.worker_threads)

    return args


def api_thread(thread_index: int) -> None:
    global dataframe, data_length, api_counter, errors, download_threads, download_jobs
    sys.stdout.add_prefix(f"[API Thread {thread_index+1}]")
    sys.stderr.add_prefix(f"[API Thread {thread_index+1}]")

    while True:
        if len(download_jobs) > download_threads * 4:
            # do nothing if there is enough work for the download threads
            time.sleep(0.1)
            continue

        with counter_lock:
            index = api_counter
            api_counter += 1

        if index >= data_length:
            print("Thread finished.")
            return

        row = dataframe.iloc[index]
        date = row["SURVEY_DATE"].split("-")
        date = f"20{date[2]}-{date[1]}-{date[0]}"
        latitude = float(row["TH_LAT"])
        longitude = float(row["TH_LONG"])
        print(f"Date: {date}, Lat: {latitude}, Long: {longitude}")

        try:
            data = SentinelApi.get_data(date, latitude, longitude)
            if not data:
                raise ValueError("No Sentinel data found!")

            data = data[0]
            sentinel_date = data["properties"]["startDate"].split("T")[0]
            print(f"Best dataset: {data["id"]} ({sentinel_date})")

            download_jobs.append(
                {
                    "point": str(row["POINTID"]),
                    "id": data["id"],
                    "date": sentinel_date,
                    "lat": latitude,
                    "lon": longitude,
                }
            )
        except Exception as e:
            print(e, file=sys.stderr)
            with counter_lock:
                errors += 1


def download_thread(thread_index: int) -> None:
    global data_length, errors, crop_threads, download_jobs, crop_jobs, api_finished
    sys.stdout.add_prefix(f"[DL Thread {thread_index+1}]")
    sys.stderr.add_prefix(f"[DL Thread {thread_index+1}]")

    while True:
        if not download_jobs and api_finished:
            print("Thread finished.")
            return
        if not download_jobs or len(crop_jobs) > crop_threads * 2:
            # do nothing if there is enough work for the crop threads
            time.sleep(0.1)
            continue

        job = download_jobs.pop(0)

        result_path = (
            Path(os.environ["SENTINEL_DIR"])
            / job["date"]
            / f"{job["lat"]} {job["lon"]}"
        )
        if result_path.is_dir():
            print("Dataset already downloaded")
            continue

        try:
            SentinelApi.download_data(job["id"], job["point"], os.environ["TMP_DIR"])
            crop_jobs.append(job)
        except Exception as e:
            print(e, file=sys.stderr)
            with counter_lock:
                errors += 1


def crop_thread(thread_index: int) -> None:
    global data_length, errors, crop_jobs, download_finished
    sys.stdout.add_prefix(f"[Crop Thread {thread_index+1}]")
    sys.stderr.add_prefix(f"[Crop Thread {thread_index+1}]")

    while True:
        if not crop_jobs:
            if download_finished:
                print("Thread finished.")
                return

            time.sleep(0.1)
            continue

        job = crop_jobs.pop(0)

        result_path = (
            Path(os.environ["SENTINEL_DIR"])
            / job["date"]
            / f"{job["lat"]} {job["lon"]}"
        )

        try:
            SentinelApi.crop_images(
                job["point"], job["lat"], job["lon"], os.environ["TMP_DIR"]
            )
            os.makedirs(Path(os.environ["SENTINEL_DIR"]) / job["date"], exist_ok=True)
            shutil.move(Path(os.environ["TMP_DIR"]) / job["point"], result_path)
            print(f"Point {job["point"]} finished.")
        except Exception as e:
            print(e, file=sys.stderr)
            with counter_lock:
                errors += 1
            shutil.rmtree(Path(os.environ["TMP_DIR"]) / job["point"], True)


def main():
    global dataframe, data_length, api_threads, download_threads, crop_threads, api_finished, download_finished

    args = setup_parser()

    data = pd.read_csv(args.input, sep=",", header=0)

    os.makedirs(os.environ["TMP_DIR"], exist_ok=True)
    os.chdir(os.environ["TMP_DIR"])

    start = int(args.start) - 1
    end = int(args.end)

    if end > 0:
        dataframe = data.iloc[start:end]
    else:
        dataframe = data.iloc[start:]

    data_length = len(dataframe)

    print(f"Loaded {data_length} rows (lines {start+1}-{end or "∞"}).")

    threads_api = []
    for i in range(api_threads):
        threads_api.append(threading.Thread(target=api_thread, args=(i,)))

    threads_download = []
    for i in range(download_threads):
        threads_download.append(threading.Thread(target=download_thread, args=(i,)))

    threads_crop = []
    for i in range(crop_threads):
        threads_crop.append(threading.Thread(target=crop_thread, args=(i,)))

    sys.stdout = ThreadPrefixStd(sys.__stdout__)
    sys.stderr = ThreadPrefixStd(sys.__stderr__)

    for thread in threads_api + threads_download + threads_crop:
        thread.start()

    for thread in threads_api:
        thread.join()
    api_finished = True

    for thread in threads_download:
        thread.join()
    download_finished = True

    for thread in threads_crop:
        thread.join()

    print(f"--- Finished downloading {data_length} Sentinel 2 datasets.")
    print(f"Failed: {errors}")


if __name__ == "__main__":
    main()
