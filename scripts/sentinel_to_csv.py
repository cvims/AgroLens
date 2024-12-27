#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn import preprocessing

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.image_utils import ImageUtils


def setup_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adds Sentinel 2 band pixel data to the given CSV table \n\n"
        "Required environment variables: \n"
        "SENTINEL_DIR: Directory with the cropped Sentinel 2 files",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", help="Path to input LUCAS SOIL CSV file", required=True
    )
    parser.add_argument(
        "--mapping", "-m", help="Path to the mapping CSV", required=True
    )
    parser.add_argument(
        "--output", "-o", help="Path to save the output CSV", required=True
    )
    parser.add_argument(
        "--normalize",
        help="Add normalized Sentinel 2 columns?",
        default=False,
        const=True,
        nargs="?",
    )
    return parser.parse_args()


def main():
    args = setup_parser()

    sentinel_path = Path(os.environ["SENTINEL_DIR"])

    input = pd.read_csv(args.input, sep=",", header=0)
    mapping = pd.read_csv(args.mapping, sep=",", header=0)

    # drop duplicate columns
    mapping.drop(columns=["SURVEY_DATE", "TH_LAT", "TH_LONG"], inplace=True)

    output = pd.merge(input, mapping, on="POINTID", how="inner")

    bands = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B11",
        "B12",
    ]

    resolutions = ["R10m", "R20m", "R60m"]

    drop = []

    # create columns in output dataset
    for band in bands:
        output[band] = None

    for index, row in output.iterrows():
        path = (
            sentinel_path
            / row["SENTINEL_DATE"]
            / f"{row["TH_LAT"]} {row["TH_LONG"]}"
            / "IMG_DATA"
        )

        # Sentinel dataset does not exist, drop row
        if not path.is_dir():
            drop.append(index)
            continue

        # iterate over every band and every resolution and try until a value is found
        for band in bands:
            for resolution in resolutions:
                value = -1
                file = path / resolution / f"{band}.jp2"
                if file.is_file():
                    try:
                        value = ImageUtils.get_pixel_value(str(file), 50, 50)
                    except:
                        print(
                            f"Could not get value of point {row["POINTID"]}, band {band} ({resolution})!",
                            file=sys.stderr,
                        )
                        pass
                if value >= 0:
                    output.at[index, band] = value
                    break

    # drop all datasets without Sentinel values
    print(f"Dropping {len(drop)} rows...")
    output.drop(output.index[drop], inplace=True)

    if args.normalize:
        scaler = preprocessing.MinMaxScaler()
        normalizer = scaler.fit_transform(output[bands])
        normalized = pd.DataFrame(normalizer)
        names = [i + "_normalized" for i in bands]
        normalized.columns = names
        output = pd.merge(output, normalized, left_index=True, right_index=True)
        # drop redundant second index
        output.drop(output.columns[0], axis=1, inplace=True)

    output.to_csv(args.output, sep=",")
    print(f"New data table saved to '{args.output}'.")


if __name__ == "__main__":
    main()
