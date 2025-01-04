#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path

import numpy as np
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
        "-n",
        help="Add normalized Sentinel 2 columns?",
        default=False,
        const=True,
        nargs="?",
    )
    parser.add_argument(
        "--flatten",
        "-f",
        help="Flatten Sentinel 2 value matrices into separate columns? (Only useful when 'shape' is a matrix)",
        default=False,
        const=True,
        nargs="?",
    )
    parser.add_argument(
        "--shape",
        "-s",
        help="Define band shape to set the amount of pixels to include (default: 1x1)",
        nargs="?",
        default="1x1",
    )

    args = parser.parse_args()
    # do not flatten a single value
    if args.shape == "1x1":
        args.flatten = False
    elif args.normalize and not args.flatten:
        raise ValueError(
            "You cannot use '--normalize' on a matrix without '--flatten'!"
        )

    return args


def main():
    args = setup_parser()

    sentinel_path = Path(os.environ["SENTINEL_DIR"])

    input = pd.read_csv(args.input, sep=",", header=0)
    mapping = pd.read_csv(args.mapping, sep=",", header=0)

    shape = args.shape.split("x")
    shape = tuple([int(x) for x in shape])

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
    normalize_columns = []
    for band in bands:
        output[band] = None
        if args.flatten:
            for i in range(shape[0] * shape[1]):
                output[f"{band}_{i+1}"] = None
                normalize_columns.append(f"{band}_{i+1}")
        else:
            normalize_columns.append(band)

    for index, row in output.iterrows():
        path = (
            sentinel_path
            / row["SENTINEL_DATE"]
            / f"{row["TH_LAT"]} {row["TH_LONG"]}"
            / "IMG_DATA"
        )

        # Sentinel dataset does not exist, drop row
        try:
            if not path.is_dir():
                drop.append(index)
                continue
        except:
            print(f"Could not open '{path}'!")
            continue

        # iterate over every band and every resolution and try until a value is found
        empty = True
        for band in bands:
            for resolution in resolutions:
                value = None
                file = path / resolution / f"{band}.jp2"
                if file.is_file():
                    try:
                        value = ImageUtils.get_pixel_value(str(file), 50, 50, shape)
                    except:
                        print(
                            f"Could not get value of point {row["POINTID"]}, band {band} ({resolution})!",
                            file=sys.stderr,
                        )
                        pass

                if value is not None:
                    if np.any(value):
                        # drop rows where all bands are zero
                        empty = False
                    output.at[index, band] = value
                    if args.flatten:
                        # copy pixel values from matrix to new columns
                        value = value.reshape(-1)
                        center = int((len(value) - 1) / 2)
                        for i in range(value.shape[0]):
                            # switch center value and first value to make sure that the center
                            # pixel is always in the first column regardless of matrix shape
                            if i == 0:
                                output.at[index, f"{band}_{i+1}"] = value[center]
                            elif i == center:
                                output.at[index, f"{band}_{i+1}"] = value[0]
                            else:
                                output.at[index, f"{band}_{i+1}"] = value[i]
                    break

        # drop rows without any band data
        if empty:
            drop.append(index)
            continue

    # drop all datasets without Sentinel values
    print(f"Dropping {len(drop)} rows...")
    output.drop(output.index[drop], inplace=True)

    # drop redundant second index and create new index
    output.drop(output.columns[0], axis=1, inplace=True)
    output.reset_index(drop=True, inplace=True)

    if args.normalize:
        scaler = preprocessing.MinMaxScaler()
        normalizer = scaler.fit_transform(output[normalize_columns])
        normalized = pd.DataFrame(normalizer)
        names = [i + "_normalized" for i in normalize_columns]
        normalized.columns = names
        output = pd.merge(output, normalized, left_index=True, right_index=True)

    if shape != (1, 1):
        # convert numpy arrays to strings
        output[bands] = output[bands].astype(str)
        output.replace("\n", "", regex=True, inplace=True)

    output.to_csv(args.output, sep=",")
    print(f"New data table saved to '{args.output}'.")


if __name__ == "__main__":
    main()
