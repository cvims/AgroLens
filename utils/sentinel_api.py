from datetime import datetime, timedelta
import glob
import os
from pathlib import Path
import re
import requests
import shutil
import sys
import time
from zipfile import ZipFile
from utils.image_utils import ImageUtils
import rasterio
import numpy as np



class SentinelApi:
    """
    Methods to query and download Sentinel 2 data
    """

    AUTH_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    API_URL = "https://catalogue.dataspace.copernicus.eu/resto/api/collections/Sentinel2/search.json"
    DOWNLOAD_URL = (
        "https://zipper.dataspace.copernicus.eu/odata/v1/Products(#product_id#)/$value"
    )

    @classmethod
    def get_data(
        cls,
        date: str,
        latitude: float,
        longitude: float,
        span: int = 30,
        cloudCover: int = 50,
        limit: int = 1,
    ) -> list[dict]:
        """
        Searches for Sentinel 2 datasets

        Parameters:
            date (str): Date in YYYY-MM-DD format
            latitude (float): GPS latitude
            longitude (float): GPS longitude
            span (int, optional): Maximum span of days to search, defaults to 30.
            cloudCover (int, optional): Maximum percent of cloud coverage in the tile, defaults to 50.
            limit (int, optional): Maximum number of returned entries, defaults to 1.

        Returns:
            list[dict]: list of Sentinel 2 datasets, ordered by newest date
        """
        cls.authenticate()

        end = datetime.fromisoformat(date) + timedelta(days=1)
        start = end - timedelta(days=span)

        query = (
            f"{cls.API_URL}"
            f"?startDate={start.isoformat()}Z"
            f"&completionDate={end.isoformat()}Z"
            f"&maxRecords={limit}"
            f"&lat={latitude}"
            f"&lon={longitude}"
            f"&cloudCover=[0,{cloudCover}]"
            "&sortParam=startDate&sortOrder=descending"
        )

        response = requests.get(
            query,
            allow_redirects=True,
            headers={"Authorization": f"Bearer {os.environ["COPERNICUS_TOKEN"]}"},
        )
        response.raise_for_status()
        data = response.json()

        if not data["features"]:
            print(
                f"No data available for the chosen date and location!", file=sys.stderr
            )
            return None

        return data["features"]

    @classmethod
    def download_data(cls, id: str, dataset_name: str, target_path: str = None) -> None:
        """
        Downloads and extracts a full Sentinel dataset

        Parameters:
            id (str): ID of the dataset (from Sentinel API)
            dataset_name (str): Name under which the data should be saved
            target_path (str, optional): Target path to extract to, defaults to $SENTINEL_DIR.
        """
        if target_path is None:
            target_path = os.environ.get("SENTINEL_DIR", "")

        if (Path(target_path) / dataset_name).exists():
            print(f"The sentinel dataset '{dataset_name}' has already been downloaded.")
            return

        os.makedirs(target_path, exist_ok=True)
        zipfile = Path(target_path) / f"{id}.zip"

        cls.authenticate()

        # download zipfile from server and display progress
        with requests.get(
            cls.DOWNLOAD_URL.replace("#product_id#", id),
            allow_redirects=True,
            stream=True,
            headers={"Authorization": f"Bearer {os.environ["COPERNICUS_TOKEN"]}"},
        ) as response:
            response.raise_for_status()
            with open(zipfile, "wb") as file:
                total = int(response.headers.get("content-length"))
                downloaded = 0
                print(f"Downloading {round(total / 1048576)} MB to '{zipfile}'...")
                for chunk in response.iter_content(chunk_size=16777216):  # 16 MB chunks
                    if chunk:
                        progress = round(downloaded * 100 / total)
                        print(f"  {progress}%", end="\r")
                        file.write(chunk)
                        downloaded += len(chunk)
                print("Download complete")

        cls._extract(zipfile, Path(target_path) / dataset_name)


    @staticmethod
    def crop_images(
        dataset_name: str, latitude: float, longitude: float, parent_path: str = None
    ) -> None:
        """
        Crops all satellite images in the dataset to 100px around the given location to save disk space

        Parameters:
            dataset_name (str): Name of the dataset folder
            latitude (float): GPS latitude
            longitude (float): GPS longitude
            target_path (str, optional): Parent path of the dataset folder, defaults to $SENTINEL_DIR.
        """
        if parent_path is None:
            parent_path = os.environ.get("SENTINEL_DIR", "")

        path = Path(parent_path) / dataset_name / "IMG_DATA"

        for file in glob.glob("**/*.jp2", root_dir=path, recursive=True):
            file_path = str(path / file)
            print(f"Cropping image '{file_path}'...")
            ImageUtils.crop_location(file_path, file_path, latitude, longitude, 50)

    @classmethod
    def authenticate(cls) -> None:
        """
        Authenticates with the Sentinel API using $COPERNICUS_USER and $COPERNICUS_PASSWORD
        and saves the authentication token to $COPERNICUS_TOKEN.
        Authentication is skipped if there is already a valid token.
        """
        if cls._is_authenticated():
            return

        response = requests.post(
            cls.AUTH_URL,
            allow_redirects=True,
            data={
                "username": os.environ["COPERNICUS_USER"],
                "password": os.environ["COPERNICUS_PASSWORD"],
                "client_id": "cdse-public",
                "grant_type": "password",
            },
        )
        response.raise_for_status()
        token = response.json()

        os.environ["COPERNICUS_TOKEN"] = token["access_token"]
        os.environ["COPERNICUS_TOKEN_EXPIRES"] = str(
            round(time.time()) + token["expires_in"] - 10
        )

        print("Copernicus Dataspace authentication successful")

    @staticmethod
    def _is_authenticated() -> bool:
        """
        Checks if the authentication token stored in $COPERNICUS_TOKEN is present and not expired.

        Returns:
            bool: Returns True if the script is already authenticated
        """
        if not os.environ.get("COPERNICUS_TOKEN"):
            return False
        if not os.environ.get("COPERNICUS_TOKEN_EXPIRES"):
            return True
        return int(os.environ["COPERNICUS_TOKEN_EXPIRES"]) < time.time()

    @staticmethod
    def _extract(source_path: Path, target_path: Path) -> None:
        """
        Extracts the contents of the sentinel dataset zipfile, deletes unneccesary files
        and renames the image files for easier access.
        The following directory structure is achieved:
        ├── AUX_DATA
        │   └── <...>
        ├── IMG_DATA
        │   ├── R10m
        │   │   ├── AOT.jp2
        │   │   ├── B02.jp2
        │   │   ├── B03.jp2
        │   │   └── <...>.jp2
        │   ├── R20m
        │   │   └── <...>.jp2
        │   └── R60m
        │       └── <...>.jp2
        ├── QI_DATA
        │   └── <...>
        ├── MTD_MSIL2A.xml
        └── MTD_TL.xml

        Parameters:
            source_path (Path): Path of the downloaded sentinel zipfile
            target_path (Path): Target folder to extract to
        """
        print("Extracting...")
        shutil.rmtree(target_path, True)

        with ZipFile(source_path, "r") as zip:
            zip.extractall(target_path.parent / source_path.stem)
            os.remove(source_path)

        # move required files to target directory
        root = target_path.parent / source_path.stem
        root = root / os.listdir(root)[0]
        path = root / "GRANULE"
        path = path / os.listdir(path)[0]
        os.rename(path, target_path)
        os.rename(root / "MTD_MSIL2A.xml", target_path / "MTD_MSIL2A.xml")

        # rename image files to "<band>.jp2"
        img_path = target_path / "IMG_DATA"
        for dir in os.listdir(img_path):
            for file in os.listdir(img_path / dir):
                band = file.split("_")[2]
                os.rename(
                    img_path / dir / file,
                    img_path / dir / f"{band}.jp2",
                )
        for file in glob.glob("*_PVI.jp2", root_dir=target_path / "QI_DATA"):
            os.rename(
                target_path / "QI_DATA" / file,
                target_path / "QI_DATA" / "PVI.jp2",
            )

        shutil.rmtree(root.parent)

    @staticmethod
    def _check_cloud_pixel(input_file, threshold=20):
        """
            Checks whether the middle pixel and the 5 neighboring pixels are covered by clouds.
            
            Parameters:
                input_file (str): Path to the input file (GeoTIFF with bands).
                threshold (int): Threshold value for cloud probability.
            
            Returns:
                bool: True if clouds cover the middle pixel or neighboring pixels, otherwise False.
        """
        with rasterio.open(input_file) as src:
            # Determine image size and center
            height, width = src.height, src.width
            center_row, center_col = height // 2, width // 2
            
            cloud_band = src.read(1)  # Band 1 for MSK_CLDPRB
            # Select middle pixel and neighboring pixel
            region = cloud_band[center_row-2:center_row+2, center_col-2:center_col+2]
            # Check whether a pixel is above the threshold value
            cloud_present = np.any(region >= threshold)

        return cloud_present

