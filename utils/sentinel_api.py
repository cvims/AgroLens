import glob
import os
import shutil
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from zipfile import ZipFile

import boto3
import numpy as np
import rasterio
import requests

from utils.image_utils import ImageUtils


class SentinelApi:
    """
    Methods to query and download Sentinel 2 data
    """

    AUTH_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    API_URL = "https://catalogue.dataspace.copernicus.eu/resto/api/collections/Sentinel2/search.json"
    DOWNLOAD_URL = (
        "https://zipper.dataspace.copernicus.eu/odata/v1/Products(#product_id#)/$value"
    )
    AWS_ENDPOINT_URL = "https://eodata.dataspace.copernicus.eu"
    AWS_BUCKET = "eodata.dataspace.copernicus.eu"

    session = boto3.session.Session()
    s3 = boto3.client(
        "s3",
        endpoint_url=AWS_ENDPOINT_URL,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name="default",
    )

    @classmethod
    def get_data(
        cls,
        date: str,
        latitude: float,
        longitude: float,
        span: int = 14,
        filter_clouds: bool = True,
        limit: int = 1,
    ) -> list[dict]:
        """
        Searches for Sentinel 2 datasets

        Parameters:
            date (str): Date in YYYY-MM-DD format
            latitude (float): GPS latitude
            longitude (float): GPS longitude
            span (int, optional): Maximum span of days to search in each direction (past and future), defaults to 14.
            filter_clouds (bool, optional): Should images with clouds above the GPS position be filtered?, defaults to True.
            limit (int, optional): Maximum number of returned entries, defaults to 1.

        Returns:
            list[dict]: list of Sentinel 2 datasets, ordered by date difference to the search date
        """
        print(f"Getting Sentinel 2 products for {date} ({latitude}, {longitude})...")
        cls.authenticate()

        date = datetime.fromisoformat(date)
        start = date - timedelta(days=span)
        end = date + timedelta(days=span + 1)

        query = (
            f"{cls.API_URL}"
            f"?productType=S2MSI2A"
            f"&startDate={start.isoformat()}Z"
            f"&completionDate={end.isoformat()}Z"
            f"&lat={latitude}"
            f"&lon={longitude}"
            "&radius=500"
            "&sortParam=startDate"
            "&sortOrder=descending"
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

        products = data["features"]
        # sort results by difference to the desired date
        products.sort(key=lambda product: cls._productTimestampDiff(product, date))

        if not filter_clouds:
            return products[:limit]

        result = []
        for product in products:
            if len(result) >= limit:
                break
            if cls._cloud_covered(product, latitude, longitude):
                print(
                    f"Product '{product["id"]}' ({product["properties"]["startDate"]}) has been cloud filtered."
                )
            else:
                result.append(product)

        return result

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
        zipfile = Path(target_path) / f"{dataset_name}.zip"

        cls.authenticate()

        try:
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
                    for chunk in response.iter_content(
                        chunk_size=8388608
                    ):  # 8 MB chunks
                        if chunk:
                            file.write(chunk)
                            if sys.stdout.isatty():
                                # display progress (only in terminal)
                                downloaded += len(chunk)
                                progress = round(downloaded * 100 / total)
                                print(f"  {progress}%", end="\r")

                    print("Download complete")

                    cls._extract(zipfile, Path(target_path) / dataset_name)
        except Exception as e:
            try:
                shutil.rmtree(Path(target_path) / dataset_name, True)
                shutil.rmtree(Path(target_path) / f"{dataset_name}_tmp", True)
                os.remove(zipfile)
            except:
                pass
            raise

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

        print(f"Cropping images in '{path}'...")

        for file in glob.glob("**/*.jp2", root_dir=path, recursive=True):
            file_path = str(path / file)
            ImageUtils.crop_location(file_path, file_path, latitude, longitude, 50)

    @staticmethod
    def _productTimestampDiff(product: dict, date: datetime) -> int:
        """
        Calculates the difference between the product date and the given date in seconds.
        Used to sort the api result list

        Args:
            product (dict): Sentinel 2 product dict
            date (datetime): Desired date

        Returns:
            int: Time difference in seconds
        """
        startDate = datetime.fromisoformat(product["properties"]["startDate"])
        return abs(date.timestamp() - startDate.timestamp())

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
        return int(os.environ["COPERNICUS_TOKEN_EXPIRES"]) > time.time()

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
            zip.extractall(target_path.parent / f"{source_path.stem}_tmp")
            os.remove(source_path)

        # move required files to target directory
        os.mkdir(target_path)
        root = target_path.parent / f"{source_path.stem}_tmp"
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

    @classmethod
    def _cloud_covered(cls, product: dict, latitude: float, longitude: float) -> bool:
        """
        Checks if there is a cloud on the given GPS position of the given product
        by downloading and checking the cloud mask from S3.

        Parameters:
            product (dict): Product dict from the Copernicus API
            latitude (float): GPS latitude
            longitude (float): GPS longitude

        Returns:
            bool: Returns True if there is a cloud on the given position
        """
        image_file = Path(os.environ["TMP_DIR"]) / "cloud_masks" / f"{uuid.uuid4()}.jp2"

        try:
            if not image_file.is_file():
                response = cls.s3.list_objects_v2(
                    Bucket="eodata",
                    Prefix=product["properties"]["productIdentifier"].split("/", 2)[2],
                )

                cloud_mask = False
                if "Contents" in response:
                    for obj in response["Contents"]:
                        if obj["Key"].endswith("/MSK_CLDPRB_20m.jp2"):
                            cloud_mask = obj["Key"]

                os.makedirs(image_file.parent, exist_ok=True)
                cls.s3.download_file("eodata", cloud_mask, image_file)

            x, y = ImageUtils.location_to_pixel(image_file, latitude, longitude)
            result = cls._check_cloud_pixel(image_file, x, y)
        except:
            raise
        finally:
            try:
                os.remove(image_file)
            except:
                pass

        return result

    @staticmethod
    def _check_cloud_pixel(
        input_file: str, x: int, y: int, radius: int = 5, threshold: int = 20
    ) -> bool:
        """
        Checks whether the given pixel and its neighboring pixels are covered by clouds.

        Parameters:
            input_file (str): Path to the input file (GeoTIFF with bands).
            x (int): Pixel x-coordinate
            y (int): Pixel y-coordinate
            radius (int): Pixel radius to check.
            threshold (int): Threshold value for cloud probability.

        Returns:
            bool: True if clouds cover the middle pixel or neighboring pixels, otherwise False.
        """
        with rasterio.open(input_file) as src:
            cloud_band = src.read(1)  # Band 1 for MSK_CLDPRB
            # Select middle pixel and neighboring pixel
            region = cloud_band[
                y - radius : y + radius,
                x - radius : x + radius,
            ]
            # Check whether a pixel is above the threshold value
            cloud_present = np.any(region >= threshold)

        return cloud_present
