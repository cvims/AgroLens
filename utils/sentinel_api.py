from datetime import datetime, timedelta
import glob
import os
from pathlib import Path
import requests
import shutil
import sys
import time
from zipfile import ZipFile

from utils.image_utils import ImageUtils


class SentinelApi:
    AUTH_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    API_URL = "https://catalogue.dataspace.copernicus.eu/resto/api/collections/Sentinel2/search.json"
    DOWNLOAD_URL = (
        "https://zipper.dataspace.copernicus.eu/odata/v1/Products(#product_id#)/$value"
    )

    @staticmethod
    def get_data(
        date: str,
        latitude: float,
        longitude: float,
        span: int = 30,
        cloudCover: int = 50,
        limit: int = 1,
    ) -> dict:
        __class__.authenticate()

        end = datetime.fromisoformat(date) + timedelta(days=1)
        start = end - timedelta(days=span)

        query = (
            f"{__class__.API_URL}"
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

    @staticmethod
    def download_data(id: str, dataset_name: str, target_path: str = None) -> None:
        if target_path is None:
            target_path = os.environ.get("SENTINEL_DIR", "")

        if (Path(target_path) / dataset_name).exists():
            print(f"The sentinel dataset '{dataset_name}' has already been downloaded.")
            return

        os.makedirs(target_path, exist_ok=True)
        zipfile = Path(target_path) / f"{id}.zip"

        __class__.authenticate()

        with requests.get(
            __class__.DOWNLOAD_URL.replace("#product_id#", id),
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

        __class__._extract(zipfile, Path(target_path) / dataset_name)

    @staticmethod
    def crop_images(
        dataset_name: str, latitude: float, longitude: float, target_path: str = None
    ) -> None:
        if target_path is None:
            target_path = os.environ.get("SENTINEL_DIR", "")

        path = Path(target_path) / dataset_name / "IMG_DATA"

        for file in glob.glob("**/*.jp2", root_dir=path, recursive=True):
            file_path = str(path / file)
            print(f"Cropping image '{file_path}'...")
            ImageUtils.crop_location(file_path, file_path, latitude, longitude, 50)

    @staticmethod
    def authenticate() -> None:
        if __class__._is_authenticated():
            return

        response = requests.post(
            __class__.AUTH_URL,
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
    def _is_authenticated() -> None:
        if not os.environ.get("COPERNICUS_TOKEN"):
            return False
        if not os.environ.get("COPERNICUS_TOKEN_EXPIRES"):
            return True
        return int(os.environ["COPERNICUS_TOKEN_EXPIRES"]) < time.time()

    @staticmethod
    def _extract(source_path: Path, target_path: Path) -> None:
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
