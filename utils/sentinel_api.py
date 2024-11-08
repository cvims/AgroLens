from datetime import datetime, timedelta
import os
import requests
import sys
import time


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
        cloudCover: int = 30,
        limit: int = 1,
    ):
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
    def download_data(id: str, target_path: str):
        with requests.get(
            __class__.DOWNLOAD_URL.replace("#product_id#", id),
            allow_redirects=True,
            stream=True,
            headers={"Authorization": f"Bearer {os.environ["COPERNICUS_TOKEN"]}"},
        ) as response:
            response.raise_for_status()
            with open(target_path, "wb") as file:
                total = int(response.headers.get("content-length"))
                downloaded = 0
                print(f"Downloading {round(total / 1048576)} MB to '{target_path}'...")
                for chunk in response.iter_content(chunk_size=16777216):  # 16 MB chunks
                    if chunk:
                        progress = round(downloaded * 100 / total)
                        print(f"  {progress}%", end="\r")
                        file.write(chunk)
                        downloaded += len(chunk)
                print("Download complete")

    @staticmethod
    def authenticate():
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
    def _is_authenticated():
        if not os.environ.get("COPERNICUS_TOKEN"):
            return False
        if not os.environ.get("COPERNICUS_TOKEN_EXPIRES"):
            return True
        return int(os.environ["COPERNICUS_TOKEN_EXPIRES"]) < time.time()
