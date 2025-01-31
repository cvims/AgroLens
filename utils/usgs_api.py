import os
import sys
import time
from datetime import datetime, timedelta

import requests


class USGSApi:
    """
    Methods to query Landsat satellite data via the USGS M2M API
    """

    AUTH_URL = "https://m2m.cr.usgs.gov/api/api/json/stable/login-token"
    API_URL = "https://m2m.cr.usgs.gov/api/api/json/stable/scene-search"

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
        Searches for Landsat 7 datasets

        Parameters:
            date (str): Date in YYYY-MM-DD format
            latitude (float): GPS latitude
            longitude (float): GPS longitude
            span (int, optional): Maximum span of days to search in each direction (past and future), defaults to 14.
            filter_clouds (bool, optional): Should images with clouds above the GPS position be filtered?, defaults to True.
            limit (int, optional): Maximum number of returned entries, defaults to 1.

        Returns:
            list[dict]: list of datasets, ordered by date difference to the search date
        """
        print(f"Getting Landsat 7 features for {date} ({latitude}, {longitude})...")
        cls.authenticate()

        date = datetime.fromisoformat(date)
        start = date - timedelta(days=span)
        end = date + timedelta(days=span + 1)

        query = {
            "datasetName": "landsat_etm_c2_l1",
            "metadataType": "full",
            "sceneFilter": {
                "acquisitionFilter": {
                    "start": start.strftime("%Y-%m-%d"),
                    "end": end.strftime("%Y-%m-%d"),
                },
                "spatialFilter": {
                    "filterType": "mbr",
                    "lowerLeft": {"longitude": longitude, "latitude": latitude},
                    "upperRight": {"longitude": longitude, "latitude": latitude},
                },
            },
        }

        response = requests.get(
            cls.API_URL,
            allow_redirects=True,
            json=query,
            headers={"X-Auth-Token": os.environ["M2M_TOKEN"]},
        )
        response.raise_for_status()
        data = response.json()

        if not data["data"]["results"]:
            print(
                f"No data available for the chosen date and location!", file=sys.stderr
            )
            return None

        features = data["data"]["results"]
        # sort results by difference to the desired date
        features.sort(key=lambda product: cls._productTimestampDiff(product, date))

        if not filter_clouds:
            return features[:limit]

        result = []
        for feature in features:
            if len(result) >= limit:
                break
            else:
                result.append(feature)

        return result

    @classmethod
    def authenticate(cls) -> None:
        """
        Authenticates with the USGS M2M API using $M2M_USER and $M2M_SECRET
        and saves the authentication token to $M2M_TOKEN.
        Authentication is skipped if there is already a valid token.
        """
        if cls._is_authenticated():
            return

        response = requests.post(
            cls.AUTH_URL,
            allow_redirects=True,
            json={
                "username": os.environ["M2M_USER"],
                "token": os.environ["M2M_SECRET"],
            },
        )
        response.raise_for_status()
        token = response.json()

        os.environ["M2M_TOKEN"] = token["data"]
        os.environ["M2M_TOKEN_EXPIRES"] = str(round(time.time()) + 7000)

        print("USGS M2M authentication successful")

    @staticmethod
    def _is_authenticated() -> bool:
        """
        Checks if the authentication token stored in $M2M_TOKEN is present and not expired.

        Returns:
            bool: Returns True if the script is already authenticated
        """
        if not os.environ.get("M2M_TOKEN"):
            return False
        if not os.environ.get("M2M_TOKEN_EXPIRES"):
            return True
        return int(os.environ["M2M_TOKEN_EXPIRES"]) > time.time()
