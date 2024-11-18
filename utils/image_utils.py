import cv2
from osgeo import gdal, osr


class ImageUtils:
    @staticmethod
    def location_to_pixel(image: str, latitude: float, longitude: float):
        gdal.UseExceptions()
        dataset = gdal.Open(image)
        if not dataset:
            raise Exception("Could not load image file!")

        transform = dataset.GetGeoTransform()
        projection = dataset.GetProjection()

        source = osr.SpatialReference()
        source.ImportFromWkt(projection)
        target = osr.SpatialReference()
        target.ImportFromEPSG(4326)  # code for WGS84 (latitude and longitude)

        coordinate_transform = osr.CoordinateTransformation(target, source)
        x, y, _ = coordinate_transform.TransformPoint(latitude, longitude)
        pixel_x = int((x - transform[0]) / transform[1])
        pixel_y = int((y - transform[3]) / transform[5])

        return pixel_x, pixel_y

    @staticmethod
    def crop(source: str, target: str, x: int, y: int, width: int, height: int):
        img = cv2.imread(source)
        cropped = img[y : y + height, x : x + width]
        cv2.imwrite(target, cropped)

    @staticmethod
    def crop_center(source: str, target: str, x: int, y: int, radius: int):
        img = cv2.imread(source)
        cropped = img[y - radius : y + radius + 1, x - radius : x + radius + 1]
        cv2.imwrite(target, cropped)

    @staticmethod
    def crop_location(
        source: str, target: str, latitude: float, longitude: float, radius: int
    ):
        x, y = __class__.location_to_pixel(source, latitude, longitude)
        __class__.crop_center(source, target, x, y, radius)

    @staticmethod
    def merge_channels(image_blue: str, image_green: str, image_red: str, target: str):
        blue = cv2.imread(image_blue, 0)
        green = cv2.imread(image_green, 0)
        red = cv2.imread(image_red, 0)
        merged = cv2.merge((blue, green, red)) * 3
        cv2.imwrite(target, merged)

    @staticmethod
    def get_pixel_value(image: str, x: int, y: int):
        img = cv2.imread(image, 0)
        return img[y, x]
