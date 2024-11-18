import cv2
from osgeo import gdal, osr


class ImageUtils:
    """
    Various methods for loading and altering image data
    """

    @staticmethod
    def location_to_pixel(
        image: str, latitude: float, longitude: float
    ) -> tuple[int, int]:
        """
        Calculates the pixel coordinates inside the given image from GPS coordinates

        Parameters:
            image (str): Path of the image
            latitude (float): GPS latitude
            longitude (float): GPS longitude

        Returns:
            tuple[int, int]: The pixel x and y coordinates in the image
        """
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
    def crop(source: str, target: str, x: int, y: int, width: int, height: int) -> None:
        """
        Crops a image to the given size

        Parameters:
            source (str): Path of the source image
            target (str): Path to save the cropped image
            x (int): Start of the cut (X)
            y (int): Start of the cut (Y)
            width (int): Cropped width
            height (int): Cropped height
        """
        img = cv2.imread(source)
        cropped = img[y : y + height, x : x + width]
        cv2.imwrite(target, cropped)

    @staticmethod
    def crop_center(source: str, target: str, x: int, y: int, radius: int) -> None:
        """
        Crops the image to a given radius around a center point (in pixels).
        The given point is in the exact center of the cropped image.

        Parameters:
            source (str): Path of the source image
            target (str): Path to save the cropped image
            x (int): Center point (X)
            y (int): Center point (Y)
            radius (int): Radius (determines the image size -> 2 x radius + 1)
        """
        img = cv2.imread(source)
        cropped = img[y - radius : y + radius + 1, x - radius : x + radius + 1]
        cv2.imwrite(target, cropped)

    @classmethod
    def crop_location(
        cls, source: str, target: str, latitude: float, longitude: float, radius: int
    ) -> None:
        """
        Crops the image to a given radius around a center point (GPS location).
        The given point is in the exact center of the cropped image.

        Parameters:
            source (str): Path of the source image
            target (str): Path to save the cropped image
            latitude (float): GPS latitude of the center point
            longitude (float): GPS longitude of the center point
            radius (int): Radius in pixels (determines the image size -> 2 x radius + 1)
        """
        x, y = cls.location_to_pixel(source, latitude, longitude)
        cls.crop_center(source, target, x, y, radius)

    @staticmethod
    def merge_channels(
        image_blue: str, image_green: str, image_red: str, target: str
    ) -> None:
        """
        Generates and saves a true color (RGB) image from three
        separate grayscale images representing red, green and blue channels.

        Parameters:
            image_blue (str): Path of the image representing the blue channel
            image_green (str): Path of the image representing the green channel
            image_red (str): Path of the image representing the red channel
            target (str): Path to save the resulting RGB image
        """
        blue = cv2.imread(image_blue, 0)
        green = cv2.imread(image_green, 0)
        red = cv2.imread(image_red, 0)
        merged = cv2.merge((blue, green, red)) * 3
        cv2.imwrite(target, merged)

    @staticmethod
    def get_pixel_value(image: str, x: int, y: int) -> int:
        """
        Returns the value of the given pixel of a grayscale image.

        Parameters:
            image (str): Path of the image
            x (int): Pixel x-coordinate
            y (int): Pixel y-coordinate

        Returns:
            int: Grayscale integer value of the pixel
        """
        img = cv2.imread(image, 0)
        return img[y, x]
