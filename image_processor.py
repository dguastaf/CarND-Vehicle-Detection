import cv2


class ImageProcessor:

    def __init__(self, mtx, dist):
        self.__mtx = mtx
        self.__dist = dist

    def process_next_image(self, image):
        image = cv2.undistort(image, self.__mtx, self.__dist, None, self.__mtx)
        cv2.imwrite('output_images/processed.jpg', image)
