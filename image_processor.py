import cv2
import numpy as np
from matplotlib import pyplot as plt

class ImageProcessor:

    def __init__(self, mtx, dist):
        self.__mtx = mtx
        self.__dist = dist

    def process_next_image(self, image, i):
        image = cv2.undistort(image, self.__mtx, self.__dist, None, self.__mtx)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        gradx = self.__abs_sobel_thresh(image, orient='x', thresh_min=20, thresh_max=100)
        grady = self.__abs_sobel_thresh(image, orient='y', thresh_min=50, thresh_max=200)

        combined = np.zeros_like(image)
        combined[((gradx == 1) & (grady == 1))] = 1

        plt.imsave('output_images/processed{}-x.jpg'.format(i), gradx , cmap='gray')
        plt.imsave('output_images/processed{}-y.jpg'.format(i), grady , cmap='gray')
        plt.imsave('output_images/processed{}-c.jpg'.format(i), combined , cmap='gray')

    def __abs_sobel_thresh(self, img, orient='x', thresh_min=0, thresh_max=255):
        
        sobelx = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
        sobely = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))

        sobel = np.sqrt(np.square(sobelx) + np.square(sobely))
        scaled_sobel = np.uint8(255 * sobel / np.max(sobel))

        binary = np.zeros_like(scaled_sobel)
        binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

        return binary
