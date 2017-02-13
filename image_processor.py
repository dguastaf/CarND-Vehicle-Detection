import cv2
import numpy as np
from matplotlib import pyplot as plt


class ImageProcessor:

    def __init__(self, mtx, dist):
        self.__mtx = mtx
        self.__dist = dist

    def process_next_image(self, image, i):
        image = cv2.undistort(image, self.__mtx, self.__dist, None, self.__mtx)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        gradx = self.__abs_sobel_thresh(gray, orient='x', thresh=(30, 100))
        # grady = self.__abs_sobel_thresh(gray, orient='y', thresh=(50, 200))

        # combined = np.zeros_like(gray)
        # combined[((gradx == 1) & (grady == 1))] = 1

        light = self.__hls_thresh(image, channel=1, thresh=(130, 255))
        s = self.__hls_thresh(image, channel=2, thresh=(140, 255))

        ls = np.zeros_like(light)
        ls[((light == 1) & (s == 1))] = 1

        total = np.zeros_like(gradx)
        total[((gradx == 1) | (ls == 1))] = 1

        # f, p = plt.subplots(2, 2)
        # # f, (ax1) = plt.subplots(1, 1)
        # f.tight_layout()

        # p[0][0].imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        # p[0][0].set_title('Original Image', fontsize=10)

        # p[0][1].imshow(gradx, cmap='gray')
        # p[0][1].set_title('Sobel X', fontsize=10)

        # p[0][2].imshow(grady, cmap='gray')
        # p[0][2].set_title('Sobel Y', fontsize=10)

        # p[1][0].imshow(combined, cmap='gray')
        # p[1][0].set_title('Combined Sobel X & Y', fontsize=10)

        # p[1][1].imshow(light, cmap='gray')
        # p[1][1].set_title('Lightness', fontsize=10)

        # p[1][2].imshow(s, cmap='gray')
        # p[1][2].set_title('Saturation', fontsize=10)

        # p[1][0].imshow(ls, cmap='gray')
        # p[1][0].set_title('Lightness & Saturation', fontsize=10)

        # p[1][1].imshow(total, cmap='gray')
        # p[1][1].set_title('Light & Sat OR Sobel X', fontsize=10)

        # plt.show()

        return total

    def __hls_thresh(self, img, channel=1, thresh=(0, 255)):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        single_channel = hsv[:, :, channel]
        binary = np.zeros_like(single_channel)
        binary[(single_channel >= thresh[0]) & (single_channel <= thresh[1])] = 1

        return binary


    def __abs_sobel_thresh(self, img, orient='x', kernel=3, thresh=(0, 255)):
        sobel = None

        if (orient == 'x'):
            sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel))
        elif (orient == 'y'):
            sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel))
        else:
            print("Returning none")
            return None

        scaled_sobel = np.uint8(255 * sobel / np.max(sobel))

        binary = np.zeros_like(scaled_sobel)
        binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        return binary

    def __mag_thresh(self, img, kernel=3, thresh=(0, 255)):
        sobelx = cv2.Sobel(img, cv2.CV_64F, 0, 1)
        sobely = cv2.Sobel(img, cv2.CV_64F, 1, 0)
        abs_sobel = np.sqrt(np.square(sobelx) + np.square(sobely))
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        sbinary = np.zeros_like(scaled_sobel)
        sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        return sbinary

    def __dir_thresh(self, img, kernel=3, thresh=(0, np.pi / 2)):
        sobelx = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))
        sobely = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))

        scaled_sobel = np.arctan2(sobely, sobelx)
        sbinary = np.zeros_like(scaled_sobel)
        sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        return sbinary
