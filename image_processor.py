import cv2
import numpy as np
from find_lane_lines import full_search, poly_search
import collections

QUEUE_SIZE = 10


class ImageProcessor:

    def __init__(self, mtx, dist):
        self.__mtx = mtx
        self.__dist = dist
        self.__detectedLast = False
        self.__prev_image_data = collections.deque(maxlen=QUEUE_SIZE)
        self.__last_image_data = None

    def process_image(self, image):
        image = self.__undistort(image)
        binary = self.__transform_to_binary(image)
        binary_warped = self.__warp_image(binary)

        imageData = None
        if (self.__last_image_data):
            imageData = poly_search(binary_warped, self.__last_image_data.left_fit(), self.__last_image_data.right_fit())

        if (not imageData):
            print("full search")
            imageData = full_search(binary_warped)

        self.__save_image_data(imageData)

        result = self.__draw_lane_box(image, binary_warped)
        self.__write_info(result)

        return result

    def __save_image_data(self, imageData):
        self.__last_image_data = imageData
        self.__prev_image_data.append(imageData)

    def __undistort(self, image):
        return cv2.undistort(image, self.__mtx, self.__dist, None, self.__mtx)

    def __transform_to_binary(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        gradx = self.__abs_sobel_thresh(gray, orient='x', thresh=(30, 100))

        light = self.__hls_thresh(image, channel=1, thresh=(130, 255))
        s = self.__hls_thresh(image, channel=2, thresh=(140, 255))

        ls = np.zeros_like(light)
        ls[((light == 1) & (s == 1))] = 1

        total = np.zeros_like(gradx)
        total[((gradx == 1) | (ls == 1))] = 1

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
            return None

        scaled_sobel = np.uint8(255 * sobel / np.max(sobel))

        binary = np.zeros_like(scaled_sobel)
        binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        return binary

    def __draw_lane_box(self, orig_image, binary_warped):
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        left = np.zeros((len(self.__prev_image_data), 720))
        right = np.zeros((len(self.__prev_image_data), 720))

        ploty = self.__last_image_data.ploty()

        for i, data in enumerate(self.__prev_image_data):
            left[i] = data.left_fit_x()
            right[i] = data.right_fit_x()

        left_fit_x = np.mean(left, axis=0)
        right_fit_x = np.mean(right, axis=0)

        pts_left = np.array([np.transpose(np.vstack([left_fit_x, ploty]))])

        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        src = self.__get_warp_src(binary_warped)
        dst = self.__get_warp_dst()
        Minv = cv2.getPerspectiveTransform(dst, src)

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (orig_image.shape[1], orig_image.shape[0]))

        # Combine the result with the original image
        return cv2.addWeighted(orig_image, 1, newwarp, 0.3, 0)

    def __write_info(self, image):
        radius = 0
        for data in self.__prev_image_data:
            radius = radius + data.getCurveRadius()
        radius = radius / len(self.__prev_image_data)

        radiusText = "Radius of curvature: {}m".format(int(radius))
        cv2.putText(image, radiusText, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), thickness=2)

        radiusText = "{0:.2f}m offset".format(self.__last_image_data.getCenterOffset())
        cv2.putText(image, radiusText, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), thickness=2)

    def __warp_image(self, image):
        src = self.__get_warp_src(image)
        dst = self.__get_warp_dst()

        M = cv2.getPerspectiveTransform(src, dst)

        return cv2.warpPerspective(image, M, (1280, 720))

    def __get_warp_src(self, image):
        image_center = image.shape[1] / 2
        top_x_offset = 50
        bottom_x_offset = 410

        return np.float32([[image_center - top_x_offset, 450],
                          [image_center + top_x_offset, 450],
                          [image_center + bottom_x_offset, 680],
                          [image_center - bottom_x_offset + 15, 680]])

    def __get_warp_dst(self):
        return np.float32([[320, 0],
                          [960, 0],
                          [960, 720],
                          [320, 720]])
