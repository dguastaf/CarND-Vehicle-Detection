import numpy as np
from matplotlib import pyplot as plt
import cv2
from image_data import ImageData

N_WINDOWS = 40

# Width of the windows +/- margin
WINDOW_MARGIN = 50

# Set minimum number of pixels found to recenter window
WINDOW_MIN_PX = 25


class DebugImage:
    def __init__(self, binary_warped):
        self.__out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    def draw_rectangle(self, low, high):
        cv2.rectangle(self.__out_img, low, high, (0, 255, 0), 2)

    def color_lane(self, y, x, color):
        self.__out_img[y, x] = color

    def show_image(self, left_fit_x, right_fit_x, ploty):
        plt.imshow(self.__out_img)
        plt.plot(left_fit_x, ploty, color='yellow')
        plt.plot(right_fit_x, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()


def full_search(binary_warped):

    debugImage = DebugImage(binary_warped)

    histogram = np.sum(binary_warped[binary_warped.shape[0] / 2:, :], axis=0)

    plt.plot(histogram)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)

    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / N_WINDOWS)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    for window in range(N_WINDOWS):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height

        win_xleft_low = leftx_current - WINDOW_MARGIN
        win_xleft_high = leftx_current + WINDOW_MARGIN
        win_xright_low = rightx_current - WINDOW_MARGIN
        win_xright_high = rightx_current + WINDOW_MARGIN

        # Draw the windows on the visualization image
        debugImage.draw_rectangle((win_xleft_low, win_y_low), (win_xleft_high, win_y_high))
        debugImage.draw_rectangle((win_xright_low, win_y_low), (win_xright_high, win_y_high))

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > WINDOW_MIN_PX:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > WINDOW_MIN_PX:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    imageData = ImageData(binary_warped, nonzerox, nonzeroy, left_lane_inds, right_lane_inds)

    debugImage.color_lane(imageData.lefty(), imageData.leftx(), [255, 0, 0])
    debugImage.color_lane(imageData.righty(), imageData.rightx(), [0, 0, 255])
    debugImage.show_image(imageData.left_fit_x(), imageData.right_fit_x(), imageData.ploty())

    return imageData


def poly_search(img, left_fit, right_fit):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_fit_left = (left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[2] - WINDOW_MARGIN)
    left_fit_right = (left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[2] + WINDOW_MARGIN)

    right_fit_left = (right_fit[0] * (nonzeroy**2) + right_fit[1] * nonzeroy + right_fit[2] - WINDOW_MARGIN)
    right_fit_right = (right_fit[0] * (nonzeroy**2) + right_fit[1] * nonzeroy + right_fit[2] + WINDOW_MARGIN)

    left_lane_inds = ((nonzerox > left_fit_left) & (nonzerox < left_fit_right))
    right_lane_inds = ((nonzerox > right_fit_left) & (nonzerox < right_fit_right))

    imageData = ImageData(img, nonzerox, nonzeroy, left_lane_inds, right_lane_inds)

    return imageData
    # Again, extract left and right line pixel positions
    # leftx = nonzerox[left_lane_inds]
    # lefty = nonzeroy[left_lane_inds]
    # rightx = nonzerox[right_lane_inds]
    # righty = nonzeroy[right_lane_inds]
    # # Fit a second order polynomial to each
    # left_fit = np.polyfit(lefty, leftx, 2)
    # right_fit = np.polyfit(righty, rightx, 2)
    # # # Generate x and y values for plotting
    # ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    # left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    # right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
