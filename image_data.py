import numpy as np

YM_PER_PIX = 30 / 720  # meters per pixel in y dimension
XM_PER_PIX = 3.7 / 700  # meters per pixel in x dimension


class ImageData:
    def __init__(self, image, nonzerox, nonzeroy, left_lane_inds, right_lane_inds):
        self.__image = image
        self.__nonzerox = nonzerox
        self.__nonzeroy = nonzeroy
        self.__left_lane_inds = left_lane_inds
        self.__right_lane_inds = right_lane_inds

    def leftx(self):
        return self.__nonzerox[self.__left_lane_inds]

    def rightx(self):
        return self.__nonzerox[self.__right_lane_inds]

    def lefty(self):
        return self.__nonzeroy[self.__left_lane_inds]

    def righty(self):
        return self.__nonzeroy[self.__right_lane_inds]

    def __left_fit(self):
        return np.polyfit(self.lefty(), self.leftx(), 2)

    def __right_fit(self):
        return np.polyfit(self.righty(), self.rightx(), 2)

    def ploty(self):
        return np.linspace(0, self.__image.shape[1] - 1, self.__image.shape[1])

    def left_fit_x(self):
        return self.__left_fit()[0] * self.ploty()**2 \
            + self.__left_fit()[1] * self.ploty() + self.__left_fit()[2]

    def right_fit_x(self):
        return self.__right_fit()[0] * self.ploty()**2 \
            + self.__right_fit()[1] * self.ploty() + self.__right_fit()[2]

    def getCurveRadius(self):
        left_fit_cr = np.polyfit(self.__lefty() * YM_PER_PIX, self.__leftx() * XM_PER_PIX, 2)
        right_fit_cr = np.polyfit(self.__righty() * YM_PER_PIX, self.__rightx() * XM_PER_PIX, 2)
        y_eval = np.max(self.__ploty())

        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * YM_PER_PIX + left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * YM_PER_PIX + right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])

        # Now our radius of curvature is in meters
        return left_curverad, right_curverad
