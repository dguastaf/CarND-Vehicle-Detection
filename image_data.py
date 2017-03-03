import numpy as np

YM_PER_PIX = 30 / 720  # meters per pixel in y dimension
XM_PER_PIX = 3.7 / 700  # meters per pixel in x dimension
LANE_WIDTH = 3.7  # assume lanes have constant width


# Holds all the lane-finding data about a specific frame
class ImageData:
    def __init__(self, image, nonzerox, nonzeroy, left_lane_inds, right_lane_inds):
        self.__image = image
        self.__nonzerox = nonzerox
        self.__nonzeroy = nonzeroy
        self.__left_lane_inds = left_lane_inds
        self.__right_lane_inds = right_lane_inds
        self.__vehicle_heatmap = None

    # X coordinates of the left lane line
    def leftx(self):
        return self.__nonzerox[self.__left_lane_inds]

    # X coordinates of the right lane line
    def rightx(self):
        return self.__nonzerox[self.__right_lane_inds]

    # Y coordinates of the left lane line
    def lefty(self):
        return self.__nonzeroy[self.__left_lane_inds]

    # Y coordinates of the right lane line
    def righty(self):
        return self.__nonzeroy[self.__right_lane_inds]

    # Fit a 2nd degree polynomial to left lane line
    def left_fit(self):
        return np.polyfit(self.lefty(), self.leftx(), 2)

    # Fit a 2nd degree polynomial to right lane line
    def right_fit(self):
        return np.polyfit(self.righty(), self.rightx(), 2)

    def ploty(self):
        return np.linspace(0, self.__image.shape[0] - 1, self.__image.shape[0])

    def left_fit_x(self):
        return self.left_fit()[0] * self.ploty()**2 \
            + self.left_fit()[1] * self.ploty() + self.left_fit()[2]

    def right_fit_x(self):
        return self.right_fit()[0] * self.ploty()**2 \
            + self.right_fit()[1] * self.ploty() + self.right_fit()[2]

    def set_vehicle_heatmap(self, vehicle_heatmap):
        self.__vehicle_heatmap = vehicle_heatmap

    def get_vehicle_heatmap(self):
        return self.__vehicle_heatmap

    def getCurveRadius(self):
        left_fit_cr = np.polyfit(self.lefty() * YM_PER_PIX, self.leftx() * XM_PER_PIX, 2)
        right_fit_cr = np.polyfit(self.righty() * YM_PER_PIX, self.rightx() * XM_PER_PIX, 2)
        y_eval = np.max(self.ploty())

        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * YM_PER_PIX + left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * YM_PER_PIX + right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])

        # Now our radius of curvature is in meters
        return left_curverad + right_curverad / 2

    def getCenterOffset(self):
        center = self.right_fit_x()[-1] - self.left_fit_x()[-1]
        m_per_px = LANE_WIDTH / self.__image.shape[1]
        offset = (center - self.__image.shape[1] / 2) * m_per_px
        return offset
