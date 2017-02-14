import cv2
from camera_calibrator import calibrate_camera
from fine_lane_lines import full_search
from image_processor import ImageProcessor
import numpy as np
from matplotlib import pyplot as plt


def cvtBinaryToRGB(binary):
    binary_trans = np.zeros((len(binary), len(binary[0]), 3))
    for i, row in enumerate(binary):
        for j, col in enumerate(row):
            binary_trans[i][j] = (0, 0, 0) if col == 0 else (1, 1, 1)
    return binary_trans


def display_binary(img):
    img = cvtBinaryToRGB(img)
    f, plts = plt.subplots(1, 1)
    plts.imshow(img)
    plt.show()


mtx, dist = calibrate_camera()

processor = ImageProcessor(mtx, dist)

# image = cv2.imread('test_images/straight_lines1.jpg')
# processor.process_next_image(image, 0)

image = cv2.imread('test_images/test6.jpg')
binary = processor.process_next_image(image, 1)

image_center = image.shape[1] / 2
top_x_offset = 50
bottom_x_offset = 410

src = np.float32([[image_center - top_x_offset, 450],
                 [image_center + top_x_offset, 450],
                 [image_center + bottom_x_offset, 680],
                 [image_center - bottom_x_offset + 15, 680]])

dst = np.float32([[320, 0],
                 [960, 0],
                 [960, 720],
                 [320, 720]])

M = cv2.getPerspectiveTransform(src, dst)

binary_warped = cv2.warpPerspective(binary, M, (1280, 720))

full_search(binary_warped)

# display_binary(binary_warped)


# image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
# image_bgr_lines = cv2.polylines(image_bgr.copy(), [src.astype(int)], True, (255, 0, 0), thickness=5)

# binary_rgb = cvtBinaryToRGB(binary)
# binary_rgb_lines = cv2.polylines(binary_rgb.copy(), [src.astype(int)], True, (1, 0, 0), thickness=5)

# warped_color = cv2.warpPerspective(image_bgr_lines, M, (1280, 720))
# warped_binary_rgb = cv2.warpPerspective(binary_rgb_lines, M, (1280, 720))

# f, plts = plt.subplots(2, 2)
# plts[0][0].imshow(image_bgr_lines)
# plts[0][1].imshow(binary_rgb_lines)
# plts[1][0].imshow(warped_color)
# plts[1][1].imshow(warped_binary_rgb)

plt.show()
