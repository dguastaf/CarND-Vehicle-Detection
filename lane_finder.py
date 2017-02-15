import cv2
from camera_calibrator import calibrate_camera
from find_lane_lines import full_search
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


def draw_lane_box(imageData):
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([imageData.left_fit_x(), imageData.ploty()]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([imageData.right_fit_x(), imageData.ploty()])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    Minv = cv2.getPerspectiveTransform(dst, src)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))

    # Combine the result with the original image
    return cv2.addWeighted(undistort_image, 1, newwarp, 0.3, 0)


mtx, dist = calibrate_camera()

processor = ImageProcessor(mtx, dist)

# image = cv2.imread('test_images/straight_lines1.jpg')
# processor.process_next_image(image, 0)

image = cv2.imread('test_images/straight_lines1.jpg')
undistort_image = processor.undistort(image)
binary = processor.process_next_image(undistort_image, 1)

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

imageData = full_search(binary_warped)
result = draw_lane_box(imageData)

plt.imshow(cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
plt.show()

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

# plt.show()
