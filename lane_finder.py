import cv2
from camera_calibrator import calibrate_camera
from image_processor import ImageProcessor
import numpy as np
from matplotlib import pyplot as plt

mtx, dist = calibrate_camera()

processor = ImageProcessor(mtx, dist)

# image = cv2.imread('test_images/straight_lines1.jpg')
# processor.process_next_image(image, 0)

image = cv2.imread('test_images/straight_lines1.jpg')
binary = processor.process_next_image(image, 1)


pts = np.array([[600, 450], [690, 450], [1050, 680], [255, 680]],
               dtype=np.int32)

color_out = cv2.polylines(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [pts], True, (255, 0, 0), thickness=5)

binary_trans = np.zeros((len(binary), len(binary[0]), 3))

for i, row in enumerate(binary):
    for j, col in enumerate(row):
        binary_trans[i][j] = (0, 0, 0) if col == 0 else (1, 1, 1)

cv2.polylines(binary_trans, [pts], True, (1, 0, 0), thickness=5)

f, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(color_out)
ax2.imshow(binary_trans)

plt.show()
