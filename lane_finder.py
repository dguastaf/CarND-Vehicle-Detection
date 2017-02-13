import cv2
from camera_calibrator import calibrate_camera
from image_processor import ImageProcessor

mtx, dist = calibrate_camera()

processor = ImageProcessor(mtx, dist)

# image = cv2.imread('test_images/straight_lines1.jpg')
# processor.process_next_image(image, 0)

image = cv2.imread('test_images/test3.jpg')
processor.process_next_image(image, 1)
