import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from vehicle_detector import find_cars
from vehicle_detection_model import train_model

img = mpimg.imread("test_images/straight_lines2.jpg")

out_img = find_cars(img)
# train_model()

plt.imshow(out_img)
plt.show()
