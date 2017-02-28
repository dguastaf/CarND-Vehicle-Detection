import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from vehicle_detector import find_cars

img = mpimg.imread("test_images/test1.jpg")

out_img = find_cars(img)

plt.imshow(out_img)
plt.show()
