import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from vehicle_detector import find_cars
from image_processor import *
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip


def draw_labels(img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                    (np.max(nonzerox), np.max(nonzeroy)))

            if (bbox[1][0] - bbox[0][0] > 30 and bbox[1][1] - bbox[0][1] > 30):
                # Draw the box on the image
                cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)


# img = mpimg.imread("test_images/test1.jpg")
clip = VideoFileClip("project_video.mp4")
img = clip.get_frame(45.5)

heatmap = find_cars(img)

labels = label(heatmap)
draw_labels(img, labels)

plt.imshow(img)
plt.show()
