from camera_calibrator import calibrate_camera
from image_processor import ImageProcessor
from moviepy.editor import VideoFileClip

import matplotlib.pyplot as plt

# Calibrate the camera
mtx, dist = calibrate_camera()

# Create processing object. This handles all the image processing
# and lane finding.
processor = ImageProcessor(mtx, dist)

clip = VideoFileClip("short_video.mp4")


# out_img = processor.process_image(clip.get_frame(55))
# plt.imsave("test1.jpg", clip.get_frame(40))
# plt.imshow(out_img)
# plt.show()

# Process each frame of the original video
processed_clip = clip.fl_image(processor.process_image)

# Write to file
processed_clip.write_videofile("short_video_out.mp4", audio=False)
