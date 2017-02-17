from camera_calibrator import calibrate_camera
from image_processor import ImageProcessor
from moviepy.editor import VideoFileClip

# Calibrate the camera
mtx, dist = calibrate_camera()

# Create processing object. This handles all the image processing
# and lane finding.
processor = ImageProcessor(mtx, dist)

clip = VideoFileClip("project_video.mp4")

# Process each frame of the original video
processed_clip = clip.fl_image(processor.process_image)

# Write to file
processed_clip.write_videofile("output_video.mp4", audio=False)
