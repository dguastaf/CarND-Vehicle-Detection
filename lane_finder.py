from camera_calibrator import calibrate_camera
from image_processor import ImageProcessor
from moviepy.editor import VideoFileClip
# from matplotlib import pyplot as plt

mtx, dist = calibrate_camera()
processor = ImageProcessor(mtx, dist)

clip = VideoFileClip("project_video.mp4")
# result = processor.process_image(clip.get_frame(0))
# plt.imshow(result)
# plt.show()

processed_clip = clip.fl_image(processor.process_image)  # NOTE: this function expects color images!!
processed_clip.write_videofile("output_video.mp4", audio=False)
