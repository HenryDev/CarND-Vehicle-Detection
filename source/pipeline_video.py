import cv2
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip

from source.converter import yolo_net_out_to_car_boxes
from source.model import make_model, load_weights
from source.overlay import draw_box


def frame_func(image):
    crop = image[300:650, 500:, :]
    resized = cv2.resize(crop, (448, 448))
    batch = np.array([resized[:, :, 0], resized[:, :, 1], resized[:, :, 2]])
    batch = 2 * (batch / 255.) - 1
    batch = np.expand_dims(batch, axis=0)
    out = model.predict(batch)
    boxes = yolo_net_out_to_car_boxes(out[0], threshold=0.17)
    return draw_box(boxes, image, [[500, 1280], [300, 650]])


model = make_model()

load_weights(model, '../yolo-tiny.weights')

# output_video = '../video result.mp4'
# clip = VideoFileClip("../project_video.mp4")

output_video = '../Motorcycle Crash Captured on Dashcam result.mp4'
clip = VideoFileClip("../Motorcycle Crash Captured on Dashcam.mp4")

lane_clip = clip.fl_image(frame_func)
lane_clip.write_videofile(output_video, audio=False)
