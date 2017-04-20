import cv2
import glob

import numpy as np

from source.converter import yolo_net_out_to_car_boxes
from source.model import make_model, load_weights
import matplotlib.pyplot as plt

from source.overlay import draw_box

model = make_model()

load_weights(model, '../yolo-tiny.weights')

images = [plt.imread(file) for file in glob.glob('../test_images/*.jpg')]
batch = np.array([np.transpose(cv2.resize(image[300:650, 500:, :], (448, 448)), (2, 0, 1)) for image in images])
batch = 2 * (batch / 255.) - 1
prediction = model.predict(batch)
f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(11, 10))
for i, ax in zip(range(len(batch)), [ax1, ax2, ax3, ax4, ax5, ax6]):
    boxes = yolo_net_out_to_car_boxes(prediction[i], threshold=0.17)
    box = draw_box(boxes, images[i], [[500, 1280], [300, 650]])
    plt.imsave('../output_images/' + 'test' + str(i) + ' result.jpg', box)
