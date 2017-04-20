import cv2
import numpy as np

from source.converter import yolo_net_out_to_car_boxes
from source.model import make_model, load_weights
import matplotlib.pyplot as plt

from source.overlay import draw_box

model = make_model()

load_weights(model, '../yolo-tiny.weights')

imagePath = '../test_images/test1.jpg'
image = plt.imread(imagePath)
image_crop = image[300:650, 500:, :]
resized = cv2.resize(image_crop, (448, 448))

batch = np.transpose(resized, (2, 0, 1))
batch = 2 * (batch / 255.) - 1
batch = np.expand_dims(batch, axis=0)
out = model.predict(batch)

boxes = yolo_net_out_to_car_boxes(out[0], threshold=0.17)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
ax1.imshow(image)
box = draw_box(boxes, plt.imread(imagePath), [[500, 1280], [300, 650]])
ax2.imshow(box)
# cv2.imwrite('../output_images/' + 'test1 result.jpg', box)
plt.imsave('../output_images/' + 'test1 result.jpg', box)
