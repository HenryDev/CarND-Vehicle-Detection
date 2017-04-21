import cv2
import glob
import numpy
from source.converter import convert_prediction_to_box
from source.model import make_model, load_weights
import matplotlib.pyplot as plt
from source.overlay import draw_box

model = make_model()

load_weights(model, '../yolo-tiny.weights')

images = [plt.imread(file) for file in glob.glob('../test_images/*.jpg')]
batch = numpy.array([numpy.transpose(cv2.resize(image[300:650, 500:, :], (448, 448)), (2, 0, 1)) for image in images])
batch = 2 * (batch / 255.) - 1
prediction = model.predict(batch)
for i in range(len(batch)):
    boxes = convert_prediction_to_box(prediction[i])
    box = draw_box(boxes, images[i], [[500, 1280], [300, 650]])
    plt.imsave('../output_images/test' + str(i) + ' result.jpg', box)
