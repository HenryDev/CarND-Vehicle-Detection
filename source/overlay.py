import cv2


def draw_box(boxes, image):
    [x_min, x_max] = [500, 1280]
    [y_min, y_max] = [300, 650]
    for box in boxes:
        height, width, _ = image.shape
        left = int((box.x - box.width / 2.) * width)
        right = int((box.x + box.width / 2.) * width)
        top = int((box.y - box.height / 2.) * height)
        bottom = int((box.y + box.height / 2.) * height)
        left = int(left * (x_max - x_min) / width + x_min)
        right = int(right * (x_max - x_min) / width + x_min)
        top = int(top * (y_max - y_min) / height + y_min)
        bottom = int(bottom * (y_max - y_min) / height + y_min)

        if left < 0:
            left = 0
        if right > width - 1:
            right = width - 1
        if top < 0:
            top = 0
        if bottom > height - 1:
            bottom = height - 1
        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 5)
    return image
