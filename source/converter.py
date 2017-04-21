from source.box import Box


def convert_prediction_to_box(prediction, threshold=0.17, sqrt=1.8, number_of_classes=20, number_of_boxes=2,
                              grid_size=7):
    class_num = 6
    boxes = []
    number_of_grid_cells = grid_size * grid_size
    class_probabilities = number_of_grid_cells * number_of_classes
    cell_confidence = number_of_grid_cells * number_of_boxes

    probabilities = prediction[0: class_probabilities]
    confidences = prediction[class_probabilities: (class_probabilities + cell_confidence)]
    coordinates = prediction[(class_probabilities + cell_confidence):]
    probabilities = probabilities.reshape([number_of_grid_cells, number_of_classes])
    confidences = confidences.reshape([number_of_grid_cells, number_of_boxes])
    coordinates = coordinates.reshape([number_of_grid_cells, number_of_boxes, 4])

    for grid in range(number_of_grid_cells):
        for b in range(number_of_boxes):
            box = Box()
            box.confidence = confidences[grid, b]
            box.x = (coordinates[grid, b, 0] + grid % grid_size) / grid_size
            box.y = (coordinates[grid, b, 1] + grid // grid_size) / grid_size
            box.width = coordinates[grid, b, 2] ** sqrt
            box.height = coordinates[grid, b, 3] ** sqrt
            p = probabilities[grid, :] * box.confidence

            if p[class_num] >= threshold:
                box.probability = p[class_num]
                boxes.append(box)

    # combine boxes that are overlap
    boxes.sort(key=lambda b: b.prob, reverse=True)
    for i in range(len(boxes)):
        boxi = boxes[i]
        if boxi.prob == 0: continue
        for j in range(i + 1, len(boxes)):
            boxj = boxes[j]
            if box_iou(boxi, boxj) >= .4:
                boxes[j].prob = 0.
    boxes = [b for b in boxes if b.prob > 0.]

    return boxes


def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b)


def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.y, a.h, b.y, b.h)
    if w < 0 or h < 0: return 0;
    area = w * h
    return area


def box_union(a, b):
    i = box_intersection(a, b)
    u = a.w * a.h + b.w * b.h - i
    return u


def overlap(x1, w1, x2, w2):
    l1 = x1 - w1 / 2.
    l2 = x2 - w2 / 2.
    left = max(l1, l2)
    r1 = x1 + w1 / 2.
    r2 = x2 + w2 / 2.
    right = min(r1, r2)
    return right - left
