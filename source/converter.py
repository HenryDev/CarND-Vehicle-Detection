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
            probability = probabilities[grid, :] * box.confidence

            if probability[class_num] >= threshold:
                box.probability = probability[class_num]
                boxes.append(box)

    boxes = combine_overlap_boxes(boxes)

    return boxes


def combine_overlap_boxes(boxes):
    boxes.sort(key=lambda b: b.probability, reverse=True)
    for i in range(len(boxes)):
        box = boxes[i]
        if box.probability == 0:
            continue
        for j in range(i + 1, len(boxes)):
            next_box = boxes[j]
            if intersection_over_union(box, next_box) >= .4:
                boxes[j].probability = 0.
    boxes = [b for b in boxes if b.probability > 0.]
    return boxes


def intersection_over_union(a, b):
    return box_intersection(a, b) / box_union(a, b)


def box_intersection(a, b):
    width = overlap(a.x, a.width, b.x, b.width)
    height = overlap(a.y, a.height, b.y, b.height)
    if width < 0 or height < 0:
        return 0
    area = width * height
    return area


def box_union(a, b):
    intersection = box_intersection(a, b)
    union = a.width * a.height + b.width * b.height - intersection
    return union


def overlap(x, a, x1, a1):
    l1 = x - a / 2.
    l2 = x1 - a1 / 2.
    left = max(l1, l2)
    r1 = x + a / 2.
    r2 = x1 + a1 / 2.
    right = min(r1, r2)
    return right - left
