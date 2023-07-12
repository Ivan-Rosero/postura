import cv2
import time
import numpy as np
import numpy as np
import json
import base64

INPUT_SIZE = 256
NAMES = []

# Preload models for performance
print("[INFO] loading YOLO...")
net = cv2.dnn.readNetFromDarknet("./yolo_configs/yolov3-obj.cfg", "./yolo_configs/posture_yolov3.weights")

print("[INFO] loading labels...")
with open("./yolo_configs/posture.names", 'rt') as f:
    NAMES = f.read().rstrip('\n').split('\n')

# Assign colors for drawing bounding boxes
COLORS = [
    [0, 200, 0], [20, 45, 144],
    [157, 224, 173], [0, 0, 232],
    [26, 147, 111], [40, 44, 100]
]


def rescale_image(input_img):
    #height, wide, _: channels
    h, w, _ = input_img.shape

    # Resize if height is more than 1000px. First numerical + 1, will be the ratio to scale to.
    # Eg. 2540px, 2540px / ( 2 + 1 ) = new height.
    return input_img if h < 1000 else cv2.resize(input_img, (int(w / (int(str(h)[0]) + 2)), int(h / (int(str(h)[0]) + 2))))


def predict_yolo(input_img):
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(input_img, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False)
    net.setInput(blob)

    return net.forward(ln)


def draw_bound(input_img, layer_outputs, confidence_level, threshold):
    boxes = []
    confidences = []
    class_id = []
    results = []
    color = []

    H, W, _ = input_img.shape
    #x, y, centerX, centerY, width, height

    # cdef float confidence

    # cdef str text

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_ids = np.argmax(scores)
            confidence = scores[class_ids]

            if confidence > confidence_level:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_id.append(class_ids)

    # Non maxima
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_level, threshold)

    if len(idxs) > 0:
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[class_id[i]]]
            cv2.rectangle(input_img, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(NAMES[class_id[i]], confidences[i])
            cv2.putText(input_img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            results.append([NAMES[class_id[i]], confidences[i]])

    return [input_img, results]


def predict(f):
    start = time.time()
    
    filePath="./json/posture.json"
    with open(filePath, 'r', encoding='utf-8') as file:
        jsonData = json.load(file)

    im = cv2.imdecode(np.fromstring(f.read(), np.uint8), cv2.IMREAD_COLOR)
    im = rescale_image(im)

    layer_outputs = predict_yolo(im)
    results = draw_bound(im, layer_outputs, 0.5, 0.4)

    predictions = []
    message = []
    reference = jsonData["reference"]
    not_found = []
    response = []

    for i in results[1]:
        status = i[0]
        predictions.append(status)

    if len(predictions) == 0:
        message.append(jsonData["suggestion"][8]['9'])
    elif len(predictions) == 1:
        for j in predictions:
            coincidence = False
            for i, sublist in enumerate(reference):
                for element in sublist:
                    if element == j:
                        coincidence = True
                        break
                if not coincidence:
                    not_found.append(sublist)
    elif len(predictions) == 2:
        for i, sublist in enumerate(reference):
            if i < len(predictions):
                coincidence = False
                for element in sublist:
                    if element == predictions[i]:
                        coincidence = True
                        break
                if not coincidence:
                    not_found.append(sublist)
    
    for sublist in not_found:
        if sublist == reference[0]:
            message.append(jsonData["suggestion"][5]['6'])
        elif sublist == reference[1]:
            message.append(jsonData["suggestion"][6]['7'])
        elif sublist == reference[2]:
            message.append(jsonData["suggestion"][7]['8'])
      
    for element in predictions:
        if element == reference[0][0]:
            response.append(jsonData["response"][2]['3'])
        elif element == reference[1][0]:
            response.append(jsonData["response"][3]['4'])
        elif element == reference[2][0]:
            response.append(jsonData["response"][4]['5'])
        elif element == reference[0][1]:
            response.append(jsonData["response"][5]['6'])
        elif element == reference[1][1]:
            response.append(jsonData["response"][6]['7'])
        elif element == reference[2][1]:
            response.append(jsonData["response"][7]['8'])
    
    end = time.time()

    return [{
        "prediction": json.dumps(results[1]),
        "duration": "{:.4f} seconds".format(end - start),
        "message": message,
        "results": response
    }]

