import cv2
import time
import numpy as np


def anonymize_face_pixelate(image, blocks=10):
    # divide the input image into NxN blocks
    (h, w) = image.shape[:2]
    xSteps = np.linspace(0, w, blocks + 1, dtype="int")
    ySteps = np.linspace(0, h, blocks + 1, dtype="int")
    # loop over the blocks in both the x and y direction
    for i in range(1, len(ySteps)):
        for j in range(1, len(xSteps)):
            # compute the starting and ending (x, y)-coordinates
            # for the current block
            startX = xSteps[j - 1]
            startY = ySteps[i - 1]
            endX = xSteps[j]
            endY = ySteps[i]
            # extract the ROI using NumPy array slicing, compute the
            # mean of the ROI, and then draw a rectangle with the
            # mean RGB values over the ROI in the original image
            roi = image[startY:endY, startX:endX]
            (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
            cv2.rectangle(image, (startX, startY), (endX, endY), (B, G, R), -1)
    # return the pixelated blurred image
    return image


def pixelate(image, x, y, w, h):
    roi = image[y : y + h, x : x + w]
    # applying a gaussian blur over this new rectangle area
    # roi = cv2.GaussianBlur(roi, (23, 23), 30)
    roi = anonymize_face_pixelate(roi)
    # impose this blurred image on original image to get final image
    image[y : y + roi.shape[0], x : x + roi.shape[1]] = roi
    return image


# List of categories and classes
categories = {
    0: "background",
    1: "aeroplane",
    2: "bicycle",
    3: "bird",
    4: "boat",
    5: "bottle",
    6: "bus",
    7: "car",
    8: "cat",
    9: "chair",
    10: "cow",
    11: "diningtable",
    12: "dog",
    13: "horse",
    14: "motorbike",
    15: "person",
    16: "pottedplant",
    17: "sheep",
    18: "sofa",
    19: "train",
    20: "tvmonitor",
}

classes = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


class ObjectDetector:
    def __init__(self, confidence_set=0.9):
        self.confidence_set = confidence_set
        self.prototxt = "MobileNetSSD_deploy.prototxt.txt"
        self.model = "MobileNetSSD_deploy.caffemodel"
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt, self.model)
        self.RESIZED_DIMENSIONS = (300, 300)  # Dimensions that SSD was trained on.
        self.IMG_NORM_RATIO = (
            0.007843  # In grayscale a pixel can range between 0 and 255
        )
        # Load the pre-trained neural network
        self.net = cv2.dnn.readNetFromCaffe(
            "MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel"
        )

    def detect(self, frame):
        x, y, w, h = 0, 0, 0, 0
        # update the images
        (h, w) = frame.shape[:2]
        x, y = -w, -h
        frame_blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, self.RESIZED_DIMENSIONS),
            self.IMG_NORM_RATIO,
            self.RESIZED_DIMENSIONS,
            127.5,
        )
        # Set the input for the neural network
        self.net.setInput(frame_blob)
        self.detections = self.net.forward()
        for i in np.arange(0, self.detections.shape[2]):
            confidence = self.detections[0, 0, i, 2]

            # Confidence must be at least 30%
            if confidence > 0.30:
                bounding_box = self.detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = bounding_box.astype("int")
                if startX > 0 and startY > 0 and endX > 0 and endY > 0:
                    x, y, w, h = startX, startY, endX - startX, endY - startY
                    break
        return x, y, w, h

    def detect_or_track(self, frame, detect_or_track, tracker):
        if True:
            bbox = (x, y, w, h) = self.detect(frame)
            if (x + y + w + h) != 0:
                frame = pixelate(frame, x, y, w, h)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # status = tracker.init(frame, bbox)
                detect_or_track = False
        else:
            # Update tracker
            status, bbox = tracker.update(frame)
            if status:
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                detect_or_track = True

        return frame, detect_or_track, bbox
