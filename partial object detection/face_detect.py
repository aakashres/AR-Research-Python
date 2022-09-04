from threading import Thread
import cv2
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


class FaceDetection:
    def __init__(self, confidence_set=0.9):
        self.confidence_set = confidence_set
        self.prototxt = "../models/Res_10/Res_10_300x300_ssd_iter_140000.txt"
        self.model = "../models/Res_10/Res_10_300x300.caffemodel"
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt, self.model)

    # def start(self):
    #     Thread(target=self.update, args=()).start()
    #     return self

    def update(self, image):
        x, y, w, h = 0, 0, 0, 0
        # update the images
        (h, w) = image.shape[:2]
        x, y = -w, -h
        img = cv2.resize(image, (300, 300))
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        self.detections = self.net.forward()
        for i in np.arange(0, self.detections.shape[2]):
            confidence = self.detections[0, 0, i, 2]
            if confidence > self.confidence_set:
                idx = int(self.detections[0, 0, i, 1])
                box = self.detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                x, y, w, h = startX, startY, endX - startX, endY - startY
                break
        return x, y, w, h

    def detect_or_track(self, frame, detect_or_track, tracker):
        if True:
            # Detect faces or start tracking
            bbox = (x, y, w, h) = self.update(frame)
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
