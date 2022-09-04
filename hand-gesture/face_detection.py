from threading import Thread
import configparser
import cv2
import numpy as np


def anonymize_face_pixelate(image, blocks=15):
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


class FaceDetection:
    def __init__(self, confidence_set=0.9):
        self.confidence_set = confidence_set
        # self.prototxt = "../models/Res_10/Res_10_300x300_ssd_iter_140000.txt"
        self.prototxt = "../models/MobileNetSSD/MobileNetSSD_deploy.prototxt.txt"
        # self.model = "../models/Res_10/Res_10_300x300.caffemodel"
        self.model = "../models/MobileNetSSD/MobileNetSSD_deploy.caffemodel"
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt, self.model)
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self, image):
        # update the images
        (h, w) = image.shape[:2]
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
                face = image[startY:endY, startX:endX]

                face = anonymize_face_pixelate(face)
            return image

    # def detections(self):
