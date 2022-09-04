import consts
import cv2 as cv
import numpy as np
import pandas as pd
from face import Face
from utilities import convert_image_to_gray
from typing import List, Tuple, Union, ClassVar, Any

class Detector(object):
    """
        Detector class that has various detection mechanism for face.
        Currently, SSD and HaarCascade model is being used and Open CV
        is used for face detection
    """
    faces: List[Face] = list()
    detector: Any   
    labels: Any
    confidence: float
    detect: Any
    def __init__(self, detector_type: str) -> None:
        if detector_type not in consts.DETECTORS:
            print("wrong")
        if detector_type == "ssd":
            self.detect = self.detect_face_ssd
            self.initialize_ssd()
        elif detector_type == "ssd_andy":
            self.detect = self.detect_ssd_andy
            self.initialize_ssd()
        elif detector_type == "haarcascade":
            self.detect = self.detect_haarcascade
            self.initialize_haarcascade()
        super().__init__()

    def initialize_ssd(self, confidence=0.9) -> None:
        """
            Initializes detectors and other variables for SSD
        """
        self.confidence = confidence
        self.detector = cv.dnn.readNetFromCaffe(consts.SSD_PROTOTEXT.as_posix(), consts.SSD_MODEL.as_posix())
        self.labels = ["img_id", "is_face", "confidence", "left", "top", "right", "bottom"]

    def initialize_haarcascade(self) -> None:
        """
            Initializes detectors and other variables for Haarcascade model
        """
        self.detector = cv.CascadeClassifier()
        if consts.HAARCASCADE_FACE_MODEL.exists():
            self.detector.load(consts.HAARCASCADE_FACE_MODEL.as_posix())

    def detect_face_ssd(self, frame: np.ndarray ) -> List[Face]:
        """
            Face detection algorithm using ssd that detects multiple faces
        """
        self.faces = []
        image = frame.copy()  
        original_size = frame.shape
        image = cv.resize(image, consts.SSD_IMAGE_TARGET_SIZE)    
        aspect_ratio_x = (original_size[1] / consts.SSD_IMAGE_TARGET_SIZE[1])
        aspect_ratio_y = (original_size[0] / consts.SSD_IMAGE_TARGET_SIZE[0])
        
        imageBlob = cv.dnn.blobFromImage(image, 1.0, consts.SSD_IMAGE_TARGET_SIZE, (104.0,177.0,123.0), True)
        self.detector.setInput(imageBlob)
        detections = self.detector.forward()

        detections_df = pd.DataFrame(detections[0][0], columns = self.labels)
            
        detections_df = detections_df[detections_df["is_face"] == 1] #0: background, 1: face
        detections_df = detections_df[detections_df["confidence"] >= 0.90]
        
        detections_df["left"] = (detections_df["left"] * 300).astype(int)
        detections_df["bottom"] = (detections_df["bottom"] * 300).astype(int)
        detections_df["right"] = (detections_df["right"] * 300).astype(int)
        detections_df["top"] = (detections_df["top"] * 300).astype(int)

        for i, instance in detections_df.iterrows():
            confidence_score = str(round(100*instance["confidence"], 2))+" %"
            left = int(instance["left"] * aspect_ratio_x)
            right = int(instance["right"] * aspect_ratio_x)
            bottom = int(instance["bottom"] * aspect_ratio_y)
            top = int(instance["top"] * aspect_ratio_y)
            
            detected_face = frame[top : bottom, left : right]
            if detected_face.shape[0] > 0 and detected_face.shape[1] > 0:
                face = Face(left, top, right - left, bottom - top, frame )
                self.faces.append(face)
        return self.faces

    def detect_ssd_andy(self, frame: np.ndarray) -> List[Face]:
        """
            Face detection algorithm using ssd that detects a single faces
        """
        self.faces = []
        (h,w) = frame.shape[:2]
        img = cv.resize(frame, consts.SSD_IMAGE_TARGET_SIZE)
        blob = cv.dnn.blobFromImage(img,1.0, consts.SSD_IMAGE_TARGET_SIZE, (104.0,177.0,123.0))
        self.detector.setInput(blob)
        detections = self.detector.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            count = 0
            if confidence > self.confidence:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = Face(startX, startY, endX - startX, endY - startY, frame )
                self.faces.append(face)

                # face = frame[startY:endY,startX:endX]
                # frame[startY:endY,startX:endX] = face
    
            return self.faces

    def detect_haarcascade(self, frame: np.ndarray) -> List[Face]:
        """
            Face detection algorithm using haarcascade that detects multiple faces
        """
        self.faces = []
        frame_gray = convert_image_to_gray(frame)
        faces = self.detector.detectMultiScale(frame_gray, scaleFactor=1.5, minNeighbors=5)
        for (x,y,w,h) in faces:
            if w > 0:
                face = Face(x, y, w, h, frame )
                self.faces.append(face)

        return self.faces
if __name__ == "__main__":
    detector = Detector("ssd")
    print(detector.get_faces())