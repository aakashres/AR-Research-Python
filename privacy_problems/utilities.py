from typing import Any
import dlib
import cv2 as cv
import numpy as np
from imutils.face_utils import FaceAligner

def convert_image_to_gray(image: np.ndarray) -> np.ndarray:
    """
        Returns a gray image frame of given image
    """
    gray_image= cv.cvtColor(image.copy(), cv.COLOR_BGR2GRAY)
    gray_image = cv.equalizeHist(gray_image)
    return gray_image

    
def draw_rectangle(image, rect):
    """
        Draws rectangle over some object
    """
    (x, y, w, h) = rect
    cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

def draw_text(img, text, x, y):
    """
        Write some text in the frame
    """
    cv.putText(img, text, (x, y), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def get_shape_predictor() -> Any:
    return dlib.shape_predictor(DLIB_SHAPE_PREDICTOR_MODEL.as_posix())


def get_face_aligner() -> FaceAligner:
    return FaceAligner(dlib.shape_predictor(DLIB_SHAPE_PREDICTOR_MODEL.as_posix()))


def get_face_encoder() -> Any:
    return dlib.face_recognition_model_v1(DLIB_FACE_ENCODE_MODEL.as_posix())