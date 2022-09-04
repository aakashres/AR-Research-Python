from pathlib import Path
import time
import cv2 as cv
import numpy as np
from typing import List, Tuple
from consts import TRAINING_DATA_PATH
from utilities import convert_image_to_gray, draw_rectangle, draw_text


class Face(object):

    def __init__(self, x: int, y: int, w: int, h: int, image: np.ndarray) -> None:
        self.x, self.y, self.w, self.h = (x, y, w, h)
        self.face_rect = (x, y, w, h)
        self.frame = image
        self.name = None
        self.face = self.frame[self.y: self.y + self.h,  self.x : self.x + self.w]
        super().__init__()
    
    def set_name(self, name) -> None:
        self.name = name

    def get_gray_face(self) -> np.ndarray:
        return convert_image_to_gray(self.face)

    def draw_rectangle(self) -> None:
        draw_rectangle(self.frame, self.face_rect)
        if self.name:
            draw_text(self.frame, self.name, self.x, self.y -5)


def save_face_for_training(name: str, detector_type: str, face: Face, encoder) -> None:
    face_data_path = TRAINING_DATA_PATH.joinpath(detector_type, name)
    face_data_path.mkdir( parents=True, exist_ok=True )
    cv.imwrite(face_data_path.joinpath(f"{name}_{time.time()}.jpg").as_posix(), encoder.get_aligned_face(face))