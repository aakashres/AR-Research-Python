from face import Face
from typing import Any
import dlib
import numpy as np
from imutils.face_utils import FaceAligner
from consts import DLIB_SHAPE_PREDICTOR_MODEL, DLIB_FACE_ENCODE_MODEL, FACE_ENCODING_DATA, FACE_ENCODING_LABEL
from utilities import convert_image_to_gray
class Recognizer(object):
    def __init__(self) -> None:
        self.faces = np.load(FACE_ENCODING_DATA.as_posix())
        self.labels = np.load(FACE_ENCODING_LABEL.as_posix())
        super().__init__()

    def recognize(self, face_encoding: Any) -> str:
        score = np.linalg.norm(self.faces - np.array(face_encoding), axis=1)
        imatches = np.argsort(score)
        score = score[imatches]
        return self.labels[imatches][0].split("_")[0]


class Encoder(object):
    pose_predictor: Any
    face_aligner: Any
    face_encoder: Any
    def __init__(self) -> None:
        self.pose_predictor = dlib.shape_predictor(DLIB_SHAPE_PREDICTOR_MODEL.as_posix())
        self.face_aligner = FaceAligner(self.pose_predictor)
        self.face_encoder = dlib.face_recognition_model_v1(DLIB_FACE_ENCODE_MODEL.as_posix())
        super().__init__()

    def get_aligned_face(self, face: Face) -> np.ndarray:
        return self.face_aligner.align(
            face.frame,
            convert_image_to_gray(face.frame),
            dlib.rectangle(face.x, face.y, face.x + face.w, face.y + face.h)
        )

    def get_face_landmark(self, aligned_face: np.ndarray):
        return self.pose_predictor(
            aligned_face, 
            dlib.rectangle(0,0,aligned_face.shape[0],aligned_face.shape[1])
        )
    
    def get_face_encoding(self, aligned_face: np.ndarray, face_landmark: Any, num_jitters: int = 2) -> np.ndarray:
        return self.face_encoder.compute_face_descriptor(aligned_face, face_landmark, num_jitters=num_jitters)

    def get_encoded_face(self, face: Face) -> np.ndarray:
        aligned_face = self.get_aligned_face(face)
        landmark = self.get_face_landmark(aligned_face)
        return self.get_face_encoding(aligned_face, landmark)
