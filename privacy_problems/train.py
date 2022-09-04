import cv2 as cv
import time
import numpy as np
import argparse
from consts import DETECTORS

from recognizer import Encoder
from consts import TRAINING_DATA_PATH, FACE_ENCODING_DATA, FACE_ENCODING_LABEL


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--detector_type", help=f"Detector Type {','.join(DETECTORS)}", type=str, default = "ssd")
    args = parser.parse_args()
    return args

args = get_args()
detector_type = args.detector_type
encoder = Encoder()
image_path = TRAINING_DATA_PATH.joinpath(detector_type)
if not image_path.exists():
    print("No data to be trained")
    exit(0)


faces = []
labels = []
for path in image_path.iterdir():
    for face in path.iterdir():
        if face.is_file() and face.suffix == '.jpg':
            image = cv.imread(str(face))
            landmarks = encoder.get_face_landmark(image)
            encoded_image = encoder.get_face_encoding(image, landmarks)
            faces.append(encoded_image)
            labels.append(face.name)

faces = np.array(faces)
labels = np.array(labels)

FACE_ENCODING_DATA.parent.mkdir(parents=True, exist_ok=True)
np.save(FACE_ENCODING_DATA.as_posix(), faces)
np.save(FACE_ENCODING_LABEL.as_posix(), labels)