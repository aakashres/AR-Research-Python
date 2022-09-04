import dlib
import time
import argparse
import cv2 as cv
from face import save_face_for_training
from consts import DETECTORS, TRAINING_PHOTO_COUNT
from detectors import Detector
from utilities import draw_text

from recognizer import Encoder, Recognizer


import sys
# Append paths for import modules from other locations
sys.path.insert(0,'../')
from utils import CvFpsCalc



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--confidence", help="Confidence Interval", type=float, default=0.9)
    parser.add_argument("-d", "--device", help="Input stream for video", type=int, default=0)
    parser.add_argument("-n", "--name", help="Name for the face. Necessary is save_face option is True", type=str, default="test"), 
    parser.add_argument("-t","--detector_type", help=f"Detector Type {','.join(DETECTORS)}", type=str, default = "ssd")
    parser.add_argument("-s","--save_face", help=f"Save faces for training. Any input value will be True. Leave empty for False", type=bool, default = False)
    parser.add_argument("-r","--recognize", help=f"Save faces for training. Any input value will be True. Leave empty for False", type=bool, default = False)
    args = parser.parse_args()
    return args



def main():
    # Argument parsing #########################################################
    args = get_args()

    cap_device = args.device
    name = args.name
    confidence = args.confidence
    detector_type = args.detector_type
    save_face = args.save_face
    if save_face and name == "test":
        print("Please provide the name for the face")
        exit(0)
    ############################################################################

    # Camera preparation #######################################################
    cap = cv.VideoCapture(cap_device, cv.CAP_DSHOW)
    if not cap.isOpened:
        print("Error opening video capture")
        exit(0)
    face_detector = Detector(detector_type)
    # for saving face
    last_capture = 0
    photo_count = 0
    encoder = Encoder()
    recognizer = Recognizer()
    frame_counter = 0
    fps_counter = CvFpsCalc(10)
    while True:
        ret, frame = cap.read()
        if frame is None:
            print("No captured frame -- Break!")
            break
        faces = face_detector.detect(frame)
        if frame_counter % 20 == 0:
            for face in faces:
                face_encoding = encoder.get_encoded_face(face)
                if face_encoding:
                    name = recognizer.recognize(face_encoding[0])
                    face.set_name(name)
    
        # to save faces
        if save_face:
            if photo_count >= TRAINING_PHOTO_COUNT:
                save_face = False
            if len(faces) != 1:
                print("Invalid number of faces. Must be one face")
                continue
            if time.time() - last_capture > 1:
                save_face_for_training(name, detector_type, faces[0], encoder)
                last_capture = time.time()
                photo_count += 1
            draw_text(frame, f"Capturing photo. {TRAINING_PHOTO_COUNT - photo_count} remaining", 10, 20)

        for face in faces:
            face.draw_rectangle()
        frame_counter += 1 
        fps_counter.draw(frame)
        cv.imshow("Capture - Face detection", frame)
        if cv.waitKey(10) == 27:
            break

if __name__ == "__main__":
    main()

import face_recognition
