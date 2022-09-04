import time
import cv2 as cv
import copy
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
import sys
# Append paths for import modules from other locations
sys.path.insert(0,'../')

from utils import CvFpsCalc, Draw_Detections
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--name", help='Name',type=str,default = 'test')
    parser.add_argument("-c","--confidence", help='Confience Interval', type=float,default=0.9)
    parser.add_argument("--device", type=int, default=0)

    args = parser.parse_args()

    return args




prototxt = '../models/Res_10/Res_10_300x300_ssd_iter_140000.txt'
model = '../models/Res_10/Res_10_300x300.caffemodel'
net = cv.dnn.readNetFromCaffe(prototxt,model)


TRAINING_DATA_PATH = Path("./training_data")
TRAINING_DATA_PATH.mkdir(parents=True,exist_ok=True)

def main():
    # Argument parsing #########################################################
    args = get_args()

    cap_device = args.device
    name = args.name
    confidence = args.confidence
    ############################################################################

    # Camera preparation #######################################################
    cap = cv.VideoCapture(cap_device)

    # FPS Measurement ##########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)


    # Prepare Objects ##########################################################
    Detections_obj = Draw_Detections(confidence)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            key = cv.waitKey(1) & 0xFF

            (h,w) = frame.shape[:2]
            img = cv.resize(frame,(300,300))
            blob = cv.dnn.blobFromImage(img,1.0, (300, 300), (104.0,177.0,123.0))
            net.setInput(blob)
            detections = net.forward()


            face,frame = Detections_obj.draw(frame,detections)
            frame = cvFpsCalc.draw(frame)

            if key == ord("a"):
                if face is not None:
                    capture_path = Path.joinpath(TRAINING_DATA_PATH, name)
                    capture_path.mkdir(exist_ok=True)
                    cv.imwrite(str(capture_path.joinpath(f"{name}_{time.time()}.jpg")), face)
                    print("Captured Face")
                else:
                    print("No Face Detected")


            # finally show frame
            cv.imshow('frame',frame)
            # Press q to close the video windows before it ends if you want
            if key == ord("q"):
                break
        else:
            break
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
