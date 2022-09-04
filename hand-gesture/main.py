import sys
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
import time

import cv2 as cv
import numpy as np
import mediapipe as mp
import csv

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

sys.path.insert(0,'../')
from utils import CvFpsCalc
#from model import KeyPointClassifier
#from model import PointHistoryClassifier
from model import KeyPointHistoryClassifier
from landgesture_history import LandMark_History,LandMarks
from face_detection import FaceDetection
from VideoGet import VideoGet
from CountsPerSec import CountsPerSec

def putIterationsPerSec(frame, iterations_per_sec):
    """
    Add iterations per second text to lower-left corner of a frame.
    """

    cv.putText(frame, "{:.0f} iterations/sec".format(iterations_per_sec),
        (10, 450), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    return frame

def calc_dist(image,hand_center,face_center):
    """
    Need to use relative coordinates instead of pixel coord
    """
    h,w,c = image.shape
    if hand_center and face_center:
        hand_center = normalizes_coordinates(hand_center,h,w)
        face_center = normalizes_coordinates(face_center,h,w)
        x_dist = (hand_center[0] - face_center[0])**2
        y_dist = (hand_center[1] - face_center[1])**2
        distance = np.sqrt(x_dist + y_dist)
        return distance

def label_center(results,debug_image):
    if results.multi_hand_landmarks:
        tempLandMark = LandMarks(debug_image,results.multi_hand_landmarks[0])
        test_output = tempLandMark._landmark_point
        labels_2_avg = [0,2,5,17]
        x = []
        y = []
        for i,points in enumerate(test_output):
            if i in labels_2_avg:
                x.append(points[0])
                y.append(points[1])
        centeroid = (int(sum(x)/len(x)),int(sum(y)/len(y)))
                # print(centeroid,end='')
        cv.circle(debug_image, centeroid, 5, (0,0,255), cv.FILLED)

def label_fingers(results,image):
    """
    Label the fingers and returns the centroid
    """
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x *w), int(lm.y*h)
                cv.circle(image, (cx,cy), 3, (255,0,255), cv.FILLED)
            mp_drawing.draw_landmarks(image,hand_landmarks,mp_hands.HAND_CONNECTIONS)

            tempLandMark = LandMarks(image,hand_landmarks)
            pixel_dist = tempLandMark._landmark_point
            labels_2_avg = [0,2,5,17]
            x = []
            y = []
            for i,points in enumerate(pixel_dist):
                if i in labels_2_avg:
                    x.append(points[0])
                    y.append(points[1])
            pix_centeroid = (int(sum(x)/len(x)),int(sum(y)/len(y)))
            cv.circle(image, pix_centeroid, 5, (255,0,0), cv.FILLED)
            return pix_centeroid


def detect_face(frame,net,confidence_set,toggle):
    (h,w) = frame.shape[:2]
    img = cv.resize(frame,(300,300))
    blob = cv.dnn.blobFromImage(img,1.0, (300, 300), (104.0,177.0,123.0))
    net.setInput(blob)
    detections = net.forward()
    count = 0

    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > confidence_set:
        #     continue
        # count +=1
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = frame[startY:endY,startX:endX]
            if toggle:
                face = pixelate(frame)
                frame[startY:endY,startX:endX] = face

    return frame

def pixelate(image, blocks=15):
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
            (B, G, R) = [int(x) for x in cv.mean(roi)[:3]]
            cv.rectangle(image, (startX, startY), (endX, endY),
                (B, G, R), -1)
    # return the pixelated blurred image
    return image

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=640)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)
    parser.add_argument("--mode",help='Mode for either recording or classifying',
                        type=int,
                        default=0)
    parser.add_argument("--number",help='Number for Gestures',
                        type=int,
                        default=1)
    parser.add_argument("-d",help='Debug',action='store_true')
    args = parser.parse_args()

    return args




def main():
    # Argument parsing #########################################################
    args = get_args()
    debug = args.d
    


    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    mode = args.mode
    number = args.number
    use_brect = True
    ############################################################################

    toggle = False


    # Camera preparation #######################################################
    # cap = cv.VideoCapture(cap_device)
    VG = VideoGet(cap_device).start()
    # cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    # cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Read Labels ##############################################################
    with open('model/keypoint_history_classifier/keypoint_history_classifier_label.csv',
      encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
        row[0] for row in keypoint_classifier_labels
    ]

    # Model load ###############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


    keypoint_history_classifier = KeyPointHistoryClassifier()
    landmark_history = LandMark_History()
    face_history = FaceDetection()
    cps = CountsPerSec().start()


    cvFpsCalc = CvFpsCalc(buffer_len=10)
    frame_count = 0

    # Coordinate history #######################################################
    history_length = 10
    gesture_history = deque(maxlen=history_length)

    while True:
        fps = cvFpsCalc.get()
        key = cv.waitKey(10)
        # Face Capture #########################################################
#         confidence_set = 0.9
#         prototxt = '../models/Res_10/Res_10_300x300_ssd_iter_140000.txt'
#         model = '../models/Res_10/Res_10_300x300.caffemodel'
#         net = cv.dnn.readNetFromCaffe(prototxt,model)

        if key == ord('q'):  # ESC
            VG.stop()
            break

        # Camera capture #####################################################
        # ret, image = cap.read()
        ret = VG.grabbed
        image = VG.frame
        if not ret:
            break
        # image = cv.flip(image, 1/)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################
        # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        ####
        landmark_history.update(image,results)
        # cv.putText(debug_image, str(np.round(fps,2)), (0,25), cv.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
        debug_image = putIterationsPerSec(debug_image, cps.countsPerSec())
        cps.increment()

        if mode == 1:
            gesture_id = -1

            if results.multi_hand_landmarks is not None:
                if None not in landmark_history._landmark_list:
                    test_landmark = landmark_history.export_single_landmark(-1)
                    centroid_2 = test_landmark.centeroid()
                    cv.circle(debug_image, centroid_2, 10, (255,0,0), cv.FILLED)
                    hand_id = keypoint_history_classifier(landmark_history.return_array())
                    gesture = keypoint_classifier_labels[hand_id]
                    cv.putText(debug_image, str(gesture), (0,50), cv.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)

        elif mode == 2:
            # cv.putText(debug_image, "Pixelation: "+str(toggle), (0,50), cv.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
            debug_image = face_history.update(debug_image)

        elif mode == 3:
            if key == ord('a'):
                toggle = toggle_switch(toggle)

                file = 'model/keypoint_history_classifier/data/arr_' + str(number) + '.csv'
                landmark_history.export(file,number)
        # Mode 4
        elif mode == 4:
            gesture_id = -1

            if results.multi_hand_landmarks is not None:
                if None not in landmark_history._landmark_list:
                    test_landmark = landmark_history.export_single_landmark(-1)
                    centroid_2 = test_landmark.centeroid()
                    cv.circle(debug_image, centroid_2, 10, (255,0,0), cv.FILLED)
                    hand_id = keypoint_history_classifier(landmark_history.return_array())
                    gesture = keypoint_classifier_labels[hand_id]
                    cv.putText(debug_image, str(gesture), (0,50), cv.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)

            debug_image = face_history.update(debug_image)

        # cv.putText(debug_image,str(toggle), (0,25), cv.FONT_HERSHEY_PLAIN, 1.5, (0,0,255), 2)
        # Export corrdinates
        hand_center = label_fingers(results,debug_image)
        # dist = calc_dist(hand_center,face_center)
        label_center(results,debug_image)


        # identify face in the system:

        # debug_image = detect_face(debug_image,net,confidence_set,toggle)
        # print(keypoint_classifier_labels[int(gesture_id)])

        cv.imshow('Hand Gesture Recognition', debug_image)

def toggle_switch(toggle):
    if toggle == False:
        toggle = True
    elif toggle == True:
        toggle = False
    return toggle


if __name__ == '__main__':
    main()
