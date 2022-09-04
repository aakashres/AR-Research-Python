import time
import pickle
import cv2 as cv
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

ACTIONS = ["train","capture", "detect", "recognize"]
TRAINING_DATA_PATH = Path("./training_data")
RECOGNIZER_MODEL_FILE = Path("./recognizer_model/known_faces.xml")
RECOGNIZER_LABEL_FILE = Path("./recognizer_model/labels.db")
TRAINING_PHOTO_COUNT = 5

face_cascade = cv.CascadeClassifier()
if not face_cascade.load(cv.samples.findFile("./opencv_files/haarcascade_frontalface_alt.xml")):
    print("Error loading face cascade")
    exit(0)
face_recognizer = cv.face.LBPHFaceRecognizer_create()


def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

def draw_text(img, text, x, y):
    cv.putText(img, text, (x, y), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def anonymize_face_pixelate(frame, face, rect, blocks=15):
	# divide the input image into NxN blocks
	(x, y, w, h) = rect
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
			roi = face[startY:endY, startX:endX]
			(B, G, R) = [int(x) for x in cv.mean(roi)[:3]]
			cv.rectangle(face, (startX, startY), (endX, endY),
				(B, G, R), -1)
	# return the pixelated blurred image
	frame[y:y+w, x:x+h] = face

def detect_face(classifier, frame):
    #convert the test image to gray image as open cv face detector expects gray images
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    #let's detect multiscale (some images may be closer to camera than others) images
    #result is a list of faces
    faces = classifier.detectMultiScale(frame_gray, scaleFactor=1.5, minNeighbors=5)
    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None

    #under the assumption that there will be only one face,
    #extract the face area
    (x, y, w, h) = faces[0]

    #return only the face part of the image
    return frame[y:y+w, x:x+h], faces[0]

def detect_faces(classifier, frame):
    #convert the test image to gray image as open cv face detector expects gray images
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    #let's detect multiscale (some images may be closer to camera than others) images
    #result is a list of faces
    faces = classifier.detectMultiScale(frame_gray, scaleFactor=1.5, minNeighbors=5)
    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None
    return faces

def train():
    faces = []
    labels = []
    for path in TRAINING_DATA_PATH.iterdir():
        for face in path.iterdir():
            if face.is_file() and face.suffix == '.jpg':
                image = cv.imread(str(face))
                image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                image = cv.equalizeHist(image)
                faces.append(image)
                labels.append(face.name)
    if faces and labels:
        with RECOGNIZER_LABEL_FILE.open("wb") as labelfile:
            pickle.dump(labels, labelfile)
        face_recognizer.train(faces, np.array(list(range(len(faces)))))
        face_recognizer.write(str(RECOGNIZER_MODEL_FILE))




def capture(classifier, frame, name):
    captured = False
    face, rect = detect_face(classifier, frame)
    if face is not None:
        capture_path = Path.joinpath(TRAINING_DATA_PATH, name)
        capture_path.mkdir(exist_ok=True)
        cv.imwrite(str(capture_path.joinpath(f"{name}_{time.time()}.jpg")), face)
        captured = True
    return captured



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
    parser.add_argument("-a", "--action", help="Train The face recognition model or detect faces. Note: First Train the model and only run detection", default="train")
    parser.add_argument("-n", "--name", help="Name of person in frame", default="")
    args = parser.parse_args()
    action = args.action
    if action not in ACTIONS:
        print(f"Invalid actions. Must be one of ({'|'.join(ACTIONS)})")
        exit(0)
    if action == "capture" and not args.name:
        print(f"Name is mandatory for capture action")
        exit(0)
    name = args.name
    if action != "train":
        cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        if not cap.isOpened:
            print("Error opening video capture")
            exit(0)
    if action == "capture":
        last_capture = 0
        #-- 1. Load the cascades
        photo_count = 0
        while True:
            ret, frame = cap.read()
            if frame is None:
                print("No captured frame -- Break!")
                break
            cv.imshow("Capture - Face detection", frame)
            if time.time() - last_capture > 2:
            	key = cv2.waitKey(1) & 0xFF

                if capture(face_cascade, frame, name):
                    last_capture = time.time()
                    photo_count += 1
            if cv.waitKey(10) == 27:
                break
                
    elif action == "train":
        train()


    elif action == "recognize":
        face_recognizer.read(str(RECOGNIZER_MODEL_FILE))
        labels = []
        with RECOGNIZER_LABEL_FILE.open("rb") as labelfile:
            labels = pickle.load(labelfile)
        while True:
            ret, frame = cap.read()
            if frame is None:
                print("No captured frame -- Break!")
                break
            # face, rect = detect_face(face_cascade, frame)
            faces = detect_faces(face_cascade, frame)
            for rect in faces:
                if rect is not None:
                    (x, y, w, h) = rect
                    face = frame[y:y+w, x:x+h]
                    face_gray = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
                    face_gray = cv.equalizeHist(face_gray)
                    label_index, confidence = face_recognizer.predict(face_gray)
                    draw_rectangle(frame, rect)
                    name = labels[label_index].split("_")[0]
                    if name:
                        anonymize_face_pixelate(frame, face, rect)
                        draw_text(frame, name, rect[0], rect[1]-5)
                    else:
                        draw_text(frame, "Unknown", rect[0], rect[1]-5)

            cv.imshow("Capture - Face detection", frame)
            if cv.waitKey(10) == 27:
                break
