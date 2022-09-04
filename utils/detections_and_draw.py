import numpy as np
import cv2 as cv

class Draw_Detections:
    def __init__(self,confidence_set=0.9):
        self.confidence_set = confidence_set
        # Assumes detections is a 4d array

        # self.detections = detections


    def draw(self,frame,detections):
        (h,w) = frame.shape[:2]


        for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
            confidence = detections[0, 0, i, 2]
            count = 0
    		# filter out weak detections by ensuring the `confidence` is
    		# greater than the minimum confidence
            if confidence > self.confidence_set:
                # 	continue
                # count +=1
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = frame[startY:endY,startX:endX]
                # draw the prediction on the frame

                # label = "{}: {:.2f}%".format(CLASSES[idx],confidence * 100)
                frame[startY:endY,startX:endX] = face

                text = "{:.2f}%".format(confidence * 100) + ", Count " + str(count)
                cv.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 2)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv.putText(frame, text, (startX, y),cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

            else:
                face = None

            return face,frame
# def detect_face(classifier, frame):
#     #convert the test image to gray image as open cv face detector expects gray images
#     frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     frame_gray = cv.equalizeHist(frame_gray)
#
#     #let's detect multiscale (some images may be closer to camera than others) images
#     #result is a list of faces
#     faces = classifier.detectMultiScale(frame_gray, scaleFactor=1.5, minNeighbors=5)
#     #if no faces are detected then return original img
#     if (len(faces) == 0):
#         return None, None
#
#     #under the assumption that there will be only one face,
#     #extract the face area
#     (x, y, w, h) = faces[0]
#
#     #return only the face part of the image
#     return frame[y:y+w, x:x+h], faces[0]

# def capture(classifier, frame, name):
#     captured = False
#     face, rect = detect_face(classifier, frame)
#     if face is not None:
#         capture_path = Path.joinpath(TRAINING_DATA_PATH, name)
#         capture_path.mkdir(exist_ok=True)
#         cv.imwrite(str(capture_path.joinpath(f"{name}_{time.time()}.jpg")), face)
#         captured = True
#     return captured
