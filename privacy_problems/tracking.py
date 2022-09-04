import uuid
import face_recognition
import cv2
import dlib
from face_recognition.api import face_encodings
import numpy as np

video_capture = cv2.VideoCapture(0)

face_locations = []
trackers = []
frame_count = 0
face_trackers = {}
while True:
    ret, frame = video_capture.read()
    image_cpy = frame.copy()

    if image_cpy is None:
        break

    small_frame = cv2.resize(image_cpy, (0, 0), fx=0.25, fy=0.25)


    if frame_count % 10 == 0:
        face_trackers = {}
        face_locations = face_recognition.face_locations(small_frame)
        if face_locations:
            # face_encodings = face_recognition.face_encodings(small_frame, face_locations)
            for i, (top, right, bottom, left) in enumerate(face_locations):
                print(i)
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(left, top, right, bottom)
                tracker.start_track(image_cpy, rect)
                face_trackers[uuid.uuid4()] = tracker

    for fid, tracker in face_trackers.items():
        tracker.update(image_cpy)
        pos = tracker.get_position()
        startX = int(pos.left())
        startY = int(pos.top())
        endX = int(pos.right()) 
        endY = int(pos.bottom())
        cv2.rectangle(image_cpy, (startX, startY), (endX, endY), (0, 255, 0), 3)
    frame_count += 1
    cv2.imshow('Video', image_cpy)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
