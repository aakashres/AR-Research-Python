import uuid
import dlib
import cv2 as cv
import face_recognition
from numpy import mat


video_capture = cv.VideoCapture(0)


faceTrackers = {}
faceNames = {}
faceCentroids = []
frameCounter = 0

rectangleColor = (0, 165, 255)
font = cv.FONT_HERSHEY_SIMPLEX


while True:
    ret, frame = video_capture.read()
    if frame is None:
        continue

    small_frame = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Increase the framecounter

    fidsToDelete = []
    for fid in faceTrackers.keys():
        trackingQuality = faceTrackers[fid].update(small_frame)

        # If the tracking quality is good enough,delete tracker
        if trackingQuality < 7:
            fidsToDelete.append(fid)

    for fid in fidsToDelete:
        print("Removing fid " + str(fid) + " from list of trackers")
        faceTrackers.pop(fid, None)
    if (frameCounter % 10 ) == 0:
        face_locations = face_recognition.face_locations(small_frame)
        for top, right, bottom, left in face_locations:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            (x, y, w, h) = (left, top, right - left, bottom - top)
            x_bar = x + 0.5 * w
            y_bar = y + 0.5 * h
            
            matchedFid = None
            for fid in faceTrackers.keys():
                tracked_position = faceTrackers[fid].get_position()
                left = int(tracked_position.left())
                top = int(tracked_position.top())
                right = int(tracked_position.right()) 
                bottom = int(tracked_position.bottom())

                (t_x, t_y, t_w, t_h) = (left, top, right - left, top - bottom)

                # calculate the centerpoint
                t_x_bar = t_x + 0.5 * t_w
                t_y_bar = t_y + 0.5 * t_h

                # check if the centerpoint of the face is within the
                # rectangleof a tracker region. Also, the centerpoint
                # of the tracker region must be within the region
                # detected as a face. If both of these conditions a match occur
                if ((t_x <= x_bar <= (t_x + t_w)) and
                        (t_y <= y_bar <= (t_y + t_h)) and
                        (x <= t_x_bar <= (x + w)) and
                        (y <= t_y_bar <= (y + h))):
                    matchedFid = fid
            if not matchedFid:
                # Create and store the tracker((
                newId = uuid.uuid4()
                tracker = dlib.correlation_tracker()

                #Convert tracking rectangle to dlib rectangle
                rect = dlib.rectangle(left, top, right, bottom)
                tracker.start_track(frame, rect)
                faceTrackers[newId] = tracker
                # face_location = [face_locations[i]]
                # Start a new thread to track faceid
                # if FACEREC:
                #     t = threading.Thread(target=doRecognizePerson,
                #                             args=(face_location,baseImageColor,
                #                                 MULTRACK,faceNames, currentFaceID,))
                #     t.start()
                # # Increase the currentFaceID counter
                # currentFaceID += 1

    for fid in faceTrackers.keys():
        tracker = faceTrackers[fid]
        tracker.update(frame)
        tracked_position = tracker.get_position()
        left = int(tracked_position.left())
        top = int(tracked_position.top())
        right = int(tracked_position.right()) 
        bottom = int(tracked_position.bottom())

        (t_x, t_y, t_w, t_h) = (left, top, right - left, bottom - top)

        # Centroid of face
        face_centX = int(t_x + 0.5 * t_x)
        face_centY = int(t_y + 0.5 * t_h)
        faceCentroids.append((face_centX, face_centY))

        cv.rectangle(frame, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 1)

        if fid in faceNames.keys():
            cv.putText(frame, str(faceNames[fid]),(int(t_x + t_w / 2), t_y), font, 0.3, (255, 255, 255), 1)
        else:
            cv.putText(frame, "Detecting...", (int(t_x + t_w / 2), t_y), font, 0.3, (255, 255, 255), 1)

    cv.imshow("test", frame)
    frameCounter += 1

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv.destroyAllWindows()