# import the opencv library
import cv2
import os
from time import time
import numpy as np
from buffer import Buffer
from face_detect import FaceDetection, pixelate

once = True

tracker = cv2.TrackerMIL_create()

backtracker = cv2.TrackerMIL_create()
detector = FaceDetection()

frame_buffer = Buffer(30)
my_img_1 = np.zeros((1280, 720, 1), dtype="uint8")


filename = "output.mp4"
frames_per_second = 30.0
res = "720p"


def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)


# Standard Video Dimensions Sizes
STD_DIMENSIONS = {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}


def get_dims(cap, res="1080p"):
    width, height = STD_DIMENSIONS["480p"]
    # if res in STD_DIMENSIONS:
    #     width, height = STD_DIMENSIONS[res]
    ## change the current caputre device
    ## to the resulting resolution
    change_res(cap, width, height)
    return width, height


VIDEO_TYPE = {
    "avi": cv2.VideoWriter_fourcc(*"XVID"),
    "mp4": cv2.VideoWriter_fourcc(*"mp4v"),
}


def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
        return VIDEO_TYPE[ext]
    return VIDEO_TYPE["mp4"]


# define a video capture object
# vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
vid = cv2.VideoCapture("input.mp4")
out = cv2.VideoWriter(filename, get_video_type(filename), 25, get_dims(vid, res))
start_time = time()
print_delay = True
detect = True
bbox = None
write_frame = None


input_frames = []
output_frames = []

while True:
    ret, frame = vid.read()
    if not ret:
        break
    input_frames.append(frame)
    out.write(frame)
vid.release()


counter = 0


while True:
    try:
        frame = input_frames.pop(0)
    except IndexError:
        frame = False

    if not isinstance(frame, bool):
        frame, detect, bbox = detector.detect_or_track(frame, detect, tracker)
        if once and not detect:
            once = False
            backtracker.init(frame, bbox)
            frames = frame_buffer.get_buffer_frames()
            for i in range(len(frames) - 2, 0, -1):
                cur_frame = frames[i]
                status, bbox = backtracker.update(cur_frame)
                if status:
                    x, y, w, h = bbox
                    cur_frame = pixelate(cur_frame, x, y, w, h)
                    cv2.rectangle(cur_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    frame_buffer.update_frame(cur_frame, i)
        status = frame_buffer.add_frames(frame)

    ret_frame = frame_buffer.get_frames()
    if isinstance(ret_frame, bool):
        if not ret_frame and isinstance(frame, bool):
            break
        if not ret_frame:
            write_frame = my_img_1
    else:
        write_frame = ret_frame
        if print_delay:
            frame_start_time = time()
            print("Frame Lag ", frame_start_time - start_time)
            print_delay = False

    cv2.imshow("frame", write_frame)
    output_frames.append(write_frame)
    counter += 1
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

for frame in output_frames:
    out.write(frame)

out.release()
# Destroy all the windows
cv2.destroyAllWindows()
