# import the opencv library
import cv2
from time import time
import numpy as np
from buffer import Buffer
from im_utils import get_video_type, get_dims
from object_detector import ObjectDetector, pixelate


start_time = time()
print_delay = True
detect = True
bbox = None
write_frame = None
once = True

tracker = cv2.TrackerMIL_create()
backtracker = cv2.TrackerMIL_create()
detector = ObjectDetector()

frame_buffer = Buffer(40)
empty_image = np.zeros((1280, 720, 1), dtype="uint8")
vid = cv2.VideoCapture("input2.mp4")


filename = "output.mp4"
frames_per_second = 30.0
file_size = (720, 1280)

# Create a VideoWriter object so we can save the video output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(filename, fourcc, frames_per_second, file_size)


input_frames = []
output_frames = []

while True:
    ret, frame = vid.read()
    if not ret:
        break
    input_frames.append(frame)
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
            start = time()
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
            print(time() - start)
            print(frame_buffer.buffer_length)
        status = frame_buffer.add_frames(frame)

    ret_frame = frame_buffer.get_frames()
    if isinstance(ret_frame, bool):
        if not ret_frame and isinstance(frame, bool):
            break
        if not ret_frame:
            write_frame = empty_image
    else:
        write_frame = ret_frame
        if print_delay:
            frame_start_time = time()
            print("Frame Lag ", frame_start_time - start_time)
            print_delay = False

    cv2.imshow("frame", write_frame)
    output_frames.append(write_frame)
    out.write(write_frame)
    counter += 1
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
out.release()
cv2.destroyAllWindows()
