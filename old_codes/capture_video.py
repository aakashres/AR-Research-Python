# Capture video from hololen stream
from FPS import FPS

import numpy as np
import cv2

src = 'https://usuer:pw@192.168.1.103/api/holographic/stream/live_high.mp4?holo=true&pv=true&mic=true&loopback=true'
cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')

# print(cap.get(3),cap.get(4))
out = cv2.VideoWriter('test.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

fps = FPS().start()
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        # frame = cv2.flip(frame,0)

        # write the flipped frame
        out.write(frame)
        cv2.putText(frame,'Text',(0,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,255,2)

        cv2.imshow('frame',frame)
        fps.update()

        # fps_label = "{:.2f}".format(fps.fps())

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        k = cv2.waitKey(33)
        if k == 27:
            break
        elif k == -1:
            continue
    # break

    else:
        break

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
