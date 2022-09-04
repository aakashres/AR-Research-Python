import numpy as np
import cv2
import time,sys
from Hololen import HololenVideoStream



CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# src_addresss = parsed_username_pw('credentials.txt')
# Open a sample video available in sample-videos
confidence_set = 0.2
prototxt = 'models/MobileNetSSD_deploy.prototxt.txt'
model = 'models/MobileNetSSD_deploy.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt,model)


vcap = HololenVideoStream(net).start()
#if not vcap.isOpened():
#    print "File Cannot be Opened"


time.sleep(2.0)

# Record the video
# fourcc = cv2.VideoWriter_fourcc(*'X264')
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(vcap.get(3)), int(vcap.get(4))))

while(True):
    # Capture frame-by-frame
    frame = vcap.read()
    #print cap.isOpened(), ret
    # if frame is not None:
        # Display the resulting frame
    # (h,w) = frame.shape[:2]
    # img = cv2.resize(frame,(300,300))
    # blob = cv2.dnn.blobFromImage(img,0.007843, (300, 300), 127.5)
    # net.setInput(blob)
    # detections = net.forward()


    cv2.imshow('frame',frame)



    key = cv2.waitKey(1) & 0xFF

        # Press q to close the video windows before it ends if you want
    if key == ord("q"):
        break

# When everything done, release the capture
# out.release()
cv2.destroyAllWindows()
vcap.stop()

print("Video stop")
