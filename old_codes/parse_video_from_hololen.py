import numpy as np
import cv2
import time,sys
import configparser
def parsed_username_pw(file):

	config = configparser.ConfigParser()
	config.read(file)

	username = config['Specs']['username']
	password = config['Specs']['password']
	ip = config['Specs']['ip']

	hololen_base = '/api/holographic/stream/live_high.mp4?holo=true&pv=true&mic=true&loopback=true'
	hololen = 'https://' + username + ':' + password + "@" + ip + hololen_base
	# print(hololen)
	return hololen

	# # return username,password

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

src_addresss = parsed_username_pw('credentials.txt')
# Open a sample video available in sample-videos
vcap = cv2.VideoCapture(src_addresss)
#if not vcap.isOpened():
#    print "File Cannot be Opened"
confidence_set = 0.2
prototxt = 'models/MobileNetSSD_deploy.prototxt.txt'
model = 'models/MobileNetSSD_deploy.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt,model)

time.sleep(2.0)

# Record the video
fourcc = cv2.VideoWriter_fourcc(*'X264')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(vcap.get(3)), int(vcap.get(4))))

while(True):
    # Capture frame-by-frame
    ret, frame = vcap.read()
    #print cap.isOpened(), ret
    # if frame is not None:
        # Display the resulting frame
    (h,w) = frame.shape[:2]
    img = cv2.resize(frame,(300,300))
    blob = cv2.dnn.blobFromImage(img,0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()



    for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
        confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
        if confidence > confidence_set:
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

			# draw the prediction on the frame

            label = "{}: {:.2f}%".format(CLASSES[idx],confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)


    cv2.imshow('frame',frame)

    if ret == True:
        out.write(frame)

    key = cv2.waitKey(1) & 0xFF

        # Press q to close the video windows before it ends if you want
    if key == ord("q"):
        break

# When everything done, release the capture
out.release()
vcap.release()
cv2.destroyAllWindows()
print("Video stop")
