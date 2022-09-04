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

labelsPath = 'models/coco.names'
LABELS = open(labelsPath).read().strip().split("\n")
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

src_addresss = parsed_username_pw('credentials.txt')
# Open a sample video available in sample-videos
vcap = cv2.VideoCapture(src_addresss)
#if not vcap.isOpened():
#    print "File Cannot be Opened"
confidence_set = 0.5
threshold_set = 0.3
configPath = 'models/yolov3.cfg'
weightPath = 'models/yolov3.weights'

net = cv2.dnn.readNetFromDarknet(configPath,weightPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


time.sleep(2.0)

# Record the video
fourcc = cv2.VideoWriter_fourcc(*'X264')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(vcap.get(3)), int(vcap.get(4))))

while(True):
	# Capture frame-by-frame
	ret, frame = vcap.read()
	(h,w) = frame.shape[:2]
	img = cv2.resize(frame,(608,608))
	blob = cv2.dnn.blobFromImage(img,1/255, (608, 608), swapRB=True,crop=False)
	net.setInput(blob)
	layOutput = net.forward(ln)
	boxes = []
	confidences = []
	classIDs = []

	for output in layOutput:

		for detections in output:

			scores = detections[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			if confidence > confidence_set:
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_set,
		threshold_set)
	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			# draw a bounding box rectangle and label on the frame
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]],
				confidences[i])
			cv2.putText(frame, text, (x, y - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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
