import numpy as np
import cv2
import time,sys
import configparser

def anonymize_face_pixelate(image, blocks=15):
	# divide the input image into NxN blocks
	(h, w) = image.shape[:2]
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
			roi = image[startY:endY, startX:endX]
			(B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
			cv2.rectangle(image, (startX, startY), (endX, endY),
				(B, G, R), -1)
	# return the pixelated blurred image
	return image
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
vcap = cv2.VideoCapture(0)
#if not vcap.isOpened():
#    print "File Cannot be Opened"
confidence_set = 0.9
# prototxt = 'models/Res_10_300x300_ssd_iter_140000.txt'
# model = 'models/Res_10_300x300.caffemodel'
model = 'models/ArcFace/resnet100.onnx'
net = cv2.dnn.readNetFromONNX(model)

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
	img = cv2.resize(frame,(112,112))
	blob = cv2.dnn.blobFromImage(img, size=(112, 112))
	print(blob.shape)
	net.setInput(blob)
	detections = net.forward()
	print(detections.shape)
	count = 0



	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > confidence_set:
		# 	continue
		# count +=1
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			face = frame[startY:endY,startX:endX]

			face = anonymize_face_pixelate(face)
				# draw the prediction on the frame

	            # label = "{}: {:.2f}%".format(CLASSES[idx],confidence * 100)
			frame[startY:endY,startX:endX] = face

			text = "{:.2f}%".format(confidence * 100) + ", Count " + str(count)
			cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 2)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

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
