from threading import Thread
import configparser
import cv2
import numpy as np

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


class HololenVideoStream:
	def __init__(self,net=None,confidence_set=0.5):
		src_address = parsed_username_pw('credentials.txt')
		self.stream = cv2.VideoCapture(src_address)
		self.net = net
		self.stopped = False
		self.confidence_set = confidence_set

		self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
			"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
			"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
			"sofa", "train", "tvmonitor"]

		self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))

	def start(self):
		Thread(target=self.update, args=()).start()
		return self

	def update(self):
		# keep looping infinitely until the thread is stopped
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				return
			# otherwise, read the next frame from the stream
			(self.grabbed, self.frame) = self.stream.read()
			(h,w) = self.frame.shape[:2]
			img = cv2.resize(self.frame,(300,300))
			blob = cv2.dnn.blobFromImage(img,0.007843, (300, 300), 127.5)
			self.net.setInput(blob)
			detections = self.net.forward()

			for i in np.arange(0, detections.shape[2]):
				# extract the confidence (i.e., probability) associated with
				# the prediction
				confidence = detections[0, 0, i, 2]

				# filter out weak detections by ensuring the `confidence` is
				# greater than the minimum confidence
				if confidence > self.confidence_set:
					# extract the index of the class label from the
					# `detections`, then compute the (x, y)-coordinates of
					# the bounding box for the object
					idx = int(detections[0, 0, i, 1])
					box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
					(startX, startY, endX, endY) = box.astype("int")

					# draw the prediction on the frame
					label = "{}: {:.2f}%".format(self.CLASSES[idx],confidence * 100)
					cv2.rectangle(self.frame, (startX, startY), (endX, endY),self.COLORS[idx], 2)
					y = startY - 15 if startY - 15 > 15 else startY + 15
					cv2.putText(self.frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS[idx], 2)

	def read(self):
		# return the frame most recently read
		return self.frame
	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True

	def get(self,val):
		return self.stream.get(val)
