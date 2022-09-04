import numpy as np
import cv2
import time,sys
from keras.models import load_model
from consts import EMOTION_MODEL, SSD_MODEL, SSD_PROTOTEXT, EMOTION_MODEL

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

vcap = cv2.VideoCapture(0)
confidence_set = 0.9
prototxt = SSD_PROTOTEXT.as_posix()
model = SSD_MODEL.as_posix()
net = cv2.dnn.readNetFromCaffe(prototxt,model)

emotion_model = load_model(EMOTION_MODEL.as_posix())

#Dictionary for emotion recognition model output and emotions
emotions = {0:'Angry',1:'Fear',2:'Happy',3:'Sad',4:'Surprised',5:'Neutral'}

#Upload images of emojis from emojis folder
emoji = []
for index in range(6):
    emotion = emotions[index]
    emoji.append(cv2.imread('./emojis/' + emotion + '.png', -1))

time.sleep(2.0)

fourcc = cv2.VideoWriter_fourcc(*'X264')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(vcap.get(3)), int(vcap.get(4))))

def detect_faces(frame):
    faces = []
    (h,w) = frame.shape[:2]
    frame_resized = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(frame_resized, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_set:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            dim  = max(endX-startX,endY-startY)
            endX = startX + dim
            endY = startY + dim
            faces.append((startX, startY, endX, endY))
    return faces 

frame_count = 0
while(True):
    # Capture frame-by-frame
    ret, frame = vcap.read()
    if not ret:
        continue
    start_time = time.time()
    faces = detect_faces(frame)
    detect_time = time.time() - start_time
    print("detect_time:",detect_time)
    emotion_times = []
    print(len(faces))
    for (startX, startY, endX, endY) in faces:
        face = frame[startY:endY,startX:endX]
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        test_image = cv2.resize(gray, (48, 48))
        test_image = test_image.reshape([-1,48,48,1])
        test_image = np.multiply(test_image, 1.0 / 255.0)
        start_time = time.time()
        probab = emotion_model.predict(test_image)[0] * 100
        emotion_times.append((time.time() - start_time))
        #Finding label from probabilities
        #Class having highest probability considered output label
        label = np.argmax(probab)
        probab_predicted = int(probab[label])
        predicted_emotion = emotions[label]
        prediction_time = time.time() - start_time
        width, height = endY-startY, endX-startX
        # Drawing on frame
        font_size = width / 300
        filled_rect_ht = int(height / 5)

        #Resizing emoji according to size of detected face
        emoji_face = emoji[(label)]
        emoji_face = cv2.resize(emoji_face, (filled_rect_ht, filled_rect_ht))

        #Positioning emojis on frame
        emoji_x1 = startX + width - filled_rect_ht
        emoji_x2 = emoji_x1 + filled_rect_ht
        emoji_y1 = startY + height
        emoji_y2 = emoji_y1 + filled_rect_ht

        #Drawing rectangle and showing output values on frame
        cv2.rectangle(frame, (startX, startY), (startX + width, startY + height),(155,155, 0),2)
        cv2.rectangle(frame, (startX-1, startY+height), (startX+1 + width, startY + height+filled_rect_ht),
                        (155, 155, 0),cv2.FILLED)
        cv2.putText(frame, predicted_emotion+' '+ str(probab_predicted)+'%',
                    (startX, startY + height+ filled_rect_ht-10), cv2.FONT_HERSHEY_SIMPLEX,font_size,(255,255,255), 1, cv2.LINE_AA)

        # Showing emoji on frame
        # for c in range(0, 3):
        #     frame[emoji_y1:emoji_y2, emoji_x1:emoji_x2, c] = emoji_face[:, :, c] * \
        #         (emoji_face[:, :, 3] / 255.0) + frame[emoji_y1:emoji_y2, emoji_x1:emoji_x2, c] * \
        #         (1.0 - emoji_face[:, :, 3] / 255.0)
        # face = anonymize_face_pixelate(face)
    cv2.imshow('frame',frame)
    frame_count += 1


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
