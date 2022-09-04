import cv2
import time
import mediapipe as mp
from FPS import FPS


def label_fingers(hand_results):
    if hand_results.multi_hand_landmarks:
        for handLms in hand_results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x *w), int(lm.y*h)
                #if id ==0:
                cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils


fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter('test_hand_tracking_output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

fps = FPS().start()
while(cap.isOpened()):
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)


    label_fingers(results)
    cv2.putText(img,str(round(fps.fps(),2)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    cv2.imshow("Image", img)
    fps.update()

    if success == True:
        out.write(img)
    key = cv2.waitKey(1) & 0xFF

        # Press q to close the video windows before it ends if you want
    if key == ord("q"):
    	break
fps.stop()
out.release()
cap.release()
cv2.destroyAllWindows()
print("Video stop")
