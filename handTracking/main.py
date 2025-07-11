import cv2 
import mediapipe as mp
import time

cap  = cv2.VideoCapture(0)

#^ Parameters of Hands()
# The Hands() constructor accepts several optional parameters to customize its behavior. Here's what can be "inside" it in terms of configuration:

## static_image_mode:
# Default: False
# If True, treats input as a static image (no temporal tracking). If False, it tracks hands across frames like in a video.
## max_num_hands:
# Default: 2
# Specifies the maximum number of hands to detect in the input.
## model_complexity:
# Default: 1
# Range: 0 (lite model) to 1 (full model). Higher complexity improves accuracy but slows down processing.
## min_detection_confidence:
# Default: 0.5
# A value between 0 and 1. The model only reports hands if it's at least this confident they exist in the frame.
## min_tracking_confidence:
# Default: 0.5
# A value between 0 and 1. Used when static_image_mode=False to ensure tracked hands meet this confidence threshold.

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime=0
cTime=0

while True:
    ret,frame = cap.read()
    if not ret :
        break

    imgRBG = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) 
    results = hands.process(imgRBG)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for  handLms in results.multi_hand_landmarks: 
            for id,lm in enumerate(handLms.landmark):
                # print(id,lm)
                # this will give some , decimal point
                h,w,c=frame.shape
                cx,cy = int(lm.x*w) ,int(lm.y*h)
                print(id,cx,cy)
                if id==0:
                    cv2.circle(frame,(cx,cy),25,(255,0,255),cv2.FILLED)
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)



    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)

    cv2.imshow("frame",frame)
    cv2.waitKey(1)
cv2.destroyAllWindows()