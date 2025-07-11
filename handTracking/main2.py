import cv2 
import mediapipe as mp 


cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
cTime=0
pTime=0
while True:
    ret,frame = cap.read()
    if not ret :
        break

    img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results=hands.process(frame)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id , lm in handLms:
                h,w,c=frame.shape
                cx,cy = int(lm.x*w) ,int(lm.y*h)
                print(id,cx,cy)
            
            mpDraw.drawLandmarks(frame,handLms,mpHands.handConnection)

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)

    cv2.imshow("frame",frame)
    cv2.waitKey(1)
cv2.destroyAllWindows()

    