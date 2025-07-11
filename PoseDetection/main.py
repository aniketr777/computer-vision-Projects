import cv2
import time
import mediapipe as mp 
cap = cv2.VideoCapture(0)
cTime = 0
pTime = 0

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

while True:
    ret, frame = cap.read()
    if not ret:
        break


    imgRBG = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results= pose.process(imgRBG)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(frame,results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id , lm in enumerate(results.pose_landmarks.landmarks):
            h,w,c = frame.shape
            cx,cy = int(lm.x*w) ,int(lm.y*h)
            print(id,cx,cy)



    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(frame, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 2)
    cv2.imshow("Webcam Feed", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()


# if __name__ =="__main__":
#     main()