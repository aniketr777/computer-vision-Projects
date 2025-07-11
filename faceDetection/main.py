import cv2
import time
import mediapipe as mp
# Initialize video capture
cap = cv2.VideoCapture('Video.mp4')

# Initialize pTime for FPS calculation
pTime = time.time()

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    imgRBG = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRBG)
    print(results)

    if results.detections:
        for id,detection in  enumerate(results.detections):
            # mpDraw.draw_detection(frame, detection)
            # print(id, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box.xmin)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = frame.shape
            # bbox  = (bboxC.xmin  * iw)
            bbox=int(bboxC.xmin*iw),int(bboxC.ymin*ih),int(bboxC.width  * iw),int(bboxC.height*ih)
            cv2.rectangle(frame,bbox,(255,0,0),2)
            cv2.putText(frame, f' {int(detection.score[0]*100)}', (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    # Calculate FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Overlay FPS on the frame
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Exit on 'q' key press or wait 1ms
    cv2.waitKey(10)

# Release resources
cap.release()
cv2.destroyAllWindows()