import cv2
import time
from module import poseDetector

def main():
    pTime = 0
    cap = cv2.VideoCapture('video.mp4')

    detector = poseDetector()
    pushup_count = 0
    direction = 0  # 0 = going down, 1 = coming up

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detector.findPose(frame)
        lmList = detector.findPosition(frame)

        if lmList:
            # Get the y-coordinates of the landmarks
            left_elbow_y = lmList[13][2]  # Left elbow (y-coordinate)
            left_shoulder_y = lmList[11][2]  # Left shoulder (y-coordinate)
            left_wrist_y = lmList[15][2]  # Left wrist (y-coordinate)

            # Check the elbow position relative to the wrist and shoulder
            if left_elbow_y > left_shoulder_y  :  # Going down
                if direction == 0:
                    direction = 1

            elif left_elbow_y < left_shoulder_y :  # Coming up
                if direction == 1:
                    pushup_count += 1
                    direction = 0  # Reset direction for the next pushup

        # Display the pushup count
        cv2.putText(frame, f"Pushups: {pushup_count}", (10, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)

        # Calculate and display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
