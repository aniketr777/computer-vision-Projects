import cv2
import mediapipe as mp
import time

class poseDetector:
    def __init__(self, mode=False, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpPose = mp.solutions.pose
        self.Pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.Pose.process(imgRGB)

        if draw and self.results.pose_landmarks:
            self.mpDraw.draw_landmarks(frame, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return frame

    def findPosition(self, frame, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 8, (255, 0, 255), cv2.FILLED)

        return lmList

    def getDistance(self, a, b):
        return int(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5)

def main():
    pTime = 0
    cap = cv2.VideoCapture(0)

    detector = poseDetector()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detector.findPose(frame)
        lmList = detector.findPosition(frame)
        
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
