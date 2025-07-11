from module import handDetector as htm
import cv2
import mediapipe as mp
import time
import pyautogui as pag

# Initialize variables
pTime = 0
last_action_time = 0  # For cooldown
cap = cv2.VideoCapture(0)

# Get screen and camera resolutions
screen_width, screen_height = pag.size()
cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

detector = htm()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Detect hands and get landmarks
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame)

    if lmList:
        indexFinger = lmList[8][1:]   # Index finger tip
        thumb = lmList[4][1:]         # Thumb tip
        middleFinger = lmList[12][1:] # Middle finger tip
        pinkyFinger = lmList[20][1:]  # Pinky finger tip
        print(f"Index Finger: {indexFinger}")

        # Map coordinates to screen
        screen_x = int((cap_width - indexFinger[0]) * screen_width / cap_width)
        screen_y = int(indexFinger[1] * screen_height / cap_height)
        pag.moveTo(screen_x, screen_y, duration=0.05)

        # Check distances and trigger actions with cooldown
        current_time = time.time()
        if current_time - last_action_time > 0.5:  # 0.5s cooldown
            # Left click (thumb-index)
            distance_index = detector.getDistance(thumb, indexFinger)
            print(f"Thumb-Index Distance: {distance_index}")
            if distance_index < 30:
                pag.click(button='left')
                last_action_time = current_time
                print("Left click")

            # Right click (thumb-middle)
            distance_middle = detector.getDistance(thumb, middleFinger)
            print(f"Thumb-Middle Distance: {distance_middle}")
            if distance_middle < 30:
                pag.click(button='right')
                last_action_time = current_time
                print("Right click")

            # Screenshot (thumb-pinky)
            distance_pinky = detector.getDistance(thumb, pinkyFinger)
            print(f"Thumb-Pinky Distance: {distance_pinky}")
            if distance_pinky < 20:  # Adjusted threshold
                pag.screenshot('screenshot.png')
                last_action_time = current_time
                print("Screenshot taken")

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()