from module import handDetector as htm
import cv2
import mediapipe as mp
import time
import pyautogui as pag
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize variables
pTime = 0
cap = cv2.VideoCapture(0)

# Get screen and camera resolutions
screen_width, screen_height = pag.size()
cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize pycaw for volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_min, vol_max, _ = volume.GetVolumeRange()

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
        indexFinger = lmList[8][1:]  # Index finger tip
        thumb = lmList[4][1:]        # Thumb tip
        print(f"Index Finger: {indexFinger}")

        # Map coordinates to screen (for mouse control)
        screen_x = int((cap_width - indexFinger[0]) * screen_width / cap_width)
        screen_y = int(indexFinger[1] * screen_height / cap_height)
        pag.moveTo(screen_x, screen_y, duration=0.05)

        # Volume control with thumb-index distance
        distance = detector.getDistance(thumb, indexFinger)
        print(f"Thumb-Index Distance: {distance}")
        
        # Map distance to volume (e.g., 20-150 pixels to 0.0-1.0 scalar)
        vol_range = min(150, max(20, distance))  # Clamp between 20 and 150
        vol_scalar = (vol_range - 20) / (150 - 20)  # Normalize to 0.0-1.0
        volume.SetMasterVolumeLevelScalar(vol_scalar, None)
        print(f"Volume set to: {vol_scalar * 100:.1f}%")

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()