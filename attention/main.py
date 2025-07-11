import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import time

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Nano version for speed

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Initialize video capture (webcam)
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or "video.mp4" for a file

# Attention tracking variables
attention_time = 0
last_attentive_time = time.time()
attentive_threshold = 0.8  # Confidence threshold for attention
sleep_threshold = 0.25  # Increased EAR threshold (less strict)
sleep_duration_threshold = 2.0  # Seconds of eye closure to classify as sleeping
min_closure_duration = 0.5  # Minimum seconds eyes must be closed to count as inattentive (ignores blinks)
closed_eyes_start = None  # Track when eyes first close

def eye_aspect_ratio(eye_landmarks):
    """Calculate the Eye Aspect Ratio (EAR) to detect if eyes are closed."""
    # Vertical distances
    v1 = np.linalg.norm([eye_landmarks[1].x - eye_landmarks[5].x, eye_landmarks[1].y - eye_landmarks[5].y])
    v2 = np.linalg.norm([eye_landmarks[2].x - eye_landmarks[4].x, eye_landmarks[2].y - eye_landmarks[4].y])
    # Horizontal distance
    h = np.linalg.norm([eye_landmarks[0].x - eye_landmarks[3].x, eye_landmarks[0].y - eye_landmarks[3].y])
    # EAR formula
    ear = (v1 + v2) / (2.0 * h)
    return ear

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions for MediaPipe (to fix the warning)
    height, width, _ = frame.shape

    # Run YOLOv8 to detect person/face
    results = model(frame)
    detected = False  # Flag to check if a person is detected

    # Check if there are any detections
    if results and results[0].boxes:  # Check if boxes exist in the first result
        for detection in results:
            if len(detection.boxes.xywh) == 0:  # Ensure there is at least one box
                continue

            # Extract bounding box (xywh format)
            x, y, w, h = detection.boxes.xywh[0].cpu().numpy()
            x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)  # Convert to xyxy
            face_roi = frame[y:y + h, x:x + w]

            # Process face with MediaPipe for gaze/head pose
            rgb_frame = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            face_results = face_mesh.process(rgb_frame)

            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    detected = True  # Set flag to true since a face is detected

                    # Get key landmarks
                    left_eye = face_landmarks.landmark[33]  # Left eye center
                    right_eye = face_landmarks.landmark[263]  # Right eye center
                    nose = face_landmarks.landmark[1]  # Nose tip

                    # Eye closure detection
                    left_eye_points = [face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]  # Left eye landmarks
                    right_eye_points = [face_landmarks.landmark[i] for i in [263, 387, 385, 362, 380, 373]]  # Right eye landmarks
                    left_ear = eye_aspect_ratio(left_eye_points)
                    right_ear = eye_aspect_ratio(right_eye_points)
                    avg_ear = (left_ear + right_ear) / 2.0

                    # Head pose (rough estimate: if nose.y is too low, head is tilted down)
                    head_tilted = nose.y > 0.7  # Adjust threshold based on normalized coordinates

                    # Attention and sleep logic
                    current_time = time.time()
                    eye_mid_x = (left_eye.x + right_eye.x) / 2
                    is_gaze_forward = abs(eye_mid_x - nose.x) < 0.1  # Gaze forward check

                    if avg_ear < sleep_threshold:  # Eyes closed
                        if closed_eyes_start is None:
                            closed_eyes_start = current_time
                        closure_duration = current_time - closed_eyes_start
                        if closure_duration > sleep_duration_threshold:
                            status = "Sleeping"
                            cv2.putText(frame, "Sleeping", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        elif closure_duration > min_closure_duration:  # Longer than a blink but not sleeping
                            status = "Not Attentive (Eyes Closed)"
                            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        else:
                            # Treat brief closures (blinks) as still attentive if gaze was forward
                            if is_gaze_forward and not head_tilted:
                                status = "Attentive"
                                attention_time += current_time - last_attentive_time
                                cv2.putText(frame, f"Attentive: {attention_time:.2f}s", (10, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            else:
                                status = "Not Attentive"
                                cv2.putText(frame, "Not Attentive", (10, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:  # Eyes open
                        closed_eyes_start = None  # Reset sleep timer
                        if is_gaze_forward and not head_tilted:
                            status = "Attentive"
                            attention_time += current_time - last_attentive_time
                            cv2.putText(frame, f"Attentive: {attention_time:.2f}s", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        else:
                            status = "Not Attentive"
                            cv2.putText(frame, "Not Attentive", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    last_attentive_time = current_time

                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # If no person is detected, update time continuity
    if not detected:
        last_attentive_time = time.time()
        closed_eyes_start = None

    # Display frame
    cv2.imshow("Attention Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Total serious attention time: {attention_time:.2f} seconds")



