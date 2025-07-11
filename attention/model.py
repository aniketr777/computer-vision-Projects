# attention_tracker_api.py
import cv2  # OpenCV for image processing (e.g., decoding frames, color conversion)
import numpy as np  # NumPy for numerical operations (e.g., array manipulation, norm calculations)
from ultralytics import YOLO  # Ultralytics library to use YOLOv8 model for object detection
import mediapipe as mp  # MediaPipe for facial landmark detection (e.g., eye tracking)
import time  # For timing operations (e.g., tracking attention duration)
from flask import Flask, request, jsonify  # Flask for creating the API, handling requests, and sending JSON responses
import base64  # For decoding base64-encoded image data from the client
from io import BytesIO  # For handling in-memory byte streams (e.g., decoding images)
import asyncio  # For asynchronous programming to handle multiple requests efficiently
from concurrent.futures import ThreadPoolExecutor  # For running blocking tasks (e.g., frame processing) in separate threads
from threading import Lock  # For thread-safe access to shared resources (e.g., sessions dictionary)
import uuid  # For generating unique session IDs (though not used directly here, likely intended for future use)

# Initialize Flask app
app = Flask(__name__)  # Creates a Flask application instance; __name__ is the module name (here, "attention_tracker_api")

# Set up thread pool for concurrent frame processing
executor = ThreadPoolExecutor(max_workers=4)  # Creates a pool of 4 worker threads to process frames concurrently; adjust based on server capacity

# Session storage
sessions = {}  # Dictionary to store session data for each user (e.g., attention time, timestamps)
sessions_lock = Lock()  # A lock to ensure thread-safe access to the sessions dictionary when multiple threads modify it

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Loads the pre-trained YOLOv8 nano model from the file "yolov8n.pt" for person/face detection

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh  # Imports the Face Mesh module from MediaPipe
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)  # Creates a Face Mesh instance to detect facial landmarks; limited to 1 face, with refined landmarks for accuracy

# Define Eye Aspect Ratio (EAR) function
def eye_aspect_ratio(eye_landmarks):
    """Calculate the Eye Aspect Ratio (EAR) to detect if eyes are closed."""
    v1 = np.linalg.norm([eye_landmarks[1].x - eye_landmarks[5].x, eye_landmarks[1].y - eye_landmarks[5].y])  # Vertical distance between landmarks 1 and 5 of the eye
    v2 = np.linalg.norm([eye_landmarks[2].x - eye_landmarks[4].x, eye_landmarks[2].y - eye_landmarks[4].y])  # Vertical distance between landmarks 2 and 4 of the eye
    h = np.linalg.norm([eye_landmarks[0].x - eye_landmarks[3].x, eye_landmarks[0].y - eye_landmarks[3].y])  # Horizontal distance between landmarks 0 and 3 of the eye
    return (v1 + v2) / (2.0 * h)  # EAR formula: (sum of vertical distances) / (2 * horizontal distance)

# Frame processing function
def process_frame(frame, session_id):
    # Thread-safe session initialization
    with sessions_lock:  # Acquires the lock to safely access/modify the sessions dictionary
        if session_id not in sessions:  # If this is a new session
            sessions[session_id] = {  # Initialize session data
                'attention_time': 0,  # Total time the user was attentive
                'last_attentive_time': time.time(),  # Last timestamp when attention was tracked
                'closed_eyes_start': None  # Timestamp when eyes first closed (for sleep detection)
            }
        session = sessions[session_id]  # Get the session data for this user

    # Extract session variables
    attention_time = session['attention_time']  # Current total attention time
    last_attentive_time = session['last_attentive_time']  # Last recorded time
    closed_eyes_start = session['closed_eyes_start']  # Time when eyes closed (if any)
    sleep_threshold = 0.25  # EAR threshold below which eyes are considered closed
    sleep_duration_threshold = 2.0  # Seconds of eye closure to classify as sleeping
    min_closure_duration = 0.5  # Minimum seconds of eye closure to count as inattentive (ignores blinks)

    # YOLOv8 inference
    results = model(frame)  # Runs YOLOv8 on the frame to detect persons/faces
    detected = False  # Flag to track if a face is detected
    status = "No Person Detected"  # Default status if no detection occurs

    # Process detection results
    if results and results[0].boxes:  # Check if there are any detected objects (boxes)
        for detection in results:  # Iterate over detections (though limited to 1 face here)
            if len(detection.boxes.xywh) == 0:  # If no boxes are present, skip
                continue

            # Extract bounding box coordinates
            x, y, w, h = detection.boxes.xywh[0].cpu().numpy()  # Get x, y, width, height from the first detection
            x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)  # Convert center-based (xywh) to corner-based (xyxy)
            face_roi = frame[y:y + h, x:x + w]  # Crop the face region from the frame

            # Process face with MediaPipe
            rgb_frame = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV) to RGB (MediaPipe requirement)
            face_results = face_mesh.process(rgb_frame)  # Run MediaPipe Face Mesh on the cropped face

            # Analyze facial landmarks
            if face_results.multi_face_landmarks:  # If landmarks are detected
                for face_landmarks in face_results.multi_face_landmarks:  # Iterate over detected faces (only 1 due to max_num_faces=1)
                    detected = True  # Mark that a face was detected

                    # Get key landmarks
                    left_eye = face_landmarks.landmark[33]  # Center of left eye
                    right_eye = face_landmarks.landmark[263]  # Center of right eye
                    nose = face_landmarks.landmark[1]  # Tip of the nose

                    # Eye closure detection
                    left_eye_points = [face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]  # Landmarks around left eye
                    right_eye_points = [face_landmarks.landmark[i] for i in [263, 387, 385, 362, 380, 373]]  # Landmarks around right eye
                    avg_ear = (eye_aspect_ratio(left_eye_points) + eye_aspect_ratio(right_eye_points)) / 2.0  # Average EAR of both eyes

                    # Head pose estimation
                    head_tilted = nose.y > 0.7  # If nose.y (normalized) is > 0.7, head is tilted down
                    current_time = time.time()  # Current timestamp
                    eye_mid_x = (left_eye.x + right_eye.x) / 2  # Midpoint of eyes horizontally
                    is_gaze_forward = abs(eye_mid_x - nose.x) < 0.1  # Check if gaze is forward (eyes aligned with nose)

                    # Attention and sleep logic
                    if avg_ear < sleep_threshold:  # Eyes are closed
                        if closed_eyes_start is None:  # If this is the first frame of closure
                            closed_eyes_start = current_time
                        closure_duration = current_time - closed_eyes_start  # How long eyes have been closed
                        if closure_duration > sleep_duration_threshold:  # Longer than 2 seconds
                            status = "Sleeping"
                        elif closure_duration > min_closure_duration:  # Longer than 0.5 seconds but not sleeping
                            status = "Not Attentive (Eyes Closed)"
                        else:  # Brief closure (e.g., blink)
                            status = "Attentive" if (is_gaze_forward and not head_tilted) else "Not Attentive"
                    else:  # Eyes are open
                        closed_eyes_start = None  # Reset closure timer
                        if is_gaze_forward and not head_tilted:  # Gaze forward and head upright
                            status = "Attentive"
                            attention_time += current_time - last_attentive_time  # Add time since last frame
                        else:  # Eyes open but not attentive
                            status = "Not Attentive"

                    last_attentive_time = current_time  # Update last timestamp

    # If no face detected, reset timers
    if not detected:
        last_attentive_time = time.time()
        closed_eyes_start = None

    # Update session data
    with sessions_lock:  # Thread-safe update
        sessions[session_id].update({
            'attention_time': attention_time,
            'last_attentive_time': last_attentive_time,
            'closed_eyes_start': closed_eyes_start
        })

    return status, attention_time  # Return current status and total attention time

# Asynchronous wrapper for frame processing
async def async_process_frame(frame, session_id):
    loop = asyncio.get_event_loop()  # Get the current event loop
    status, attention_time = await loop.run_in_executor(executor, process_frame, frame, session_id)  # Run process_frame in a thread
    return status, attention_time  # Return results asynchronously

# API endpoint for processing frames
@app.route('/attention', methods=['POST'])
async def attention_tracker():
    try:
        data = request.json  # Get JSON data from the POST request
        if not data or 'frame' not in data or 'session_id' not in data:  # Validate input
            return jsonify({'error': 'Missing frame or session_id'}), 400  # Return error if data is incomplete

        session_id = data['session_id']  # Extract session ID
        frame_data = base64.b64decode(data['frame'].split(',')[1])  # Decode base64 frame (remove "data:image/jpeg;base64," prefix)
        np_frame = np.frombuffer(frame_data, dtype=np.uint8)  # Convert byte data to NumPy array
        frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)  # Decode array into an OpenCV image

        if frame is None:  # Check if decoding failed
            return jsonify({'error': 'Invalid frame data'}), 400

        status, attention_time = await async_process_frame(frame, session_id)  # Process frame asynchronously

        return jsonify({  # Return JSON response
            'status': status,
            'attention_time': round(attention_time, 2)
        }), 200  # HTTP 200 OK

    except Exception as e:  # Catch any errors
        return jsonify({'error': str(e)}), 500  # Return error with HTTP 500 Internal Server Error

# API endpoint to end session and get final results
@app.route('/end_session', methods=['GET'])
def end_session():
    try:
        session_id = request.args.get('session_id')  # Get session_id from query parameters
        if not session_id or session_id not in sessions:  # Validate session_id
            return jsonify({'error': 'Invalid or missing session_id'}), 400

        with sessions_lock:  # Thread-safe access
            attention_time = sessions[session_id]['attention_time']  # Get final attention time
            del sessions[session_id]  # Delete session to free memory

        return jsonify({  # Return JSON response
            'message': 'Session ended',
            'total_attention_time': round(attention_time, 2)
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Root endpoint
@app.route('/')
def index():
    return "Attention Tracker API is running."  # Simple message to confirm API is up

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # Start the server on all interfaces, port 5000, with debug mode enabled