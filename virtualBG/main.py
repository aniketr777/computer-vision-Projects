import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk  # To get screen dimensions

# Get screen resolution dynamically
root = tk.Tk()
width = root.winfo_screenwidth()
height = root.winfo_screenheight()
root.destroy()  # Close the Tkinter root after getting dimensions

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set webcam resolution to match screen (optional, depends on camera capability)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Load the background image
background = cv2.imread('img2.jpg')  # Replace with your image path
if background is None:
    print("Error: Could not load background image.")
    exit()

# Resize background image to match screen resolution
background_resized = cv2.resize(background, (width, height))

# Set up MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)  # 1 for general model

# Main loop to process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Resize the frame to match screen resolution
    frame = cv2.resize(frame, (width, height))

    # Flip frame horizontally for a mirror-like effect
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get segmentation mask from MediaPipe
    results = selfie_segmentation.process(frame_rgb)
    mask = results.segmentation_mask

    # Create binary mask by thresholding
    condition = mask > 0.6
    condition = condition.astype(np.uint8) * 255

    # Combine original frame and background using the mask
    output = np.where(condition[:, :, None], frame, background_resized)

    # Display the result in fullscreen
    cv2.namedWindow('Virtual Background', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Virtual Background', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Virtual Background', output)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()