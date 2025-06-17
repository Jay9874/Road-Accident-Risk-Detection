import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

# --- Configuration ---
# Name of your trained yawn detection model file
YAWN_MODEL_FILENAME = 'yawn_cnn.keras'

# Model input dimensions (must match what the model was trained on)
YAWN_IMG_HEIGHT = 64
YAWN_IMG_WIDTH = 64

# Labels mapping (must match the labels used during training from the yawn_labels dictionary)
# From our previous discussions, it's typically:
# {'No_Yawn': 0, 'Yawn': 1} (or similar, depending on alphabetical order of directories)
LABELS = {0: 'No_Yawn', 1: 'Yawn'} # Adjust if your training output shows different mapping

# --- 1. Load the Trained Yawn Detection Model ---
print(f"Attempting to load yawn model from {YAWN_MODEL_FILENAME}...")
try:
    yawn_model = load_model(YAWN_MODEL_FILENAME)
    print("Yawn model loaded successfully.")
    yawn_model.summary()
except Exception as e:
    print(f"Error loading yawn model: {e}")
    print("Please ensure you have trained the yawn model (using the 'yawn-detection-standalone-app' script) and that 'yawn_detection_cnn_model.h5' exists in the same directory.")
    exit()

# --- 2. Load OpenCV's Pre-trained Haar Cascade Classifiers ---
print("\nLoading Haar Cascade classifiers for face and mouth detection...")
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
# Using haarcascade_smile.xml as a proxy for detecting open mouth regions.
# For more accurate mouth detection and landmark-based analysis, `dlib` would be preferred
# but this allows for a quick start with standard OpenCV cascades.
mouth_cascade_path = cv2.data.haarcascades + 'haarcascade_smile.xml'

face_cascade = cv2.CascadeClassifier(face_cascade_path)
mouth_cascade = cv2.CascadeClassifier(mouth_cascade_path)

if face_cascade.empty():
    print(f"Error: Face cascade not loaded from {face_cascade_path}. Make sure the XML file exists.")
    exit()
if mouth_cascade.empty():
    print(f"Error: Mouth cascade not loaded from {mouth_cascade_path}. Make sure the XML file exists.")
    print("Warning: Mouth cascade not found or loaded. Yawning detection might be less reliable without it.")
print("Haar Cascades loaded successfully.")

# --- 3. Start Live Camera Feed and Real-time Detection ---
print("\nStarting real-time yawning detection with webcam. Press 'q' to quit.")

cap = cv2.VideoCapture(0) # 0 for default webcam

if not cap.isOpened():
    print("Error: Could not open webcam. Make sure your camera is connected and not in use by another application.")
    exit()

while True:
    ret, frame = cap.read() # Read a frame from the webcam
    if not ret:
        print("Failed to grab frame from camera. Exiting...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert frame to grayscale for cascade detection

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # Draw rectangle around face

        roi_gray = gray[y:y+h, x:x+w] # Region of Interest (ROI) for mouth in grayscale
        roi_color = frame[y:y+h, x:x+w] # ROI for drawing rectangles in color

        is_yawning = False
        # Search for mouth in the lower half of the face ROI (typically where the mouth is)
        # The coordinates for detectMultiScale are relative to the roi_gray.
        # We start searching from roughly the middle of the face downwards (h/2)
        mouths = mouth_cascade.detectMultiScale(roi_gray[int(h*0.5):h, :], 1.7, 11) # Tuned parameters for mouth/smile cascade

        for (mx, my, mw, mh) in mouths:
            # Adjust 'my' to be relative to the original 'roi_gray' for correct cropping
            # since we searched in roi_gray[int(h*0.5):h, :].
            actual_my = int(h*0.5) + my

            # Ensure the detected mouth region is valid and large enough
            # Add a small buffer around the mouth region for better capture
            buffer_x = int(mw * 0.1)
            buffer_y = int(mh * 0.1)
            mx_start = max(0, mx - buffer_x)
            my_start = max(0, actual_my - buffer_y)
            mx_end = min(roi_gray.shape[1], mx + mw + buffer_x)
            my_end = min(roi_gray.shape[0], actual_my + mh + buffer_y)

            mouth_img_gray = roi_gray[my_start:my_end, mx_start:mx_end]

            if mouth_img_gray.shape[0] < YAWN_IMG_HEIGHT/2 or mouth_img_gray.shape[1] < YAWN_IMG_WIDTH/2:
                # Skip if the detected mouth region is too small after buffering
                continue

            # Resize mouth image to model's input size
            mouth_img_resized = cv2.resize(mouth_img_gray, (YAWN_IMG_WIDTH, YAWN_IMG_HEIGHT))
            
            # Normalize pixel values to [0, 1]
            mouth_input = mouth_img_resized / 255.0

            # Expand dimensions to match model input shape (BATCH, HEIGHT, WIDTH, CHANNELS)
            mouth_input = np.expand_dims(mouth_input, axis=-1) # Add channel dimension (1 for grayscale)
            mouth_input = np.expand_dims(mouth_input, axis=0)      # Add batch dimension (for a single image)

            # Predict yawn state
            prediction_yawn = yawn_model.predict(mouth_input, verbose=0)
            predicted_yawn_class_index = np.argmax(prediction_yawn)
            predicted_yawn_label = LABELS[predicted_yawn_class_index]
            yawn_confidence = prediction_yawn[0][predicted_yawn_class_index] * 100

            # Check if prediction is 'Yawn' (ensure this matches your dataset's class name exactly)
            if predicted_yawn_label == 'Yawn':
                is_yawning = True
                # Draw red rectangle around yawning mouth
                cv2.rectangle(roi_color, (mx_start, my_start), (mx_end, my_end), (0, 0, 255), 2)
                cv2.putText(roi_color, f"Yawning: {yawn_confidence:.1f}%", (mx_start, my_start - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                break # Flag as yawning if at least one strong yawn detection is made

        # Display overall yawning status above the face
        status_text = "No Yawn Detected"
        if is_yawning:
            status_text = "YAWNING!"
        
        cv2.putText(frame, status_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
    # Display the processed frame
    cv2.imshow('Live Yawning Detection', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
