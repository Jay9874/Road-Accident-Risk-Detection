import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
import dlib # Import dlib
from imutils import face_utils # Helper functions for dlib landmarks

# --- Configuration ---
# Name of your trained eye detection model file
EYE_MODEL_FILENAME = 'eye_cnn.h5'

# Dlib's pre-trained facial landmark predictor model
SHAPE_PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat' # Make sure you download and place this file!

# Model input dimensions (must match what the model was trained on)
EYE_IMG_HEIGHT = 64
EYE_IMG_WIDTH = 64

# Labels mapping (must match the labels used during training)
LABELS = {0: 'closed_eye', 1: 'open_eye'} # VERIFY THIS FROM YOUR TRAINING SCRIPT'S OUTPUT!

# Indices for left and right eye landmarks (from dlib's 68-point model)
# These define the points around each eye.
(L_START, L_END) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(R_START, R_END) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# --- 1. Load the Trained Eye Detection Model ---
print(f"Attempting to load eye model from {EYE_MODEL_FILENAME}...")
try:
    eye_model = load_model(EYE_MODEL_FILENAME)
    print("Eye model loaded successfully.")
    eye_model.summary()
except Exception as e:
    print(f"Error loading eye model: {e}")
    print("Please ensure you have trained the eye model and that 'eye_state_cnn_model.h5' exists in the same directory as this script.")
    exit()

# --- 2. Load Dlib's Face Detector and Facial Landmark Predictor ---
print("\nLoading Dlib's face detector and facial landmark predictor...")
detector = dlib.get_frontal_face_detector() # HOG-based face detector
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH) # Facial landmark predictor

if not os.path.exists(SHAPE_PREDICTOR_PATH):
    print(f"Error: Dlib shape predictor not found at {SHAPE_PREDICTOR_PATH}. Please download and place it as instructed.")
    exit()
print("Dlib models loaded successfully.")

# --- 3. Start Live Camera Feed and Real-time Detection ---
print("\nStarting real-time eye status detection with webcam. Press 'q' to quit.")
print("DEBUG: Observe the 'Cropped Eye for CNN' window and console output for each eye.")

cap = cv2.VideoCapture(0) # 0 for default webcam

if not cap.isOpened():
    print("Error: Could not open webcam. Make sure your camera is connected and not in use by another application. Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` or higher if you have multiple cameras.")
    exit()

while True:
    ret, frame = cap.read() # Read a frame from the webcam
    if not ret:
        print("Failed to grab frame from camera. Exiting...")
        break

    # Flip frame horizontally, often useful for selfie-style webcam feeds
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert frame to grayscale for Dlib

    # Detect faces in the grayscale frame using Dlib's detector
    rects = detector(gray, 0) # The '0' indicates no upsampling

    for rect in rects:
        # Convert dlib's rectangle to OpenCV bounding box format (x, y, w, h)
        x, y, w, h = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # Draw rectangle around face

        # Determine the facial landmarks for the face region
        shape = predictor(gray, rect)
        # Convert the (x, y)-coordinates to a NumPy array
        shape = face_utils.shape_to_np(shape)

        # Extract the left and right eye coordinates
        left_eye = shape[L_START:L_END]
        right_eye = shape[R_START:R_END]

        # List to store eye states for the current face
        current_face_eye_states = []

        # Process each eye
        for i, eye_points in enumerate([left_eye, right_eye]):
            # Calculate the bounding box for the eye region based on its landmarks
            (ex, ey, ew, eh) = cv2.boundingRect(eye_points)

            # Expand eye region slightly to ensure full eye is captured for the CNN
            buffer_x = int(ew * 0.2) # Increased buffer
            buffer_y = int(eh * 0.2)
            ex_start = max(0, ex - buffer_x)
            ey_start = max(0, ey - buffer_y)
            ex_end = min(gray.shape[1], ex + ew + buffer_x)
            ey_end = min(gray.shape[0], ey + eh + buffer_y)

            # Ensure the coordinates are valid and within frame bounds
            eye_img_gray_cropped = gray[ey_start:ey_end, ex_start:ex_end]

            # Skip if cropped eye image is invalid or too small after buffering
            # if eye_img_gray_cropped.shape[0] < EYE_IMG_HEIGHT * 0.5 or eye_img_gray_cropped.shape[1] < EYE_IMG_WIDTH * 0.5:
            #     print(f"DEBUG: Skipped eye {i+1} due to small/invalid crop size: {eye_img_gray_cropped.shape}")
            #     continue

            # Resize eye image to model's input size
            eye_img_resized = cv2.resize(eye_img_gray_cropped, (EYE_IMG_WIDTH, EYE_IMG_HEIGHT))
            
            # DEBUG: Display the cropped and resized eye image that goes into the CNN
            cv2.imshow(f'Cropped Eye for CNN {i+1}', eye_img_resized)

            # Normalize pixel values to [0, 1]
            eye_input = eye_img_resized / 255.0

            # Expand dimensions to match model input shape (BATCH, HEIGHT, WIDTH, CHANNELS)
            eye_input = np.expand_dims(eye_input, axis=-1) # Add channel dimension (1 for grayscale)
            eye_input = np.expand_dims(eye_input, axis=0)      # Add batch dimension (for a single image)

            # Predict eye state
            prediction = eye_model.predict(eye_input, verbose=0)
            predicted_class_index = np.argmax(prediction)
            predicted_label = LABELS[predicted_class_index]
            confidence = prediction[0][predicted_class_index] * 100

            current_face_eye_states.append(predicted_label)

            # DEBUG: Print raw prediction values to console
            print(f"DEBUG: Eye {i+1} Raw Prediction: {prediction[0]}, Predicted Index: {predicted_class_index}, Label: {predicted_label}, Confidence: {confidence:.2f}%")

            # Draw rectangle around individual eye and display label
            # Coordinates for rectangle and text need to be relative to the full 'frame'
            cv2.rectangle(frame, (ex_start, ey_start), (ex_end, ey_end), (0, 200, 255), 1)
            text_individual_eye = f"{predicted_label.replace('_', ' ').capitalize()}: {confidence:.1f}%"
            cv2.putText(frame, text_individual_eye, (ex_start, ey_start - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

            # Optionally draw the facial landmarks for the eyes for visualization
            # for (x_lm, y_lm) in eye_points:
            #     cv2.circle(frame, (x_lm, y_lm), 1, (0, 255, 0), -1)

        # Determine overall eye state for the face based on detected eyes
        overall_eye_status = "No Eyes Detected" # Default if no eyes are found or processed

        if len(current_face_eye_states) == 2: # Ideal scenario: both eyes detected
            left_eye_status = current_face_eye_states[0] # Assuming sorted by x, left is first
            right_eye_status = current_face_eye_states[1]

            if left_eye_status == 'open_eye' and right_eye_status == 'open_eye':
                overall_eye_status = "Eyes Open"
            elif left_eye_status == 'closed_eye' and right_eye_status == 'closed_eye':
                overall_eye_status = "Both Eyes Closed"
            else: # One open, one closed, or mixed
                overall_eye_status = "Eyes Mixed State"
        elif len(current_face_eye_states) == 1:
            overall_eye_status = f"One Eye: {current_face_eye_states[0].replace('_', ' ').capitalize()}"
        
        # Display the overall face status above the face rectangle
        cv2.putText(frame, overall_eye_status, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
    # Display the processed frame
    cv2.imshow('Live Eye Status Detection', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
