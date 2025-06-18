import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
import dlib # For face and facial landmark detection
from imutils import face_utils # Helper functions for dlib landmarks

# --- Configuration ---
# Name of your trained yawn detection model file
YAWN_MODEL_FILENAME = 'yawn_cnn_combine_dataset.keras'

# Dlib's pre-trained facial landmark predictor model path
SHAPE_PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat' # Make sure you download and place this file!

# Model input dimensions (MUST match what the yawn model was trained on)
YAWN_IMG_HEIGHT = 64
YAWN_IMG_WIDTH = 64

# Labels mapping (must match the labels used during training from your generator's class_indices)
# Assuming 'Yawn' is 0 and 'No Yawn' is 1 as per the new dataset and typical alphabetical sorting.
# It's good practice to verify this from your training script's output (yawn_labels_actual).
LABELS = {0: 'Yawn', 1: 'No Yawn'} 

# Indices for mouth landmarks (from dlib's 68-point model)
# These define the points around the inner and outer mouth (points 48-67)
(M_START, M_END) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

# --- 1. Load the Trained Yawn Detection Model ---
print(f"Attempting to load yawn model from {YAWN_MODEL_FILENAME}...")
try:
    yawn_model = load_model(YAWN_MODEL_FILENAME)
    print("Yawn model loaded successfully.")
    yawn_model.summary()
except Exception as e:
    print(f"Error loading yawn model: {e}")
    print("Please ensure you have trained the yawn model (using the 'yawn-detection-standalone-app' Canvas) and that 'yawn_detection_cnn_model.h5' exists in the same directory.")
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
print("\nStarting real-time yawning detection with webcam. Press 'q' to quit.")
print("DEBUG: Observe the 'Cropped Mouth for CNN Input' window and console output for each prediction.")

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

    yawn_detected_in_frame = False # Flag to track if any yawn is detected in the current frame

    # Iterate over each detected face
    for rect in rects:
        # Get face bounding box (x, y, w, h) - used for positioning status text
        x_face, y_face, w_face, h_face = face_utils.rect_to_bb(rect)

        # Determine the facial landmarks for the face region
        shape = predictor(gray, rect)
        # Convert the (x, y)-coordinates to a NumPy array
        shape = face_utils.shape_to_np(shape)

        # Extract mouth coordinates from the landmarks
        mouth_points = shape[M_START:M_END]

        # Calculate the bounding box for the mouth region based on its landmarks
        (mx, my, mw, mh) = cv2.boundingRect(mouth_points)

        # Expand mouth region slightly to ensure full mouth is captured for the CNN
        # Increased buffer to ensure enough context for the CNN, as it was trained on these
        buffer_x = int(mw * 0.25)
        buffer_y = int(mh * 0.3)
        
        mx_start = max(0, mx - buffer_x)
        my_start = max(0, my - buffer_y)
        mx_end = min(gray.shape[1], mx + mw + buffer_x)
        my_end = min(gray.shape[0], my + mh + buffer_y)

        # Ensure the coordinates are valid and within frame bounds for cropping
        # Only proceed if the calculated region is valid
        if my_start >= my_end or mx_start >= mx_end:
            print(f"DEBUG: Invalid mouth crop coordinates: ({mx_start},{my_start}) to ({mx_end},{my_end}). Skipping mouth processing.")
            continue

        mouth_img_gray_cropped = gray[my_start:my_end, mx_start:mx_end]

        # Skip if cropped mouth image is invalid or too small after buffering
        # if mouth_img_gray_cropped.shape[0] < YAWN_IMG_HEIGHT * 0.5 or \
        #    mouth_img_gray_cropped.shape[1] < YAWN_IMG_WIDTH * 0.5:
        #     print(f"DEBUG: Skipped mouth due to very small or invalid crop size: {mouth_img_gray_cropped.shape}. Expected ~{YAWN_IMG_HEIGHT}x{YAWN_IMG_WIDTH}.")
        #     continue

        # Resize mouth image to model's input size (64x64)
        input_for_model = cv2.resize(mouth_img_gray_cropped, (YAWN_IMG_WIDTH, YAWN_IMG_HEIGHT))
        
        # DEBUG: Display the cropped and resized mouth image that goes into the CNN
        cv2.imshow('Cropped Mouth for CNN Input', input_for_model)

        # Normalize pixel values to [0, 1]
        input_for_model = input_for_model / 255.0

        # Expand dimensions to match model input shape (BATCH, HEIGHT, WIDTH, CHANNELS)
        input_for_model = np.expand_dims(input_for_model, axis=-1) # Add channel dimension (1 for grayscale)
        input_for_model = np.expand_dims(input_for_model, axis=0)      # Add batch dimension (for a single image)

        # Predict yawn state
        prediction = yawn_model.predict(input_for_model, verbose=0)
        predicted_class_index = np.argmax(prediction)
        predicted_label = LABELS[predicted_class_index]
        confidence = prediction[0][predicted_class_index] * 100

        # DEBUG: Print raw prediction values to console
        print(f"DEBUG: Mouth Raw Prediction: {prediction[0]}, Predicted Index: {predicted_class_index}, Label: {predicted_label}, Confidence: {confidence:.2f}%")

        # Check if prediction is 'Yawn' and update overall status for the frame
        if predicted_label == 'Yawn':
            yawn_detected_in_frame = True
            # Draw red rectangle around yawning mouth
            cv2.rectangle(frame, (mx_start, my_start), (mx_end, my_end), (0, 0, 255), 2)
            cv2.putText(frame, f"Yawning: {confidence:.1f}%", (mx_start, my_start - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else: # No Yawn
            # Draw green rectangle around non-yawning mouth
            cv2.rectangle(frame, (mx_start, my_start), (mx_end, my_end), (0, 255, 0), 1)
            cv2.putText(frame, f"No Yawn: {confidence:.1f}%", (mx_start, my_start - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Facial landmarks for mouth (optional: uncomment to visualize the landmark points)
        # for (lm_x, lm_y) in mouth_points:
        #     cv2.circle(frame, (lm_x, lm_y), 1, (255, 0, 255), -1)

    # Display overall yawning status for the frame
    # Position this text relative to the detected face's top-left corner if faces are found
    overall_status_text = "No Yawn Detected"
    if yawn_detected_in_frame:
        overall_status_text = "YAWNING!"
    
    if len(rects) > 0: # If at least one face was detected
        cv2.putText(frame, overall_status_text, (x_face, y_face - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    else: # If no face was detected
        cv2.putText(frame, "No Face Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
    # Display the processed frame
    cv2.imshow('Live Yawning Detection (Dlib Mouth Cropping)', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

# Release the camera and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
