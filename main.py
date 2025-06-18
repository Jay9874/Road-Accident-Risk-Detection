import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
import dlib # For face and facial landmark detection
from imutils import face_utils # Helper functions for dlib landmarks

# --- Configuration ---
# --- Yawning Model Configuration ---
YAWN_MODEL_FILENAME = 'yawn_cnn_combine_dataset.keras'
YAWN_IMG_HEIGHT = 64
YAWN_IMG_WIDTH = 64
# Yawn Labels (verify this from your yawn training script's output for yaw_labels_actual)
YAWN_LABELS = {0: 'Yawn', 1: 'No Yawn'} 

# --- Eye Model Configuration ---
EYE_MODEL_FILENAME = 'eye_cnn.h5' # Make sure this is your trained eye model
EYE_IMG_HEIGHT = 64
EYE_IMG_WIDTH = 64
# Eye Labels (verify this from your eye training script's output for its class_indices)
EYE_LABELS = {0: 'closed_eye', 1: 'open_eye'} 

# --- Dlib Configuration ---
SHAPE_PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat' # Download and place this file!

# Dlib facial landmark indices for specific regions
(M_START, M_END) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]
(L_EYE_START, L_EYE_END) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(R_EYE_START, R_EYE_END) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]


# --- 1. Load Trained Models ---
print(f"Attempting to load Yawn model from {YAWN_MODEL_FILENAME}...")
try:
    yawn_model = load_model(YAWN_MODEL_FILENAME)
    print("Yawn model loaded successfully.")
except Exception as e:
    print(f"Error loading yawn model: {e}")
    print("Please ensure 'yawn_detection_cnn_model.h5' exists in the same directory.")
    exit()

print(f"\nAttempting to load Eye model from {EYE_MODEL_FILENAME}...")
try:
    eye_model = load_model(EYE_MODEL_FILENAME)
    print("Eye model loaded successfully.")
except Exception as e:
    print(f"Error loading eye model: {e}")
    print("Please ensure 'eye_state_cnn_model.h5' exists in the same directory.")
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
print("\nStarting live drowsiness detection (Eyes & Yawn). Press 'q' to quit.")
print("DEBUG: Watch console for prediction details and separate cropped image windows.")

cap = cv2.VideoCapture(0) # 0 for default webcam

if not cap.isOpened():
    print("Error: Could not open webcam. Make sure your camera is connected and not in use by another application. Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` or higher if you have multiple cameras.")
    exit()

while True:
    ret, frame = cap.read() # Read a frame from the webcam
    if not ret:
        print("Failed to grab frame from camera. Exiting...")
        break

    # Flip frame horizontally for selfie-style webcam feeds
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert frame to grayscale for Dlib

    # Detect faces in the grayscale frame using Dlib's detector
    rects = detector(gray, 0) # The '0' indicates no upsampling

    # Flags for overall frame status
    yawn_detected_in_frame = False
    eyes_closed_in_frame = False

    # Iterate over each detected face
    for rect in rects:
        # Get face bounding box (x, y, w, h) - used for positioning status text
        x_face, y_face, w_face, h_face = face_utils.rect_to_bb(rect)
        # Optional: Draw rectangle around face (uncomment if you want to see face box)
        # cv2.rectangle(frame, (x_face, y_face), (x_face+w_face, y_face+h_face), (255, 0, 0), 2)

        # Determine the facial landmarks for the face region
        shape = predictor(gray, rect)
        # Convert the (x, y)-coordinates to a NumPy array
        shape = face_utils.shape_to_np(shape)

        # --- Yawning Detection Logic ---
        mouth_points = shape[M_START:M_END]
        (mx, my, mw, mh) = cv2.boundingRect(mouth_points)
        
        # Buffer for mouth crop
        mouth_buffer_x = int(mw * 0.25)
        mouth_buffer_y = int(mh * 0.3)
        mx_start = max(0, mx - mouth_buffer_x)
        my_start = max(0, my - mouth_buffer_y)
        mx_end = min(gray.shape[1], mx + mw + mouth_buffer_x)
        my_end = min(gray.shape[0], my + mh + mouth_buffer_y)

        if my_start < my_end and mx_start < mx_end: # Ensure valid crop
            mouth_img_gray_cropped = gray[my_start:my_end, mx_start:mx_end]

            if mouth_img_gray_cropped.shape[0] > 0 and mouth_img_gray_cropped.shape[1] > 0:
                input_yawn_model = cv2.resize(mouth_img_gray_cropped, (YAWN_IMG_WIDTH, YAWN_IMG_HEIGHT))
                cv2.imshow('Cropped Mouth for Yawn CNN', input_yawn_model) # DEBUG: show cropped mouth
                
                input_yawn_model = input_yawn_model / 255.0
                input_yawn_model = np.expand_dims(input_yawn_model, axis=-1)
                input_yawn_model = np.expand_dims(input_yawn_model, axis=0)

                prediction_yawn = yawn_model.predict(input_yawn_model, verbose=0)
                predicted_yawn_class_index = np.argmax(prediction_yawn)
                predicted_yawn_label = YAWN_LABELS[predicted_yawn_class_index]
                yawn_confidence = prediction_yawn[0][predicted_yawn_class_index] * 100

                if predicted_yawn_label == 'Yawn':
                    yawn_detected_in_frame = True
                    cv2.rectangle(frame, (mx_start, my_start), (mx_end, my_end), (0, 0, 255), 2) # Red for Yawn
                    cv2.putText(frame, f"Yawn: {yawn_confidence:.1f}%", (mx_start, my_start - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                else:
                    cv2.rectangle(frame, (mx_start, my_start), (mx_end, my_end), (0, 255, 0), 1) # Green for No Yawn
                    cv2.putText(frame, f"No Yawn: {yawn_confidence:.1f}%", (mx_start, my_start - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                print(f"DEBUG: Yawn Prediction: {prediction_yawn[0]}, Label: {predicted_yawn_label}, Conf: {yawn_confidence:.2f}%")
            else:
                print("DEBUG: Mouth crop invalid (0 dimensions). Skipping yawn prediction.")


        # --- Eye Status Detection Logic ---
        left_eye_points = shape[L_EYE_START:L_EYE_END]
        right_eye_points = shape[R_EYE_START:R_EYE_END]
        
        current_face_eye_states = []

        for i, eye_points in enumerate([left_eye_points, right_eye_points]):
            (ex, ey, ew, eh) = cv2.boundingRect(eye_points)
            
            # Buffer for eye crop
            eye_buffer_x = int(ew * 0.15)
            eye_buffer_y = int(eh * 0.15)
            ex_start = max(0, ex - eye_buffer_x)
            ey_start = max(0, ey - eye_buffer_y)
            ex_end = min(gray.shape[1], ex + ew + eye_buffer_x)
            ey_end = min(gray.shape[0], ey + eh + eye_buffer_y)

            if ey_start < ey_end and ex_start < ex_end: # Ensure valid crop
                eye_img_gray_cropped = gray[ey_start:ey_end, ex_start:ex_end]

                if eye_img_gray_cropped.shape[0] > 0 and eye_img_gray_cropped.shape[1] > 0:
                    input_eye_model = cv2.resize(eye_img_gray_cropped, (EYE_IMG_WIDTH, EYE_IMG_HEIGHT))
                    cv2.imshow(f'Cropped Eye {i+1} for Eye CNN', input_eye_model) # DEBUG: show cropped eye
                    
                    input_eye_model = input_eye_model / 255.0
                    input_eye_model = np.expand_dims(input_eye_model, axis=-1)
                    input_eye_model = np.expand_dims(input_eye_model, axis=0)

                    prediction_eye = eye_model.predict(input_eye_model, verbose=0)
                    predicted_eye_class_index = np.argmax(prediction_eye)
                    predicted_eye_label = EYE_LABELS[predicted_eye_class_index]
                    eye_confidence = prediction_eye[0][predicted_eye_class_index] * 100

                    current_face_eye_states.append(predicted_eye_label)

                    color = (0, 200, 255) # Yellow-ish for open
                    if predicted_eye_label == 'closed_eye':
                        eyes_closed_in_frame = True # Set flag if any eye is closed
                        color = (0, 0, 255) # Red for closed
                    
                    cv2.rectangle(frame, (ex_start, ey_start), (ex_end, ey_end), color, 1)
                    cv2.putText(frame, f"{predicted_eye_label.replace('_', ' ').capitalize()}: {eye_confidence:.1f}%", (ex_start, ey_start - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    print(f"DEBUG: Eye {i+1} Prediction: {prediction_eye[0]}, Label: {predicted_eye_label}, Conf: {eye_confidence:.2f}%")
                else:
                    print(f"DEBUG: Eye {i+1} crop invalid (0 dimensions). Skipping eye prediction.")
            else:
                print(f"DEBUG: Eye {i+1} invalid crop coordinates: ({ex_start},{ey_start}) to ({ex_end},{ey_end}). Skipping eye processing.")

        # --- Overall Drowsiness Status ---
        overall_drowsiness_status = "Alert"
        status_color = (0, 255, 0) # Green for alert

        if eyes_closed_in_frame and yawn_detected_in_frame:
            overall_drowsiness_status = "Drowsy (Eyes Closed & Yawning!)"
            status_color = (0, 0, 255) # Red for drowsy
        elif eyes_closed_in_frame:
            overall_drowsiness_status = "Eyes Closed (Possible Drowsiness)"
            status_color = (0, 165, 255) # Orange for eyes closed
        elif yawn_detected_in_frame:
            overall_drowsiness_status = "Yawning (Possible Drowsiness)"
            status_color = (0, 165, 255) # Orange for yawning

        # Display overall status for the face
        cv2.putText(frame, overall_drowsiness_status, (x_face, y_face - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
    # --- No Face Detected Status ---
    if len(rects) == 0:
        cv2.putText(frame, "No Face Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
    # Display the main processed frame
    cv2.imshow('Live Drowsiness Detection', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
