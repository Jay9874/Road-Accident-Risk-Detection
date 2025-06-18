import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

# --- Configuration ---
# Name of your trained yawn detection model file
YAWN_MODEL_FILENAME = 'yawn_detection_cnn_model.h5'

# Model input dimensions (MUST match what the yawn model was trained on)
# Based on your training script, this is 64x64
YAWN_IMG_HEIGHT = 64
YAWN_IMG_WIDTH = 64

# Labels mapping (must match the labels used during training from the yawn_labels_actual dictionary)
# From the "Yawn Eye Dataset (New)", it's typically:
LABELS = {0: 'Yawn', 1: 'No Yawn'} # Verify this based on your generator's class_indices output

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

# --- 2. Start Live Camera Feed and Real-time Detection ---
print("\nStarting real-time yawning detection with webcam. Press 'q' to quit.")
print("The entire resized frame will be fed into the model.")

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

    # Convert frame to grayscale (as the model was trained on grayscale images)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize the entire grayscale frame to the model's input dimensions
    # This is crucial because your model was trained on 64x64 inputs from potentially larger images
    input_for_model = cv2.resize(gray_frame, (YAWN_IMG_WIDTH, YAWN_IMG_HEIGHT))
    
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

    # Display status on the original frame
    status_text = f"{predicted_label.replace('_', ' ').capitalize()}: {confidence:.1f}%"
    text_color = (0, 255, 0) # Green for No Yawn
    if predicted_label == 'Yawn':
        text_color = (0, 0, 255) # Red for Yawn
        status_text = f"YAWNING! ({confidence:.1f}%)"
    
    # Position the text at the top-left of the frame
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        
    # Display the processed frame
    cv2.imshow('Live Yawning Detection (Full Frame Input)', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
