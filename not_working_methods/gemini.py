import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# --- Configuration ---
# Path to your pre-trained Keras model for eye status classification.
# Make sure this file is in the same directory as your script, or provide the full path.
MODEL_PATH = "./cnn.h5"

# Haar Cascade XML file for face detection.
# This path typically points to where OpenCV installs its cascade files.
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Target size for the input image to your CNN model.
# Your model was trained on 128x128 images.
MODEL_INPUT_SIZE = (128, 128)

# Threshold for classifying eye status.
# If the model's output probability is greater than this, it's "Open", otherwise "Closed".
PREDICTION_THRESHOLD = 0.5

# --- Load Models and Classifiers ---
try:
    # Load the pre-trained Keras model
    model = load_model(MODEL_PATH)
    print(f"Successfully loaded eye status model from: {MODEL_PATH}")
except Exception as e:
    print(f"Error loading Keras model: {e}")
    print("Please ensure 'eye_status_cnn_model.h5' is in the correct path.")
    exit()

try:
    # Load the Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if face_cascade.empty():
        raise IOError(f"Could not load face cascade from {FACE_CASCADE_PATH}")
    print(f"Successfully loaded face cascade from: {FACE_CASCADE_PATH}")
except IOError as e:
    print(f"Error loading face cascade: {e}")
    print(
        "Please ensure 'haarcascade_frontalface_default.xml' is correctly installed with OpenCV."
    )
    exit()

# --- Initialize Webcam ---
# 0 typically refers to the default built-in webcam.
# If you have multiple cameras, you might need to try 1, 2, etc.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(
        "Error: Could not open webcam. Please check if the camera is connected and not in use."
    )
    exit()

print("\n--- Live Eye Status Detection Started ---")
print("Press 'q' to quit the application.")

# --- Main Loop for Live Detection ---
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # If frame is not read successfully, break the loop
    if not ret:
        print("Failed to grab frame, exiting...")
        break

    # Flip the frame horizontally for a mirrored view (optional, but common for webcams)
    frame = cv2.flip(frame, 1)

    # Convert the frame to grayscale for face detection (Haar cascades work on grayscale)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    # scaleFactor: How much the image size is reduced at each image scale.
    # minNeighbors: How many neighbors each candidate rectangle should have to retain it.
    faces = face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    # Iterate over each detected face
    for x, y, w, h in faces:
        # Extract the face region from the original color frame
        face_roi = frame[y : y + h, x : x + w]

        # Check if the face region is valid (not empty)
        if face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
            continue

        # Resize the face region to the model's expected input size (e.g., 128x128)
        resized_face = cv2.resize(face_roi, MODEL_INPUT_SIZE)

        # Normalize pixel values to be between 0 and 1
        normalized_face = resized_face / 255.0

        # Expand dimensions to create a batch of 1 image
        # Keras models expect input in the shape (batch_size, height, width, channels)
        input_image_for_model = np.expand_dims(normalized_face, axis=0)

        # Make a prediction using the loaded CNN model
        # The model outputs a probability (e.g., close to 1 for open, close to 0 for closed)
        prediction_prob = model.predict(input_image_for_model)[0][0]
        print(f"The accuracy: {prediction_prob}")
        # Determine the label based on the prediction threshold
        # If probability is > 0.5, classify as "Open", otherwise "Closed"
        if prediction_prob > PREDICTION_THRESHOLD:
            label = "Open"
            color = (0, 255, 0)  # Green color for "Open"
        else:
            label = "Closed"
            color = (0, 0, 255)  # Red color for "Closed"

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Put the predicted label text above the rectangle
        # FONT_HERSHEY_SIMPLEX is a common font type
        # 0.9 is font scale, 2 is thickness
        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
            cv2.LINE_AA,
        )

        # Optional: Print prediction details to console for debugging
        # print(f"Face at ({x},{y}) - Predicted Probability: {prediction_prob:.4f}, Label: {label}")

    # Display the resulting frame
    cv2.imshow("Live Eye Status Detection (Press 'q' to Quit)", frame)

    # Wait for 1 millisecond and check for 'q' key press
    # `cv2.waitKey(1)` returns the ASCII value of the pressed key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("\n'q' pressed. Exiting application.")
        break

# --- Cleanup ---
# Release the webcam
cap.release()
# Destroy all OpenCV windows
cv2.destroyAllWindows()
print("Application closed.")
