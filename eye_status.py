import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import cv2
import numpy as np
import pathlib # For path manipulation
# Load model and face detector
IMG_HEIGHT = 64
IMG_WIDTH = 64
BATCH_SIZE = 32
path = "./eye_cnn.h5"
model = load_model("./eye_cnn.h5")

# Map class indices to labels (e.g., {0: 'closed_eye', 1: 'open_eye'})
# labels = {v: k for k, v in train_generator.class_indices.items()}
labels = {0: 'closed_eye', 1: 'open_eye'}
# With camera live capture
# --- 4. Real-time Eye Detection with Webcam ---
print("Starting real-time eye detection with webcam. Press 'q' to quit.")

# Load the trained model
try:
    print(f"Loaded model from: {path}")
except Exception as e:
    print(
        f"Error loading model: {e}. Please ensure the model was trained and saved correctly."
    )
    print("Exiting real-time detection.")
    exit()

# Load OpenCV's pre-trained Haar Cascade classifiers for face and eye detection
# You might need to locate these XML files on your system.
# Common locations:
#   - Anaconda: C:\Users\<YourUser>\anaconda3\Lib\site-packages\cv2\data\
#   - General Python: <Python_Install_Dir>\Lib\site-packages\cv2\data\
# Or download from: https://github.com/opencv/opencv/tree/master/data/haarcascades
face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
eye_cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

if face_cascade.empty():
    print(
        f"Error: Face cascade not loaded from {face_cascade_path}. Make sure the XML file exists."
    )
    exit()
if eye_cascade.empty():
    print(
        f"Error: Eye cascade not loaded from {eye_cascade_path}. Make sure the XML file exists."
    )
    exit()

cap = cv2.VideoCapture(0)  # 0 for default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(
        frame, cv2.COLOR_BGR2GRAY
    )  # Convert frame to grayscale for cascade detection

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        cv2.rectangle(
            frame, (x, y), (x + w, y + h), (255, 0, 0), 2
        )  # Draw rectangle around face

        roi_gray = gray[
            y : y + h, x : x + w
        ]  # Region of Interest (ROI) for eyes in grayscale
        roi_color = frame[y : y + h, x : x + w]  # ROI for drawing rectangles in color

        # List to store eye states for the current face
        current_face_eye_states = []

        # Detect eyes within the face ROI
        # Adjust eye detection region to focus on upper half of face to avoid mouth/nose
        eyes = eye_cascade.detectMultiScale(roi_gray[0 : int(h / 2), :], 1.3, 5)

        for ex, ey, ew, eh in eyes:
            # Ensure the eye region is within bounds and large enough to process
            eye_img_gray = roi_gray[ey : ey + eh, ex : ex + ew]

            if eye_img_gray.shape[0] == 0 or eye_img_gray.shape[1] == 0:
                continue  # Skip if eye region is empty or invalid

            # Resize eye image to model's input size
            eye_img_resized = cv2.resize(eye_img_gray, (IMG_WIDTH, IMG_HEIGHT))
            # Expand dimensions to match model input shape (BATCH, HEIGHT, WIDTH, CHANNELS)
            eye_input = np.expand_dims(
                eye_img_resized, axis=-1
            )  # Add channel dimension
            eye_input = np.expand_dims(eye_input, axis=0)  # Add batch dimension
            eye_input = eye_input / 255.0  # Normalize pixel values

            # Predict eye state
            prediction = model.predict(eye_input, verbose=0)
            predicted_class_index = np.argmax(prediction)
            predicted_label = labels[predicted_class_index]
            confidence = prediction[0][predicted_class_index] * 100

            # Add the detected eye's label to the list for this face
            current_face_eye_states.append(predicted_label)

            # Optionally, draw a smaller rectangle around each detected eye
            cv2.rectangle(
                roi_color, (ex, ey), (ex + ew, ey + eh), (0, 200, 255), 1
            )  # Yellow-ish for individual eye

        # Determine overall eye state for the face based on detected eyes
        face_status_text = "No Eyes Detected"  # Default if no eyes are found

        if len(current_face_eye_states) > 0:
            # If 'open_eye' is found in any of the detected eyes, the overall status is "Eyes Open"
            if "open_eye" in current_face_eye_states:
                face_status_text = "Eyes Open"
            # If all detected eyes are 'closed_eye', then set status to "Both Eyes Closed"
            elif all(label == "closed_eye" for label in current_face_eye_states):
                face_status_text = "Both Eyes Closed"
            else:
                # This case might happen if some eyes are closed and others are not detected,
                # or if there's a mix of 'open_eye' and 'closed_eye' (e.g., one eye closed).
                # For this specific request, if 'open_eye' is not explicitly found but not all are 'closed_eye',
                # we'll default to "Eyes Open" as the dominant state.
                face_status_text = "Eyes Open (Mixed/Partial)"

        # Display the overall face status above the face rectangle
        cv2.putText(
            frame,
            face_status_text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    cv2.imshow("Eye State Detection", frame)  # Display the frame

    if cv2.waitKey(1000) & 0xFF == ord("q"):  # Press 'q' to quit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
