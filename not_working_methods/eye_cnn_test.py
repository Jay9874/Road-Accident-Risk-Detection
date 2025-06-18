import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import cv2
import numpy as np
import pathlib  # For path manipulation

# --- Configuration ---
# Path to your extracted dataset. Make sure to adjust this!
# Example: If you unzipped `mrl-eye-dataset.zip` into a folder named 'mrl-eye-dataset'
# in the same directory as your script, this path should be correct.

# Model training parameters
IMAGE_PATH = "./images/test_closed.png"
EYE_IMG_HEIGHT = 64
EYE_IMG_WIDTH = 64
BATCH_SIZE = 32
EPOCHS = (
    15  # You might need more epochs for better accuracy, or fewer for quicker testing
)
NUM_CLASSES = 2  # open_eye, closed_eye
EYE_MODEL_FILENAME = "eye_cnn.h5"  # Name to save the trained model


# Map class indices to labels (e.g., {0: 'closed_eye', 1: 'open_eye'})
# LABELS = {v: k for k, v in train_generator.class_indices.items()} # Adjust if your training output shows different mapping
LABELS = {0: "closed_eye", 1: "open_eye"}
# --- 1. Load the Trained Eye Detection Model ---
print(f"Attempting to load eye model from {EYE_MODEL_FILENAME}...")
try:
    eye_model = load_model(EYE_MODEL_FILENAME)
    print("Eye model loaded successfully.")
    eye_model.summary()
except Exception as e:
    print(f"Error loading eye model: {e}")
    print(
        "Please ensure you have trained the eye model and that 'eye_state_cnn_model.h5' exists in the same directory."
    )
    exit()

# --- 2. Load OpenCV's Pre-trained Haar Cascade Classifiers ---
print("\nLoading Haar Cascade classifiers for face and eye detection...")
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
print("Haar Cascades loaded successfully.")

# --- 3. Load and Process the Static Image ---
print(f"\nLoading and processing image from {IMAGE_PATH}...")
if not os.path.exists(IMAGE_PATH):
    print(f"Error: Image not found at {IMAGE_PATH}. Please check the path.")
    exit()

frame = cv2.imread(IMAGE_PATH)  # Load the image
if frame is None:
    print(
        f"Error: Could not load image from {IMAGE_PATH}. Check if it's a valid image file (e.g., .jpg, .png)."
    )
    exit()

gray = cv2.cvtColor(
    frame, cv2.COLOR_BGR2GRAY
)  # Convert to grayscale for cascade detection

# Detect faces in the grayscale frame
faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))

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

    # Detect eyes within the upper half of the face ROI to avoid mouth/nose
    eyes = eye_cascade.detectMultiScale(roi_gray[0 : int(h * 0.6), :], 1.2, 8)

    # Sort eyes by x-coordinate to consistently identify left/right
    eyes = sorted(eyes, key=lambda e: e[0])

    for ex, ey, ew, eh in eyes:
        # Ensure the detected eye region is valid and large enough
        # Add a small buffer around the eye region for better capture
        buffer_x = int(ew * 0.1)
        buffer_y = int(eh * 0.1)
        ex_start = max(0, ex - buffer_x)
        ey_start = max(0, ey - buffer_y)
        ex_end = min(roi_gray.shape[1], ex + ew + buffer_x)
        ey_end = min(roi_gray.shape[0], ey + eh + buffer_y)

        eye_img_gray = roi_gray[ey_start:ey_end, ex_start:ex_end]

        if (
            eye_img_gray.shape[0] < EYE_IMG_HEIGHT / 2
            or eye_img_gray.shape[1] < EYE_IMG_WIDTH / 2
        ):
            # Skip if the detected eye region is too small after buffering
            continue

        # Resize eye image to model's input size
        eye_img_resized = cv2.resize(eye_img_gray, (EYE_IMG_WIDTH, EYE_IMG_HEIGHT))

        # Normalize pixel values to [0, 1]
        eye_input = eye_img_resized / 255.0

        # Expand dimensions to match model input shape (BATCH, HEIGHT, WIDTH, CHANNELS)
        eye_input = np.expand_dims(
            eye_input, axis=-1
        )  # Add channel dimension (1 for grayscale)
        eye_input = np.expand_dims(
            eye_input, axis=0
        )  # Add batch dimension (for a single image)

        # Predict eye state
        prediction = eye_model.predict(eye_input, verbose=0)
        predicted_class_index = np.argmax(prediction)
        predicted_label = LABELS[predicted_class_index]
        confidence = prediction[0][predicted_class_index] * 100

        # Add the detected eye's label to the list for this face
        current_face_eye_states.append(predicted_label)

        # Draw rectangle around eye and display individual label
        cv2.rectangle(
            roi_color, (ex, ey), (ex + ew, ey + eh), (0, 200, 255), 1
        )  # Yellow-ish for individual eye
        text_individual_eye = f"{predicted_label}: {confidence:.1f}%"
        cv2.putText(
            roi_color,
            text_individual_eye,
            (ex, ey - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 200, 255),
            1,
        )

    # Determine overall eye state for the face based on detected eyes
    overall_eye_status = "No Eyes Detected"

    if len(current_face_eye_states) == 2:
        if (
            "open_eye" in current_face_eye_states[0]
            and "open_eye" in current_face_eye_states[1]
        ):
            overall_eye_status = "Eyes Open"
        elif (
            "closed_eye" in current_face_eye_states[0]
            and "closed_eye" in current_face_eye_states[1]
        ):
            overall_eye_status = "Both Eyes Closed"
        else:  # One eye open, one closed, or mixed
            overall_eye_status = "Eyes Mixed State"
    elif len(current_face_eye_states) == 1:
        overall_eye_status = (
            f"One Eye: {current_face_eye_states[0].replace('_', ' ').capitalize()}"
        )

    # Display the overall face status above the face rectangle
    cv2.putText(
        frame,
        overall_eye_status,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

# Display the processed frame
cv2.imshow(f"Eye Status Detection for: {os.path.basename(IMAGE_PATH)}", frame)
print("\nPress any key on the image window to close it.")
cv2.waitKey(0)  # Wait indefinitely until a key is pressed
cv2.destroyAllWindows()
