import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

# --- Configuration ---
# Name of your trained eye detection model file
EYE_MODEL_FILENAME = 'eye_cnn.h5'

# Path to the image you want to test.
# IMPORTANT: This image should ALREADY BE CROPPED to contain only an eye.
# Example: 'path/to/your/cropped_eye.jpg'
IMAGE_PATH = 'images/latest_closed_eye.jpg' # <--- UPDATE THIS LINE WITH YOUR CROPPED EYE IMAGE PATH

# Model input dimensions (must match what the model was trained on)
EYE_IMG_HEIGHT = 64
EYE_IMG_WIDTH = 64

# Labels mapping (must match the labels used during training)
# From our previous discussions for the MRL-Eye dataset, it's typically:
LABELS = {0: 'closed_eye', 1: 'open_eye'} # Adjust if your training output shows different mapping

# --- 1. Load the Trained Eye Detection Model ---
print(f"Attempting to load eye model from {EYE_MODEL_FILENAME}...")
try:
    eye_model = load_model(EYE_MODEL_FILENAME)
    print("Eye model loaded successfully.")
    eye_model.summary()
except Exception as e:
    print(f"Error loading eye model: {e}")
    print("Please ensure you have trained the eye model and that 'eye_state_cnn_model.h5' exists in the same directory.")
    exit()

# --- 2. Load and Preprocess the Cropped Eye Image ---
print(f"\nLoading and preprocessing cropped eye image from {IMAGE_PATH}...")
if not os.path.exists(IMAGE_PATH):
    print(f"Error: Image not found at {IMAGE_PATH}. Please check the path.")
    exit()

# Load image in grayscale directly as the eye image
eye_img_gray = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

if eye_img_gray is None:
    print(f"Error: Could not load image from {IMAGE_PATH}. Check if it's a valid image file (e.g., .jpg, .png).")
    exit()

# Resize image to the model's expected input dimensions
eye_img_resized = cv2.resize(eye_img_gray, (EYE_IMG_WIDTH, EYE_IMG_HEIGHT))

# Normalize pixel values to [0, 1]
eye_input = eye_img_resized / 255.0

# Expand dimensions to match model input shape (BATCH, HEIGHT, WIDTH, CHANNELS)
# The model expects a batch dimension (even for a single image) and a channel dimension (1 for grayscale)
eye_input = np.expand_dims(eye_input, axis=-1) # Add channel dimension
eye_input = np.expand_dims(eye_input, axis=0)      # Add batch dimension

print("Image preprocessed successfully.")

# --- 3. Make a Prediction ---
print("\nMaking prediction...")
prediction = eye_model.predict(eye_input, verbose=0) # verbose=0 to suppress prediction log

# Get the predicted class (index with highest probability)
predicted_class_index = np.argmax(prediction)

# Get the corresponding label and confidence
predicted_label = LABELS[predicted_class_index]
confidence = prediction[0][predicted_class_index] * 100

# --- 4. Display Results ---
print("\n--- Prediction Result ---")
print(f"Image File: {os.path.basename(IMAGE_PATH)}")
print(f"Predicted Eye Status: {predicted_label.replace('_', ' ').capitalize()}") # Format nicely
print(f"Confidence: {confidence:.2f}%")
print("------------------------")

# Optional: Display the image with the prediction (requires OpenCV's GUI capabilities)
try:
    # Display the original image, not just the resized version, for better context
    original_display_img = cv2.imread(IMAGE_PATH)
    if original_display_img is not None:
        # Put text on the image
        display_text = f"{predicted_label.replace('_', ' ').capitalize()}: {confidence:.1f}%"
        cv2.putText(original_display_img, display_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) # Green text for prediction
        
        cv2.imshow(f"Eye Status: {predicted_label} ({os.path.basename(IMAGE_PATH)})", original_display_img)
        print("\nPress any key on the image window to close it.")
        cv2.waitKey(0) # Wait indefinitely until a key is pressed
        cv2.destroyAllWindows()
    else:
        print("Could not load original image for display. Showing only text output.")
except Exception as e:
    print(f"Could not display image using OpenCV: {e}. This might happen in environments without GUI support.")
    print("You can comment out the `cv2.imshow` and related lines if not needed.")

