import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import cv2
import numpy as np
import pathlib # For path manipulation

# --- Configuration ---
# Path to your extracted dataset. Make sure to adjust this!
# Example: If you unzipped `mrl-eye-dataset.zip` into a folder named 'mrl-eye-dataset'
# in the same directory as your script, this path should be correct.
DATA_DIR = 'mrl-eye-dataset'

# Model training parameters
IMG_HEIGHT = 64
IMG_WIDTH = 64
BATCH_SIZE = 32
EPOCHS = 15 # You might need more epochs for better accuracy, or fewer for quicker testing
NUM_CLASSES = 2 # open_eye, closed_eye
MODEL_FILENAME = 'eye_state_cnn_model.h5' # Name to save the trained model

# --- 1. Dataset Loading and Preprocessing ---
print("Loading and preprocessing dataset...")

# Create ImageDataGenerators for data augmentation and scaling
# Data augmentation helps prevent overfitting by creating new training samples
# from existing ones (e.g., rotating, shifting, zooming).
train_datagen = ImageDataGenerator(
    rescale=1./255,          # Normalize pixel values to [0, 1]
    rotation_range=10,       # Rotate images by up to 10 degrees
    width_shift_range=0.1,   # Shift images horizontally by up to 10%
    height_shift_range=0.1,  # Shift images vertically by up to 10%
    shear_range=0.1,         # Apply shear transformation
    zoom_range=0.1,          # Zoom in/out by up to 10%
    horizontal_flip=True,    # Randomly flip images horizontally
    fill_mode='nearest'      # Fill newly created pixels with the nearest values
)

test_datagen = ImageDataGenerator(rescale=1./255) # Only rescale for test data, no augmentation

# Flow training images in batches from the directory
train_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical', # 'categorical' for one-hot encoded labels (2 classes)
    color_mode='grayscale' # Eyes are often processed in grayscale for simplicity and speed
)

# Flow validation images in batches from the directory
validation_generator = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'test'),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='grayscale'
)

# Map class indices to labels (e.g., {0: 'closed_eye', 1: 'open_eye'})
labels = {v: k for k, v in train_generator.class_indices.items()}
print(f"Class labels: {labels}")

# --- 2. CNN Model Definition ---
print("Defining CNN model...")

model = Sequential([
    # First Convolutional Block
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25), # Dropout to prevent overfitting

    # Second Convolutional Block
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Third Convolutional Block
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Flatten the output for the Dense layers
    Flatten(),

    # Fully Connected Layers
    Dense(128, activation='relu'),
    Dropout(0.5), # Another dropout layer
    Dense(NUM_CLASSES, activation='softmax') # Output layer with softmax for classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy', # Use categorical_crossentropy for one-hot labels
              metrics=['accuracy'])

model.summary()

# --- 3. Model Training ---
print("Training model...")

# Callbacks for early stopping and model checkpointing
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(MODEL_FILENAME, save_best_only=True, monitor='val_loss', mode='min')

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=[early_stopping, model_checkpoint]
)

print(f"Model training complete. Best model saved to {MODEL_FILENAME}")

# --- 4. Real-time Eye Detection with Webcam ---
print("Starting real-time eye detection with webcam. Press 'q' to quit.")

# Load the trained model
try:
    model = load_model(MODEL_FILENAME)
    print(f"Loaded model from {MODEL_FILENAME}")
except Exception as e:
    print(f"Error loading model: {e}. Please ensure the model was trained and saved correctly.")
    print("Exiting real-time detection.")
    exit()

# Load OpenCV's pre-trained Haar Cascade classifiers for face and eye detection
# You might need to locate these XML files on your system.
# Common locations:
#   - Anaconda: C:\Users\<YourUser>\anaconda3\Lib\site-packages\cv2\data\
#   - General Python: <Python_Install_Dir>\Lib\site-packages\cv2\data\
# Or download from: https://github.com/opencv/opencv/tree/master/data/haarcascades
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

if face_cascade.empty():
    print(f"Error: Face cascade not loaded from {face_cascade_path}. Make sure the XML file exists.")
    exit()
if eye_cascade.empty():
    print(f"Error: Eye cascade not loaded from {eye_cascade_path}. Make sure the XML file exists.")
    exit()

cap = cv2.VideoCapture(0) # 0 for default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read() # Read a frame from the webcam
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert frame to grayscale for cascade detection

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # Draw rectangle around face

        roi_gray = gray[y:y+h, x:x+w] # Region of Interest (ROI) for eyes in grayscale
        roi_color = frame[y:y+h, x:x+w] # ROI for drawing rectangles in color

        # Detect eyes within the face ROI
        # Adjust eye detection region to focus on upper half of face to avoid mouth/nose
        eyes = eye_cascade.detectMultiScale(roi_gray[0:int(h/2), :], 1.3, 5)

        for (ex, ey, ew, eh) in eyes:
            # Ensure the eye region is within bounds and large enough to process
            eye_img_gray = roi_gray[ey:ey+eh, ex:ex+ew]

            if eye_img_gray.shape[0] == 0 or eye_img_gray.shape[1] == 0:
                continue # Skip if eye region is empty

            # Resize eye image to model's input size
            eye_img_resized = cv2.resize(eye_img_gray, (IMG_WIDTH, IMG_HEIGHT))
            # Expand dimensions to match model input shape (BATCH, HEIGHT, WIDTH, CHANNELS)
            eye_input = np.expand_dims(eye_img_resized, axis=-1) # Add channel dimension
            eye_input = np.expand_dims(eye_input, axis=0)      # Add batch dimension
            eye_input = eye_input / 255.0 # Normalize pixel values

            # Predict eye state
            prediction = model.predict(eye_input, verbose=0)
            predicted_class_index = np.argmax(prediction)
            predicted_label = labels[predicted_class_index]
            confidence = prediction[0][predicted_class_index] * 100

            # Draw rectangle around eye and display label
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            text = f"{predicted_label}: {confidence:.2f}%"
            cv2.putText(roi_color, text, (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Eye State Detection', frame) # Display the frame

    if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to quit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

