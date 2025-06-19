import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import cv2
import numpy as np
import pathlib # For path manipulation

import matplotlib.pyplot as plt # Import matplotlib for plotting

from sklearn.metrics import confusion_matrix # Import for confusion matrix
import seaborn as sns # Import seaborn for heatmap visualization

# --- Configuration ---
# Path to your extracted dataset. Make sure to adjust this!
# Example: If you unzipped `mrl-eye-dataset.zip` into a folder named 'mrl-eye-dataset'
# in the same directory as your script, this path should be correct.
DATA_DIR = '../data'

# Model training parameters
IMG_HEIGHT = 64
IMG_WIDTH = 64
BATCH_SIZE = 32
EPOCHS = 15 # You might need more epochs for better accuracy, or fewer for quicker testing
NUM_CLASSES = 2 # open_eye, closed_eye
MODEL_FILENAME = 'eye_cnn_with_graph.keras' # Name to save the trained model

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


# --- 5. Plotting Learning Curves ---
print("\nPlotting learning curves...")

# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st plot
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Eye Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid(True)

# Plot training & validation loss values
plt.subplot(1, 2, 2) # 1 row, 2 columns, 2nd plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Eye Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid(True)

plt.tight_layout() # Adjust layout to prevent overlapping titles/labels
plt.show() # Display the plots


# --- 6. Generate and Plot Confusion Matrix ---
print("\nGenerating confusion matrix...")

# Get predictions on the validation set
# Reset the generator to ensure predictions are made from the start
validation_generator.reset()
Y_pred = model.predict(validation_generator, steps=validation_generator.samples // BATCH_SIZE + 1)
y_pred_classes = np.argmax(Y_pred, axis=1)

# Get true labels for the validation set
y_true = validation_generator.classes[validation_generator.index_array]

# Get class names from the generator
class_names = list(validation_generator.class_indices.keys())

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Eye Model Confusion Matrix')
plt.show()

print("Confusion matrix plot displayed.")