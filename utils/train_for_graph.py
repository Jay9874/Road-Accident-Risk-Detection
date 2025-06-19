import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import numpy as np
import matplotlib.pyplot as plt # Import matplotlib for plotting

from sklearn.metrics import confusion_matrix # Import for confusion matrix
import seaborn as sns # Import seaborn for heatmap visualization

# --- Configuration ---
# Path to your processed yawn dataset, which contains mouth-cropped images.
# This should point to the output directory from the preprocessing script.
YAWN_DATA_DIR = '../processed_yawn_dataset' # <--- NOW POINTS TO PROCESSED DATA

# Model training parameters for yawning
YAWN_IMG_HEIGHT = 64
YAWN_IMG_WIDTH = 64
YAWN_BATCH_SIZE = 32
YAWN_EPOCHS = 15 # You can adjust epochs as needed for optimal performance
YAWN_NUM_CLASSES = 2 # Yawn, No Yawn
YAWN_MODEL_FILENAME = 'yawn_cnn_with_graph.keras' # Name to save the trained model

# --- 1. Yawn Dataset Loading and Preprocessing ---
print("Loading and preprocessing yawn dataset from the processed source...")

# Create ImageDataGenerators for data augmentation and scaling
train_yawn_datagen = ImageDataGenerator(
    rescale=1./255,          # Normalize pixel values to [0, 1]
    rotation_range=10,       # Rotate images by up to 10 degrees
    width_shift_range=0.1,   # Shift images horizontally by up to 10%
    height_shift_range=0.1,  # Shift images vertically by up to 10%
    shear_range=0.1,         # Apply shear transformation
    zoom_range=0.1,          # Zoom in/out by up to 10%
    horizontal_flip=True,    # Randomly flip images horizontally
    fill_mode='nearest'      # Fill newly created pixels with the nearest values
)

test_yawn_datagen = ImageDataGenerator(rescale=1./255) # Only rescale for test data, no augmentation

# Flow training images in batches from the 'train' subdirectory of the processed dataset
train_yawn_generator = train_yawn_datagen.flow_from_directory(
    os.path.join(YAWN_DATA_DIR, 'train'),
    target_size=(YAWN_IMG_HEIGHT, YAWN_IMG_WIDTH),
    batch_size=YAWN_BATCH_SIZE,
    class_mode='categorical', # 'categorical' for one-hot encoded labels (2 classes)
    color_mode='grayscale',   # Model expects grayscale input
    classes=['Yawn', 'No Yawn'] # Explicitly specify classes to ensure correct loading and order
)

# Flow validation images in batches from the 'test' subdirectory of the processed dataset
validation_yawn_generator = test_yawn_datagen.flow_from_directory(
    os.path.join(YAWN_DATA_DIR, 'test'),
    target_size=(YAWN_IMG_HEIGHT, YAWN_IMG_WIDTH),
    batch_size=YAWN_BATCH_SIZE,
    class_mode='categorical',
    color_mode='grayscale',
    classes=['Yawn', 'No Yawn'] # Explicitly specify classes for consistency
)

# Map class indices to labels (useful for interpreting predictions)
yawn_labels_actual = {v: k for k, v in train_yawn_generator.class_indices.items()}
print(f"Yawn class labels detected by generator: {yawn_labels_actual}")
# Use the generator's detected labels for robust mapping in subsequent prediction scripts
LABELS_FOR_PREDICTION = yawn_labels_actual


# --- 2. CNN Model Definition for Yawning ---
print("\nDefining Yawn CNN model...")

yawn_model = Sequential([
    # First Convolutional Block
    Conv2D(32, (3, 3), activation='relu', input_shape=(YAWN_IMG_HEIGHT, YAWN_IMG_WIDTH, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25), # Dropout to prevent overfitting

    # Second Convolutional Block
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Flatten the output for the Dense layers
    Flatten(),

    # Fully Connected Layers
    Dense(64, activation='relu'), # Simpler dense layer, can be adjusted
    Dropout(0.5), # Another dropout layer
    Dense(YAWN_NUM_CLASSES, activation='softmax') # Output layer with softmax for classification
])

# Compile the model
yawn_model.compile(optimizer='adam',
                   loss='categorical_crossentropy', # Use categorical_crossentropy for one-hot labels
                   metrics=['accuracy'])

yawn_model.summary()


# --- 3. Model Training for Yawning ---
print("\nTraining Yawn model...")

# Callbacks for early stopping and model checkpointing
early_stopping_yawn = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint_yawn = ModelCheckpoint(YAWN_MODEL_FILENAME, save_best_only=True, monitor='val_loss', mode='min')

history_yawn = yawn_model.fit(
    train_yawn_generator,
    steps_per_epoch=train_yawn_generator.samples // YAWN_BATCH_SIZE,
    epochs=YAWN_EPOCHS,
    validation_data=validation_yawn_generator,
    validation_steps=validation_yawn_generator.samples // YAWN_BATCH_SIZE,
    callbacks=[early_stopping_yawn, model_checkpoint_yawn]
)


print(f"Yawn model training complete. Best model saved to {YAWN_MODEL_FILENAME}")

# --- 4. Evaluate Model Metrics ---
print("\n--- Model Evaluation on Test Data ---")
loss, accuracy = yawn_model.evaluate(validation_yawn_generator, verbose=1)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# --- 4. Evaluate Model Metrics ---
print("\n--- Model Evaluation on Test Data ---")
loss, accuracy = yawn_model.evaluate(validation_yawn_generator, verbose=1)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")


# --- 5. Plotting Learning Curves ---
print("\nPlotting learning curves...")

# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st plot
plt.plot(history_yawn.history['accuracy'])
plt.plot(history_yawn.history['val_accuracy'])
plt.title('Yawn Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid(True)

# Plot training & validation loss values
plt.subplot(1, 2, 2) # 1 row, 2 columns, 2nd plot
plt.plot(history_yawn.history['loss'])
plt.plot(history_yawn.history['val_loss'])
plt.title('Yawn Model Loss')
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
validation_yawn_generator.reset()
Y_pred = yawn_model.predict(validation_yawn_generator, steps=validation_yawn_generator.samples // YAWN_BATCH_SIZE + 1)
y_pred_classes = np.argmax(Y_pred, axis=1)

# Get true labels for the validation set
y_true = validation_yawn_generator.classes[validation_yawn_generator.index_array]

# Get class names from the generator
class_names = list(validation_yawn_generator.class_indices.keys())

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Yawn Model Confusion Matrix')
plt.show()

print("Confusion matrix plot displayed.")
