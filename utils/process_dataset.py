import cv2
import os
import dlib
import numpy as np
from imutils import face_utils
from tqdm import tqdm # For progress bar
import random # For splitting data

# --- Configuration ---
# Path to your original downloaded and extracted 'dataset_new' (with full driver/car images)
ORIGINAL_DATA_DIR = 'dataset_new'

# Path to your NEW pre-cropped dataset (e.g., 'yawn_data' folder from the screenshot)
NEW_PRE_CROPPED_DATA_DIR = 'yawn_data' # <--- UPDATE THIS TO YOUR NEW DATASET'S DIRECTORY

# Directory where ALL processed (mouth-cropped) images will be saved
PROCESSED_DATA_DIR = 'processed_yawn_dataset'

# Dlib's pre-trained facial landmark predictor model path
SHAPE_PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'

# Target dimensions for the cropped mouth images (must match your CNN model's input)
TARGET_IMG_HEIGHT = 64
TARGET_IMG_WIDTH = 64

# Indices for mouth landmarks (from dlib's 68-point model)
(M_START, M_END) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

# Split ratio for the NEW_PRE_CROPPED_DATA_DIR (e.g., 0.2 means 20% for test, 80% for train)
VALIDATION_SPLIT_NEW_DATA = 0.2

# --- Initialize Dlib (only if ORIGINAL_DATA_DIR processing is enabled) ---
detector = None
predictor = None

# We only load Dlib if we are actually processing the original dataset
# that requires face/mouth detection.
if ORIGINAL_DATA_DIR and os.path.exists(ORIGINAL_DATA_DIR):
    print("Loading Dlib's face detector and facial landmark predictor for original dataset processing...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

    if not os.path.exists(SHAPE_PREDICTOR_PATH):
        print(f"Error: Dlib shape predictor not found at {SHAPE_PREDICTOR_PATH}. Please download and place it.")
        exit()
    print("Dlib models loaded successfully.")
else:
    print(f"Skipping Dlib loading as '{ORIGINAL_DATA_DIR}' not found or not specified for processing.")


# --- Data Processing Function for Dlib Cropping (Original Dataset) ---
def process_original_dataset_with_dlib(input_folder, output_folder, label_class):
    """
    Processes images from an input folder, crops mouth regions using Dlib,
    resizes them, and saves them to an output folder. Used for the original dataset.
    """
    if detector is None or predictor is None:
        print(f"Dlib not loaded. Skipping Dlib cropping for '{input_folder}'.")
        return

    os.makedirs(output_folder, exist_ok=True)
    
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    print(f"\nProcessing images for class: '{label_class}' in '{input_folder}' (Dlib cropping)...")
    processed_count = 0
    skipped_count = 0

    for image_name in tqdm(image_files, desc=f"Cropping {label_class}"):
        image_path = os.path.join(input_folder, image_name)
        frame = cv2.imread(image_path)

        if frame is None:
            skipped_count += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0) # Detect faces

        if len(rects) == 0:
            skipped_count += 1
            continue

        rect = rects[0] # Assuming one face per image
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        mouth_points = shape[M_START:M_END]
        (mx, my, mw, mh) = cv2.boundingRect(mouth_points)

        buffer_x = int(mw * 0.25)
        buffer_y = int(mh * 0.3)
        mx_start = max(0, mx - buffer_x)
        my_start = max(0, my - buffer_y)
        mx_end = min(gray.shape[1], mx + mw + buffer_x)
        my_end = min(gray.shape[0], my + mh + buffer_y)

        cropped_mouth = gray[my_start:my_end, mx_start:mx_end]

        if cropped_mouth.shape[0] == 0 or cropped_mouth.shape[1] == 0:
            skipped_count += 1
            continue

        resized_mouth = cv2.resize(cropped_mouth, (TARGET_IMG_WIDTH, TARGET_IMG_HEIGHT))
        output_image_path = os.path.join(output_folder, image_name)
        cv2.imwrite(output_image_path, resized_mouth)
        processed_count += 1
            
    print(f"Finished processing '{label_class}'. Processed: {processed_count}, Skipped: {skipped_count}")

# --- Data Processing Function for Pre-Cropped Images (New Dataset with Train/Test Split) ---
def process_and_split_pre_cropped_dataset(input_folder, base_output_dir, label_class, split_ratio=0.2):
    """
    Processes already cropped images, resizes them, converts to grayscale,
    and then splits them into train/test subfolders within the base_output_dir.
    """
    output_train_folder = os.path.join(base_output_dir, 'train', label_class)
    output_test_folder = os.path.join(base_output_dir, 'test', label_class)
    
    os.makedirs(output_train_folder, exist_ok=True)
    os.makedirs(output_test_folder, exist_ok=True)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    random.shuffle(image_files) # Shuffle to ensure random split

    num_test_images = int(len(image_files) * split_ratio)
    test_images = image_files[:num_test_images]
    train_images = image_files[num_test_images:]

    print(f"\nProcessing images for class: '{label_class}' in '{input_folder}' (Resizing & Splitting)...")
    print(f"  Total images: {len(image_files)}, Train: {len(train_images)}, Test: {len(test_images)}")

    processed_count = 0
    skipped_count = 0

    # Process training images
    for image_name in tqdm(train_images, desc=f"Resizing & Saving Train {label_class}"):
        image_path = os.path.join(input_folder, image_name)
        img = cv2.imread(image_path)
        if img is None:
            skipped_count += 1
            continue
        if len(img.shape) == 3: # Convert to grayscale if color
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized_img = cv2.resize(img, (TARGET_IMG_WIDTH, TARGET_IMG_HEIGHT))
        output_image_path = os.path.join(output_train_folder, image_name)
        cv2.imwrite(output_image_path, resized_img)
        processed_count += 1
    
    # Process testing images
    for image_name in tqdm(test_images, desc=f"Resizing & Saving Test {label_class}"):
        image_path = os.path.join(input_folder, image_name)
        img = cv2.imread(image_path)
        if img is None:
            skipped_count += 1
            continue
        if len(img.shape) == 3: # Convert to grayscale if color
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized_img = cv2.resize(img, (TARGET_IMG_WIDTH, TARGET_IMG_HEIGHT))
        output_image_path = os.path.join(output_test_folder, image_name)
        cv2.imwrite(output_image_path, resized_img)
        processed_count += 1

    print(f"Finished processing '{label_class}'. Processed: {processed_count}, Skipped: {skipped_count}")


# --- Main Processing Logic ---
print("\nStarting dataset preprocessing...")
print(f"Processed dataset will be saved to: {PROCESSED_DATA_DIR}")

# Step 1: Process the ORIGINAL_DATA_DIR (if it exists) using Dlib for cropping
if os.path.exists(ORIGINAL_DATA_DIR):
    print(f"\n--- Processing Original Dataset '{ORIGINAL_DATA_DIR}' (Dlib Cropping) ---")
    # Process Training Data (Original)
    process_original_dataset_with_dlib(
        os.path.join(ORIGINAL_DATA_DIR, 'train', 'Yawn'),
        os.path.join(PROCESSED_DATA_DIR, 'train', 'Yawn'),
        'Yawn_Original'
    )
    process_original_dataset_with_dlib(
        os.path.join(ORIGINAL_DATA_DIR, 'train', 'No Yawn'),
        os.path.join(PROCESSED_DATA_DIR, 'train', 'No Yawn'),
        'No_Yawn_Original'
    )
    # Process Testing Data (Original)
    process_original_dataset_with_dlib(
        os.path.join(ORIGINAL_DATA_DIR, 'test', 'Yawn'),
        os.path.join(PROCESSED_DATA_DIR, 'test', 'Yawn'),
        'Yawn_Original_Test'
    )
    process_original_dataset_with_dlib(
        os.path.join(ORIGINAL_DATA_DIR, 'test', 'No Yawn'),
        os.path.join(PROCESSED_DATA_DIR, 'test', 'No Yawn'),
        'No_Yawn_Original_Test'
    )
else:
    print(f"Original dataset directory '{ORIGINAL_DATA_DIR}' not found. Skipping Dlib cropping for this dataset.")


# Step 2: Process the NEW_PRE_CROPPED_DATA_DIR (if it exists)
if os.path.exists(NEW_PRE_CROPPED_DATA_DIR):
    print(f"\n--- Processing New Pre-cropped Dataset '{NEW_PRE_CROPPED_DATA_DIR}' (Resizing & Splitting) ---")
    
    # Process 'yawn' images from the new pre-cropped dataset and split into train/test
    process_and_split_pre_cropped_dataset(
        os.path.join(NEW_PRE_CROPPED_DATA_DIR, 'yawn'),
        PROCESSED_DATA_DIR, # Base output dir
        'Yawn', # Class name (will map to Yawn subfolder in train/test)
        split_ratio=VALIDATION_SPLIT_NEW_DATA
    )

    # Process 'no yawn' images from the new pre-cropped dataset and split into train/test
    process_and_split_pre_cropped_dataset(
        os.path.join(NEW_PRE_CROPPED_DATA_DIR, 'no yawn'),
        PROCESSED_DATA_DIR, # Base output dir
        'No Yawn', # Class name (will map to No Yawn subfolder in train/test)
        split_ratio=VALIDATION_SPLIT_NEW_DATA
    )
else:
    print(f"New pre-cropped dataset directory '{NEW_PRE_CROPPED_DATA_DIR}' not found. Skipping processing for this dataset.")


print(f"\nCombined dataset preprocessing complete. All processed images saved to: {PROCESSED_DATA_DIR}")
print("\n--- NEXT IMPORTANT STEP ---")
print(f"1. Open your 'yawn-detection-standalone-app' (the training script).")
print(f"2. Change the 'YAWN_DATA_DIR' variable to: '{PROCESSED_DATA_DIR}'")
print(f"3. Run the 'yawn-detection-standalone-app' script again to retrain your model with the new, combined, and consistently-processed mouth images.")
print("This will create a model that is much better aligned with the input it receives during live detection.")
