import os
import cv2
import numpy as np
import dlib
from imutils import face_utils

face_dir = "output/face"
status_dir = "output/status"
os.makedirs(status_dir, exist_ok=True)

predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def compute(ptA, ptB):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(ptA - ptB)

def blinked(a, b, c, d, e, f):
    """Determine blink ratio based on eye landmarks."""
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    if ratio > 0.25:
        return 2  # Active
    elif 0.21 <= ratio <= 0.25:
        return 1  # Drowsy
    else:
        return 0  # Sleeping

for filename in os.listdir(face_dir):
    if not (filename.endswith(".jpg") or filename.endswith(".png")):
        continue

    filepath = os.path.join(face_dir, filename)
    image = cv2.imread(filepath)
    if image is None:
        print(f"Failed to load image: {filename}")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    if len(faces) == 0:
        print(f"No face detected in {filename}.")
        continue

    face = faces[0]
    landmarks = predictor(gray, face)
    landmarks = face_utils.shape_to_np(landmarks)

    left_blink = blinked(
        landmarks[36], landmarks[37], landmarks[38], 
        landmarks[41], landmarks[40], landmarks[39]
    )
    right_blink = blinked(
        landmarks[42], landmarks[43], landmarks[44], 
        landmarks[47], landmarks[46], landmarks[45]
    )

    if left_blink == 0 or right_blink == 0:
        status = "SLEEPING"
    elif left_blink == 1 or right_blink == 1:
        status = "DROWSY"
    else:
        status = "ACTIVE"

    status_file = os.path.join(status_dir, f"{os.path.splitext(filename)[0]}.txt")
    with open(status_file, "w") as f:
        f.write(status)

    print(f"Processed {filename}: {status}")
