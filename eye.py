import os
import cv2
import mediapipe as mp

face_dir = "face"
eye_dir = "eye"
os.makedirs(eye_dir, exist_ok=True)

face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5
)

def invert_normalization(x, y, w, h):
    return int(x * w), int(y * h)

def fix_aspect_ratio(image, y, y1, required_ratio):
    new_h = int(image.shape[1] * required_ratio)
    diff_h = int((new_h - image.shape[0]) / 2)
    return max(0, y - diff_h), min(image.shape[0], y1 + diff_h)

def get_aspect_ratio(region):
    return float(region.shape[0]) / float(region.shape[1])

for filename in os.listdir(face_dir):
    filepath = os.path.join(face_dir, filename)
    
    if not (filename.endswith(".jpg") or filename.endswith(".png")):
        continue

    image = cv2.imread(filepath)
    if image is None:
        print(f"Failed to load image: {filepath}")
        continue
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        print(f"No face detected in {filename}.")
        continue

    landmarks = results.multi_face_landmarks[0].landmark
    image_width, image_height = image.shape[1], image.shape[0]

    up_left_x, up_left_y = invert_normalization(landmarks[71].x, landmarks[71].y, image_width, image_height)
    down_left_x, down_left_y = invert_normalization(landmarks[71].x, landmarks[123].y, image_width, image_height)
    up_right_x, up_right_y = invert_normalization(landmarks[301].x, landmarks[71].y, image_width, image_height)
    down_right_x, down_right_y = invert_normalization(landmarks[301].x, landmarks[123].y, image_width, image_height)

    width_eyes = (down_right_x - down_left_x)
    safe_increase = int(width_eyes * 0.2)
    down_right_x = min(image_width, down_right_x + safe_increase)
    down_left_x = max(0, down_left_x - safe_increase)

    eye_region = rgb_image[up_left_y:down_left_y, down_left_x:down_right_x, :]
    aspect_ratio = get_aspect_ratio(eye_region)

    if aspect_ratio <= 0.5:
        up_left_y, down_left_y = fix_aspect_ratio(eye_region, up_left_y, down_left_y, 0.5)
        eye_region = rgb_image[up_left_y:down_left_y, down_left_x:down_right_x, :]

    eye_image_resized = cv2.resize(cv2.cvtColor(eye_region, cv2.COLOR_RGB2BGR), (64, 64))
    eye_output_path = os.path.join(eye_dir, filename)
    cv2.imwrite(eye_output_path, eye_image_resized)
    print(f"Saved eye image to {eye_output_path}.")
