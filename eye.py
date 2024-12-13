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

    x_min = min(landmarks[33].x, landmarks[133].x, landmarks[362].x, landmarks[263].x)
    x_max = max(landmarks[33].x, landmarks[133].x, landmarks[362].x, landmarks[263].x)
    y_min = min(landmarks[159].y, landmarks[145].y, landmarks[386].y, landmarks[374].y)
    y_max = max(landmarks[159].y, landmarks[145].y, landmarks[386].y, landmarks[374].y)

    x_min, y_min = invert_normalization(x_min, y_min, image_width, image_height)
    x_max, y_max = invert_normalization(x_max, y_max, image_width, image_height)

    eye_region = rgb_image[y_min:y_max, x_min:x_max]

    if eye_region.size > 0:
        eye_bgr = cv2.cvtColor(eye_region, cv2.COLOR_RGB2BGR)
        eye_output_path = os.path.join(eye_dir, filename)
        cv2.imwrite(eye_output_path, eye_bgr)
        print(f"Saved eye image to {eye_output_path}.")
