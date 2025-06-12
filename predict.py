import cv2
import time
import os
import numpy as np
from tensorflow.keras.models import load_model

# Load model and face detector
model = load_model('./eye_status_cnn_model.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create save directory
save_dir = 'face_classified'
os.makedirs(save_dir, exist_ok=True)

# Start webcam
cap = cv2.VideoCapture(0)

print("Detecting face and classifying eyes every second. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Crop the face
        face_img = frame[y:y+h, x:x+w]

        # Preprocess for model
        resized = cv2.resize(face_img, (128, 128))
        norm = resized / 255.0
        input_img = np.expand_dims(norm, axis=0)

        # Save the cropped face image
        timestamp = int(time.time())
        filename = f"{save_dir}/{label}_{timestamp}.jpg"
        cv2.imwrite(filename, face_img)
        print(f"[{label}] Saved: {filename}")
        # Resize and preprocess
        resized = cv2.resize(img, (128, 128))
        normalized = resized / 255.0
        input_img = np.expand_dims(normalized, axis=0)

        # Predict
        pred = model.predict(input_img)[0][0]
        label = "Open" if pred > 0.5 else "Closed"
        # Predict
        # pred = model.predict(input_img)[0][0]
        # label = 'Open' if pred > 0.5 else 'Closed'



        print(f"Pred: {model.predict(input_img)}\n")
        print(f"predict: {model.predict(input_img)[0][0]}\n")
        print(f"Prediction: Eyes are {label}")

        # Draw rectangle and label
        color = (0,255,0) if label == 'Open' else (0,0,255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f'{label}', (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Show full frame
    cv2.imshow("Eye State Detection", frame)

    # Wait 1 second or exit on 'q'
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
