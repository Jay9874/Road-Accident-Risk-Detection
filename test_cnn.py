import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("./eye_status_cnn_model.h5")

# Load the image
image_path = "face_classified/Closed_1749574500.jpg"
img = cv2.imread(image_path)

# Optional: Show original image
cv2.imshow("Input Image", img)
cv2.waitKey(1000)
cv2.destroyAllWindows()

# Resize and preprocess
resized = cv2.resize(img, (128, 128))
normalized = resized / 255.0
input_img = np.expand_dims(normalized, axis=0)

# Predict
pred = model.predict(input_img)[0][0]
label = "Open" if pred > 0.5 else "Closed"

print(f"Pred: {model.predict(input_img)}\n")
print(f"predict: {model.predict(input_img)[0][0]}\n")
print(f"Prediction: Eyes are {label}")
