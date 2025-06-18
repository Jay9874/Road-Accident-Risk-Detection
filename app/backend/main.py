# backend/main.py
from dotenv import load_dotenv
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
import sys
import dlib
from imutils import face_utils
from twilio.rest import Client
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import base64

load_dotenv()

# --- Configuration (same as your original code) ---
YAWN_MODEL_FILENAME = "yawn_cnn_combine_dataset.keras"
YAWN_IMG_HEIGHT = 64
YAWN_IMG_WIDTH = 64
YAWN_LABELS = {0: "Yawn", 1: "No Yawn"}

EYE_MODEL_FILENAME = "eye_cnn.h5"
EYE_IMG_HEIGHT = 64
EYE_IMG_WIDTH = 64
EYE_LABELS = {0: "closed_eye", 1: "open_eye"}

SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

(M_START, M_END) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]
(L_EYE_START, L_EYE_END) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(R_EYE_START, R_EYE_END) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
RECIPIENT_PHONE_NUMBER = os.getenv("RECIPIENT_PHONE_NUMBER")

TARGET_DROWSINESS_DURATION_SECONDS = 5
drowsy_frame_count = 0
alert_sent = False
last_alert_time = 0 # To prevent immediate re-alerts

SIMULATED_LATITUDE = 34.0522
SIMULATED_LONGITUDE = -118.2437

# Global variables for models and detector/predictor
yawn_model = None
eye_model = None
detector = None
predictor = None
twilio_client = None

# This will store connection-specific drowsiness state if multiple clients connect
# For a single user, you can keep it simple. For multiple, you'd need a dict
# of client_id: drowsy_frame_count and client_id: alert_sent
# For simplicity, we'll assume one client for now or reset for each new connection
connection_drowsiness_state = {}


# Function to send SMS alert
def send_drowsiness_alert(latitude, longitude):
    global alert_sent, last_alert_time
    if twilio_client is None:
        print("Twilio client not initialized. Cannot send SMS.")
        return False

    # Add a cooldown period for sending alerts (e.g., 60 seconds)
    if time.time() - last_alert_time < 60:
        print("Alert cooldown active. Not sending SMS.")
        return False

    message_body = (
        f"URGENT: Drowsiness detected!\n"
        f"Location (Simulated): Lat {latitude}, Lon {longitude}\n"
        f"This is a test alert. Do NOT reply if you are receiving this unexpectedly."
    )
    try:
        message = twilio_client.messages.create(
            to=RECIPIENT_PHONE_NUMBER, from_=TWILIO_PHONE_NUMBER, body=message_body
        )
        print(f"SMS alert sent! Message SID: {message.sid}")
        alert_sent = True
        last_alert_time = time.time()
        return True
    except Exception as e:
        print(f"Error sending SMS: {e}")
        print(
            "Please check your Twilio phone number, recipient number (must be verified for trial accounts), and account balance."
        )
        return False

# --- Drowsiness Detection Function ---
def process_frame_for_drowsiness(frame_rgb, client_id):
    global drowsy_frame_count, alert_sent

    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR) # Convert to BGR for OpenCV processing

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    results = {
        "status": "Alert",
        "status_color": (0, 255, 0), # Green
        "yawn_detected": False,
        "eyes_closed": False,
        "face_detected": False,
        "face_boxes": [],
        "eye_boxes": [],
        "mouth_boxes": [],
        "drowsy_frame_count": connection_drowsiness_state.get(client_id, {}).get("drowsy_count", 0),
        "alert_triggered": False
    }

    yawn_detected_this_frame = False
    eyes_closed_this_frame = False

    if len(rects) > 0:
        results["face_detected"] = True
        for rect in rects:
            x_face, y_face, w_face, h_face = face_utils.rect_to_bb(rect)
            results["face_boxes"].append([x_face, y_face, x_face + w_face, y_face + h_face])

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # --- Yawning Detection Logic ---
            mouth_points = shape[M_START:M_END]
            (mx, my, mw, mh) = cv2.boundingRect(mouth_points)
            mouth_buffer_x = int(mw * 0.25)
            mouth_buffer_y = int(mh * 0.3)
            mx_start = max(0, mx - mouth_buffer_x)
            my_start = max(0, my - mouth_buffer_y)
            mx_end = min(gray.shape[1], mx + mw + mouth_buffer_x)
            my_end = min(gray.shape[0], my + mh + mouth_buffer_y)

            if my_start < my_end and mx_start < mx_end:
                mouth_img_gray_cropped = gray[my_start:my_end, mx_start:mx_end]
                if mouth_img_gray_cropped.shape[0] > 0 and mouth_img_gray_cropped.shape[1] > 0:
                    input_yawn_model = cv2.resize(mouth_img_gray_cropped, (YAWN_IMG_WIDTH, YAWN_IMG_HEIGHT))
                    input_yawn_model = input_yawn_model / 255.0
                    input_yawn_model = np.expand_dims(input_yawn_model, axis=-1)
                    input_yawn_model = np.expand_dims(input_yawn_model, axis=0)
                    prediction_yawn = yawn_model.predict(input_yawn_model, verbose=0)
                    predicted_yawn_class_index = np.argmax(prediction_yawn)
                    predicted_yawn_label = YAWN_LABELS[predicted_yawn_class_index]
                    yawn_confidence = prediction_yawn[0][predicted_yawn_class_index] * 100

                    if predicted_yawn_label == "Yawn":
                        yawn_detected_this_frame = True
                        results["mouth_boxes"].append([mx_start, my_start, mx_end, my_end, "Yawn", f"{yawn_confidence:.1f}%"])
                    else:
                        results["mouth_boxes"].append([mx_start, my_start, mx_end, my_end, "No Yawn", f"{yawn_confidence:.1f}%"])


            # --- Eye Status Detection Logic ---
            left_eye_points = shape[L_EYE_START:L_EYE_END]
            right_eye_points = shape[R_EYE_START:R_EYE_END]

            for i, eye_points in enumerate([left_eye_points, right_eye_points]):
                (ex, ey, ew, eh) = cv2.boundingRect(eye_points)
                eye_buffer_x = int(ew * 0.15)
                eye_buffer_y = int(eh * 0.15)
                ex_start = max(0, ex - eye_buffer_x)
                ey_start = max(0, ey - eye_buffer_y)
                ex_end = min(gray.shape[1], ex + ew + eye_buffer_x)
                ey_end = min(gray.shape[0], ey + eh + eye_buffer_y)

                if ey_start < ey_end and ex_start < ex_end:
                    eye_img_gray_cropped = gray[ey_start:ey_end, ex_start:ex_end]
                    if eye_img_gray_cropped.shape[0] > 0 and eye_img_gray_cropped.shape[1] > 0:
                        input_eye_model = cv2.resize(eye_img_gray_cropped, (EYE_IMG_WIDTH, EYE_IMG_HEIGHT))
                        input_eye_model = input_eye_model / 255.0
                        input_eye_model = np.expand_dims(input_eye_model, axis=-1)
                        input_eye_model = np.expand_dims(input_eye_model, axis=0)
                        prediction_eye = eye_model.predict(input_eye_model, verbose=0)
                        predicted_eye_class_index = np.argmax(prediction_eye)
                        predicted_eye_label = EYE_LABELS[predicted_eye_class_index]
                        eye_confidence = prediction_eye[0][predicted_eye_class_index] * 100

                        if predicted_eye_label == "closed_eye":
                            eyes_closed_this_frame = True
                            results["eye_boxes"].append([ex_start, ey_start, ex_end, ey_end, "Closed", f"{eye_confidence:.1f}%"])
                        else:
                            results["eye_boxes"].append([ex_start, ey_start, ex_end, ey_end, "Open", f"{eye_confidence:.1f}%"])

    # Update connection-specific drowsiness state
    current_drowsy_count = connection_drowsiness_state.get(client_id, {}).get("drowsy_count", 0)
    current_alert_sent_status = connection_drowsiness_state.get(client_id, {}).get("alert_sent", False)


    if results["face_detected"]:
        if eyes_closed_this_frame and yawn_detected_this_frame:
            results["status"] = "Drowsy (Eyes Closed & Yawning!)"
            results["status_color"] = (0, 0, 255) # Red
            current_drowsy_count += 1
        elif eyes_closed_this_frame:
            results["status"] = "Eyes Closed (Possible Drowsiness)"
            results["status_color"] = (0, 165, 255) # Orange
            current_drowsy_count = 0
        elif yawn_detected_this_frame:
            results["status"] = "Yawning (Possible Drowsiness)"
            results["status_color"] = (0, 165, 255) # Orange
            current_drowsy_count = 0
        else:
            current_drowsy_count = 0
            current_alert_sent_status = False # Allow new alerts once user is alert again
    else:
        current_drowsy_count = 0
        current_alert_sent_status = False

    connection_drowsiness_state[client_id] = {
        "drowsy_count": current_drowsy_count,
        "alert_sent": current_alert_sent_status
    }
    results["drowsy_frame_count"] = current_drowsy_count
    results["yawn_detected"] = yawn_detected_this_frame
    results["eyes_closed"] = eyes_closed_this_frame

    # Alert Trigger (using the global logic for now, but better to be per-client)
    # Note: The Twilio SMS alert should still use your global 'alert_sent' and 'last_alert_time'
    # to prevent spamming based on multiple clients or rapid re-triggers from one client.
    if current_drowsy_count >= 30 and not alert_sent: # 30 frames is approx 1 second at 30fps
         print(f"\n--- DROWSINESS ALERT TRIGGERED! ({current_drowsy_count} consecutive frames) ---")
         if send_drowsiness_alert(SIMULATED_LATITUDE, SIMULATED_LONGITUDE):
            results["alert_triggered"] = True
            # The global alert_sent flag is set inside send_drowsiness_alert
         print("----------------------------------------------------------------")


    return results


app = FastAPI()

# Pre-load models when the FastAPI application starts
@app.on_event("startup")
async def startup_event():
    global yawn_model, eye_model, detector, predictor, twilio_client

    print("Loading models and Dlib resources...")
    try:
        yawn_model = load_model(YAWN_MODEL_FILENAME)
        print("Yawn model loaded successfully.")
    except Exception as e:
        print(f"Error loading yawn model: {e}. Exiting.")
        sys.exit(1)

    try:
        eye_model = load_model(EYE_MODEL_FILENAME)
        print("Eye model loaded successfully.")
    except Exception as e:
        print(f"Error loading eye model: {e}. Exiting.")
        sys.exit(1)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    if not os.path.exists(SHAPE_PREDICTOR_PATH):
        print(f"Error: Dlib shape predictor not found at {SHAPE_PREDICTOR_PATH}. Exiting.")
        sys.exit(1)
    print("Dlib models loaded successfully.")

    try:
        twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        print("Twilio client initialized.")
    except Exception as e:
        print(f"Error initializing Twilio client: {e}. Twilio alerts will be disabled.")
        twilio_client = None


@app.websocket("/ws/drowsiness")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = websocket.client.host + ":" + str(websocket.client.port)
    print(f"Client {client_id} connected.")
    # Initialize drowsiness state for this client
    connection_drowsiness_state[client_id] = {"drowsy_count": 0, "alert_sent": False}

    try:
        while True:
            # Receive base64 encoded image string from frontend
            data = await websocket.receive_text()
            # Decode base64 to bytes
            image_bytes = base64.b64decode(data)
            # Convert bytes to numpy array
            np_arr = np.frombuffer(image_bytes, np.uint8)
            # Decode numpy array to OpenCV image
            frame_rgb = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB) # Ensure it's RGB if frontend expects it, or just use BGR throughout
           
           # show the image got from frontend
            cv2.imshow(f"Frontend image", frame_rgb)

            if frame_rgb is None:
                print("Could not decode image from client.")
                continue

            # Process the frame for drowsiness
            results = process_frame_for_drowsiness(frame_rgb, client_id)

            # Send results back to frontend
            await websocket.send_json(results)

    except WebSocketDisconnect:
        print(f"Client {client_id} disconnected.")
        # Clean up client's drowsiness state
        if client_id in connection_drowsiness_state:
            del connection_drowsiness_state[client_id]
    except Exception as e:
        print(f"WebSocket error for client {client_id}: {e}")
    finally:
        # Important: Reset global alert_sent and drowsy_frame_count if this is the only client,
        # or if you are managing them per-client globally.
        # For this simplified example, we'll keep the global alert_sent behavior.
        pass