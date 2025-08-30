import os
import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model
import mediapipe as mp


# -------------------- Suppress TF and MP logs --------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs

# -------------------- Config --------------------
MODEL_PATH = "asl_model2.h5"
IMG_SIZE = (224, 224)
DATA_DIR = "my_webcam_data"
CONFIDENCE_THRESHOLD = 0.6
FRAME_SMOOTH = 5  # Number of frames to average predictions

# -------------------- Load model --------------------
model = load_model(MODEL_PATH)
class_labels = sorted(os.listdir(DATA_DIR))

# -------------------- MediaPipe setup --------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -------------------- Webcam & prediction --------------------
def predict_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    prediction_queue = deque(maxlen=FRAME_SMOOTH)
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        label, confidence = "Unknown", 0.0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Compute bounding box
                h, w, _ = frame.shape
                x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
                y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
                x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
                y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)
                
                margin = 20
                x_min, y_min = max(0, x_min - margin), max(0, y_min - margin)
                x_max, y_max = min(w, x_max + margin), min(h, y_max + margin)

                roi = frame[y_min:y_max, x_min:x_max]
                if roi.size != 0:
                    img = cv2.resize(roi, IMG_SIZE)
                    img = img.astype("float32") / 255.0
                    img = np.expand_dims(img, axis=0)

                    preds = model.predict(img, verbose=0)[0]
                    prediction_queue.append(preds)

                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Compute average prediction over frames
        if prediction_queue:
            avg_preds = np.mean(prediction_queue, axis=0)
            confidence = np.max(avg_preds)
            label = class_labels[np.argmax(avg_preds)] if confidence >= CONFIDENCE_THRESHOLD else "Unknown"

        # Display label and confidence
        cv2.putText(
            frame,
            f"{label}: {confidence*100:.2f}%",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        cv2.imshow("ASL Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    predict_webcam()
