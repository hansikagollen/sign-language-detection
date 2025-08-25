import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from collections import deque

MODEL_PATH = "asl_mobilenet_model.h5"
model = load_model(MODEL_PATH)

IMG_SIZE = (224, 224)
DATA_DIR = "my_webcam_data"  
CONFIDENCE_THRESHOLD = 0.6   
FRAME_SMOOTH = 5             

class_labels = sorted(os.listdir(DATA_DIR))  

def predict_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found:", image_path)
        return
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    
    preds = model.predict(img, verbose=0)
    confidence = np.max(preds)
    label = class_labels[np.argmax(preds)] if confidence >= CONFIDENCE_THRESHOLD else "Unknown"
    print(f"Predicted: {label} ({confidence*100:.2f}%)")
    return label, confidence

def predict_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return
    
    print("Press 'q' to quit")
    frame_count = 0
    prediction_queue = deque(maxlen=FRAME_SMOOTH)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        x, y, w, h = 200, 100, 300, 300
        roi = frame[y:y+h, x:x+w]

        label = "..."
        confidence = 0.0

        if roi.size != 0:
            frame_count += 1
            if frame_count % 3 == 0:  
                img = cv2.resize(roi, IMG_SIZE)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype("float32") / 255.0
                img = np.expand_dims(img, axis=0)
                
                preds = model.predict(img, verbose=0)
                prediction_queue.append(preds[0])

            if prediction_queue:
                avg_preds = np.mean(prediction_queue, axis=0)
                confidence = np.max(avg_preds)
                label = class_labels[np.argmax(avg_preds)] if confidence >= CONFIDENCE_THRESHOLD else "Unknown"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label}: {confidence*100:.2f}%" if label != "..." else "...",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        cv2.imshow("ASL Prediction", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    mode = input("Enter 'i' for image or 'w' for webcam: ").strip().lower()
    if mode == "i":
        path = input("Enter image path: ").strip()
        predict_image(path)
    elif mode == "w":
        predict_webcam()
    else:
        print("Invalid option")
