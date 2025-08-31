import cv2
import os
import mediapipe as mp

import mediapipe as mp


DATA_DIR = "my_webcam_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


label_name = input("Enter label name (e.g., A, B, Hello): ").strip().upper()
label_dir = os.path.join(DATA_DIR, label_name)
if not os.path.exists(label_dir):
    os.makedirs(label_dir)


label_name = input("Enter label name (e.g., A, B, Hello): ").strip().upper()
label_dir = os.path.join(DATA_DIR, label_name)
if not os.path.exists(label_dir):
    os.makedirs(label_dir)

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

count = 0
print("Press 'q' to stop...")
count = 0
print("Press 'q' to stop...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

           
            h, w, _ = frame.shape
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, y_min = min(x, x_min), min(y, y_min)
                x_max, y_max = max(x, x_max), max(y, y_max)

           
            x_min, y_min = max(0, x_min-20), max(0, y_min-20)
            x_max, y_max = min(w, x_max+20), min(h, y_max+20)

            hand_img = frame[y_min:y_max, x_min:x_max]

            if hand_img.size > 0:
                img_path = os.path.join(label_dir, f"{count}.jpg")
                cv2.imwrite(img_path, hand_img)
                count += 1

    cv2.putText(frame, f"Saved: {count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Capture Hand Data", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

           
            h, w, _ = frame.shape
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, y_min = min(x, x_min), min(y, y_min)
                x_max, y_max = max(x, x_max), max(y, y_max)

           
            x_min, y_min = max(0, x_min-20), max(0, y_min-20)
            x_max, y_max = min(w, x_max+20), min(h, y_max+20)

            hand_img = frame[y_min:y_max, x_min:x_max]

            if hand_img.size > 0:
                img_path = os.path.join(label_dir, f"{count}.jpg")
                cv2.imwrite(img_path, hand_img)
                count += 1

    cv2.putText(frame, f"Saved: {count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Capture Hand Data", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Saved {count} images for label '{label_name}'")