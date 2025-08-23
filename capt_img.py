import cv2
import os

# Base folder where data will be stored
DATA_DIR = "my_webcam_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Default label (A)
current_label = "A"
label_dir = os.path.join(DATA_DIR, current_label)
os.makedirs(label_dir, exist_ok=True)

# Start webcam
cap = cv2.VideoCapture(0)

print("Press A-Z to change label/class.")
print("Press 's' to save image.")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip image for natural movement
    frame = cv2.flip(frame, 1)

    # Define green box
    x, y, w, h = 200, 100, 300, 300
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show current label on frame
    cv2.putText(
        frame,
        f"Label: {current_label}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

    cv2.imshow("Capture ASL Dataset", frame)

    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord("q"):
        break

    # Save image
    elif key == ord("s"):
        roi = frame[y : y + h, x : x + w]
        count = len(os.listdir(label_dir))
        img_name = os.path.join(label_dir, f"{count}.jpg")
        cv2.imwrite(img_name, roi)
        print(f"Saved {img_name}")

    # Change label when A-Z keys pressed
    elif 65 <= key <= 90:  # ASCII A-Z
        current_label = chr(key)
        label_dir = os.path.join(DATA_DIR, current_label)
        os.makedirs(label_dir, exist_ok=True)
        print(f"Switched to label: {current_label}")

cap.release()
cv2.destroyAllWindows()
