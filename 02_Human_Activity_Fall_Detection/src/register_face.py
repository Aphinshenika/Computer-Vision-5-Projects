
import cv2
import os

name = input("Enter your name: ").strip()

if not name:
    print("Please enter a name.")
    raise SystemExit

save_dir = os.path.join("dataset", name)
os.makedirs(save_dir, exist_ok=True)

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

count = 0
print("Look at the camera. Press Q to stop.")

while count < 50:
    success, frame = camera.read()

    if not success:
        print("Could not read from camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
    )

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        count += 1
        cv2.imwrite(os.path.join(save_dir, f"{count:03d}.jpg"), face)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Saved: {count}/50", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Register Face", frame)

    if cv2.waitKey(100) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
print(f"Saved {count} face images in {save_dir}")
