
import cv2
from pathlib import Path

# Project folders
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATASET_DIR = PROJECT_DIR / "dataset"

# Face detector XML file
CASCADE_PATH = (
    BASE_DIR / "data" / "haarcascade_frontalface_default.xml"
)

if not CASCADE_PATH.exists():
    print("ERROR: Face detector XML file is missing!")
    print("Expected file:", CASCADE_PATH)
    raise SystemExit

detector = cv2.CascadeClassifier(str(CASCADE_PATH))

if detector.empty():
    print("ERROR: Could not load the face detector.")
    raise SystemExit

# Get person's name
name = input("Enter your name: ").strip()

if (
    not name
    or name in {".", ".."}
    or any(c in name for c in '\\/:*?"<>|')
):
    print("Please enter a valid name.")
    raise SystemExit

# Create a folder for this person's images
save_dir = DATASET_DIR / name
save_dir.mkdir(parents=True, exist_ok=True)

# Open webcam using the default backend
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    camera.release()
    camera = cv2.VideoCapture(1)

if not camera.isOpened():
    print("ERROR: Camera could not open.")
    print("Close other camera apps and try again.")
    raise SystemExit

count = 0
target = 50

print("Look at the camera and move your head slightly.")
print("Press Q to stop early.")

try:
    while count < target:
        success, frame = camera.read()

        if not success or frame is None:
            print("Could not read a camera frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80)
        )

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (200, 200))

            count += 1
            image_path = save_dir / f"{count:03d}.jpg"
            cv2.imwrite(str(image_path), face)

            cv2.rectangle(
                frame, (x, y), (x + w, y + h), (0, 255, 0), 2
            )

            cv2.putText(
                frame,
                f"Saved: {count}/{target}",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            if count >= target:
                break

        cv2.imshow("Face Registration", frame)

        if cv2.waitKey(100) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()

print(f"Finished! Saved {count} images in: {save_dir}")

