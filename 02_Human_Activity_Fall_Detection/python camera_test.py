import cv2
import time

print("Testing available camera backends...")

backends = [
    ("Default", cv2.CAP_ANY),
    ("Media Foundation", cv2.CAP_MSMF),
    ("DirectShow", cv2.CAP_DSHOW),
]

camera_found = False

for backend_name, backend in backends:
    print(f"\nTesting {backend_name}...")

    camera = cv2.VideoCapture(0, backend)
    time.sleep(1)

    if not camera.isOpened():
        print(f"{backend_name}: Could not open camera.")
        camera.release()
        continue

    success, frame = camera.read()

    if success and frame is not None:
        print(f"{backend_name}: CAMERA WORKS!")

        cv2.imshow("Camera Test - Press Q to quit", frame)

        while True:
            success, frame = camera.read()

            if success and frame is not None:
                cv2.imshow("Camera Test - Press Q to quit", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

        camera_found = True
        camera.release()
        cv2.destroyAllWindows()
        break

    print(f"{backend_name}: Opened, but no frame received.")
    camera.release()

cv2.destroyAllWindows()

if camera_found:
    print("\nSUCCESS: Your camera is working!")
else:
    print("\nAll tested backends failed.")
    print("Check Windows camera permissions and close other camera apps.")