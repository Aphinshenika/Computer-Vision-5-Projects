import cv2
import time
from ultralytics import YOLO


def main():
    print("Loading YOLO model...")

    # Load the lightweight YOLO model.
    # The first run may download yolov8n.pt automatically.
    model = YOLO("yolov8n.pt")

    print("YOLO model loaded successfully!")
    print("Opening webcam...")

    # Try different Windows camera backends.
    camera = None

    backends = [
        ("Default", cv2.CAP_ANY),
        ("Media Foundation", cv2.CAP_MSMF),
        ("DirectShow", cv2.CAP_DSHOW),
    ]

    for backend_name, backend in backends:
        print(f"Trying {backend_name}...")

        candidate = cv2.VideoCapture(0, backend)
        time.sleep(1)

        if candidate.isOpened():
            success, frame = candidate.read()

            if success and frame is not None:
                camera = candidate
                print(f"Camera opened using {backend_name}!")
                break

        candidate.release()

    if camera is None:
        print("ERROR: Could not open the webcam.")
        print("Close other camera applications and check permissions.")
        return

    print("\nObject detection is running!")
    print("Press Q to quit.")

    previous_time = time.time()

    try:
        while True:
            success, frame = camera.read()

            if not success or frame is None:
                print("Could not read a camera frame.")
                break

            # Detect common objects using YOLO.
            results = model.predict(
                source=frame,
                conf=0.4,
                imgsz=640,
                verbose=False
            )

            object_count = 0

            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    label = model.names[class_id]

                    x1, y1, x2, y2 = map(
                        int, box.xyxy[0].tolist()
                    )

                    object_count += 1

                    # Draw the detected object's bounding box.
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    text = f"{label} {confidence:.2f}"

                    cv2.putText(
                        frame,
                        text,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

                    # Estimate the object's center in image pixels.
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    cv2.circle(
                        frame,
                        (center_x, center_y),
                        5,
                        (0, 0, 255),
                        -1
                    )

                    cv2.putText(
                        frame,
                        f"Center: ({center_x}, {center_y})",
                        (x1, min(y2 + 22, frame.shape[0] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 0),
                        1
                    )

            # Calculate frames per second.
            current_time = time.time()
            elapsed = current_time - previous_time
            fps = 1.0 / elapsed if elapsed > 0 else 0
            previous_time = current_time

            cv2.putText(
                frame,
                f"Objects detected: {object_count}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.imshow(
                "Project 5 - Vision-Based Object Detection",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Camera closed. Program finished.")


if __name__ == "__main__":
    main()