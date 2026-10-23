import cv2
from ultralytics import YOLO
import time

# ==========================================
# 1. LOAD YOLO MODEL
# ==========================================
print("Loading YOLO model...")
model = YOLO("yolov8n.pt")
print("YOLO model loaded successfully!")

# ==========================================
# 2. OPEN WEBCAM
# ==========================================
# Try camera backends in order.
backends = [
    ("Default", cv2.CAP_ANY),
    ("Media Foundation", cv2.CAP_MSMF),
    ("DirectShow", cv2.CAP_DSHOW),
]

camera = None
working_backend = None

for backend_name, backend in backends:
    print(f"Trying camera backend: {backend_name}")

    test_camera = cv2.VideoCapture(0, backend)
    time.sleep(1)

    if test_camera.isOpened():
        success, frame = test_camera.read()

        if success and frame is not None:
            camera = test_camera
            working_backend = backend_name
            print(f"Camera works with {working_backend}!")
            break

    test_camera.release()

if camera is None:
    print("ERROR: No camera backend could provide a video frame.")
    print("Check Windows camera permissions and close other camera apps.")
    raise SystemExit

print("Press R to set the signal to RED.")
print("Press G to set the signal to GREEN.")
print("Press Q to quit.")

# ==========================================
# 3. INITIAL SETTINGS
# ==========================================
signal = "RED"

vehicle_classes = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}

previous_y = {}
reported_violations = set()

# Read frame dimensions.
success, frame = camera.read()

if not success or frame is None:
    print("ERROR: Camera opened but cannot provide frames.")
    camera.release()
    raise SystemExit

height, width = frame.shape[:2]
stop_line_y = int(height * 0.65)

# ==========================================
# 4. MAIN DETECTION LOOP
# ==========================================
try:
    while True:
        success, frame = camera.read()

        if not success or frame is None:
            print("Unable to read camera frame.")
            break

        # Detect and track objects.
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.4,
            verbose=False,
        )

        vehicle_count = 0
        current_time = time.strftime("%H:%M:%S")

        # ======================================
        # 5. PROCESS VEHICLES
        # ======================================
        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0])

                if class_id not in vehicle_classes:
                    continue

                vehicle_count += 1

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist(),
                )

                center_y = (y1 + y2) // 2

                track_id = None

                if box.id is not None:
                    track_id = int(box.id[0])

                vehicle_name = vehicle_classes[class_id]
                box_color = (0, 255, 0)
                label = vehicle_name

                # ==================================
                # 6. CHECK VIRTUAL STOP-LINE CROSSING
                # ==================================
                if track_id is not None:
                    if track_id in previous_y:
                        old_y = previous_y[track_id]

                        crossed_line = (
                            old_y < stop_line_y
                            and center_y >= stop_line_y
                        )

                        if (
                            crossed_line
                            and signal == "RED"
                            and track_id not in reported_violations
                        ):
                            reported_violations.add(track_id)

                            print(
                                f"TRAFFIC VIOLATION: {vehicle_name} "
                                f"ID {track_id} at {current_time}"
                            )

                    previous_y[track_id] = center_y

                    if track_id in reported_violations:
                        box_color = (0, 0, 255)
                        label = (
                            f"{vehicle_name} "
                            f"ID:{track_id} VIOLATION"
                        )
                    else:
                        label = f"{vehicle_name} ID:{track_id}"

                # Draw bounding box.
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    2,
                )

                # Draw label.
                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    box_color,
                    2,
                )

        # ======================================
        # 7. DRAW STOP LINE
        # ======================================
        cv2.line(
            frame,
            (0, stop_line_y),
            (width, stop_line_y),
            (0, 0, 255),
            3,
        )

        cv2.putText(
            frame,
            "VIRTUAL STOP LINE",
            (20, stop_line_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        # ======================================
        # 8. DRAW SIMULATED TRAFFIC SIGNAL
        # ======================================
        signal_color = (
            (0, 0, 255) if signal == "RED"
            else (0, 255, 0)
        )

        cv2.rectangle(
            frame,
            (15, 15),
            (300, 115),
            (40, 40, 40),
            -1,
        )

        cv2.circle(
            frame,
            (50, 55),
            16,
            signal_color,
            -1,
        )

        cv2.putText(
            frame,
            f"SIGNAL: {signal}",
            (80, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            signal_color,
            2,
        )

        # ======================================
        # 9. DISPLAY COUNTERS AND INSTRUCTIONS
        # ======================================
        cv2.putText(
            frame,
            f"Vehicles detected: {vehicle_count}",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Violations detected: {len(reported_violations)}",
            (20, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        cv2.putText(
            frame,
            "R: RED | G: GREEN | Q: QUIT",
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # Display the camera frame.
        cv2.imshow(
            "Intelligent Traffic Violation Detection",
            frame,
        )

        # ======================================
        # 10. KEYBOARD CONTROLS
        # ======================================
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            signal = "RED"
            print("Traffic signal changed to RED.")

        elif key == ord("g"):
            signal = "GREEN"
            print("Traffic signal changed to GREEN.")

        elif key == ord("q"):
            print("Closing application...")
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("Application closed.")
