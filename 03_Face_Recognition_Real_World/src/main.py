import cv2

def main():
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    camera = cv2.VideoCapture(0, cv2.CAP_MSMF)

    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Camera could not open. Close other camera apps and try again.")
        return

    print("Face detection started! Press Q to quit.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read camera frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Face Detected", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, f"Faces: {len(faces)}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Project 3 - Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
