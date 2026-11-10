import cv2
import os

def register_user():

    # Ask for user's name
    name = input("Enter your name: ").strip()

    if not name:
        print("Name cannot be empty!")
        return

    # Create user's database folder
    folder = os.path.join("database", name)
    os.makedirs(folder, exist_ok=True)

    # Initialize webcam
    camera = cv2.VideoCapture(0)

    # Load Haar Cascade face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    if face_cascade.empty():
        print("Error: Face detection model could not be loaded!")
        camera.release()
        return

    if not camera.isOpened():
        print("Error: Camera could not be opened!")
        return

    print("\nFace Registration Started")
    print("-------------------------")
    print("Press SPACE to capture your face.")
    print("Press Q to cancel.")

    while True:

        # Capture camera frame
        ret, frame = camera.read()

        if not ret:
            print("Failed to capture image!")
            break

        # Create grayscale image for detection
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        # Display detection rectangles
        for (x, y, w, h) in faces:

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Face Detected",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        # Display detection status
        if len(faces) == 0:
            status = "No Face Detected"
        elif len(faces) == 1:
            status = "One Face Detected"
        else:
            status = "Multiple Faces Detected"

        cv2.putText(
            frame,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Display camera window
        cv2.imshow("Face Registration", frame)

        # Read keyboard input
        key = cv2.waitKey(1) & 0xFF

        # Capture face when SPACE is pressed
        if key == ord(' '):

            if len(faces) == 1:

                # Extract face from ORIGINAL frame
                x, y, w, h = faces[0]

                face_image = frame[y:y+h, x:x+w].copy()

                # Generate unique filename
                existing_images = [
                    f for f in os.listdir(folder)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]

                image_number = len(existing_images) + 1

                image_path = os.path.join(
                    folder,
                    f"face{image_number}.jpg"
                )

                # Save clean facial image
                cv2.imwrite(image_path, face_image)

                print(f"\nFace registered successfully!")
                print(f"User: {name}")
                print(f"Saved image: {image_path}")

                break

            else:
                print("Please ensure exactly ONE face is visible.")

        # Cancel registration
        elif key == ord('q'):

            print("Registration cancelled.")
            break

    # Release camera resources
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    register_user()