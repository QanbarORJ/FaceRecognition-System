import cv2
import os
from deepface import DeepFace


def recognize_face():

    # Initialize webcam
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Camera could not be opened!")
        return

    print("\nFACE RECOGNITION SYSTEM")
    print("-----------------------")
    print("Press SPACE to recognize a face.")
    print("Press Q to exit.")

    while True:

        ret, frame = camera.read()

        if not ret:
            print("Camera frame error!")
            break

        cv2.putText(
            frame,
            "Press SPACE to recognize",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow("Face Recognition", frame)

        key = cv2.waitKey(1) & 0xFF

        # Start recognition
        if key == ord(' '):

            print("\nAnalyzing face...")

            try:

                results = DeepFace.find(
                    img_path=frame,
                    db_path="database",
                    model_name="Facenet512",
                    detector_backend="opencv",
                    enforce_detection=True,
                    silent=True
                )

                matches = results[0]

                if not matches.empty:

                    best_match = matches.iloc[0]

                    identity_path = best_match["identity"]

                    # Extract registered person's name
                    person_name = os.path.basename(
                        os.path.dirname(identity_path)
                    )

                    distance = best_match["distance"]

                    print("\nMATCH FOUND!")
                    print("Person:", person_name)
                    print("Distance:", distance)

                    message = f"Recognized: {person_name}"

                    color = (0, 255, 0)

                else:

                    print("\nUNKNOWN PERSON")

                    message = "Unknown Person"

                    color = (0, 0, 255)

            except Exception as error:

                print("Recognition error:", error)

                message = "Recognition Failed"

                color = (0, 0, 255)

            # Display recognition result
            result_frame = frame.copy()

            cv2.putText(
                result_frame,
                message,
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

            cv2.imshow("Face Recognition", result_frame)

            cv2.waitKey(2000)

        elif key == ord('q'):

            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    recognize_face()