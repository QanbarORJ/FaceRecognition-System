import cv2

# Initialize webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Camera could not be opened!")
    exit()

print("Camera started successfully!")

while True:

    # Capture frame
    ret, frame = camera.read()

    if not ret:
        print("Failed to capture image")
        break

    # Display camera feed
    cv2.imshow("Face Recognition System", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
camera.release()
cv2.destroyAllWindows()