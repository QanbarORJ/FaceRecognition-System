import os
import re
import cv2
import shutil
import threading
import queue
import tkinter as tk

from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
from deepface import DeepFace


# ==========================================
# CONFIGURATION
# ==========================================

DATABASE_PATH = "database"
MODEL_NAME = "Facenet512"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

os.makedirs(DATABASE_PATH, exist_ok=True)


# ==========================================
# MAIN APPLICATION
# ==========================================

class FaceRecognitionApp:

    def __init__(self, root):

        self.root = root

        self.root.title("Face Recognition System")
        self.root.geometry("1100x750")
        self.root.minsize(950, 680)
        self.root.configure(bg="#172033")

        # Camera initialization
        self.camera = cv2.VideoCapture(0)

        if not self.camera.isOpened():
            messagebox.showerror(
                "Camera Error",
                "Unable to access webcam."
            )
            self.root.destroy()
            return

        # Haar Cascade face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

        if self.face_cascade.empty():
            messagebox.showerror(
                "Detection Error",
                "Unable to load face detection model."
            )
            self.camera.release()
            self.root.destroy()
            return

        # Application variables
        self.latest_frame = None
        self.processing = False
        self.result_queue = queue.Queue()

        # Build interface
        self.create_interface()

        # Initialize application
        self.refresh_users()
        self.update_camera()
        self.check_results()

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close_application
        )

    # ======================================
    # INTERFACE DESIGN
    # ======================================

    def create_interface(self):

        # HEADER
        header = tk.Frame(
            self.root,
            bg="#172033"
        )

        header.pack(
            fill="x",
            padx=25,
            pady=(20, 10)
        )

        tk.Label(
            header,
            text="FACE RECOGNITION SYSTEM",
            font=("Segoe UI", 24, "bold"),
            fg="white",
            bg="#172033"
        ).pack(anchor="w")

        tk.Label(
            header,
            text="Biometric Identification Platform",
            font=("Segoe UI", 11),
            fg="#94A3B8",
            bg="#172033"
        ).pack(anchor="w", pady=(3, 0))

        # MAIN CONTAINER
        main = tk.Frame(
            self.root,
            bg="#172033"
        )

        main.pack(
            fill="both",
            expand=True,
            padx=25,
            pady=10
        )

        # ==================================
        # LEFT SIDE - CAMERA
        # ==================================

        left = tk.Frame(
            main,
            bg="#263449",
            padx=15,
            pady=15
        )

        left.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(0, 15)
        )

        tk.Label(
            left,
            text="LIVE CAMERA",
            font=("Segoe UI", 12, "bold"),
            fg="white",
            bg="#263449"
        ).pack(
            anchor="w",
            pady=(0, 12)
        )

        self.camera_label = tk.Label(
            left,
            bg="#0F172A",
            text="Initializing camera...",
            fg="white"
        )

        self.camera_label.pack(
            fill="both",
            expand=True
        )

        # ==================================
        # RIGHT SIDE - CONTROL PANEL
        # ==================================

        right = tk.Frame(
            main,
            bg="#263449",
            width=285,
            padx=15,
            pady=15
        )

        right.pack(
            side="right",
            fill="y"
        )

        right.pack_propagate(False)

        tk.Label(
            right,
            text="CONTROL PANEL",
            font=("Segoe UI", 12, "bold"),
            fg="white",
            bg="#263449"
        ).pack(
            anchor="w",
            pady=(0, 15)
        )

        # Register button
        ttk.Button(
            right,
            text="Register New User",
            command=self.register_user
        ).pack(
            fill="x",
            pady=5
        )

        # Recognition button
        self.recognize_button = ttk.Button(
            right,
            text="Start Face Recognition",
            command=self.recognize_user
        )

        self.recognize_button.pack(
            fill="x",
            pady=5
        )

        # ==================================
        # RECOGNITION RESULT CARD
        # ==================================

        result_frame = tk.Frame(
            right,
            bg="#172033",
            padx=12,
            pady=12
        )

        result_frame.pack(
            fill="x",
            pady=(18, 5)
        )

        tk.Label(
            result_frame,
            text="RECOGNITION RESULT",
            font=("Segoe UI", 10, "bold"),
            fg="#94A3B8",
            bg="#172033"
        ).pack(anchor="w")

        self.result_label = tk.Label(
            result_frame,
            text="Waiting for recognition...",
            font=("Segoe UI", 12, "bold"),
            fg="white",
            bg="#172033",
            wraplength=235,
            justify="left"
        )

        self.result_label.pack(
            anchor="w",
            pady=(10, 3)
        )

        self.distance_label = tk.Label(
            result_frame,
            text="Distance: --",
            font=("Segoe UI", 9),
            fg="#94A3B8",
            bg="#172033"
        )

        self.distance_label.pack(
            anchor="w"
        )

        # ==================================
        # REGISTERED USERS
        # ==================================

        tk.Label(
            right,
            text="REGISTERED USERS",
            font=("Segoe UI", 10, "bold"),
            fg="white",
            bg="#263449"
        ).pack(
            anchor="w",
            pady=(20, 8)
        )

        self.user_list = tk.Listbox(
            right,
            bg="#172033",
            fg="white",
            selectbackground="#2563EB",
            font=("Segoe UI", 10),
            height=8,
            borderwidth=0
        )

        self.user_list.pack(
            fill="both",
            expand=True
        )

        # Delete user
        ttk.Button(
            right,
            text="Delete Selected User",
            command=self.delete_user
        ).pack(
            fill="x",
            pady=(10, 5)
        )

        # Refresh list
        ttk.Button(
            right,
            text="Refresh User List",
            command=self.refresh_users
        ).pack(
            fill="x",
            pady=5
        )

        # ==================================
        # FOOTER - SYSTEM STATUS
        # ==================================

        footer = tk.Frame(
            self.root,
            bg="#263449",
            padx=20,
            pady=15
        )

        footer.pack(
            fill="x",
            padx=25,
            pady=(5, 15)
        )

        tk.Label(
            footer,
            text="SYSTEM STATUS:",
            font=("Segoe UI", 10, "bold"),
            fg="#94A3B8",
            bg="#263449"
        ).pack(side="left")

        self.status_label = tk.Label(
            footer,
            text="Camera Active - Waiting",
            font=("Segoe UI", 10, "bold"),
            fg="#22C55E",
            bg="#263449"
        )

        self.status_label.pack(
            side="left",
            padx=10
        )

        # Privacy notice
        tk.Label(
            self.root,
            text="Biometric images are stored locally. Register only with consent.",
            font=("Segoe UI", 8),
            fg="#94A3B8",
            bg="#172033"
        ).pack(pady=(0, 8))

    # ======================================
    # FACE DETECTION AND CROPPING
    # ======================================

    def detect_and_crop_face(self, frame):

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        if len(faces) != 1:
            return None

        x, y, w, h = faces[0]

        # Add margins around face
        margin_x = int(w * 0.2)
        margin_y = int(h * 0.2)

        height, width = frame.shape[:2]

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)

        x2 = min(width, x + w + margin_x)
        y2 = min(height, y + h + margin_y)

        face_image = frame[y1:y2, x1:x2].copy()

        return face_image

    # ======================================
    # CAMERA AND LIVE FACE DETECTION
    # ======================================

    def update_camera(self):

        if not self.camera.isOpened():
            return

        ret, frame = self.camera.read()

        if ret:

            # Keep original frame for processing
            self.latest_frame = frame.copy()

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50)
            )

            # Draw rectangles
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
                    0.7,
                    (0, 255, 0),
                    2
                )

            # Convert BGR to RGB
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            image = Image.fromarray(frame)

            image = image.resize((640, 480))

            photo = ImageTk.PhotoImage(image)

            self.camera_label.configure(
                image=photo,
                text=""
            )

            self.camera_label.image = photo

        self.root.after(
            30,
            self.update_camera
        )

    # ======================================
    # USER REGISTRATION
    # ======================================

    def register_user(self):

        name = simpledialog.askstring(
            "Registration",
            "Enter the person's name:",
            parent=self.root
        )

        if not name:
            return

        name = name.strip()

        if not re.fullmatch(r"[\w -]{1,40}", name):

            messagebox.showerror(
                "Invalid Name",
                "Use only letters, numbers, spaces, underscores or hyphens."
            )

            return

        if self.latest_frame is None:

            messagebox.showerror(
                "Error",
                "Camera image is unavailable."
            )

            return

        frame = self.latest_frame.copy()

        face_image = self.detect_and_crop_face(frame)

        if face_image is None:

            messagebox.showwarning(
                "Registration Failed",
                "Please ensure exactly one face is visible."
            )

            return

        folder = os.path.join(
            DATABASE_PATH,
            name
        )

        os.makedirs(folder, exist_ok=True)

        existing = [
            f for f in os.listdir(folder)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ]

        image_number = len(existing) + 1

        image_path = os.path.join(
            folder,
            f"face{image_number}.jpg"
        )

        success = cv2.imwrite(
            image_path,
            face_image
        )

        if success:

            messagebox.showinfo(
                "Registration Successful",
                f"{name} has been registered successfully!"
            )

            self.status_label.config(
                text=f"Registered: {name}",
                fg="#22C55E"
            )

            self.refresh_users()

        else:

            messagebox.showerror(
                "Error",
                "Unable to save facial image."
            )

    # ======================================
    # CHECK REGISTERED IMAGES
    # ======================================

    def get_registered_images(self):

        registered_images = []

        if not os.path.exists(DATABASE_PATH):
            return registered_images

        for name in os.listdir(DATABASE_PATH):

            folder = os.path.join(
                DATABASE_PATH,
                name
            )

            if not os.path.isdir(folder):
                continue

            for filename in os.listdir(folder):

                if filename.lower().endswith(IMAGE_EXTENSIONS):

                    registered_images.append(
                        os.path.join(folder, filename)
                    )

        return registered_images

    # ======================================
    # START FACE RECOGNITION
    # ======================================

    def recognize_user(self):

        if self.processing:
            return

        if self.latest_frame is None:
            return

        # Check database first
        registered_images = self.get_registered_images()

        if not registered_images:

            messagebox.showwarning(
                "Empty Database",
                "No registered users found!\n\nPlease register a user first."
            )

            self.status_label.config(
                text="No Registered Users",
                fg="#EF4444"
            )

            return

        # Capture current frame
        frame = self.latest_frame.copy()

        # Detect and crop face
        face_image = self.detect_and_crop_face(frame)

        if face_image is None:

            messagebox.showwarning(
                "Face Detection",
                "Please ensure exactly one face is visible."
            )

            return

        # Update interface
        self.processing = True

        self.recognize_button.config(
            state="disabled"
        )

        self.status_label.config(
            text="Analyzing Face...",
            fg="#FACC15"
        )

        self.result_label.config(
            text="Processing...",
            fg="#FACC15"
        )

        self.distance_label.config(
            text="Distance: Calculating..."
        )

        # Start background thread
        thread = threading.Thread(
            target=self.perform_recognition,
            args=(face_image,),
            daemon=True
        )

        thread.start()

    # ======================================
    # DEEPFACE RECOGNITION
    # ======================================

    def perform_recognition(self, face_image):

        try:

            print("\nStarting face recognition...")

            results = DeepFace.find(
                img_path=face_image,
                db_path=DATABASE_PATH,
                model_name=MODEL_NAME,
                detector_backend="skip",
                enforce_detection=False,
                silent=True
            )

            if not results:

                result = {
                    "success": False
                }

            else:

                matches = results[0]

                if not matches.empty:

                    best_match = matches.iloc[0]

                    identity_path = best_match["identity"]

                    person_name = os.path.basename(
                        os.path.dirname(identity_path)
                    )

                    distance = float(
                        best_match["distance"]
                    )

                    result = {
                        "success": True,
                        "name": person_name,
                        "distance": distance
                    }

                else:

                    result = {
                        "success": False
                    }

        except Exception as error:

            result = {
                "success": None,
                "error": str(error)
            }

        # Send result to main GUI thread
        self.result_queue.put(result)

    # ======================================
    # UPDATE GUI WITH RECOGNITION RESULTS
    # ======================================

    def check_results(self):

        try:

            while True:

                result = self.result_queue.get_nowait()

                self.processing = False

                self.recognize_button.config(
                    state="normal"
                )

                # MATCH FOUND
                if result["success"] is True:

                    name = result["name"]
                    distance = result["distance"]

                    self.status_label.config(
                        text="MATCH FOUND",
                        fg="#22C55E"
                    )

                    self.result_label.config(
                        text=f"IDENTIFIED: {name}",
                        fg="#22C55E"
                    )

                    self.distance_label.config(
                        text=f"Distance: {distance:.6f}"
                    )

                    print("\nMATCH FOUND!")
                    print("Person:", name)
                    print("Distance:", distance)

                # UNKNOWN PERSON
                elif result["success"] is False:

                    self.status_label.config(
                        text="NO MATCH FOUND",
                        fg="#EF4444"
                    )

                    self.result_label.config(
                        text="UNKNOWN PERSON",
                        fg="#EF4444"
                    )

                    self.distance_label.config(
                        text="No matching identity found."
                    )

                    print("\nNO MATCH FOUND!")

                # ERROR
                else:

                    self.status_label.config(
                        text="RECOGNITION ERROR",
                        fg="#EF4444"
                    )

                    self.result_label.config(
                        text="Recognition failed.",
                        fg="#EF4444"
                    )

                    self.distance_label.config(
                        text="Check terminal for details."
                    )

                    print(
                        "\nRecognition error:",
                        result["error"]
                    )

                # Force interface refresh
                self.root.update_idletasks()

        except queue.Empty:
            pass

        # Check again after 200 milliseconds
        self.root.after(
            200,
            self.check_results
        )

    # ======================================
    # REGISTERED USERS LIST
    # ======================================

    def refresh_users(self):

        self.user_list.delete(
            0,
            tk.END
        )

        if not os.path.exists(DATABASE_PATH):
            return

        for name in sorted(os.listdir(DATABASE_PATH)):

            folder = os.path.join(
                DATABASE_PATH,
                name
            )

            if os.path.isdir(folder):

                images = [
                    f for f in os.listdir(folder)
                    if f.lower().endswith(IMAGE_EXTENSIONS)
                ]

                if images:

                    self.user_list.insert(
                        tk.END,
                        f"{name} ({len(images)} images)"
                    )

    # ======================================
    # DELETE REGISTERED USER
    # ======================================

    def delete_user(self):

        selection = self.user_list.curselection()

        if not selection:

            messagebox.showwarning(
                "Selection Required",
                "Please select a registered user."
            )

            return

        selected_text = self.user_list.get(
            selection[0]
        )

        name = selected_text.rsplit(" (", 1)[0]

        confirm = messagebox.askyesno(
            "Delete User",
            f"Are you sure you want to delete {name} and all their facial images?"
        )

        if confirm:

            folder = os.path.join(
                DATABASE_PATH,
                name
            )

            if os.path.isdir(folder):

                shutil.rmtree(folder)

                messagebox.showinfo(
                    "Deleted",
                    f"{name} has been removed."
                )

                self.refresh_users()

                self.status_label.config(
                    text="User Deleted",
                    fg="#22C55E"
                )

                self.result_label.config(
                    text="Waiting for recognition...",
                    fg="white"
                )

                self.distance_label.config(
                    text="Distance: --"
                )

    # ======================================
    # CLOSE APPLICATION
    # ======================================

    def close_application(self):

        if self.camera.isOpened():
            self.camera.release()

        cv2.destroyAllWindows()

        self.root.destroy()


# ==========================================
# APPLICATION ENTRY POINT
# ==========================================

if __name__ == "__main__":

    root = tk.Tk()

    app = FaceRecognitionApp(root)

    root.mainloop()