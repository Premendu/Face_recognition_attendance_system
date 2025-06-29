import tkinter as tk
from tkinter import messagebox
import cv2
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import os

# Define Base Path
BASE_PATH = r"D:/My_Learning/Projects/Git/Face_recognition_attendance_system/classes/"  # Change this to your actual project folder


def update_encodings():
    try:
        class_name = class_entry.get().strip()  # Get user input for class
        if not class_name:
            messagebox.showerror("Error", "Please enter a class name before updating encodings.")
            return

        excel_file = BASE_PATH + f"{class_name}/{class_name}.xlsx"  # Class-specific Excel file
        if not os.path.exists(excel_file):
            messagebox.showerror("Error", f"File '{class_name}/{class_name}.xlsx' not found!")
            return

        file = pd.read_excel(excel_file)
        known_faces_names = file["Name"].tolist()
        st_number = len(known_faces_names)
        known_face_encodings = []
        encodings_file = BASE_PATH + f"{class_name}/{class_name}_encodings.pkl"

        # Track students with missing images
        missing_students = []

        status_label.config(text="Updating face encodings...", fg="blue")
        root.update()

        for i in range(1, st_number + 1):
            status_label.config(text=f"Processing encoding for roll number {i}...", fg="#040531")
            root.update()
            img_path = BASE_PATH + f"{class_name}/faces/{i}.jpg"
            
            if os.path.exists(img_path):  
                try:
                    st_img = face_recognition.load_image_file(img_path)
                    encodings = face_recognition.face_encodings(st_img)
                    if encodings:
                        known_face_encodings.append(encodings[0])
                    else:
                        raise ValueError("No face detected in image.")
                except Exception as e:
                    status_label.config(text=f"⚠️ Error with {img_path}: {e}", fg="red")
                    root.update()
                    known_face_encodings.append(np.zeros(128))  # Placeholder encoding
                    missing_students.append(f"Roll {i} - {known_faces_names[i-1]}")
            else:
                known_face_encodings.append(np.zeros(128))  # Placeholder encoding
                missing_students.append(f"Roll {i} - {known_faces_names[i-1]}")

        # Save encodings
        with open(encodings_file, "wb") as f:
            pickle.dump((known_faces_names, known_face_encodings), f)

        if missing_students:
            missing_text = "✅ Encodings updated successfully! \n ⚠️ Placeholder encoding used for:\n" + "\n".join(missing_students)
            status_label.config(text=missing_text, fg="#f06902")
        else:
            status_label.config(text="✅ Encodings updated successfully!", fg="#040531")

    except Exception as e:
        status_label.config(text=f"❌ Error: {e}", fg="red")
    root.update()



def take_attendance():
    try:
        class_name = class_entry.get().strip()  # Get user input for class
        if not class_name:
            messagebox.showerror("Error", "Please enter a class name before taking attendance.")
            return

        excel_file = BASE_PATH + f"{class_name}/{class_name}.xlsx"  # Class-specific Excel file
        encodings_file = BASE_PATH +  f"{class_name}/{class_name}_encodings.pkl"

        if not os.path.exists(excel_file):
            messagebox.showerror("Error", f"File '{class_name}.xlsx' not found!")
            return
        
        if not os.path.exists(encodings_file):
            messagebox.showerror("Error", f"Encodings file '{class_name}_encodings.pkl' not found! Please update encodings first.")
            return

        now = datetime.now()
        date = now.strftime("%d.%m.%Y")
        file = pd.read_excel(excel_file)

        with open(encodings_file, "rb") as f:
            known_faces_names, known_face_encodings = pickle.load(f)

        students = known_faces_names.copy()
        video_capture = cv2.VideoCapture(0)

        frame_count = 0

        while True:
            frame_count += 1
            if frame_count%5 != 0:
                continue
            _, frame = video_capture.read()

            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_locations = [(top * 2, right * 2, bottom * 2, left * 2) for (top, right, bottom, left) in face_locations]

            for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)  
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if face_distances[best_match_index] < 0.4:  # Set strict threshold to false attendence
                    name = known_faces_names[best_match_index]
                    cv2.rectangle(frame, (left, top), (right, bottom), (6, 145, 43), 2)
                    # Calculate text width and height
                    text = f"{name} is present"
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)

                    # Center the text above the face rectangle
                    text_x = left + (right - left) // 2 - text_width // 2
                    text_y = top - 10

                    # Draw the text
                    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, 0.8, (6, 145, 43), 2)

                    if name in students:
                        students.remove(name)
                        time = datetime.now().strftime("%H:%M:%S")
                        file.at[best_match_index, date] = "P " + time
                        file.to_excel(excel_file, index=False)

            cv2.imshow("Attendance", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("ℹ️ Exiting attendance system...")
                break

        video_capture.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error: {e}")
        status_label.config(text=f"❌ Error: {e}", fg="red")
    root.update()


# GUI Setup
root = tk.Tk()
root.title("Face Recognition Attendance System")
root.geometry("450x400")
root.configure(bg="#4914cf")  # Light background color

# Title Label
label = tk.Label(root, text="Face Recognition Attendance", font=("Arial", 16, "bold"), bg="#4914cf", fg="#333")
label.pack(pady=15)

# Class Input Field
class_label = tk.Label(root, text="Enter Class Name:", font=("Arial", 12), bg="#4914cf", fg="black")
class_label.pack()
class_entry = tk.Entry(root, font=("Arial", 12))
class_entry.pack(pady=5)

# Status Label
status_label = tk.Label(root, text="", font=("Arial", 12), bg="#4914cf", fg="black")
status_label.pack(pady=10)

# Buttons
update_btn = tk.Button(root, text="Update Face Encodings", command=update_encodings, width=25, height=2, bg="#007bff", fg="white", font=("Arial", 10, "bold"), relief="ridge")
update_btn.pack(pady=10)

take_attendance_btn = tk.Button(root, text="Take Attendance", command=take_attendance, width=25, height=2, bg="#112e6b", fg="white", font=("Arial", 10, "bold"), relief="ridge")
take_attendance_btn.pack(pady=10)

exit_btn = tk.Button(root, text="Exit", command=root.quit, width=25, height=2, bg="#dc3545", fg="white", font=("Arial", 10, "bold"), relief="ridge")
exit_btn.pack(pady=10)

root.mainloop()