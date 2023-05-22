import cv2
import face_recognition
from tkinter import filedialog
from tkinter import Tk, Button, Label, StringVar
import time


def choose_file():
    root = Tk()
    root.filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                               filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    root.withdraw()
    return root.filename


def load_and_encode_image(image_path):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    return image, face_locations, face_encodings


def compare_two_images():
    image1_path = choose_file()
    image2_path = choose_file()

    image1, face_locations1, face_encodings1 = load_and_encode_image(image1_path)
    image2, face_locations2, face_encodings2 = load_and_encode_image(image2_path)

    # Porównaj każdą twarz na pierwszym obrazie z każdą twarz na drugim obrazie
    for i, (encoding1, location1) in enumerate(zip(face_encodings1, face_locations1)):
        for j, (encoding2, location2) in enumerate(zip(face_encodings2, face_locations2)):
            # Oblicz stopień podobieństwa między twarzami
            face_distance = face_recognition.face_distance([encoding1], encoding2)

            # Jeżeli twarze są podobne, oznacz je na obrazach
            if face_distance < 0.6:
                top1, right1, bottom1, left1 = location1
                top2, right2, bottom2, left2 = location2
                cv2.rectangle(image1, (left1, top1), (right1, bottom1), (0, 0, 255), 2)
                cv2.rectangle(image2, (left2, top2), (right2, bottom2), (0, 0, 255), 2)
                print(
                    f"Face {i + 1} on image1 is similar to Face {j + 1} on image2 with a similarity score of {1 - face_distance[0]}")

    # Konwertuj obrazy z BGR na RGB
    image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    # Wyświetl obrazy z oznaczonymi twarzami
    cv2.imshow('Image1', image1_rgb)
    cv2.imshow('Image2', image2_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def compare_image_to_camera():
    image_path = choose_file()
    image, face_locations, face_encodings = load_and_encode_image(image_path)

    video_capture = cv2.VideoCapture(0)

    # Czekaj, aż kamera będzie gotowa
    while True:
        if video_capture.isOpened():
            break
        print("Waiting for the camera to be ready...")
        time.sleep(1)  # czekaj 1 sekundę

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to read frame from camera. Please check your camera.")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations_frame = face_recognition.face_locations(frame)
        face_encodings_frame = face_recognition.face_encodings(frame, face_locations_frame)

        for i, (encoding1, location1) in enumerate(zip(face_encodings, face_locations)):
            for j, (encoding2, location2) in enumerate(zip(face_encodings_frame, face_locations_frame)):
                face_distance = face_recognition.face_distance([encoding1], encoding2)

                if face_distance < 0.6:
                    top1, right1, bottom1, left1 = location1
                    top2, right2, bottom2, left2 = location2

                    cv2.rectangle(image, (left1, top1), (right1, bottom1), (0, 0, 255), 2)
                    cv2.rectangle(frame, (left2, top2), (right2, bottom2), (0, 0, 255), 2)
                    print(
                        f"Face {i + 1} on image is similar to Face {j + 1} on camera feed with a similarity score of {1 - face_distance[0]}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cv2.imshow('Image', image_rgb)
        cv2.imshow('Webcam', frame_rgb)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


def main():
    root = Tk()
    root.geometry("300x100")

    Button(root, text="Compare Two Images", command=compare_two_images).pack(pady=10)
    Button(root, text="Compare Image to Camera", command=compare_image_to_camera).pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    main()
