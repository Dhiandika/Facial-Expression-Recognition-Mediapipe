import cv2
import mediapipe as mp

# Inisialisasi FaceMesh MediaPipe di luar fungsi untuk efisiensi
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True,
                                            max_num_faces=1,
                                            min_detection_confidence=0.5)

def get_face_landmarks(image, draw=False):
    # Konversi gambar ke RGB karena FaceMesh membutuhkan input RGB
    image_input_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Proses gambar untuk mendeteksi face landmarks
    results = face_mesh.process(image_input_rgb)

    # Untuk menyimpan hasil landmark
    image_landmarks = []

    # Jika ada landmark wajah yang terdeteksi
    if results.multi_face_landmarks:
        # Jika draw=True, gambarkan landmarks pada gambar
        if draw:
            mp_drawing = mp.solutions.drawing_utils
            drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=results.multi_face_landmarks[0],
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

        # Ambil landmark dari wajah pertama yang terdeteksi
        face_landmarks = results.multi_face_landmarks[0].landmark

        # Loop setiap landmark, masukkan koordinat (x, y, z)
        for landmark in face_landmarks:
            image_landmarks.append(landmark.x)
            image_landmarks.append(landmark.y)
            image_landmarks.append(landmark.z)

    # Kembalikan hasil landmark (1404 elemen jika 468 titik wajah terdeteksi)
    return image_landmarks
