import cv2
import numpy as np
import pickle
import mediapipe as mp
from utils import get_face_landmarks

# Load the trained model
with open('model/model_file_augmented.pkl', 'rb') as f:
    rf_classifier = pickle.load(f)

# Labels for emotions (sesuaikan urutannya dengan urutan yang benar)
emotion_labels = ['marah', 'netral', 'sedih', 'senang', 'terkejut']

# Initialize MediaPipe FaceMesh for drawing skeleton
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize webcam
cap = cv2.VideoCapture(0)

# Function to draw facemesh skeleton with customization
def draw_custom_face_mesh(image, face_landmarks, opacity=0.5, color=(0, 255, 0), thickness=1):
    # Copy original image to apply customizations on top
    overlay = image.copy()

    # Draw face landmarks
    mp_drawing.draw_landmarks(
        image=overlay,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,  # Use mesh tessellation
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=thickness, circle_radius=1),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=thickness)
    )

    # Blend overlay with the original image based on opacity
    cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0, image)

while True:
    # Capture frame from the camera
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to capture image.")
        break

    ### Deteksi Emosi dan Skeleton Bersamaan ###
    # Extract face landmarks and draw skeleton from the camera frame
    face_landmarks = get_face_landmarks(frame, draw=False)  # draw=False karena kita akan menggambar sendiri

    if len(face_landmarks) == 1404:
        face_landmarks_array = np.array(face_landmarks).reshape(1, -1)  # Reshape menjadi (1, 1404)

        # Predict emotion using the trained model
        predicted_emotion_index = rf_classifier.predict(face_landmarks_array)
        predicted_emotion = emotion_labels[int(predicted_emotion_index[0])]

        # Display the predicted emotion on the frame
        cv2.putText(frame, f"Emotion: {predicted_emotion}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        # If no face landmarks detected
        cv2.putText(frame, "No face detected", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    ### Skeleton Drawing ###
    # Convert the BGR image to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe to get the facemesh
    with mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Custom draw facemesh skeleton with adjustable parameters (color, thickness, opacity)
                draw_custom_face_mesh(frame, face_landmarks, opacity=0.7, color=(255, 0, 0), thickness=1)

    # Display the frame with both emotion detection and skeleton
    cv2.imshow('Emotion Detection with Customized Facemesh Skeleton', frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
