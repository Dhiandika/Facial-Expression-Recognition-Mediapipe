import csv
import copy
import itertools
import pickle
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp
from model import KeyPointClassifier
from utils import get_face_landmarks


# Class untuk menghitung FPS
class CvFpsCalc(object):
    def __init__(self, buffer_len=1):
        self._start_tick = cv.getTickCount()
        self._freq = 1000.0 / cv.getTickFrequency()
        self._difftimes = deque(maxlen=buffer_len)

    def get(self):
        current_tick = cv.getTickCount()
        different_time = (current_tick - self._start_tick) * self._freq
        self._start_tick = current_tick

        self._difftimes.append(different_time)

        fps = 1000.0 / (sum(self._difftimes) / len(self._difftimes))
        fps_rounded = round(fps, 2)

        return fps_rounded


# Function to calculate landmarks from image
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point


# Pre-process landmarks for classification
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    return list(map(lambda n: n / max_value, temp_landmark_list))


# Calculate bounding rectangle from landmarks
def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_array = np.append(landmark_array, np.array([[landmark_x, landmark_y]]), axis=0)
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]


# Draw bounding box on the image
def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image


# Draw information text on the image
def draw_info_text(image, brect, facial_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
    if facial_text != "":
        info_text = 'Emotion: ' + facial_text
        cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image


# Custom function to draw face mesh skeleton
def draw_custom_face_mesh(image, face_landmarks, opacity=0.5, color=(0, 255, 0), thickness=1):
    overlay = image.copy()
    mp_drawing.draw_landmarks(
        image=overlay,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=thickness, circle_radius=1),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=thickness)
    )
    cv.addWeighted(overlay, opacity, image, 1 - opacity, 0, image)


# Initialize webcam and model
cap_device = 0
cap_width = 1920
cap_height = 1080
use_brect = True
cap = cv.VideoCapture(cap_device)
cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

keypoint_classifier = KeyPointClassifier()

# Load emotion recognition model and labels
with open('model/model_file_augmented.pkl', 'rb') as f:
    rf_classifier = pickle.load(f)
emotion_labels = ['marah', 'netral', 'sedih', 'senang', 'terkejut']

# Read Keypoint classifier labels
with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
    keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

# Initialize FPS calculation
fps_calculator = CvFpsCalc(buffer_len=10)

while True:
    key = cv.waitKey(10)
    if key == 27:  # ESC key
        break

    ret, image = cap.read()
    if not ret:
        break
    image = cv.flip(image, 1)  # Mirror display
    debug_image = copy.deepcopy(image)
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    # Calculate FPS
    fps = fps_calculator.get()

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Bounding box calculation
            brect = calc_bounding_rect(debug_image, face_landmarks)
            # Landmark calculation
            landmark_list = calc_landmark_list(debug_image, face_landmarks)
            # Convert to relative coordinates / normalized coordinates
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            # Classify emotion based on landmarks
            facial_emotion_id = keypoint_classifier(pre_processed_landmark_list)
            facial_emotion = keypoint_classifier_labels[facial_emotion_id]
            
            # Drawing part
            debug_image = draw_bounding_rect(use_brect, debug_image, brect)
            debug_image = draw_info_text(debug_image, brect, facial_emotion)

            # Draw custom face mesh skeleton
            draw_custom_face_mesh(debug_image, face_landmarks, opacity=0.7, color=(255, 0, 0), thickness=1)

            # Emotion prediction using RF model
            if len(landmark_list) == 1404:
                face_landmarks_array = np.array(landmark_list).reshape(1, -1)
                predicted_emotion_index = rf_classifier.predict(face_landmarks_array)
                predicted_emotion = emotion_labels[int(predicted_emotion_index[0])]
                cv.putText(debug_image, f"RF Emotion: {predicted_emotion}", (30, 80),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

    # Display FPS
    cv.putText(debug_image, f"FPS: {fps}", (30, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

    cv.imshow('Emotion and Face Mesh Recognition', debug_image)

cap.release()
cv.destroyAllWindows()
