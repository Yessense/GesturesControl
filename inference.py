import os

import cv2  # type: ignore
import mediapipe as mp  # type: ignore
import numpy as np
from scipy.special import softmax  # type: ignore

np.set_printoptions(precision=3, suppress=True)
from joblib import dump, load  # type: ignore

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

INPUT_SOURCE = 0

LABELS = {
    0: 'stand',
    1: 'jump',
    2: 'place',
    3: 'down',
    4: 'sit',
    5: 'come',
    6: 'paw',
}

DISTANCES = {
    0: 'down_close',
    1: 'down_far',
    2: 'middle_close',
}

LABELS_R = {value: key for key, value in LABELS.items()}
DISTANSES_R = {value: key for key, value in DISTANCES.items()}

DATASET_DIR = '/media/yessense/Transcend/gestures_dataset'
ANNOTATED_DIR = '/media/yessense/Transcend/annotated_images'

CLASSIFIER_PATH = '/home/yessense/PycharmProjects/RoboDogControl/data/classifier.joblib'
classifier = load(CLASSIFIER_PATH)
cap = cv2.VideoCapture(INPUT_SOURCE)
with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if not results.pose_landmarks:
            continue

        person = []
        if not results.pose_landmarks:
            continue
        for point in results.pose_landmarks.landmark:
            person.append([point.x,
                           point.y,
                           point.z,
                           point.visibility])

        person = np.array(person).reshape(-1, 33 * 4)  # type: ignore
        pred = softmax(classifier.predict(person, raw_score=True))[0]
        pred_argmax = np.argmax(pred)
        print(pred)
        if pred[pred_argmax] > 0.9:
            label = LABELS[pred_argmax]  # type: ignore
        else:
            label = 'none'
        # label = LABELS[pred[0]]

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        image = cv2.flip(image, 1)
        cv2.putText(image, f'{label}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('Prediction', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()
