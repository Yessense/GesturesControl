from abc import abstractmethod, ABC
from typing import Union, Optional, Tuple, Any

import cv2  # type: ignore
import mediapipe as mp  # type: ignore
import numpy as np

from argparse import ArgumentParser
from joblib import dump, load  # type: ignore

from body_pose_embedder import FullBodyPoseEmbedder

# python3 inference.py --display_image=--source
parser = ArgumentParser()

# add PROGRAM level args
program_parser = parser.add_argument_group('Options')
program_parser.add_argument("--display_image", type=bool, default=False)
program_parser.add_argument("--source", type=int, default=0)
program_parser.add_argument("--confidence", type=bool, default=True)
program_parser.add_argument("--delay", type=int, default=5)
program_parser.add_argument("--classifier_path", type=str, default='classifier.joblib')

# parse input
args = parser.parse_args()


class Capture(ABC):
    @abstractmethod
    def isOpened(self) -> bool:
        pass

    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def release(self):
        pass


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


class PoseClassifier(object):
    def __init__(self, classifier_path: str, delay: int = 5):
        # Load classifier
        self.classifier = load(classifier_path)

        # Create mediapipe objects
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose

        # Pose embedder
        self.embedder = FullBodyPoseEmbedder()
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Delay between camera frames
        self.delay = delay

        # Labels description
        self.LABELS = {0: 'stand',
                       1: 'jump',
                       2: 'place',
                       3: 'down',
                       4: 'sit',
                       5: 'come',
                       6: 'paw',
                       }

        self.DISTANCES = {0: 'down_close',
                          1: 'down_far',
                          2: 'middle_close',
                          }

        self.LABELS_R = {value: key for key, value in self.LABELS.items()}
        self.DISTANSES_R = {value: key for key, value in self.DISTANCES.items()}

    def start_predicting(self, capture, imshow=False):
        """Start an infinite processing of capture
        output data is processing with self.process_prediction function
        """

        # check if capture is valid
        Capture.register(type(capture))
        assert isinstance(capture, Capture)

        while capture.isOpened():
            # getting camera frame
            success, image = cap.read()

            if not success:
                print("Ignoring empty camera frame.")
                continue

            # get predictions
            label, confidence, results = self.predict_pose(image)

            # display image
            if imshow:
                self.imshow(image, label, results)

            # processing predictions
            if label is None:
                print("No person is found on the image")
                continue
            else:
                self.process_prediction(label, confidence)

            # Press Esc to exit
            if cv2.waitKey(args.delay) & 0xFF == 27:
                break

        cap.release()

    def imshow(self, image, label: Optional[int], results: Any):
        """Draw the pose annotation on the image. """

        image.flags.writeable = True
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # draw landmarks
        if results is not None:
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

        # Flip the image horizontally for a selfie-view display.
        image = cv2.flip(image, 1)

        # Draw classname text on image
        if label is not None:
            cv2.putText(image, f'{self.LABELS[label]}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        # Display image
        cv2.imshow('Prediction', image)

    def predict_pose(self, image) -> Tuple[Optional[int], Optional[float], Any]:
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)

        # No detections
        if not results.pose_landmarks:
            return None, None, None

        # Get all points in numpy.array
        points = [[point.x, point.y, point.z] for point in results.pose_landmarks.landmark]
        points = np.array(points)

        # Get embeddings
        embeddings = self.embedder(points)
        embeddings = embeddings.reshape(-1, np.prod(embeddings.shape))  # type: ignore

        # Predict probabilities
        pred = self.classifier.predict(embeddings, raw_score=True)[0]
        pred = softmax(pred)

        # Get label and confidence
        label: int = np.argmax(pred).item()
        confidence: float = pred[label]

        return label, confidence, results

    def process_prediction(self, label, confidence):
        """CHANGE OUTCOME HERE"""
        out = f'{self.LABELS[label]} {confidence:.3f}'
        print(out)


if __name__ == '__main__':
    # Start reading capture
    cap = cv2.VideoCapture(args.source)
    classifier = PoseClassifier(classifier_path=args.classifier_path, delay=args.delay)

    classifier.start_predicting(capture=cap, draw_landmarks=args.display_image)
