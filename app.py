"""
Key bindings:
    t        toggle Train / Test mode
    r/p/s    (Train mode) add current hand pose as rock/paper/scissors
    u        undo the last added sample
    c        clear ALL samples for this session
    w        save now (also saves automatically on quit)
    q / Esc  quit
    n        normalization (on/off)
"""
import argparse
import os

import cv2
import mediapipe as mp
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


#############################################
# IGNORE THIS CODE UNTIL NEXT COMMENT BLOCK #
#############################################

def draw_overlay(frame, session, mode, prediction, confidence, normalize, hand_found):
    h, w = frame.shape[:2]
    panel_h = 110
    cv2.rectangle(frame, (0, 0), (w, panel_h), (30, 30, 30), -1)

    mode_color = (50, 200, 50) if mode == "TRAIN" else (50, 170, 230)
    cv2.putText(frame, f"Mode: {mode}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

    counts = session.counts()
    counts_text = "  ".join(f"{c}:{counts[c]}" for c in CLASSES)
    cv2.putText(frame, f"Samples  {counts_text}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(frame, f"Normalize: {'ON' if normalize else 'OFF'}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    cv2.putText(frame, "[t] mode  [r/p/s] add  [u] undo  [c] clear  [w] save  [q] quit  [n] normalize",
                (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv2.putText(frame, f"session: {session.path}", (10, 118),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)

    if not hand_found:
        cv2.putText(frame, "No hand detected", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    elif mode == "TEST":
        color = COLORS.get(prediction, COLORS[None])
        label = prediction.upper() if prediction else "..."
        cv2.rectangle(frame, (0, h - 60), (w, h), color, -1)
        cv2.putText(frame, f"{label}  ({confidence * 100:.0f}%)", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="practice",
                         help="Name for this round's saved samples, "
                              "e.g. round0, round1 (default: practice)")
    return parser.parse_args()

def create_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. See README.md > Troubleshooting.")
    return cap

def create_hand_detector():
    return mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1,
        min_detection_confidence=0.6, min_tracking_confidence=0.5,
    )




#############################
# WORKSHOP CODE STARTS HERE #
#############################

# CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
# KEY_TO_LABEL = {el: el for el in CLASSES}
# COLORS = {
#     "0": (60, 60, 220),
#     "1": (80, 180, 80),
#     "2": (200, 140, 40),
#     "3": (200, 40, 40),
#     "4": (200, 40, 200),
#     "5": (40, 200, 200),
#     "6": (200, 200, 40),
#     "7": (40, 200, 40),
#     "8": (40, 40, 200),
#     "9": (200, 40, 40),
#     None: (90, 90, 90),
# }


CLASSES = ["rock", "paper", "scissors"]
KEY_TO_LABEL = {"r": "rock", "p": "paper", "s": "scissors"}
COLORS = {
    "r": (60, 60, 220),
    "p": (80, 180, 80),
    "s": (200, 140, 40),
    None: (90, 90, 90),
}

class GestureSession:
    def __init__(self, path):
        self.path = path
        self.X = np.zeros((0, 63), dtype=np.float32)
        self.y = np.array([], dtype="<U16")
        if os.path.exists(path):
            data = np.load(path)
            self.X, self.y = data["X"], data["y"]
            print(f"Loaded {len(self.y)} existing samples from {path}")

    def add(self, label, feature_vector):
        self.X = np.vstack([self.X, feature_vector[None, :]])
        self.y = np.append(self.y, label)

    def undo(self):
        if len(self.y):
            self.X = self.X[:-1]
            self.y = self.y[:-1]

    def clear(self):
        self.X = np.zeros((0, 63), dtype=np.float32)
        self.y = np.array([], dtype="<U16")

    def counts(self):
        return {c: int(np.sum(self.y == c)) for c in CLASSES}

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        np.savez(self.path, X=self.X, y=self.y)

def capture_frame(cap):
    success, frame = cap.read()
    return frame if success else None

def preprocess_frame(frame):
    # display_frame = cv2.flip(frame, 1)
    # rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    # return display_frame, rgb_frame
    return frame, frame

def detect_landmarks(hand_detector, rgb_frame):
    results = hand_detector.process(rgb_frame)
    if results.multi_hand_landmarks:
        return results.multi_hand_landmarks[0]
    return None


def landmarks_to_features(hand_landmarks, normalize = False):
    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32
    )  
    if normalize:
        mins = coords.min(axis=0)
        ranges = coords.max(axis=0) - mins
        ranges[ranges == 0] = 1  # avoid divide-by-zero if a column is constant
        normalized = (coords - mins) / ranges
        return normalized.flatten()
    return coords.flatten()


def classify(session, feature_vector):
    n = len(session.y)
    if n == 0:
        return None, 0.0
    k = min(3, n)
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(session.X, session.y)
    pred = clf.predict(feature_vector[None, :])[0]
    neighbor_labels = session.y[
        np.argsort(np.linalg.norm(session.X - feature_vector, axis=1))[:k]
    ]
    confidence = float(np.mean(neighbor_labels == pred))
    return pred, confidence


def postprocess_prediction(prediction, confidence):
    # if confidence < 0.5:
    #     return None, confidence
    return prediction, confidence



def main():
    args = parse_args()
    session = GestureSession(os.path.join("sessions", f"{args.session}.npz"))
    cap = create_webcam()
    hand_detector = create_hand_detector()
    landmark_drawer = mp.solutions.drawing_utils

    mode = "TEST"
    normalize = False
    try:
        while True:
            # 1. INPUT
            frame = capture_frame(cap)
            if frame is None:
                break

            # 2. PREPROCESSING
            display_frame, rgb_frame = preprocess_frame(frame)

            # 3. LANDMARK DETECTION
            hand_landmarks = detect_landmarks(hand_detector, rgb_frame)

            feature_vector = None
            prediction, confidence = None, 0.0

            if hand_landmarks is not None:
                landmark_drawer.draw_landmarks(
                    display_frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                # 4. FEATURE PROCESSING
                feature_vector = landmarks_to_features(hand_landmarks, normalize = normalize)

                if mode == "TEST":
                    # 5. CLASSIFICATION
                    raw_prediction, raw_confidence = classify(session, feature_vector)
                    # 6. POSTPROCESSING
                    prediction, confidence = postprocess_prediction(raw_prediction, raw_confidence)

            # 7. OUTPUT
            draw_overlay(display_frame, session, mode, prediction, confidence, normalize,
                         hand_found=hand_landmarks is not None)
            cv2.imshow("AI Elective iteration Workshop", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("t"):
                mode = "TRAIN" if mode == "TEST" else "TEST"
            elif key == ord("c"):
                session.clear()
            elif key == ord("n"):
                normalize = not normalize
            elif key == ord("u"):
                session.undo()
            elif key == ord("w"):
                session.save()
                print(f"Saved {len(session.y)} samples to {session.path}")
            elif mode == "TRAIN" and chr(key) in KEY_TO_LABEL:
                if feature_vector is not None:
                    print(feature_vector)
                    session.add(KEY_TO_LABEL[chr(key)], feature_vector)
                else:
                    print("No hand detected -- sample not added.")
    finally:
        session.save()
        print(f"Saved {len(session.y)} samples to {session.path}")
        cap.release()
        cv2.destroyAllWindows()
        hand_detector.close()


if __name__ == "__main__":
    main()
