import time
import csv
import os
import re
import cv2
import numpy as np
from collections import deque
from joblib import load
import mediapipe as mp


MODEL_PATH = r"C:\Users\Janu\Desktop\hack\ASL-Gesture-App-Debug\ASL-Gesture-App-Hack\models\asl_model.joblib"
LETTERS_CSV = r"C:\Users\Janu\Desktop\hack\ASL-Gesture-App-Debug\ASL-Gesture-App-Hack\letters.csv"

mp_holistic = mp.solutions.holistic
mp_hands    = mp.solutions.hands
mp_draw     = mp.solutions.drawing_utils
mp_style    = mp.solutions.drawing_styles

LANDMARKS = 21
POSE_IDS  = [11, 12, 13, 14, 15, 16]

# ---------- helpers ----------
def pose_xyz_subset(pose_lms, ids):
    out = {}
    if pose_lms:
        lm = pose_lms.landmark
        for idx in ids:
            if lm[idx].visibility > 0.5:
                out[idx] = (lm[idx].x, lm[idx].y, lm[idx].z)
    return out

def flatten_hand_norm(hand_lms):
    if not hand_lms:
        return [0.0]*(LANDMARKS*3)
    flat = []
    for lm in hand_lms.landmark:
        flat += [lm.x, lm.y, lm.z]
    return flat

def build_feature_vector(res, feature_order):
    left_hand_flat  = flatten_hand_norm(res.left_hand_landmarks)
    right_hand_flat = flatten_hand_norm(res.right_hand_landmarks)

    pose_pts = {}
    left_shoulder  = [0.0,0.0,0.0]
    right_shoulder = [0.0,0.0,0.0]
    left_elbow     = [0.0,0.0,0.0]
    right_elbow    = [0.0,0.0,0.0]
    left_wrist     = [0.0,0.0,0.0]
    right_wrist    = [0.0,0.0,0.0]

    if res.pose_landmarks:
        pose_pts = pose_xyz_subset(res.pose_landmarks, POSE_IDS)
        left_shoulder  = list(pose_pts.get(11, [0.0,0.0,0.0]))
        right_shoulder = list(pose_pts.get(12, [0.0,0.0,0.0]))
        left_elbow     = list(pose_pts.get(13, [0.0,0.0,0.0]))
        right_elbow    = list(pose_pts.get(14, [0.0,0.0,0.0]))
        left_wrist     = list(pose_pts.get(15, [0.0,0.0,0.0]))
        right_wrist    = list(pose_pts.get(16, [0.0,0.0,0.0]))

        if left_wrist == [0.0,0.0,0.0] and res.left_hand_landmarks:
            w0 = res.left_hand_landmarks.landmark[0]
            left_wrist = [w0.x, w0.y, w0.z]
        if right_wrist == [0.0,0.0,0.0] and res.right_hand_landmarks:
            w0 = res.right_hand_landmarks.landmark[0]
            right_wrist = [w0.x, w0.y, w0.z]

    parts = {}
    for i in range(LANDMARKS):
        parts[f"L_x{i}"] = left_hand_flat[i*3 + 0]
        parts[f"L_y{i}"] = left_hand_flat[i*3 + 1]
        parts[f"L_z{i}"] = left_hand_flat[i*3 + 2]
        parts[f"R_x{i}"] = right_hand_flat[i*3 + 0]
        parts[f"R_y{i}"] = right_hand_flat[i*3 + 1]
        parts[f"R_z{i}"] = right_hand_flat[i*3 + 2]

    for joint, vals in zip(["shoulder","elbow","wrist"], [left_shoulder, left_elbow, left_wrist]):
        parts[f"L_x{joint}"], parts[f"L_y{joint}"], parts[f"L_z{joint}"] = vals
    for joint, vals in zip(["shoulder","elbow","wrist"], [right_shoulder, right_elbow, right_wrist]):
        parts[f"R_x{joint}"], parts[f"R_y{joint}"], parts[f"R_z{joint}"] = vals

    vec = [parts.get(k, 0.0) for k in feature_order]
    return np.array(vec, dtype=np.float32)

class MajorityVote:
    def __init__(self, k=9):
        self.q = deque(maxlen=k)
    def push(self, label):
        self.q.append(label)
    def get(self):
        if not self.q: return None
        vals, counts = np.unique(self.q, return_counts=True)
        return vals[np.argmax(counts)]

# ---------- main ----------
def main():
    artifact = load(MODEL_PATH)
    model = artifact["model"]
    classes = artifact["label_encoder_classes_"]
    feature_order = artifact["feature_order"]

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    smoother = MajorityVote(k=9)

    # ---------- CSV setup ----------
    buffer_letter = None
    buffer_start_time = None
    last_logged_times = {}  # {letter: last_timestamp}
    csv_filename = LETTERS_CSV

    if not os.path.exists(csv_filename):
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["letter"])
    # --------------------------------------------------------

    with mp_holistic.Holistic(
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holo:

        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = holo.process(rgb)
            rgb.flags.writeable = True
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            if res.left_hand_landmarks:
                mp_draw.draw_landmarks(frame, res.left_hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_style.get_default_hand_landmarks_style(),
                    mp_style.get_default_hand_connections_style())
            if res.right_hand_landmarks:
                mp_draw.draw_landmarks(frame, res.right_hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_style.get_default_hand_landmarks_style(),
                    mp_style.get_default_hand_connections_style())

            x = build_feature_vector(res, feature_order).reshape(1, -1)

            try:
                proba = model.predict_proba(x)[0]
                idx = int(np.argmax(proba))
                conf = float(proba[idx])
                pred_label = classes[idx]
            except Exception:
                pred = model.predict(x)[0]
                pred_label = classes[int(pred)]
                conf = 1.0

            smoother.push(pred_label)
            stable = smoother.get()

            # ---------- CSV logic ----------
            current_time = time.time()
            if conf >= 0.5:
                letter_only = re.sub(r'[^A-Z]', '', pred_label.upper())

                if buffer_letter != letter_only:
                    buffer_letter = letter_only
                    buffer_start_time = current_time
                elif current_time - buffer_start_time >= 1.5:
                    last_time = last_logged_times.get(letter_only, 0)
                    if current_time - last_time >= 2.0:
                        with open(csv_filename, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([letter_only])
                        last_logged_times[letter_only] = current_time
                        print(f"Saved letter '{letter_only}' to {csv_filename}")
                        buffer_letter = None
                        buffer_start_time = None
            else:
                buffer_letter = None
                buffer_start_time = None
            # ------------------------------------------------

            h, w = frame.shape[:2]
            cv2.putText(frame, f"Pred: {pred_label}  ({conf:.2f})", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            if stable:
                cv2.putText(frame, f"Stable: {stable}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.putText(frame, "ESC to quit", (10, h-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)

            cv2.imshow("ASL Inference (Hands + Pose)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
