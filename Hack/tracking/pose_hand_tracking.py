import cv2, csv, os
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_hands    = mp.solutions.hands
mp_draw     = mp.solutions.drawing_utils
mp_style    = mp.solutions.drawing_styles

# Pose: shoulders (11,12), elbows (13,14), wrists (15,16)
POSE_IDS = [11, 12, 13, 14, 15, 16]

PATH = "asl_data_c.csv"
LANDMARKS = 21  # per hand

# ---- Build header once (also used for row-length validation) ----
header = ["label"]
# left hand 21 * (x,y,z)
for i in range(LANDMARKS):
    header += [f"L_x{i}", f"L_y{i}", f"L_z{i}"]
# right hand 21 * (x,y,z)
for i in range(LANDMARKS):
    header += [f"R_x{i}", f"R_y{i}", f"R_z{i}"]
# left arm (normalized xyz)
for joint in ["shoulder", "elbow", "wrist"]:
    header += [f"L_x{joint}", f"L_y{joint}", f"L_z{joint}"]
# right arm (normalized xyz)
for joint in ["shoulder", "elbow", "wrist"]:
    header += [f"R_x{joint}", f"R_y{joint}", f"R_z{joint}"]

# ---- Create CSV if missing or empty ----
abs_path = os.path.abspath(PATH)
if not os.path.exists(PATH) or os.path.getsize(PATH) == 0:
    with open(PATH, "w", newline="") as f:
        csv.writer(f).writerow(header)
print(f"[INFO] Writing CSV to: {abs_path}")
print(f"[INFO] Columns per row: {len(header)}")

def pose_xyz_subset(pose_lms, ids):
    """Return {id: (x,y,z)} for selected pose ids with visibility > 0.5."""
    out = {}
    if pose_lms:
        lm = pose_lms.landmark
        for idx in ids:
            if lm[idx].visibility > 0.5:
                out[idx] = (lm[idx].x, lm[idx].y, lm[idx].z)
    return out

# ---- label input (kept as you had it) ----
label_id = "None"
label_id = input("Add a label: ")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open camera 0")

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

frames_written = 0

try:
    with mp_holistic.Holistic(
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ) as holo:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = holo.process(rgb)  # pose (33) + left hand (21) + right hand (21)
            rgb.flags.writeable = True
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # ---- initialize per-frame arrays (normalized xyz everywhere) ----
            left_hand_data  = [0.0] * (LANDMARKS * 3)
            right_hand_data = [0.0] * (LANDMARKS * 3)
            left_shoulder   = [0.0, 0.0, 0.0]
            left_elbow      = [0.0, 0.0, 0.0]
            left_wrist      = [0.0, 0.0, 0.0]
            right_shoulder  = [0.0, 0.0, 0.0]
            right_elbow     = [0.0, 0.0, 0.0]
            right_wrist     = [0.0, 0.0, 0.0]

            # pixel wrists only for drawing elbow->hand bridge
            left_hand_root_px  = None
            right_hand_root_px = None

            # ---------- LEFT HAND ----------
            if res.left_hand_landmarks:
                # flatten 21*(x,y,z) normalized
                flat = []
                for lm in res.left_hand_landmarks.landmark:
                    flat += [lm.x, lm.y, lm.z]
                left_hand_data = flat

                mp_draw.draw_landmarks(
                    frame,
                    res.left_hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_style.get_default_hand_landmarks_style(),
                    mp_style.get_default_hand_connections_style()
                )

                # pixel wrist for drawing only
                wrist0 = res.left_hand_landmarks.landmark[0]
                left_hand_root_px = (int(wrist0.x * w), int(wrist0.y * h))
            else:
                cv2.putText(frame, "NO LEFT HAND", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # ---------- RIGHT HAND ----------
            if res.right_hand_landmarks:
                flat = []
                for lm in res.right_hand_landmarks.landmark:
                    flat += [lm.x, lm.y, lm.z]
                right_hand_data = flat

                mp_draw.draw_landmarks(
                    frame,
                    res.right_hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_style.get_default_hand_landmarks_style(),
                    mp_style.get_default_hand_connections_style()
                )

                wrist0 = res.right_hand_landmarks.landmark[0]
                right_hand_root_px = (int(wrist0.x * w), int(wrist0.y * h))
            else:
                cv2.putText(frame, "NO RIGHT HAND", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # ---------- POSE (shoulders + elbows + wrists) ----------
            pose_pts = {}
            if res.pose_landmarks:
                pose_pts = pose_xyz_subset(res.pose_landmarks, POSE_IDS)

                # shoulders/elbows
                left_shoulder  = list(pose_pts.get(11, [0.0, 0.0, 0.0]))
                right_shoulder = list(pose_pts.get(12, [0.0, 0.0, 0.0]))
                left_elbow     = list(pose_pts.get(13, [0.0, 0.0, 0.0]))
                right_elbow    = list(pose_pts.get(14, [0.0, 0.0, 0.0]))

                # wrists from pose (15/16) if visible; else fallback to hand wrist (landmark 0) normalized
                left_wrist  = list(pose_pts.get(15, [0.0, 0.0, 0.0]))
                right_wrist = list(pose_pts.get(16, [0.0, 0.0, 0.0]))
                if left_wrist == [0.0, 0.0, 0.0] and res.left_hand_landmarks:
                    w0 = res.left_hand_landmarks.landmark[0]
                    left_wrist = [w0.x, w0.y, w0.z]
                if right_wrist == [0.0, 0.0, 0.0] and res.right_hand_landmarks:
                    w0 = res.right_hand_landmarks.landmark[0]
                    right_wrist = [w0.x, w0.y, w0.z]

                # draw pose points + lines (pixel space)
                px = {}
                for idx, (nx, ny, _) in pose_pts.items():
                    cx, cy = int(nx * w), int(ny * h)
                    px[idx] = (cx, cy)
                    cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1)

                # shoulder → elbow
                if 11 in px and 13 in px:
                    cv2.line(frame, px[11], px[13], (0, 255, 255), 2)
                if 12 in px and 14 in px:
                    cv2.line(frame, px[12], px[14], (0, 255, 255), 2)

                # elbow → (hand wrist in pixels for a clean visual bridge)
                if 13 in px and left_hand_root_px is not None:
                    cv2.line(frame, px[13], left_hand_root_px, (0, 255, 255), 2)
                if 14 in px and right_hand_root_px is not None:
                    cv2.line(frame, px[14], right_hand_root_px, (0, 255, 255), 2)

            # ---------- Write CSV row (validate shape) ----------
            if label_id != "None":
                row = [label_id] + left_hand_data + right_hand_data + \
                      left_shoulder + left_elbow + left_wrist + \
                      right_shoulder + right_elbow + right_wrist

                if len(row) != len(header):
                    # Silent guard so you don't end up with corrupt rows
                    print(f"[WARN] Row length {len(row)} != header {len(header)} — skipping frame.")
                else:
                    with open(PATH, "a", newline="") as f:
                        csv.writer(f).writerow(row)
                    frames_written += 1
                    if frames_written % 30 == 0:
                        print(f"[INFO] Frames written: {frames_written}")

            cv2.putText(frame, f"Label: {label_id}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 250, 0), 2)

            cv2.imshow("Hands + Pose (normalized xyz)", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print(f"[DONE] Total frames written: {frames_written}")
    print(f"[DONE] File: {abs_path}")
