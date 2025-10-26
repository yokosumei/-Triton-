#!/usr/bin/env python
# Triton_pose / pose_test.py — runtime v2 complet (YOLO-Pose + XGB) cu suport USB cam / PiCamera2
# Triton_pose / pose_test.py — runtime v2 complet (showcase doar Pose + XGB)
# - 4 fire + cozi (PoseEstimator -> Normalizer -> FeatureExtractor -> Classifier)
# - Fereastra temporizata 4s, HOP=0.5s (ca la training v2)
# - Filtru calitate frame: ≥80% kp cu conf≥0.5
# - Normalizare T/S: pelvis center + shoulder distance scale (fara rotatie)
# - Features v2 (nume identice cu make_features_v2.py)
# - Pipeline XGB (Imputer+Scaler+XGB) + calibrare Platt (optional)
# - Histerezis (tau_on/tau_off) + cooldown (din threshold.json sau defaults)
# - UI minimal: / (Start/Stop), /video_feed (RAW), /pose_feed (overlay), SocketIO "decision"

import os, json, time, threading, queue
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO

# ===================== Camera source =====================
USE_PICAM       = True   # <-- pe Raspberry Pi: True; pe laptop: False (USB cam)
CAM_INDEX       = 0
W, H            = 640, 480
JPEG_QUALITY    = 70
MJPEG_HZ        = 30     # ~30 fps în feed (independent de inferență)

# ===================== Pose/Decision config ==============
POSE_IMGSZ       = 352
POSE_FPS         = 5.0
KP_OK_CONF       = 0.50
KP_MIN_RATIO     = 0.80

WINDOW_SEC       = 4.0
HOP_SEC          = 0.5

DEFAULT_TAU_ON   = 0.60
DEFAULT_TAU_OFF  = 0.55
DEFAULT_COOLDOWN = 7.0

MODELS_DIR       = "models"
POSE_WEIGHTS     = os.path.join(MODELS_DIR, "yolo11n-pose.pt")
XGB_PIPELINE     = os.path.join(MODELS_DIR, "xgb_pipeline.joblib")
THRESHOLDS_JSON  = os.path.join(MODELS_DIR, "threshold.json")
FEATURE_SCHEMA   = os.path.join(MODELS_DIR, "features_schema.json")
CALIBRATOR_PATH  = os.path.join(MODELS_DIR, "calibrator.joblib")

# ===================== App/State =========================
app = Flask(__name__, template_folder="templates", static_folder="static")
socketio = SocketIO(app, async_mode="threading")

raw_lock   = threading.Lock()
jpeg_lock  = threading.Lock()
pose_lock  = threading.Lock()
latest_raw  = None
latest_jpeg = None
pose_jpeg   = None

q_kp   = queue.Queue(maxsize=256)
q_norm = queue.Queue(maxsize=256)
q_feat = queue.Queue(maxsize=64)

last_label = "N/A"
last_proba = 0.0
cooldown_until = 0.0
consec_on  = 0
consec_off = 0
running = False

tau_on = DEFAULT_TAU_ON
tau_off = DEFAULT_TAU_OFF
cooldown_s = DEFAULT_COOLDOWN

try:
    if os.path.exists(THRESHOLDS_JSON):
        with open(THRESHOLDS_JSON, "r", encoding="utf-8") as f:
            th = json.load(f)
        if "tau_on" in th:
            tau_on = float(th["tau_on"])
            tau_off = float(th.get("tau_off", max(0.5, tau_on - 0.05)))
            cooldown_s = float(th.get("cooldown_s", DEFAULT_COOLDOWN))
        elif "threshold" in th:
            tau_on = float(th["threshold"])
            tau_off = max(0.5, tau_on - 0.05)
            cooldown_s = DEFAULT_COOLDOWN
except Exception:
    pass

CALIBRATOR = None
try:
    if os.path.exists(CALIBRATOR_PATH):
        CALIBRATOR = joblib.load(CALIBRATOR_PATH)
except Exception:
    CALIBRATOR = None

from ultralytics import YOLO
try:
    import torch
    DEVICE = 0 if torch.cuda.is_available() else "cpu"
    HALF   = (DEVICE != "cpu")
except Exception:
    DEVICE, HALF = "cpu", False

pose_model = YOLO(POSE_WEIGHTS)
XGB        = joblib.load(XGB_PIPELINE)

# ===== Keypoints idx
NOSE=0; L_EYE=1; R_EYE=2; L_EAR=3; R_EAR=4
L_SH=5; R_SH=6; L_EL=7; R_EL=8; L_WR=9; R_WR=10
L_HIP=11; R_HIP=12; L_KNE=13; R_KNE=14; L_ANK=15; R_ANK=16

@dataclass
class KPPacket:
    t: float
    frame: np.ndarray
    xy: np.ndarray
    conf: np.ndarray

@dataclass
class NormPacket:
    t: float
    xy_norm: np.ndarray
    conf: np.ndarray

@dataclass
class FeatPacket:
    t_start: float
    t_end: float
    feats: dict

def blank_jpeg():
    img = np.zeros((H,W,3), np.uint8)
    return cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])[1].tobytes()

def extract_xy_conf(res):
    kpts = getattr(res, "keypoints", None)
    if kpts is None or getattr(kpts, "xy", None) is None or len(kpts.xy) == 0:
        return None, None
    xy = kpts.xy[0]
    try: xy = xy.detach().cpu().numpy()
    except: xy = np.asarray(xy)
    xy = xy[:, :2].astype(np.float32)

    if getattr(kpts, "conf", None) is not None and len(kpts.conf) > 0:
        c = kpts.conf[0]
        try: conf = c.detach().cpu().numpy().astype(np.float32)
        except: conf = np.asarray(c, dtype=np.float32)
    else:
        conf = np.ones(xy.shape[0], dtype=np.float32)

    if xy.shape[0] < 17:
        pad_xy = np.full((17 - xy.shape[0], 2), -1.0, np.float32)
        xy = np.vstack([xy, pad_xy])
        pad_cf = np.zeros(17 - conf.shape[0], np.float32)
        conf = np.concatenate([conf, pad_cf])
    elif xy.shape[0] > 17:
        xy = xy[:17]; conf = conf[:17]
    return xy, conf

def frame_quality_ok(conf, kp_ok_conf=KP_OK_CONF, min_ratio=KP_MIN_RATIO):
    return float((conf >= kp_ok_conf).mean()) >= min_ratio

def normalize_TS(xy):
    xy = xy.copy()
    valid = (xy[:,0]>=0)&(xy[:,1]>=0)
    if valid[L_HIP] and valid[R_HIP]:
        pelvis = (xy[L_HIP] + xy[R_HIP]) / 2.0
    elif valid[L_HIP]:
        pelvis = xy[L_HIP]
    elif valid[R_HIP]:
        pelvis = xy[R_HIP]
    else:
        pelvis = np.array([0.0,0.0], np.float32)
    xy[valid] -= pelvis
    scale = 1.0
    if valid[L_SH] and valid[R_SH]:
        d = np.linalg.norm(xy[L_SH] - xy[R_SH])
        if d > 1e-6: scale = d
    xy[valid] /= (scale + 1e-6)
    return xy

def draw_pose_overlay(frame, xy, conf):
    CONN = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
            (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    for i,(x,y) in enumerate(xy):
        if conf[i] >= KP_OK_CONF and x>=0 and y>=0:
            cv2.circle(frame,(int(x),int(y)),3,(0,255,0),-1)
    for i,j in CONN:
        if conf[i] >= KP_OK_CONF and conf[j] >= KP_OK_CONF:
            x1,y1 = xy[i]; x2,y2 = xy[j]
            if x1>=0 and y1>=0 and x2>=0 and y2>=0:
                cv2.line(frame,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),2)

def features_on_window(xs):
    X = np.asarray(xs, dtype=np.float32)
    T = X.shape[0]
    if T < 2: return None

    vL = np.linalg.norm(np.diff(X[:, L_WR, :], axis=0), axis=1)
    vR = np.linalg.norm(np.diff(X[:, R_WR, :], axis=0), axis=1)
    v_wrist_L_mean = float(np.nanmean(vL))
    v_wrist_R_mean = float(np.nanmean(vR))
    v_wrist_L_var  = float(np.nanvar (vL))
    v_wrist_R_var  = float(np.nanvar (vR))

    sh_y = (X[:, L_SH, 1] + X[:, R_SH, 1]) / 2.0
    frac_hands_above_shoulders = float(((X[:, L_WR, 1] < sh_y) & (X[:, R_WR, 1] < sh_y)).mean())

    armL = np.linalg.norm(X[:, L_SH, :] - X[:, L_WR, :], axis=1)
    armR = np.linalg.norm(X[:, R_SH, :] - X[:, R_WR, :], axis=1)
    arm_len_mean = float(np.nanmean(np.concatenate([armL, armR])))
    arm_asym     = float(abs(np.nanmean(armL) - np.nanmean(armR)))

    trunk = ((X[:, L_SH, :] + X[:, R_SH, :]) / 2.0) - ((X[:, L_HIP, :] + X[:, R_HIP, :]) / 2.0)
    ang = np.arctan2(trunk[:,0], -trunk[:,1])
    trunk_angle_std = float(np.nanstd(ang))

    pelvis = ((X[:, L_HIP, :] + X[:, R_HIP, :]) / 2.0)
    step = np.linalg.norm(np.diff(pelvis, axis=0), axis=1)
    pelvis_disp_mean = float(np.nanmean(step))
    pelvis_disp_var  = float(np.nanvar (step))

    return {
        "v_wrist_L_mean": v_wrist_L_mean,
        "v_wrist_R_mean": v_wrist_R_mean,
        "v_wrist_L_var":  v_wrist_L_var,
        "v_wrist_R_var":  v_wrist_R_var,
        "frac_hands_above_shoulders": frac_hands_above_shoulders,
        "arm_len_mean": arm_len_mean,
        "arm_asym": arm_asym,
        "trunk_angle_std": trunk_angle_std,
        "pelvis_disp_mean": pelvis_disp_mean,
        "pelvis_disp_var":  pelvis_disp_var,
    }

# ====================== CAMERA THREAD ======================
def camera_thread():
    """Suport USB cam (OpenCV) sau PiCamera2 (RGB→BGR). MJPEG ~30Hz."""
    global latest_raw, latest_jpeg
    use_picam = False
    picam = None

    if USE_PICAM:
        try:
            from picamera2 import Picamera2
            picam = Picamera2()
            # Video config: 640x480, RGB888 pentru conversie rapidă la BGR
            cfg = picam.create_video_configuration(main={"size": (W, H), "format": "RGB888"})
            picam.configure(cfg)
            picam.start()
            use_picam = True
            print("[Cam] Using PiCamera2 at", (W, H))
        except Exception as e:
            print("[Cam] PiCamera2 not available, falling back to USB cam:", e)

    if not use_picam:
        cap = cv2.VideoCapture(CAM_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        if not cap.isOpened():
            print("[ERR] Camera (USB) nu poate fi deschisa."); return
        print("[Cam] Using USB camera at", (W, H))

    next_jpeg_at = 0.0
    period = 1.0 / MJPEG_HZ

    while True:
        if use_picam:
            # Picamera2 returnează RGB
            frame_rgb = picam.capture_array()
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        else:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.005); continue

        with raw_lock:
            latest_raw = frame

        now = time.time()
        if now >= next_jpeg_at:
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if ok:
                with jpeg_lock:
                    latest_jpeg = buf.tobytes()
            next_jpeg_at = now + period
        else:
            time.sleep(0.001)

# ====================== OTHER THREADS (identic) ============
def pose_estimator_thread():
    last_t = 0.0
    while True:
        if not running:
            time.sleep(0.02); continue
        now = time.monotonic()
        if now - last_t < 1.0/POSE_FPS:
            time.sleep(0.001); continue
        last_t = now

        with raw_lock:
            frame = None if latest_raw is None else latest_raw.copy()
        if frame is None:
            time.sleep(0.005); continue

        res = pose_model.predict(frame, imgsz=POSE_IMGSZ, device=DEVICE, half=HALF, verbose=False)
        xy, conf = None, None
        for r in (res if isinstance(res,(list,tuple)) else [res]):
            xy0, cf0 = extract_xy_conf(r)
            if xy0 is None: continue
            if frame_quality_ok(cf0):
                xy, conf = xy0, cf0
                break

        if xy is not None:
            overlay = frame.copy()
            draw_pose_overlay(overlay, xy, conf)
            ok, buf = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if ok:
                with pose_lock:
                    global pose_jpeg
                    pose_jpeg = buf.tobytes()

            pkt = KPPacket(t=now, frame=frame, xy=xy, conf=conf)
            try: q_kp.put(pkt, timeout=0.01)
            except queue.Full: pass

def normalizer_thread():
    while True:
        if not running:
            time.sleep(0.02); continue
        try:
            pkt: KPPacket = q_kp.get(timeout=0.2)
        except queue.Empty:
            continue
        xy_norm = normalize_TS(pkt.xy)
        npkt = NormPacket(t=pkt.t, xy_norm=xy_norm, conf=pkt.conf)
        try: q_norm.put(npkt, timeout=0.01)
        except queue.Full: pass

def feature_extractor_thread():
    win = deque()
    next_eval_at = 0.0
    while True:
        if not running:
            time.sleep(0.02); continue
        try:
            npkt: NormPacket = q_norm.get(timeout=0.3)
        except queue.Empty:
            continue
        now = npkt.t
        win.append((now, npkt.xy_norm))
        t_min = now - WINDOW_SEC - 1e-6
        while win and win[0][0] < t_min:
            win.popleft()

        if now < next_eval_at:
            continue
        next_eval_at = now + HOP_SEC

        xs = [xy for (t, xy) in win if t >= now - WINDOW_SEC - 1e-6]
        if len(xs) < 4:
            continue
        feats = features_on_window(xs)
        if feats is None:
            continue
        fpkt = FeatPacket(t_start=now - WINDOW_SEC, t_end=now, feats=feats)
        try: q_feat.put(fpkt, timeout=0.01)
        except queue.Full: pass

def classifier_thread():
    global last_label, last_proba, cooldown_until, consec_on, consec_off

    feature_order = None
    try:
        if os.path.exists(FEATURE_SCHEMA):
            with open(FEATURE_SCHEMA, "r", encoding="utf-8") as f:
                js = json.load(f)
            feature_order = list(js.get("feature_names", [])) or None
    except Exception:
        feature_order = None
    if feature_order is None and hasattr(XGB, "feature_names_in_"):
        feature_order = list(XGB.feature_names_in_)

    while True:
        if not running:
            time.sleep(0.02); continue
        try:
            fpkt: FeatPacket = q_feat.get(timeout=0.3)
        except queue.Empty:
            continue

        feats = fpkt.feats
        cols = feature_order if feature_order is not None else list(feats.keys())
        row = {c: feats.get(c, np.nan) for c in cols}
        X = pd.DataFrame([row], columns=cols)

        if hasattr(XGB, "predict_proba"):
            p = XGB.predict_proba(X)[0]
            classes = getattr(XGB, "classes_", [0,1])
            pos_idx = list(classes).index(1) if 1 in classes else 1
            proba = float(p[pos_idx])
        else:
            yhat = XGB.predict(X)[0]
            proba = float(yhat) if isinstance(yhat,(int,float,np.floating)) else 0.0

        if CALIBRATOR is not None:
            try:
                proba = float(CALIBRATOR.predict_proba(np.array([[proba]]))[:,1][0])
            except Exception:
                pass

        now = time.monotonic()
        state_changed = False

        if now < cooldown_until:
            last_proba = proba
            socketio.emit("decision", {"label": last_label, "proba": round(last_proba,3)})
            continue

        if proba >= tau_on:
            consec_on += 1;  consec_off = 0
        elif proba <= tau_off:
            consec_off += 1; consec_on = 0

        new_label = last_label
        if last_label != "INEC" and consec_on >= 2:
            new_label = "INEC"
            consec_on = consec_off = 0
            cooldown_until = now + cooldown_s
            state_changed = True
        elif last_label != "INOT" and consec_off >= 2:
            new_label = "INOT"
            consec_on = consec_off = 0
            state_changed = True

        last_proba = proba
        if state_changed:
            last_label = new_label

        socketio.emit("decision", {"label": last_label, "proba": round(last_proba,3)})

# ====================== Routes / Boot ======================
@app.route("/")
def index():
    return render_template("index.test.html")

@app.route("/start", methods=["POST"])
def start():
    global running, cooldown_until, consec_on, consec_off, last_label, last_proba
    running = True
    cooldown_until = 0.0
    consec_on = consec_off = 0
    last_label, last_proba = "N/A", 0.0
    return jsonify({"status":"started"})

@app.route("/stop", methods=["POST"])
def stop():
    global running
    running = False
    return jsonify({"status":"stopped"})

@app.route("/video_feed")
def video_feed():
    def gen():
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while True:
            with jpeg_lock:
                frame = latest_jpeg if latest_jpeg is not None else blank_jpeg()
            yield boundary + frame + b"\r\n"
            time.sleep(0.03)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/pose_feed")
def pose_feed():
    def gen():
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while True:
            with pose_lock:
                frame = pose_jpeg if pose_jpeg is not None else blank_jpeg()
            yield boundary + frame + b"\r\n"
            time.sleep(0.03)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

def boot():
    threading.Thread(target=camera_thread,            daemon=True, name="Camera").start()
    threading.Thread(target=pose_estimator_thread,    daemon=True, name="PoseEstimator").start()
    threading.Thread(target=normalizer_thread,        daemon=True, name="Normalizer").start()
    threading.Thread(target=feature_extractor_thread, daemon=True, name="FeatExtractor").start()
    threading.Thread(target=classifier_thread,        daemon=True, name="Classifier").start()

boot()

if __name__ == "__main__":
    print("Triton_pose v2 live at http://0.0.0.0:5000 (PiCamera2=" + str(USE_PICAM) + ")")
    socketio.run(app, host="0.0.0.0", port=5000)
