#!/usr/bin/env python
# Triton_pose / pose_test.py — YOLO Pose + XGB (runtime v2 complet, cu DEBUG/UI minimal)
# - 4 fire + cozi (PoseEstimator -> Normalizer -> FeatureExtractor -> Classifier)
# - Fereastra temporizată 4s, HOP=0.5s
# - Filtru calitate frame pentru decizie (≥50% kp cu conf≥0.5 — relaxat pentru demo)
# - Normalizare T/S: pelvis center + shoulder-distance scale
# - Features v2 identice cu make_features_v2.py
# - Pipeline XGB (Imputer+Scaler+XGB) + calibrare Platt (opțional)
# - Histerezis (tau_on/tau_off) + cooldown (din threshold.json sau defaults)
# - UI minimal: / (Start/Stop), /video_feed, /pose_feed, SocketIO "decision"
# - DEBUG: /selftest (testează XGB), /debug (stare runtime), loguri în toate firele

import os, json, time, threading, queue
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO

# ===================== Helpers / Logging =====================
LOG = True
def log(*a):
    if LOG:
        print("[POSE]", *a, flush=True)

# ===================== Camera/Feed config ====================
USE_PICAM       = True   # Raspberry Pi: True; Laptop: False (USB cam)
CAM_INDEX       = 0
W, H            = 640, 480
JPEG_QUALITY    = 70
MJPEG_HZ        = 30     # ~30 fps feed (independent de inferență)

# ===================== Pose/Decision config ==================
POSE_IMGSZ       = 352
POSE_FPS         = 5.0
KP_OK_CONF       = 0.50
KP_MIN_RATIO     = 0.50  # DEBUG: relaxat (în training era 0.80)
MIN_DRAW_CONF    = 0.20  # prag mic pentru a vedea scheletul pe overlay

WINDOW_SEC       = 4.0
HOP_SEC          = 0.5

DEFAULT_TAU_ON   = 0.60
DEFAULT_TAU_OFF  = 0.55
DEFAULT_COOLDOWN = 7.0

# ===================== Paths (independente de CWD) ===========
BASE_DIR    = Path(__file__).resolve().parent.parent     # repo root (~/Triton)
APP_DIR     = Path(__file__).resolve().parent            # Triton_pose/
MODELS_DIR  = BASE_DIR / "models"

POSE_WEIGHTS    = str(MODELS_DIR / "yolo11n-pose.pt")    # sau pune yolov8n-pose.pt dacă folosești YOLOv8
XGB_PIPELINE    = str(MODELS_DIR / "xgb_pipeline.joblib")
THRESHOLDS_JSON = str(MODELS_DIR / "threshold.json")
FEATURE_SCHEMA  = str(MODELS_DIR / "features_schema.json")
CALIBRATOR_PATH = str(MODELS_DIR / "calibrator.joblib")  # opțional

# ===================== App/State =============================
app = Flask(
    __name__,
    template_folder=str(APP_DIR / "templates"),   # <<— UI minimal de test e aici
    static_folder=str(BASE_DIR / "static")
)
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

# ===================== Thresholds / Calibrator ===============
try:
    if os.path.exists(THRESHOLDS_JSON):
        with open(THRESHOLDS_JSON, "r", encoding="utf-8") as f:
            th = json.load(f)
        if "tau_on" in th:
            tau_on = float(th["tau_on"])
            tau_off = float(th.get("tau_off", max(0.5, tau_on - 0.05)))
            cooldown_s = float(th.get("cooldown_s", DEFAULT_COOLDOWN))
        elif "threshold" in th:
            tau_on = float(th["threshold"])  # din OOF/Platt
            tau_off = max(0.5, tau_on - 0.05)
            cooldown_s = DEFAULT_COOLDOWN
except Exception as e:
    log("WARN thresholds load:", e)

CALIBRATOR = None
try:
    if os.path.exists(CALIBRATOR_PATH):
        CALIBRATOR = joblib.load(CALIBRATOR_PATH)
except Exception as e:
    log("WARN calibrator load:", e)
    CALIBRATOR = None

# ===================== YOLO / Torch ==========================
from ultralytics import YOLO
try:
    import torch
    DEVICE = 0 if torch.cuda.is_available() else "cpu"
    HALF   = (DEVICE != "cpu")
except Exception:
    DEVICE, HALF = "cpu", False

pose_model = YOLO(POSE_WEIGHTS); log("YOLO loaded:", POSE_WEIGHTS)
XGB        = joblib.load(XGB_PIPELINE); log("XGB loaded:", XGB_PIPELINE)
log(f"thresholds: tau_on={tau_on:.3f} tau_off={tau_off:.3f} cooldown={cooldown_s:.1f}s")

# ===== Feature schema tolerantă (acceptă `columns` sau `feature_names`)
FEATURE_LIST = None
try:
    with open(FEATURE_SCHEMA, "r", encoding="utf-8") as f:
        _schema = json.load(f)
    feats = _schema.get("feature_names")
    if feats is None:
        cols = _schema.get("columns", [])
        drop = {"video_id", "t_start", "t_end", "label"}
        feats = [c for c in cols if c not in drop]
    if not feats:
        raise ValueError("no feature names in schema")
    FEATURE_LIST = list(feats)
    log("feature_schema OK, features:", len(FEATURE_LIST))
except Exception as e:
    log("feature_schema missing/unreadable:", e)
    FEATURE_LIST = None

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

# ===================== Utils =================================
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

    # pad/trunc la 17
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

def draw_pose_overlay(frame, xy, conf, min_conf=MIN_DRAW_CONF):
    CONN = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
            (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    for i,(x,y) in enumerate(xy):
        if conf[i] >= min_conf and x>=0 and y>=0:
            cv2.circle(frame,(int(x),int(y)),3,(0,255,0),-1)
    for i,j in CONN:
        if conf[i] >= min_conf and conf[j] >= min_conf:
            x1,y1 = xy[i]; x2,y2 = xy[j]
            if x1>=0 and y1>=0 and x2>=0 and y2>=0:
                cv2.line(frame,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),2)
    return frame

def pick_best_xy(res_obj, draw_conf=MIN_DRAW_CONF):
    """Alege predicția cu cei mai mulți KP peste draw_conf, pentru overlay."""
    best = None
    best_count = -1
    seq = res_obj if isinstance(res_obj, (list, tuple)) else [res_obj]
    for r in seq:
        xy0, cf0 = extract_xy_conf(r)
        if xy0 is None or cf0 is None:
            continue
        count = int((cf0 >= draw_conf).sum())
        if count > best_count:
            best = (xy0, cf0)
            best_count = count
    return best

def features_on_window(xs):
    X = np.asarray(xs, dtype=np.float32)
    if X.shape[0] < 2: return None

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

# ====================== CAMERA THREAD =======================
def camera_thread():
    """Suport USB cam (OpenCV) sau PiCamera2 (RGB→BGR). MJPEG ~30Hz."""
    global latest_raw, latest_jpeg
    use_picam = False
    picam = None

    if USE_PICAM:
        try:
            from picamera2 import Picamera2
            picam = Picamera2()
            cfg = picam.create_video_configuration(main={"size": (W, H), "format": "RGB888"})
            picam.configure(cfg)
            picam.start()
            use_picam = True
            log("Using PiCamera2 at", (W, H))
        except Exception as e:
            log("PiCamera2 not available, fallback to USB:", e)

    if not use_picam:
        cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            log("[ERR] USB camera cannot open."); return
        log("Using USB camera at", (W, H))

    next_jpeg_at = 0.0
    period = 1.0 / MJPEG_HZ

    while True:
        if use_picam:
            frame_rgb = picam.capture_array()
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        else:
            ret, frame = cap.read()
            if not ret:
                cap.grab(); ret, frame = cap.retrieve()
                if not ret:
                    time.sleep(0.01); continue

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

# ====================== OTHER THREADS =======================
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
        best = pick_best_xy(res, MIN_DRAW_CONF)

        if best is not None:
            xy, conf = best
            overlay = frame.copy()
            draw_pose_overlay(overlay, xy, conf, min_conf=MIN_DRAW_CONF)

            vis_ratio = float((conf >= KP_OK_CONF).mean())
            cv2.putText(overlay, f"vis={vis_ratio:.2f}  {last_label} p={last_proba:.2f}", (10,24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0,0,255) if last_label=='INEC' else (0,255,0), 2)

            ok, buf = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if ok:
                with pose_lock:
                    global pose_jpeg
                    pose_jpeg = buf.tobytes()

            pushed = False
            if frame_quality_ok(conf):
                pkt = KPPacket(t=now, frame=frame, xy=xy, conf=conf)
                try:
                    q_kp.put(pkt, timeout=0.01)
                    pushed = True
                except queue.Full:
                    pass
            if pushed:
                log(f"pose->kp   vis={vis_ratio:.2f}  q_kp={q_kp.qsize()}")
        else:
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if ok:
                with pose_lock:
                    pose_jpeg = buf.tobytes()

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
        try:
            q_norm.put(npkt, timeout=0.01)
            log(f"kp->norm    q_norm={q_norm.qsize()}")
        except queue.Full:
            pass

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
        try:
            q_feat.put(fpkt, timeout=0.01)
            log(f"norm->feat  window={len(xs)}  q_feat={q_feat.qsize()}")
        except queue.Full:
            pass

def classifier_thread():
    global last_label, last_proba, cooldown_until, consec_on, consec_off

    feature_order = FEATURE_LIST
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
            log(f"feat->cls  (cooldown) p={proba:.3f} label={last_label}")
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
        log(f"feat->cls   p={proba:.3f}  label={last_label}  on={consec_on} off={consec_off}")

# ====================== Routes / Debug =======================
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
    log("== START ==")
    return jsonify({"status":"started"})

@app.route("/stop", methods=["POST"])
def stop():
    global running
    running = False
    log("== STOP ==")
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

@app.route("/selftest")
def selftest():
    cols = FEATURE_LIST
    if not cols and hasattr(XGB, "feature_names_in_"):
        cols = list(XGB.feature_names_in_)
    if not cols:
        return jsonify({"ok": False, "err": "no feature schema"}), 500
    row = {c: np.nan for c in cols}
    X = pd.DataFrame([row], columns=cols)
    try:
        classes = getattr(XGB, "classes_", [0,1])
        pos_idx = list(classes).index(1) if 1 in classes else 1
        p = float(XGB.predict_proba(X)[0, pos_idx])
    except Exception as e:
        return jsonify({"ok": False, "err": str(e)}), 500
    return jsonify({"ok": True, "proba": round(p,3), "n_features": len(cols)})

@app.route("/debug")
def debug_status():
    return jsonify({
        "running": running,
        "queues": {"kp": q_kp.qsize(), "norm": q_norm.qsize(), "feat": q_feat.qsize()},
        "label": last_label, "proba": round(last_proba,3),
        "tau_on": tau_on, "tau_off": tau_off, "cooldown_s": cooldown_s
    })

# ====================== Boot threads =========================
def boot():
    threading.Thread(target=camera_thread,            daemon=True, name="Camera").start()
    threading.Thread(target=pose_estimator_thread,    daemon=True, name="PoseEstimator").start()
    threading.Thread(target=normalizer_thread,        daemon=True, name="Normalizer").start()
    threading.Thread(target=feature_extractor_thread, daemon=True, name="FeatExtractor").start()
    threading.Thread(target=classifier_thread,        daemon=True, name="Classifier").start()

boot()

if __name__ == "__main__":
    print(f"Triton_pose v2 live at http://0.0.0.0:5000 (PiCamera2={USE_PICAM})")
    socketio.run(app, host="0.0.0.0", port=5000)
