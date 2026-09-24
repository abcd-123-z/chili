"""
app.py
=======
เว็บเซิร์ฟเวอร์รับภาพจาก ESP32-CAM ผ่าน HTTP POST, รันโมเดล TFLite ตรวจสอบ
พริกปกติ/เป็นโรค แล้วแสดงผลบนแดชบอร์ดแบบเรียลไทม์

เพิ่มเติมจากเดิม: หน้าเว็บตอนนี้ทำได้ครบวงจร —
    - กดบันทึกภาพล่าสุดจากกล้องลง dataset/train หรือ val (สุ่มแบ่ง 85/15) ตาม label ที่เลือก
    - กดเทรนโมเดลได้จากหน้าเว็บ (รัน pipeline เดียวกับ train_chili_classifier.py แบบ background)
    - เลือก/สลับโมเดลที่เคยเทรนไว้ได้ทุกเมื่อ (รวมถึง chili_model.tflite เดิมที่มีอยู่ก่อนแล้ว)

วิธีใช้:
    1. (ถ้ามี) เอาไฟล์ chili_model.tflite มาวางไว้ในโฟลเดอร์เดียวกับ app.py — ถ้าไม่มีก็เทรนใหม่จากหน้าเว็บได้เลย
    2. pip install -r requirements.txt --break-system-packages
    3. python app.py
    4. เปิดเบราว์เซอร์ไปที่ http://<IP เครื่องนี้>:5000

ESP32-CAM ต้อง POST ภาพ (Content-Type: image/jpeg) ไปที่:
    http://<IP เครื่องนี้>:5000/upload
"""

import os
import io
import json
import time
import random
import uuid
import shutil
import threading
from datetime import datetime
from collections import deque

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory
import tensorflow as tf

import train_web  # โมดูลเทรนโมเดล (ดู train_web.py) — ใช้โดย /api/train


# ---------------------------------------------------------------------------
# ตั้งค่า
# ---------------------------------------------------------------------------
LEGACY_MODEL_PATH = "chili_model.tflite"  # โมเดลเดิมที่มีอยู่ก่อนแล้วในโปรเจกต์ (นอกระบบ versioning)
IMG_SIZE = (224, 224)
CLASS_NAMES = ["unhealthy", "healthy"]   # ค่าเริ่มต้น/ของโมเดลเดิม (โมเดลใหม่แต่ละตัวมีของตัวเองใน metadata.json)
HISTORY_DIR = os.path.join("static", "history")
LATEST_IMAGE_PATH = os.path.join("static", "latest.jpg")
MAX_HISTORY = 50                          # เก็บผลลัพธ์ล่าสุดไว้กี่รายการในหน้าจอ

DATASET_DIR = "dataset"
DATASET_CLASSES = ("healthy", "unhealthy", "green")  # green = พริกสีเขียว/ยังไม่สุก
MODELS_DIR = "models"
ACTIVE_MODEL_FILE = os.path.join(MODELS_DIR, "active_model.json")
VAL_SPLIT_RATIO = 0.15   # สัดส่วนรูปใหม่ที่จะสุ่มไปลง val แทน train (85/15)

os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
for split in ("train", "val"):
    for cls in DATASET_CLASSES:
        os.makedirs(os.path.join(DATASET_DIR, split, cls), exist_ok=True)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# ตัวจัดการโมเดล — รองรับหลายเวอร์ชัน เลือกใช้งาน/สลับได้จากหน้าเว็บ
# โมเดลที่เทรนผ่านหน้าเว็บจะถูกเก็บที่ models/<model_id>/ (model.tflite + metadata.json)
# ส่วนโมเดลเดิม (chili_model.tflite ที่ root) จะถูกจดทะเบียนเป็นโมเดลชื่อ "legacy" ให้เลือกใช้ได้เหมือนกัน
# ---------------------------------------------------------------------------
current_model = {
    "model_id": None,
    "interpreter": None,
    "input_details": None,
    "output_details": None,
    "img_size": IMG_SIZE,
    "classes": CLASS_NAMES,
}


def list_models():
    entries = []
    if os.path.exists(LEGACY_MODEL_PATH):
        entries.append({
            "model_id": "legacy",
            "path": LEGACY_MODEL_PATH,
            "created_at": datetime.fromtimestamp(os.path.getmtime(LEGACY_MODEL_PATH)).isoformat(),
            "classes": CLASS_NAMES,
            "img_size": list(IMG_SIZE),
            "val_accuracy": None,
            "num_train_images": None,
            "num_val_images": None,
            "note": "โมเดลเดิมจาก chili_model.tflite (นอกระบบ versioning)",
        })
    for name in sorted(os.listdir(MODELS_DIR)):
        meta_path = os.path.join(MODELS_DIR, name, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["model_id"] = name
            meta["path"] = os.path.join(MODELS_DIR, name, "model.tflite")
            entries.append(meta)
    entries.sort(key=lambda m: m["created_at"], reverse=True)
    return entries


def get_saved_active_model_id():
    if os.path.exists(ACTIVE_MODEL_FILE):
        with open(ACTIVE_MODEL_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("model_id")
    return None


def load_model(model_id):
    entries = {m["model_id"]: m for m in list_models()}
    if model_id not in entries:
        raise ValueError("ไม่พบโมเดลนี้")
    meta = entries[model_id]
    interpreter = tf.lite.Interpreter(model_path=meta["path"])
    interpreter.allocate_tensors()
    current_model.update({
        "model_id": model_id,
        "interpreter": interpreter,
        "input_details": interpreter.get_input_details(),
        "output_details": interpreter.get_output_details(),
        "img_size": tuple(meta["img_size"]),
        "classes": meta["classes"],
    })
    with open(ACTIVE_MODEL_FILE, "w", encoding="utf-8") as f:
        json.dump({"model_id": model_id}, f)


def init_model():
    models = list_models()
    if not models:
        print("⚠ ไม่พบโมเดลใดๆ เลย — ระบบจะยังทำนายไม่ได้จนกว่าจะเทรนโมเดลใหม่หรือมี chili_model.tflite")
        return
    active_id = get_saved_active_model_id()
    if active_id not in {m["model_id"] for m in models}:
        active_id = models[0]["model_id"]  # ตัวล่าสุด (list ถูก sort โดย created_at ใหม่สุดก่อนแล้ว)
    load_model(active_id)


init_model()

# ---------------------------------------------------------------------------
# สถานะที่เก็บไว้ในหน่วยความจำ (สำหรับ dashboard ดึงไปแสดง)
# ---------------------------------------------------------------------------
state = {
    "latest_result": None,      # {"label": ..., "confidence": ..., "timestamp": ...}
    "total_count": 0,
    "class_counts": {},          # {"healthy": 12, "unhealthy": 3, "green": 5, ...} เพิ่ม key อัตโนมัติตามคลาสที่เจอจริง
    "history": deque(maxlen=MAX_HISTORY),   # รายการล่าสุดอยู่หน้าสุด
    "connected": False,          # ESP32-CAM เคยส่งภาพเข้ามาหรือยัง
    "last_seen": None,
    "esp32_ip": None,            # IP ของ ESP32-CAM สำหรับดึงวิดีโอสดมาแสดง
    "sensor": {                  # ค่าล่าสุดจากเซ็นเซอร์ DHT11 (ส่งเข้ามาทาง /api/sensor)
        "temperature": None,
        "humidity": None,
        "timestamp": None,
    },
}


MIN_OBJECT_RATIO = 0.06   # สัดส่วนพิกเซลสีพริกขั้นต่ำ (6%) ถึงจะถือว่ามีพริกอยู่ในเฟรม ปรับได้ตามสภาพจริง


def has_chili(image: Image.Image) -> bool:
    """เช็คว่ามีพริกอยู่ในภาพจริงไหม โดยดูสัดส่วนพิกเซลที่มีสีอิ่มตัวสูง
    (พริกมักมีสีแดง/เขียว/เหลืองสด ส่วนสายพาน/พื้นหลังมักเป็นสีทึบ ไม่ฉูดฉาด)"""
    hsv = np.array(image.convert("HSV").resize((160, 120)))
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    # นับพิกเซลที่ทั้งอิ่มตัว (สีสด) และไม่มืดเกินไป (ไม่ใช่เงา)
    object_mask = (saturation > 70) & (value > 40)
    object_ratio = object_mask.mean()
    return object_ratio >= MIN_OBJECT_RATIO


def predict(image: Image.Image):
    """รันโมเดล TFLite ที่กำลังใช้งานอยู่กับภาพที่รับเข้ามา คืนค่า (label, confidence)
    คืน (None, None) ถ้ายังไม่มีโมเดลถูกเลือกใช้งาน (เช่นยังไม่เคยเทรน/ไม่มี chili_model.tflite)

    รองรับทั้งโมเดลเก่า (2 คลาส, เอาต์พุตเดียวแบบ sigmoid) และโมเดลใหม่ (3 คลาสขึ้นไป,
    เอาต์พุตหลายค่าแบบ softmax) โดยดูจากรูปร่างของเอาต์พุตอัตโนมัติ ไม่ต้องตั้งค่าแยก"""
    if current_model["interpreter"] is None:
        return None, None

    img = image.convert("RGB").resize(current_model["img_size"])
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    interpreter = current_model["interpreter"]
    interpreter.set_tensor(current_model["input_details"][0]["index"], arr)
    interpreter.invoke()
    output = interpreter.get_tensor(current_model["output_details"][0]["index"])[0]

    classes = current_model["classes"]

    if output.shape[0] == 1:
        # โมเดลแบบ binary (sigmoid เอาต์พุตเดียว) — ใช้กับโมเดลเก่า/2 คลาสที่เทรนไว้ก่อนหน้านี้
        score = float(output[0])
        label = classes[1] if score > 0.5 else classes[0]
        confidence = score if score > 0.5 else 1 - score
    else:
        # โมเดลหลายคลาส (softmax) — เลือกคลาสที่ค่าความน่าจะเป็นสูงสุด
        idx = int(np.argmax(output))
        label = classes[idx]
        confidence = float(output[idx])

    return label, confidence


@app.route("/upload", methods=["POST"])
def upload():
    """endpoint ที่รับภาพเข้ามา ใช้ได้ 2 ทาง:
    1. ESP32-CAM ส่งภาพจริงมา (raw bytes, Content-Type: image/jpeg)
    2. ทดสอบจากหน้าเว็บโดยอัปโหลดภาพเอง (แนบ header X-Source: manual-test)
       กรณีนี้จะยังรันโมเดลตรวจให้ปกติ แต่จะไม่ไปทำให้สถานะ "เชื่อมต่อ ESP32-CAM" เปลี่ยนเป็นเชื่อมต่อแล้ว
       เพื่อไม่ให้สับสนกับการเชื่อมต่อฮาร์ดแวร์จริง"""
    is_test = request.headers.get("X-Source") == "manual-test"

    image_bytes = request.get_data()
    if not image_bytes:
        return jsonify({"error": "ไม่พบข้อมูลภาพ"}), 400

    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception:
        return jsonify({"error": "ไฟล์ภาพเสียหายหรือไม่ใช่ไฟล์ภาพ"}), 400

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # บันทึกภาพล่าสุดไว้แสดงบนแดชบอร์ดเสมอ ไม่ว่าจะมีพริกหรือไม่
    image.convert("RGB").save(LATEST_IMAGE_PATH, "JPEG")

    if not has_chili(image):
        # ไม่พบพริกในเฟรม ข้ามการจำแนกโรค ไม่ต้องเดา
        state["latest_result"] = {
            "label": "no_object",
            "confidence": 0,
            "timestamp": timestamp,
        }
        if not is_test:
            state["connected"] = True
            state["last_seen"] = timestamp
            state["esp32_ip"] = request.headers.get("X-Esp32-Ip", state["esp32_ip"])
        return jsonify({"label": "no_object", "confidence": 0})

    label, confidence = predict(image)

    if label is None:
        # ยังไม่มีโมเดลให้ใช้งาน (ยังไม่เคยเทรน/ยังไม่เลือก) — เก็บภาพไว้เฉยๆ ไม่ต้องทำนาย
        state["latest_result"] = {"label": "no_model", "confidence": 0, "timestamp": timestamp}
        if not is_test:
            state["connected"] = True
            state["last_seen"] = timestamp
            state["esp32_ip"] = request.headers.get("X-Esp32-Ip", state["esp32_ip"])
        return jsonify({"label": "no_model", "confidence": 0})

    # บันทึกภาพลง history (ตั้งชื่อด้วย timestamp กันชื่อซ้ำ)
    history_filename = f"{int(time.time()*1000)}_{label}.jpg"
    image.convert("RGB").save(os.path.join(HISTORY_DIR, history_filename), "JPEG")

    # อัปเดตสถานะ
    state["latest_result"] = {
        "label": label,
        "confidence": round(confidence * 100, 1),
        "timestamp": timestamp,
    }
    state["total_count"] += 1
    state["class_counts"][label] = state["class_counts"].get(label, 0) + 1
    state["history"].appendleft({
        "label": label,
        "confidence": round(confidence * 100, 1),
        "timestamp": timestamp,
        "image": history_filename,
    })
    if not is_test:
        state["connected"] = True
        state["last_seen"] = timestamp
        state["esp32_ip"] = request.headers.get("X-Esp32-Ip", state["esp32_ip"])

    # TODO: ในอนาคตตรงนี้คือจุดที่ควรสั่งงานเซอร์โว/กลไกคัดแยกตามผล label
    # เช่น ส่งคำสั่งกลับไปที่ ESP32 หรือควบคุม GPIO ของเซิร์ฟเวอร์เอง

    return jsonify({"label": label, "confidence": confidence})


@app.route("/api/status")
def api_status():
    """endpoint ให้หน้าเว็บ dashboard ดึงข้อมูลไปแสดงผล (polling ทุกไม่กี่วินาที)"""
    return jsonify({
        "latest_result": state["latest_result"],
        "total_count": state["total_count"],
        "class_counts": state["class_counts"],
        "history": list(state["history"]),
        "connected": state["connected"],
        "last_seen": state["last_seen"],
        "esp32_ip": state["esp32_ip"],
        "sensor": state["sensor"],
    })


@app.route("/api/sensor", methods=["POST"])
def api_sensor():
    """endpoint ที่ ESP32 (ตัวคุมมอเตอร์ หรือตัวใดก็ได้ที่ต่อ DHT11) ส่งค่าอุณหภูมิ/ความชื้นเข้ามา
    ส่งเป็น JSON: {"temperature": 29.5, "humidity": 65.2}"""
    data = request.get_json(force=True, silent=True) or {}
    temperature = data.get("temperature")
    humidity = data.get("humidity")
    if temperature is None or humidity is None:
        return jsonify({"ok": False, "error": "ต้องส่ง temperature และ humidity มาด้วย"}), 400

    state["sensor"] = {
        "temperature": round(float(temperature), 1),
        "humidity": round(float(humidity), 1),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return jsonify({"ok": True})


@app.route("/static/history/<path:filename>")
def history_image(filename):
    return send_from_directory(HISTORY_DIR, filename)


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


# =============================================================================
# ส่วนที่เพิ่มใหม่: จัดการ dataset จากหน้าเว็บ + เทรนโมเดล + เลือกโมเดล
# =============================================================================

# ---------------------------------------------------------------------------
# Dataset: บันทึกภาพล่าสุดที่ได้จากกล้อง (static/latest.jpg) หรืออัปโหลดไฟล์เดิม
# ลงโฟลเดอร์ dataset/train|val/healthy|unhealthy/ ตามโครงสร้างที่ train_chili_classifier.py ใช้อยู่แล้ว
# ---------------------------------------------------------------------------

def save_to_dataset(label, img_bytes):
    if label not in DATASET_CLASSES:
        raise ValueError("label ไม่ถูกต้อง")
    split = "val" if random.random() < VAL_SPLIT_RATIO else "train"
    folder = os.path.join(DATASET_DIR, split, label)
    os.makedirs(folder, exist_ok=True)
    fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.jpg"
    with open(os.path.join(folder, fname), "wb") as f:
        f.write(img_bytes)
    return split, fname


def dataset_stats():
    stats = {"train": {}, "val": {}}
    for split in ("train", "val"):
        for label in DATASET_CLASSES:
            folder = os.path.join(DATASET_DIR, split, label)
            n = 0
            if os.path.isdir(folder):
                n = len([f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            stats[split][label] = n
    return stats


@app.route("/api/dataset/save", methods=["POST"])
def api_dataset_save():
    """บันทึกภาพล่าสุดที่ ESP32-CAM ส่งเข้ามา (static/latest.jpg) ลง dataset ตาม label ที่เลือก
    หมายเหตุ: เป็นภาพนิ่งล่าสุดตามรอบส่งของ ESP32 (ทุก ๆ CAPTURE_INTERVAL_MS) ไม่ใช่เฟรมสดจากวิดีโอ
    ซึ่งอาจช้ากว่าภาพในวิดีโอสดเล็กน้อย"""
    data = request.get_json(force=True)
    label = data.get("label")
    if label not in DATASET_CLASSES:
        return jsonify({"ok": False, "error": "label ไม่ถูกต้อง"}), 400
    if not os.path.exists(LATEST_IMAGE_PATH):
        return jsonify({"ok": False, "error": "ยังไม่มีภาพจากกล้องเข้ามาเลย รอ ESP32-CAM ส่งภาพก่อน"}), 400
    with open(LATEST_IMAGE_PATH, "rb") as f:
        img_bytes = f.read()
    split, fname = save_to_dataset(label, img_bytes)
    return jsonify({"ok": True, "split": split, "filename": fname, "stats": dataset_stats()})


@app.route("/api/dataset/upload", methods=["POST"])
def api_dataset_upload():
    """อัปโหลดรูปที่มีอยู่แล้วเป็นชุด (เช่นถ่ายด้วยมือถือ) เข้า dataset"""
    label = request.form.get("label")
    if label not in DATASET_CLASSES:
        return jsonify({"ok": False, "error": "label ไม่ถูกต้อง"}), 400
    saved = []
    for f in request.files.getlist("files"):
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in (".jpg", ".jpeg", ".png"):
            continue
        split, fname = save_to_dataset(label, f.read())
        saved.append(f"{split}/{fname}")
    return jsonify({"ok": True, "saved": saved, "stats": dataset_stats()})


@app.route("/api/dataset/stats")
def api_dataset_stats():
    return jsonify(dataset_stats())


@app.route("/api/dataset/images")
def api_dataset_images():
    split = request.args.get("split", "train")
    label = request.args.get("label", "healthy")
    if split not in ("train", "val") or label not in DATASET_CLASSES:
        return jsonify({"error": "invalid split/label"}), 400
    folder = os.path.join(DATASET_DIR, split, label)
    files = []
    if os.path.isdir(folder):
        files = sorted(
            [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))],
            reverse=True,
        )
    return jsonify({"split": split, "label": label, "files": files})


@app.route("/dataset_files/<split>/<label>/<path:filename>")
def dataset_file(split, label, filename):
    if split not in ("train", "val") or label not in DATASET_CLASSES:
        return "not found", 404
    return send_from_directory(os.path.join(DATASET_DIR, split, label), filename)


@app.route("/api/dataset/delete", methods=["POST"])
def api_dataset_delete():
    data = request.get_json(force=True)
    split, label, filename = data.get("split"), data.get("label"), data.get("filename")
    if split not in ("train", "val") or label not in DATASET_CLASSES or not filename:
        return jsonify({"ok": False, "error": "invalid request"}), 400
    path = os.path.join(DATASET_DIR, split, label, filename)
    if os.path.exists(path):
        os.remove(path)
    return jsonify({"ok": True, "stats": dataset_stats()})


# ---------------------------------------------------------------------------
# เทรนโมเดล: รัน pipeline เดียวกับ train_chili_classifier.py แบบ background
# (ดู train_web.py) พร้อมรายงานความคืบหน้าให้หน้าเว็บ poll ทุก ๆ 1-2 วินาที
# ---------------------------------------------------------------------------

TRAIN_LOCK = threading.Lock()
train_state = {
    "status": "idle",       # idle | running | completed | error
    "phase": "",
    "epoch": 0,
    "total_epochs": 0,
    "accuracy": None,
    "val_accuracy": None,
    "loss": None,
    "val_loss": None,
    "message": "",
    "model_id": None,
}


def update_train_state(**kwargs):
    with TRAIN_LOCK:
        train_state.update(kwargs)


def get_train_state():
    with TRAIN_LOCK:
        return dict(train_state)


def run_training_job(epochs_stage1, epochs_stage2):
    try:
        update_train_state(
            status="running", phase="กำลังเตรียมข้อมูล", epoch=0,
            total_epochs=epochs_stage1 + epochs_stage2, accuracy=None, val_accuracy=None,
            loss=None, val_loss=None, message="", model_id=None,
        )

        def on_progress(phase, epoch, total_epochs, logs):
            update_train_state(
                phase=phase, epoch=epoch, total_epochs=total_epochs,
                accuracy=logs.get("accuracy"), val_accuracy=logs.get("val_accuracy"),
                loss=logs.get("loss"), val_loss=logs.get("val_loss"),
            )

        model_id, meta = train_web.run_training(epochs_stage1, epochs_stage2, on_progress)

        # ใช้งานโมเดลที่เพิ่งเทรนเสร็จทันที (ยังสลับกลับไปโมเดลอื่นได้ทีหลังจากหน้าเว็บ)
        load_model(model_id)

        update_train_state(
            status="completed",
            phase="เสร็จสิ้น",
            model_id=model_id,
            message=f"เทรนสำเร็จ ({model_id}) — val_accuracy {meta['val_accuracy']*100:.1f}%",
        )
    except Exception as e:  # noqa: BLE001
        update_train_state(status="error", message=str(e))


@app.route("/api/train", methods=["POST"])
def api_train():
    st = get_train_state()
    if st["status"] == "running":
        return jsonify({"ok": False, "error": "กำลังเทรนอยู่แล้ว กรุณารอให้เสร็จก่อน"}), 409

    stats = dataset_stats()
    for split in ("train", "val"):
        for label in DATASET_CLASSES:
            if stats[split][label] == 0:
                return jsonify({
                    "ok": False,
                    "error": f"ยังไม่มีรูปใน dataset/{split}/{label}/ กรุณาเพิ่มรูปก่อนเทรน",
                }), 400

    data = request.get_json(silent=True) or {}
    epochs_stage1 = int(data.get("epochs_stage1", 15))
    epochs_stage2 = int(data.get("epochs_stage2", 10))

    t = threading.Thread(target=run_training_job, args=(epochs_stage1, epochs_stage2), daemon=True)
    t.start()
    return jsonify({"ok": True})


@app.route("/api/train/status")
def api_train_status():
    return jsonify(get_train_state())


# ---------------------------------------------------------------------------
# เลือกใช้งานโมเดล
# ---------------------------------------------------------------------------

@app.route("/api/models")
def api_models():
    return jsonify({"models": list_models(), "active_model_id": current_model["model_id"]})


@app.route("/api/models/select", methods=["POST"])
def api_models_select():
    data = request.get_json(force=True)
    model_id = data.get("model_id")
    try:
        load_model(model_id)
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 404
    return jsonify({"ok": True, "active_model_id": model_id})


@app.route("/api/models/delete", methods=["POST"])
def api_models_delete():
    data = request.get_json(force=True)
    model_id = data.get("model_id")
    if model_id in (None, "legacy"):
        return jsonify({"ok": False, "error": "ลบโมเดลนี้ไม่ได้"}), 400
    if model_id == current_model["model_id"]:
        return jsonify({"ok": False, "error": "ลบโมเดลที่กำลังใช้งานอยู่ไม่ได้ กรุณาเลือกโมเดลอื่นก่อน"}), 400
    path = os.path.join(MODELS_DIR, model_id)
    if not os.path.isdir(path):
        return jsonify({"ok": False, "error": "ไม่พบโมเดลนี้"}), 404
    shutil.rmtree(path)
    return jsonify({"ok": True})


if __name__ == "__main__":
    # host="0.0.0.0" เพื่อให้ ESP32-CAM ในวงแลนเดียวกันเชื่อมต่อเข้ามาได้
    app.run(host="0.0.0.0", port=5000, debug=True)