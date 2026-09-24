"""
train_web.py
=============
โมดูลเทรนโมเดลสำหรับเรียกใช้จาก app.py (ปุ่ม "เทรนโมเดล" บนหน้าเว็บ)
ใช้ pipeline เดียวกับ train_chili_classifier.py (MobileNetV2 transfer learning
+ fine-tune 2 เฟส) แต่ทำเป็นฟังก์ชันที่ app.py เรียกแบบ background thread ได้
และรายงานความคืบหน้าทีละ epoch กลับไปให้หน้าเว็บ poll ผ่าน on_progress callback

app.py เรียกใช้แบบนี้:
    model_id, meta = train_web.run_training(epochs_stage1, epochs_stage2, on_progress)

โครงสร้างที่คาดหวัง (ต้องตรงกับ dataset ที่ app.py จัดการให้):
    dataset/train/healthy/*.jpg
    dataset/train/unhealthy/*.jpg
    dataset/train/green/*.jpg
    dataset/val/healthy/*.jpg
    dataset/val/unhealthy/*.jpg
    dataset/val/green/*.jpg

ผลลัพธ์: models/<model_id>/model.tflite + models/<model_id>/metadata.json
"""

import os
import json
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ---------------------------------------------------------------------------
# ค่าคงที่ — ต้อง "ตรงกัน" กับที่ app.py ใช้ (path, ขนาดภาพ)
# CLASS_NAMES คือลำดับ index ที่โมเดล softmax จะทำนายออกมา (index 0/1/2 ตามลำดับนี้)
# app.py ฝั่งทำนาย (predict()) อ่านลำดับนี้จาก metadata.json ของแต่ละโมเดลโดยตรง
# จึงไม่จำเป็นต้องตรงกับลำดับใน CLASS_NAMES ของไฟล์นี้เป๊ะ ๆ แต่ต้องมีครบทุกคลาส
# ที่มีโฟลเดอร์อยู่จริงใน dataset/
# ---------------------------------------------------------------------------
DATASET_DIR = "dataset"
MODELS_DIR = "models"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["healthy", "unhealthy", "green"]  # ต้องตรงกับโฟลเดอร์ dataset/train(หรือval)/<ชื่อ>


def _count_images(split, label):
    folder = os.path.join(DATASET_DIR, split, label)
    if not os.path.isdir(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))])


def _dataset_counts():
    return {
        split: {label: _count_images(split, label) for label in CLASS_NAMES}
        for split in ("train", "val")
    }


def _pick_batch_size(counts):
    """เลือก batch size ให้เหมาะกับจำนวนภาพที่มีจริง (dataset จากหน้าเว็บอาจมีน้อย
    ตอนเริ่มต้น) ป้องกัน error จาก batch size ที่ใหญ่กว่าจำนวนภาพทั้งหมด"""
    total_train = sum(counts["train"].values())
    total_val = sum(counts["val"].values())
    return max(1, min(16, total_train, total_val))


def _build_data_generators(batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode="nearest",
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, "train"),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=True,
    )
    val_gen = val_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, "val"),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
    )
    return train_gen, val_gen


def _build_model():
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(len(CLASS_NAMES), activation="softmax"),  # หลายคลาส: 1 นิวรอนต่อคลาส
    ])
    return model, base_model


class _ProgressCallback(tf.keras.callbacks.Callback):
    """เรียก on_progress(phase, epoch, total_epochs, logs) ทุกครั้งที่จบ 1 epoch
    เพื่อให้ app.py อัปเดต train_state ให้หน้าเว็บ poll เห็นความคืบหน้าแบบเรียลไทม์"""

    def __init__(self, phase, epoch_offset, total_epochs, on_progress):
        super().__init__()
        self.phase = phase
        self.epoch_offset = epoch_offset
        self.total_epochs = total_epochs
        self.on_progress = on_progress

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.on_progress(self.phase, self.epoch_offset + epoch + 1, self.total_epochs, logs)


def _convert_to_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    return converter.convert()


def run_training(epochs_stage1, epochs_stage2, on_progress):
    """เทรนโมเดลใหม่จาก dataset ปัจจุบัน แล้วบันทึกเป็นโมเดลเวอร์ชันใหม่

    Args:
        epochs_stage1: จำนวน epoch เฟสเทรนเฉพาะ head (freeze backbone)
        epochs_stage2: จำนวน epoch เฟส fine-tune บางส่วนของ backbone
        on_progress: callback(phase: str, epoch: int, total_epochs: int, logs: dict)
                     เรียกทุกจบ epoch เพื่อรายงานความคืบหน้า

    Returns:
        (model_id, meta) โดย meta คือ dict เดียวกับที่จะเขียนลง metadata.json
    """
    total_epochs = epochs_stage1 + epochs_stage2

    counts = _dataset_counts()
    for split in ("train", "val"):
        for label in CLASS_NAMES:
            if counts[split][label] == 0:
                raise ValueError(f"ไม่มีรูปใน dataset/{split}/{label}/ ไม่สามารถเทรนได้")

    batch_size = _pick_batch_size(counts)
    train_gen, val_gen = _build_data_generators(batch_size)

    model, base_model = _build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # เฟส 1: เทรนเฉพาะ classification head
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs_stage1,
        callbacks=[_ProgressCallback("เทรน head (freeze backbone)", 0, total_epochs, on_progress)],
        verbose=0,
    )

    # เฟส 2: fine-tune เลเยอร์บนสุดของ backbone (30 เลเยอร์สุดท้าย)
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    # สำคัญ: freeze เลเยอร์ BatchNormalization ไว้เสมอแม้อยู่ใน 30 เลเยอร์บนสุด
    # เพราะถ้าปล่อยให้เทรน สถิติ mean/variance ที่เรียนมาจาก ImageNet (ภาพนับล้าน)
    # จะถูกเขียนทับด้วย batch เล็ก ๆ ของเรา ทำให้ loss พุ่งกระโดดและโมเดลไม่เสถียร
    for layer in base_model.layers[-30:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs_stage2,
        callbacks=[
            _ProgressCallback("fine-tune backbone", epochs_stage1, total_epochs, on_progress),
            # เก็บน้ำหนักของ epoch ที่ val_loss ดีที่สุดไว้ใช้จริง แทนที่จะใช้ epoch สุดท้าย
            # เผื่อโมเดล overfit ในช่วงท้าย (val_loss แย่ลงแม้ accuracy จะยังดูดีอยู่)
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=6, restore_best_weights=True
            ),
        ],
        verbose=0,
    )

    # ประเมินผลสุดท้ายบนชุด validation ทั้งหมด (ให้ค่าที่นิ่งกว่า epoch สุดท้ายเฉยๆ)
    val_loss, val_accuracy = model.evaluate(val_gen, verbose=0)

    # แปลงและบันทึกเป็นโมเดลเวอร์ชันใหม่
    model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_dir = os.path.join(MODELS_DIR, model_id)
    os.makedirs(model_dir, exist_ok=True)

    tflite_bytes = _convert_to_tflite(model)
    with open(os.path.join(model_dir, "model.tflite"), "wb") as f:
        f.write(tflite_bytes)

    meta = {
        "created_at": datetime.now().isoformat(),
        "classes": CLASS_NAMES,
        "img_size": list(IMG_SIZE),
        "val_accuracy": float(val_accuracy),
        "val_loss": float(val_loss),
        "num_train_images": sum(counts["train"].values()),
        "num_val_images": sum(counts["val"].values()),
        "epochs_stage1": epochs_stage1,
        "epochs_stage2": epochs_stage2,
        "note": "เทรนจากหน้าเว็บ (MobileNetV2 transfer learning + fine-tune)",
    }
    with open(os.path.join(model_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return model_id, meta