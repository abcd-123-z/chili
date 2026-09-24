"""
train_chili_classifier.py
==========================
เทรนโมเดลจำแนกพริก "ปกติ" vs "เป็นโรค" ด้วย Transfer Learning จาก MobileNetV2
แล้วแปลงเป็น TensorFlow Lite เพื่อรันบน Raspberry Pi / เครื่องประมวลผลได้เร็ว
 
วิธีใช้:
    1. เตรียม dataset ตามโครงสร้าง:
         dataset/train/normal/*.jpg
         dataset/train/diseased/*.jpg
         dataset/val/normal/*.jpg
         dataset/val/diseased/*.jpg
    2. ติดตั้ง dependencies:
         pip install tensorflow pillow numpy --break-system-packages
    3. รัน:
         python3 train_chili_classifier.py
 
ผลลัพธ์ที่ได้:
    - chili_model.h5          (โมเดลเต็ม สำหรับใช้บนเครื่องแรง)
    - chili_model.tflite       (โมเดลย่อ สำหรับ Raspberry Pi / edge device)
    - training_history.png     (กราฟผลการเทรน accuracy/loss)
"""
 
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

 
# ---------------------------------------------------------------------------
# 1) การตั้งค่าหลัก (ปรับได้ตามข้อมูลของคุณ)
# ---------------------------------------------------------------------------
DATASET_DIR = "dataset"          # โฟลเดอร์ dataset ตามโครงสร้างด้านบน
IMG_SIZE = (224, 224)            # ขนาดภาพที่ MobileNetV2 ใช้
BATCH_SIZE = 16
EPOCHS_STAGE1 = 15               # เทรนเฉพาะ head (freeze backbone)
EPOCHS_STAGE2 = 10               # fine-tune บางส่วนของ backbone
CLASS_NAMES = ["healthy", "unhealthy", "green"]  # ต้องตรงกับชื่อโฟลเดอร์ใน dataset/train และ dataset/val (เพิ่ม green = พริกสีเขียว/ยังไม่สุก)
 
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
 
 
def check_dataset():
    """ตรวจสอบว่ามีข้อมูลครบก่อนเริ่มเทรน เพื่อไม่ให้พังกลางทาง"""
    for split_dir in [TRAIN_DIR, VAL_DIR]:
        for cls in CLASS_NAMES:
            path = os.path.join(split_dir, cls)
            if not os.path.isdir(path):
                raise FileNotFoundError(
                    f"ไม่พบโฟลเดอร์ {path}\n"
                    f"กรุณาจัดเรียง dataset ตามโครงสร้างที่ระบุไว้ด้านบนของไฟล์นี้"
                )
            n_images = len([f for f in os.listdir(path)
                             if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            print(f"  {path}: {n_images} ภาพ")
            if n_images < 20:
                print(f"  ⚠ คำเตือน: {path} มีภาพน้อยมาก ({n_images}) "
                      f"แนะนำอย่างน้อย 100-200 ภาพต่อคลาสเพื่อความแม่นยำที่ใช้งานได้จริง")
 
 
def build_data_generators():
    """สร้าง data pipeline พร้อม augmentation สำหรับชุด train"""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=25,          # พริกบนสายพานอาจหมุนได้ทุกมุม
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],  # จำลองสภาพแสงที่ต่างกัน
        fill_mode="nearest",
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)
 
    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=True,
    )
    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
    )
    return train_gen, val_gen
 
 
def build_model():
    """สร้างโมเดลจาก MobileNetV2 (transfer learning) + custom classification head"""
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # เฟสแรก freeze backbone ทั้งหมดก่อน
 
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(len(CLASS_NAMES), activation="softmax"),  # หลายคลาส: 1 นิวรอนต่อคลาส
    ])
    return model, base_model
 
 
def plot_history(history1, history2, save_path="training_history.png"):
    acc = history1.history["accuracy"] + history2.history["accuracy"]
    val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]
    loss = history1.history["loss"] + history2.history["loss"]
    val_loss = history1.history["val_loss"] + history2.history["val_loss"]
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(acc, label="train")
    axes[0].plot(val_acc, label="validation")
    axes[0].set_title("Accuracy")
    axes[0].legend()
 
    axes[1].plot(loss, label="train")
    axes[1].plot(val_loss, label="validation")
    axes[1].set_title("Loss")
    axes[1].legend()
 
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"บันทึกกราฟผลการเทรนที่ {save_path}")
 
 
def convert_to_tflite(model, output_path="chili_model.tflite"):
    """แปลงเป็น TFLite พร้อม quantization เพื่อให้รันเร็วขึ้นบน Raspberry Pi"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    size_kb = len(tflite_model) / 1024
    print(f"บันทึกโมเดล TFLite ที่ {output_path} (ขนาด {size_kb:.1f} KB)")
 
 
def main():
    print("=== ตรวจสอบ dataset ===")
    check_dataset()
 
    print("\n=== เตรียม data pipeline ===")
    train_gen, val_gen = build_data_generators()
    print(f"class_indices: {train_gen.class_indices}")
 
    print("\n=== สร้างโมเดล (MobileNetV2 transfer learning) ===")
    model, base_model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
 
    print("\n=== เฟส 1: เทรนเฉพาะ classification head ===")
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_STAGE1,
    )
 
    print("\n=== เฟส 2: Fine-tune บางเลเยอร์บนสุดของ backbone ===")
    base_model.trainable = True
    # freeze เลเยอร์ล่างส่วนใหญ่ไว้ ปลดเฉพาะ 30 เลเยอร์บนสุดให้ fine-tune
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    # สำคัญ: freeze เลเยอร์ BatchNormalization ไว้เสมอแม้อยู่ใน 30 เลเยอร์บนสุด
    # เพราะถ้าปล่อยให้เทรน สถิติ mean/variance ที่เรียนมาจาก ImageNet จะถูกเขียนทับ
    # ด้วย batch เล็ก ๆ ของเรา ทำให้ loss พุ่งกระโดดตอนเริ่ม fine-tune (เป็นสาเหตุที่พบบ่อย)
    for layer in base_model.layers[-30:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # lr ต่ำลงมากตอน fine-tune
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_STAGE2,
        callbacks=[
            # เก็บน้ำหนักของ epoch ที่ val_loss ดีที่สุดไว้ใช้จริง แทนที่จะใช้ epoch สุดท้าย
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=6, restore_best_weights=True
            ),
        ],
    )
 
    print("\n=== บันทึกผลลัพธ์ ===")
    model.save("chili_model.h5")
    print("บันทึกโมเดลเต็มที่ chili_model.h5")
 
    convert_to_tflite(model)
    plot_history(history1, history2)
 
    val_loss, val_acc = model.evaluate(val_gen)
    print(f"\nความแม่นยำสุดท้ายบนชุด validation: {val_acc * 100:.2f}%")
 
 
if __name__ == "__main__":
    main()