"""
split_dataset.py
==================
สุ่มย้ายภาพบางส่วนจาก dataset/train/<class> ไปไว้ที่ dataset/val/<class>
เพื่อแก้ปัญหา val ว่างเปล่า (ValueError: The PyDataset has length 0)

วิธีใช้:
    python split_dataset.py

ค่าเริ่มต้น: ย้าย 20% ของภาพในแต่ละคลาสจาก train ไป val
ปรับสัดส่วนได้ที่ VAL_RATIO ด้านล่าง
"""

import os
import random
import shutil

DATASET_DIR = "dataset"
CLASS_NAMES = ["healthy", "unhealthy"]
VAL_RATIO = 0.2      # ย้าย 20% ของภาพแต่ละคลาสไป val
RANDOM_SEED = 42      # ทำให้ผลการสุ่มเหมือนเดิมทุกครั้งที่รัน


def split_class(class_name):
    train_dir = os.path.join(DATASET_DIR, "train", class_name)
    val_dir = os.path.join(DATASET_DIR, "val", class_name)
    os.makedirs(val_dir, exist_ok=True)

    images = [f for f in os.listdir(train_dir)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    existing_val = [f for f in os.listdir(val_dir)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if existing_val:
        print(f"  [{class_name}] val มีภาพอยู่แล้ว {len(existing_val)} ภาพ ข้ามการแบ่งคลาสนี้")
        return

    random.seed(RANDOM_SEED)
    random.shuffle(images)

    n_val = max(1, int(len(images) * VAL_RATIO))
    val_images = images[:n_val]

    for fname in val_images:
        src = os.path.join(train_dir, fname)
        dst = os.path.join(val_dir, fname)
        shutil.move(src, dst)

    print(f"  [{class_name}] ย้าย {len(val_images)} ภาพไป val "
          f"(เหลือใน train: {len(images) - len(val_images)} ภาพ)")


def main():
    print(f"=== แบ่ง dataset (val ratio = {VAL_RATIO*100:.0f}%) ===")
    for class_name in CLASS_NAMES:
        split_class(class_name)
    print("\nเสร็จแล้ว! ตอนนี้ dataset/val มีภาพพร้อมใช้เทรนได้เลย")


if __name__ == "__main__":
    main()
