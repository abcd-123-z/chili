"""
predict_single_image.py
========================
ทดสอบโมเดล chili_model.tflite กับภาพเดียว เพื่อเช็คว่าโมเดลทำงานถูกต้อง
ก่อนนำไปต่อกับระบบ ESP32-CAM จริง

วิธีใช้:
    python3 predict_single_image.py path/to/image.jpg
"""

import sys
import numpy as np
from PIL import Image
import tensorflow as tf

IMG_SIZE = (224, 224)
CLASS_NAMES = ["healthy", "unhealthy", "green"]  # ต้องตรงลำดับกับตอนเทรน (เพิ่ม green = พริกสีเขียว/ยังไม่สุก)
MODEL_PATH = "chili_model.tflite"


def load_and_preprocess(image_path):
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # เพิ่ม batch dimension


def predict(image_path):
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = load_and_preprocess(image_path)
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])[0]

    if output.shape[0] == 1:
        # โมเดลเก่าแบบ binary (sigmoid เอาต์พุตเดียว)
        score = float(output[0])
        label = CLASS_NAMES[1] if score > 0.5 else CLASS_NAMES[0]
        confidence = score if score > 0.5 else 1 - score
    else:
        # โมเดลใหม่แบบหลายคลาส (softmax)
        idx = int(np.argmax(output))
        label = CLASS_NAMES[idx]
        confidence = float(output[idx])

    print(f"ผลลัพธ์: {label}")
    print(f"ความมั่นใจ: {confidence * 100:.1f}%")
    return label, confidence


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("วิธีใช้: python3 predict_single_image.py path/to/image.jpg")
        sys.exit(1)
    predict(sys.argv[1])