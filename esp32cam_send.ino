/*
  esp32cam_send.ino
  ==================
  ESP32-CAM ถ่ายภาพทุก ๆ CAPTURE_INTERVAL_MS แล้วส่งไปที่เว็บเซิร์ฟเวอร์ (app.py)
  ผ่าน HTTP POST เพื่อให้เซิร์ฟเวอร์รันโมเดล AI ตรวจสอบ

  ก่อนอัปโหลด ต้องแก้ 3 ค่านี้ให้ตรงกับของจริง:
    1. WIFI_SSID / WIFI_PASSWORD
    2. SERVER_URL (IP ของเครื่องที่รัน app.py ในวงแลนเดียวกัน)
    3. เลือกรุ่นบอร์ดกล้องให้ตรงกับที่ใช้ (ค่าเริ่มต้นคือ AI-Thinker ซึ่งเป็นรุ่นที่นิยมที่สุด)

  บอร์ด: AI-Thinker ESP32-CAM
  Arduino IDE > Tools > Board > ESP32 Wrover Module (หรือ AI Thinker ESP32-CAM ถ้ามีในลิสต์)
  ต้อง flash ผ่านขา GPIO0 ต่อ GND ตอนอัปโหลดโค้ด (โหมด flashing) แล้วถอดออกตอนรันจริง
*/

#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>

// ---------------------------------------------------------------------------
// ตั้งค่าที่ต้องแก้
// ---------------------------------------------------------------------------
const char* WIFI_SSID = "ชื่อ WiFi ของคุณ";
const char* WIFI_PASSWORD = "รหัสผ่าน WiFi";
const char* SERVER_URL = "http://192.168.1.100:5000/upload";  // เปลี่ยนเป็น IP เครื่องที่รัน app.py

const unsigned long CAPTURE_INTERVAL_MS = 5000;  // ถ่ายภาพทุก 5 วินาที ปรับได้ตามความเร็วสายพาน

// ---------------------------------------------------------------------------
// ขากล้องสำหรับบอร์ด AI-Thinker ESP32-CAM (ค่ามาตรฐาน ไม่ต้องแก้ถ้าใช้บอร์ดรุ่นนี้)
// ---------------------------------------------------------------------------
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

unsigned long lastCaptureTime = 0;

void setupCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // ถ้ามี PSRAM ใช้ความละเอียดสูงขึ้นและคุณภาพดีขึ้นได้ ไม่มีก็ยังทำงานได้ปกติ
  if (psramFound()) {
    config.frame_size = FRAMESIZE_VGA;   // 640x480 พอเหมาะสำหรับส่งผ่าน WiFi + โมเดล resize เป็น 224x224 อยู่แล้ว
    config.jpeg_quality = 12;            // ยิ่งน้อยยิ่งคุณภาพสูง (ช่วง 10-20 ใช้งานได้ดี)
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_VGA;
    config.jpeg_quality = 15;
    config.fb_count = 1;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("กล้องเริ่มต้นไม่สำเร็จ error 0x%x\n", err);
    while (true) delay(1000);  // หยุดรอถ้ากล้องไม่ทำงาน จะได้เห็น error ชัดเจนใน Serial Monitor
  }
}

void connectWiFi() {
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("กำลังเชื่อมต่อ WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("เชื่อมต่อสำเร็จ IP: ");
  Serial.println(WiFi.localIP());
}

void captureAndSend() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("ถ่ายภาพไม่สำเร็จ");
    return;
  }

  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(SERVER_URL);
    http.addHeader("Content-Type", "image/jpeg");

    int httpResponseCode = http.POST(fb->buf, fb->len);

    if (httpResponseCode > 0) {
      String response = http.getString();
      Serial.printf("ส่งภาพสำเร็จ [%d]: %s\n", httpResponseCode, response.c_str());
    } else {
      Serial.printf("ส่งภาพไม่สำเร็จ error: %s\n", http.errorToString(httpResponseCode).c_str());
    }
    http.end();
  } else {
    Serial.println("WiFi หลุด กำลังเชื่อมต่อใหม่...");
    connectWiFi();
  }

  esp_camera_fb_return(fb);  // คืนหน่วยความจำ frame buffer ทุกครั้งหลังใช้เสร็จ (สำคัญมาก ไม่งั้นหน่วยความจำเต็ม)
}

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(false);

  setupCamera();
  connectWiFi();

  Serial.println("พร้อมทำงาน เริ่มถ่ายภาพและส่งเข้าเซิร์ฟเวอร์");
}

void loop() {
  unsigned long now = millis();
  if (now - lastCaptureTime >= CAPTURE_INTERVAL_MS) {
    lastCaptureTime = now;
    captureAndSend();
  }
}
