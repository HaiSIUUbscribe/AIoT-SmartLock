# 🔐 Multimodal AIoT Smart Lock
### Thiết bị Nhận dạng Đa Phương thức (Multimodal AI) cho Xác thực Thông minh trong Môi trường IoT/AIoT

---

**Báo cáo Đồ án môn học: Phát triển Ứng dụng IoT**  
**Học viện Công nghệ Bưu chính Viễn thông – Cơ sở TP.HCM**  
**Giảng viên hướng dẫn: Đàm Minh Lịnh**

| Sinh viên | MSSV |
|---|---|
| Nguyễn Văn Hợp | N22DCPT034 |
| Lê Đức Hải | N22DCPT023 |

*TP.HCM, tháng 6 năm 2026*

</div>

---

## 📋 Mục lục

1. [Tổng quan dự án](#1-tổng-quan-dự-án)
2. [Kiến trúc hệ thống](#2-kiến-trúc-hệ-thống)
3. [Yêu cầu phần cứng](#3-yêu-cầu-phần-cứng)
4. [Yêu cầu phần mềm](#4-yêu-cầu-phần-mềm)
5. [Cài đặt và cấu hình](#5-cài-đặt-và-cấu-hình)
6. [Hướng dẫn sử dụng](#6-hướng-dẫn-sử-dụng)
7. [Mô hình AI và huấn luyện](#7-mô-hình-ai-và-huấn-luyện)
8. [Kết quả thực nghiệm](#8-kết-quả-thực-nghiệm)
9. [Đóng góp kỹ thuật nổi bật](#9-đóng-góp-kỹ-thuật-nổi-bật)
10. [Cấu trúc thư mục](#10-cấu-trúc-thư-mục)
11. [Hạn chế và hướng phát triển](#11-hạn-chế-và-hướng-phát-triển)
12. [Tài liệu tham khảo](#12-tài-liệu-tham-khảo)

---

## 1. Tổng quan dự án

### 1.1 Mô tả

**Multimodal AIoT Smart Lock** là hệ thống khóa thông minh xác thực danh tính người dùng thông qua **hai phương thức sinh trắc học kết hợp**: nhận dạng khuôn mặt và xác thực giọng nói. Hệ thống được triển khai trên nền tảng IoT chi phí thấp với tổng chi phí phần cứng **dưới 300.000 VNĐ**, phù hợp cho các ứng dụng nhà thông minh (Smart Home) và kiểm soát truy cập (Access Control).

### 1.2 Điểm nổi bật

- ✅ **Đa phương thức (Multimodal):** Kết hợp Face + Voice, tăng độ bảo mật lên đáng kể so với đơn phương thức
- ✅ **Chống giả mạo (Anti-Spoofing):** Phát hiện tấn công replay audio và ảnh in 2D
- ✅ **Chi phí thấp:** Toàn bộ phần cứng < 300.000 VNĐ (ESP32-CAM + INMP441)
- ✅ **Kiến trúc linh hoạt:** Edge-Capture + Cloud-Inference, dễ mở rộng và cập nhật model
- ✅ **Giám sát real-time:** Giao diện web SSE hiển thị kết quả xác thực tức thì

### 1.3 Từ khóa công nghệ

`Multimodal Authentication` · `Face Recognition` · `Voice Anti-Spoofing` · `ESP32-CAM` · `INMP441` · `Late Fusion` · `ECAPA-TDNN` · `LCNN` · `Transfer Learning` · `Domain Adaptation` · `IoT/AIoT` · `FastAPI` · `Firebase`

---

## 2. Kiến trúc hệ thống

Hệ thống được thiết kế theo mô hình **Edge-Capture + Local Server Inference** gồm 3 tầng:

```
┌─────────────────────────────────────────────────────────────────┐
│                     TẦNG EDGE (ESP32-CAM)                       │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐    │
│  │  Camera      │   │  Microphone  │   │  WiFi 802.11n    │    │
│  │  OV2640      │   │  INMP441     │   │  mDNS: smart-    │    │
│  │  (MJPEG)     │   │  (I2S 16-bit)│   │  lock.local      │    │
│  └──────┬───────┘   └──────┬───────┘   └────────┬─────────┘    │
└─────────┼─────────────────┼────────────────────┼───────────────┘
          │  HTTP / WiFi    │                    │
┌─────────▼─────────────────▼────────────────────▼───────────────┐
│                  TẦNG SERVER (FastAPI Python)                    │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────────┐   │
│  │FaceAuthService│  │VoiceAuthService│  │  AIFusionService  │   │
│  │  dlib ResNet  │  │  ECAPA-TDNN   │  │  Late Fusion MLP  │   │
│  │  128-dim emb  │  │  192-dim emb  │  │  Accept / Reject  │   │
│  │  Face Score   │  │  Voice Score  │  │  Fusion Score     │   │
│  └───────┬───────┘  └──────┬────────┘  └────────┬──────────┘   │
│          │                 │   ┌──────────────┐  │              │
│          │                 │   │VoiceAntiSpoof│  │              │
│          │                 │   │  LCNN+MFM    │  │              │
│          │                 │   │ Anti-Spoof   │  │              │
│          │                 │   │    Score     │  │              │
│          └─────────────────┴───┴──────────────┴──┘              │
│                         SSE Real-time Dashboard                  │
└──────────────────────────────┬──────────────────────────────────┘
                               │ Firebase SDK
┌──────────────────────────────▼──────────────────────────────────┐
│               TẦNG LƯU TRỮ (Firebase Storage)                   │
│         Model Checkpoints (.ckpt) · User Profiles               │
└─────────────────────────────────────────────────────────────────┘
```

### 2.1 Luồng xác thực

```
[Người dùng đứng trước thiết bị]
        │
        ▼
[ESP32-CAM chụp ảnh khuôn mặt]──────┐
[INMP441 ghi âm 3 giây]─────────────┤
        │                            │
        ▼                            ▼
[Gửi ảnh + audio về FastAPI Server via WiFi]
        │
        ├──► FaceAuthService ──► Face Score (0–1)
        │
        ├──► VoiceAuthService ──► Voice Match Score (0–1)
        │
        ├──► VoiceAntiSpoof ──► Anti-Spoof Score (0–1)
        │
        └──► AIFusionService (Late Fusion MLP)
                    │
        ┌───────────▼───────────┐
        │  voice_match ≥ 0.70   │
        │  AND fusion ≥ 0.65    │──► ✅ ACCEPT / ❌ REJECT
        └───────────────────────┘
                    │
        [SSE → Giao diện web real-time]
```

### 2.2 Công thức quyết định

| Thành phần | Công thức | Ngưỡng |
|---|---|---|
| Face Score | `1 − min‖e_query − e_i‖₂` | ≥ 0.50 |
| Voice Match | `(cosine_sim + 1) / 2` | ≥ 0.70 |
| Fusion Score | `ŷ[REAL_CLASS]` (Softmax MLP) | ≥ 0.65 |

---

## 3. Yêu cầu phần cứng

| Linh kiện | Thông số | Vai trò | Ghi chú |
|---|---|---|---|
| **ESP32-CAM** (AI-Thinker) | Xtensa LX6 240MHz, 520KB SRAM, 4MB PSRAM | Vi điều khiển chính + WiFi | Có tích hợp LED Flash |
| **Camera OV2640** | 2MP, JPEG, 320×240 → 1600×1200 | Thu ảnh khuôn mặt | Tích hợp sẵn trên ESP32-CAM |
| **Microphone INMP441** | I2S MEMS, SNR 61dB, 24-bit, 3.3V | Thu âm giọng nói | Cần kết nối I2S |
| **Module Relay MH-FMD** | Kích hoạt mức thấp | Điều khiển khóa điện tử | Tùy chọn |
| **Dây kết nối + nguồn** | 3.3V / 5V | — | Nguồn ổn định cho INMP441 |

### 3.1 Sơ đồ kết nối GPIO

| Thiết bị | Tín hiệu | GPIO ESP32 |
|---|---|---|
| INMP441 | WS (Word Select) | GPIO 15 |
| INMP441 | SCK (Serial Clock) | GPIO 14 |
| INMP441 | SD (Serial Data) | GPIO 13 |
| INMP441 | L/R Select | GND (chọn kênh trái) |
| MH-FMD Relay | Input điều khiển | GPIO 12 |
| Flash LED | LED tích hợp | GPIO 4 |

> ⚠️ **Lưu ý:** INMP441 hoạt động ở mức 3.3V. Không cấp nguồn 5V trực tiếp.

---

## 4. Yêu cầu phần mềm

### 4.1 Server (Python)

```
Python >= 3.11
FastAPI >= 0.100
uvicorn
torch >= 2.1          # PyTorch
torchaudio
speechbrain           # ECAPA-TDNN pretrained
face_recognition      # dlib ResNet-34
librosa               # Xử lý âm thanh
numpy, scipy
firebase-admin        # Firebase Storage SDK
jinja2                # Template giao diện web
```

### 4.2 Firmware ESP32

- **Arduino IDE** hoặc **PlatformIO**
- Board: `AI-Thinker ESP32-CAM`
- Thư viện: `ESP32 Camera`, `ESP32 I2S`, `ESPmDNS`, `HTTPClient`

### 4.3 Huấn luyện model (tùy chọn)

```
Google Colab / Kaggle (GPU Tesla T4)
PyTorch Lightning >= 2.1
CUDA 11.8
kaggle (API download dataset)
```

---

## 5. Cài đặt và cấu hình

### 5.1 Clone repository

```bash
git clone https://github.com/<username>/AIoT_SmartLock.git
cd AIoT_SmartLock
```

### 5.2 Cài đặt dependencies Python

```bash
pip install -r requirements.txt
```

### 5.3 Cấu hình Firebase

1. Tạo project trên [Firebase Console](https://console.firebase.google.com)
2. Tải file `serviceAccountKey.json` về
3. Đặt vào thư mục `server/config/`
4. Chỉnh sửa `server/config/firebase_config.py`:

```python
FIREBASE_BUCKET = "your-project-id.appspot.com"
SERVICE_ACCOUNT_KEY = "config/serviceAccountKey.json"
```

### 5.4 Nạp firmware ESP32

1. Mở `firmware/esp32_smartlock/esp32_smartlock.ino` bằng Arduino IDE
2. Cập nhật thông tin WiFi:

```cpp
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* SERVER_URL    = "http://YOUR_SERVER_IP:5000";
```

3. Chọn board `AI-Thinker ESP32-CAM`, nạp firmware

### 5.5 Khởi động server

```bash
cd server
uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

Truy cập giao diện quản trị tại: `http://localhost:5000/dashboard`

---

## 6. Hướng dẫn sử dụng

### 6.1 Đăng ký người dùng mới

1. Truy cập `http://localhost:5000/dashboard` → chọn **"Enroll Biometrics"**
2. Nhập tên người dùng
3. Đứng trước ESP32-CAM, hệ thống chụp **3–5 ảnh** khuôn mặt tự động
4. Đọc câu xác nhận (ví dụ: "Tôi là [tên], xác nhận mở khóa") vào INMP441
5. Hệ thống lưu `face_embedding` và `speaker_embedding` vào Gallery

> 💡 **Mẹo:** Đăng ký ở nhiều góc nhìn và điều kiện ánh sáng khác nhau để tăng độ chính xác Face Score.

### 6.2 Xác thực (mở khóa)

Hệ thống hoạt động **tự động** sau khi khởi động:

```
1. Camera giám sát liên tục (mỗi 15 frame ≈ 0.5 giây)
2. Phát hiện khuôn mặt → kích hoạt chu kỳ xác thực
3. Chụp ảnh + ghi âm 3 giây đồng thời
4. Gửi dữ liệu về server → AI inference
5. Kết quả hiển thị real-time trên dashboard
6. Relay mở khóa nếu ACCEPT (Fusion Score ≥ 0.65)
```

### 6.3 Giám sát real-time

Dashboard tại `/dashboard` hiển thị:

| Thông số | Ý nghĩa |
|---|---|
| **Face Identity** | Điểm khớp khuôn mặt (0–1) |
| **Voice Match** | Điểm khớp giọng nói (0–1) |
| **AI Fusion Live** | Điểm quyết định tổng hợp (0–1) |
| **System Status** | ACCEPT / REJECT + timestamp |

### 6.4 Cập nhật model tự động

Khi có checkpoint mới upload lên Firebase Storage, server tự đồng bộ qua:

```python
sync_cloud_model()   # Gọi tự động khi khởi động server
```

Không cần nạp lại firmware ESP32.

---

## 7. Mô hình AI và huấn luyện

### 7.1 Tổng quan các model

| Model | Kiến trúc | Nhiệm vụ | Tham số |
|---|---|---|---|
| **FaceEncoder** | MobileNetV3-Large + ArcFace | Trích xuất face embedding 512-dim | ~4.97M |
| **VoiceAntiSpoof** | Light CNN (LCNN) + MaxFeatureMap | Phân loại Real/Spoof | ~97.8K |
| **VoiceAuthService** | ECAPA-TDNN (SpeechBrain pretrained) | Speaker verification 192-dim | ~22M |
| **Fusion MLP** | Linear(514→128→2) + BN + Dropout | Late Fusion quyết định | ~66.4K |
| **Tổng** | — | — | **~5.13M** (trainable) |

### 7.2 Pipeline xử lý âm thanh (7 bước)

```
Raw I2S bytes
    │
    ▼ 1. Decode: int16 / 32768.0 → float32 [-1, 1]   ← Lỗi quan trọng đã sửa!
    │
    ▼ 2. Quality Gate: peak ≥ 0.01, RMS ≥ 0.003, voiced_ratio ≥ 0.1
    │
    ▼ 3. Trim Silence: librosa.effects.trim(top_db=30)
    │
    ▼ 4. Peak Normalize: y = y / max(|y|)
    │
    ▼ 5. Wrap Padding: np.tile(y, ceil(target/len))[:target]
    │
    ▼ 6. Mel-spectrogram: n_mels=80, n_fft=512, hop=160 → [80×300]
    │
    ▼ 7. Z-score Norm: (mel_dB − μ) / (σ + 1e-8)
    │
    ▼
[Đầu vào LCNN / ECAPA-TDNN]
```

### 7.3 Chiến lược huấn luyện 3 giai đoạn (Transfer Learning)

```
Giai đoạn 1: Pre-train VoiceAntiSpoof
├── Dataset: ASVspoof 2019 LA (25.380 train / 24.844 val)
├── Config: max_epochs=20, lr=3e-4, AMP 16-mixed, GPU T4
└── Kết quả: val_acc = 95.92% (Epoch 3, EarlyStopping)

Giai đoạn 2: Base Fusion (Freeze encoders)
├── Dataset: LFW + ASVspoof ghép (~8.000 train / ~2.000 val)
├── Config: max_epochs=10, lr=1e-4, batch=32
└── Kết quả: val_loss = 0.103 (Epoch 2)

Giai đoạn 3: Domain Adaptation (Fine-tune ESP32)
├── Dataset: Tự thu thập từ ESP32 (82 train / 21 val)
├── Config: max_epochs=15, lr=5e-5, batch=8, EarlyStopping
└── Kết quả: val_acc = 82.6%, val_loss = 0.419 (Epoch 9)
```

Chạy huấn luyện (trên Google Colab):

```python
# Giai đoạn 1
python train_voice_antispoofing.py --stage 1 --dataset asvspoof2019

# Giai đoạn 2
python train_fusion.py --stage 2 --freeze-encoders

# Giai đoạn 3
python train_fusion.py --stage 3 --data esp32_collected/ --lr 5e-5
```

---

## 8. Kết quả thực nghiệm

### 8.1 Đánh giá Online (30 chu kỳ thực tế)

| Tiêu chí | Kết quả | Ngưỡng | Đạt? |
|---|---|---|---|
| Face Score – chủ nhân | 0.70 ± 0.08 | ≥ 0.50 | ✅ |
| Voice Score – chủ nhân | 0.97 ± 0.03 | ≥ 0.70 | ✅ |
| Fusion Score – chủ nhân | ~0.82 | ≥ 0.65 | ✅ |
| Face Score – người lạ | 0.21 ± 0.12 | < 0.50 | ✅ Từ chối đúng |
| Voice Score – im lặng/nhiễu | 0.00 | < 0.70 | ✅ Từ chối đúng |
| Tỷ lệ phát hiện khuôn mặt | ~85% | ≥ 80% | ✅ |
| Tỷ lệ chống giả mạo | ~84% | ≥ 80% | ✅ |
| Tổng latency / chu kỳ | 4.2 ± 0.5 giây | < 6 giây | ✅ |

### 8.2 Đánh giá Offline (Google Colab – GPU T4)

| Model | Accuracy | Loss | Epoch hội tụ |
|---|---|---|---|
| VoiceAntiSpoof (ASVspoof 2019) | **95.92%** | — | 3 / 20 |
| Fusion MLP (ESP32 domain) | **82.60%** | 0.419 | 9 / 15 |

### 8.3 Sửa lỗi decode I2S – kết quả trước và sau

| Phiên bản | Công thức decode | MFCC std | Voice Score |
|---|---|---|---|
| ❌ Lỗi (ban đầu) | `int32 / 2^31` | ~10⁵ (bất thường) | **0.00** |
| ✅ Đã sửa | `int16 / 32768.0` | ~18–22 (bình thường) | **0.97** |

---

## 9. Đóng góp kỹ thuật nổi bật

### 🔧 1. Phát hiện và sửa lỗi decode I2S 32-bit

Phát hiện lỗi nghiêm trọng khi firmware ESP32 xuất dữ liệu PCM 16-bit nhưng server đọc nhầm là `int32`, gây Voice Score = 0.00 hoàn toàn:

```python
# ❌ SAI – gây Voice Score = 0.00
audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0

# ✅ ĐÚNG – Voice Score đạt 0.97
audio = np.frombuffer(raw[:usable_len], dtype=np.int16).astype(np.float32) / 32768.0
```

### 🎙️ 2. Robust Audio Pipeline cho thiết bị nhúng

Pipeline 7 bước xử lý nhiễu thực tế của INMP441 bao gồm: high-pass filter Butterworth 150Hz, quality gating đa tiêu chí (Peak/RMS/Voiced Ratio), Wrap Padding (thay vì Zero Padding), và Z-score normalization trên Mel-spectrogram.

### 🔄 3. Domain Adaptation 3 giai đoạn với dữ liệu hạn chế

Đạt 82.6% accuracy trên thiết bị thực chỉ với 103 mẫu tự thu thập, giải quyết bài toán khan hiếm dữ liệu trong triển khai AI thực tế.

### 🛡️ 4. Kiến trúc bảo mật giọng nói 2 lớp

Kết hợp song song ECAPA-TDNN (xác minh danh tính) và LCNN Anti-Spoofing (phát hiện giả mạo), tạo hàng rào bảo mật kép chống tấn công replay audio.

---

## 10. Cấu trúc thư mục

```
AIoT_SmartLock/
│
├── firmware/                    # Code Arduino cho ESP32-CAM
│   └── esp32_smartlock/
│       ├── esp32_smartlock.ino  # Main firmware
│       ├── camera_config.h      # Cấu hình OV2640
│       └── i2s_audio.h          # Driver INMP441 I2S
│
├── server/                      # FastAPI backend
│   ├── main.py                  # Entry point, routes
│   ├── services/
│   │   ├── face_auth.py         # FaceAuthService (dlib)
│   │   ├── voice_auth.py        # VoiceAuthService (ECAPA-TDNN)
│   │   ├── voice_antispoofing.py# VoiceAntiSpoof (LCNN)
│   │   └── ai_fusion.py         # AIFusionService (Late Fusion MLP)
│   ├── models/                  # Kiến trúc PyTorch
│   │   ├── lcnn.py              # Light CNN + MaxFeatureMap
│   │   ├── face_encoder.py      # MobileNetV3 + ArcFace
│   │   └── fusion_mlp.py        # Multimodal Late Fusion
│   ├── audio_pipeline.py        # Pipeline xử lý âm thanh 7 bước
│   ├── firebase_sync.py         # Đồng bộ model checkpoint
│   ├── config/
│   │   └── serviceAccountKey.json  # Firebase credentials (không commit!)
│   └── templates/
│       └── dashboard.html       # Giao diện SSE real-time
│
├── training/                    # Script huấn luyện (Google Colab)
│   ├── train_voice_antispoofing.py
│   ├── train_fusion.py
│   └── domain_adaptation.py
│
├── data/                        # Dữ liệu thu thập từ ESP32
│   ├── real/                    # Mẫu giọng nói thật
│   └── spoof/                   # Mẫu giọng nói giả mạo
│
├── checkpoints/                 # Model checkpoints (.ckpt)
│   ├── voice_antispoofing_v1.ckpt
│   └── late_fusion_v3.ckpt
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 11. Hạn chế và hướng phát triển

### 11.1 Hạn chế hiện tại

| Hạn chế | Mô tả |
|---|---|
| Dữ liệu nhỏ | Chỉ 103 mẫu ESP32, dễ overfit, chưa đủ đa dạng |
| Face Score chưa cao | Trung bình 0.70, cần cải thiện enrollment và model |
| Latency cao | 4–5 giây/chu kỳ do thu âm cố định 3 giây |
| Phụ thuộc Local Server | Server tắt → thiết bị không hoạt động |
| Chỉ Face + Voice | Chưa tích hợp RFID, PIN, vân tay |

### 11.2 Hướng phát triển tương lai

- 🎯 **Voice Activity Detection (VAD):** Dừng thu âm sớm khi đã đủ tín hiệu, giảm latency xuống < 2.5 giây
- 📸 **Nâng cấp Face Recognition:** Thay dlib bằng ArcFace/InsightFace (512-dim), mục tiêu Face Score > 0.85
- 🖥️ **Edge AI Inference:** Triển khai trên Raspberry Pi 5 / Jetson Nano với ONNX Runtime
- 🧠 **Cross-Modal Attention Fusion:** Thay Late Fusion MLP bằng Transformer khi có đủ dữ liệu
- 🔑 **OTP Challenge-Response:** Yêu cầu đọc số ngẫu nhiên để chống replay hoàn toàn
- 🔒 **Federated Learning:** Bảo vệ dữ liệu sinh trắc học phân tán

---

## 12. Tài liệu tham khảo

| # | Tài liệu |
|---|---|
| [1] | Ericsson Mobility Report 2023 |
| [2] | Bonneau et al., "The quest to replace passwords", IEEE S&P 2012 |
| [3] | Sun et al., IEEE Transactions on Industrial Informatics, 2018 |
| [4] | Zhang et al., IEEE Internet of Things Journal, 2021 |
| [5] | Marcel et al., Handbook of Biometric Anti-Spoofing, Springer 2014 |
| [6] | Kinnunen et al., IEEE ICASSP 2012 |
| [7] | Ross & Jain, "Multimodal biometrics: An overview", EUSIPCO 2004 |
| [8] | Wazid et al., IEEE Access 2019 |
| [9] | Howard et al., "Searching for MobileNetV3", IEEE ICCV 2019 |
| [10] | Lavrentyeva et al., "Audio replay attack detection with deep learning", Interspeech 2017 |
| [11] | Taigman et al., "DeepFace", IEEE CVPR 2014 |
| [12] | Deng et al., "ArcFace", IEEE CVPR 2019 |
| [13] | Geitgey, face_recognition library, GitHub 2018 |
| [14] | Nautsch et al., "ASVspoof 2019", IEEE TBIOM 2021 |
| [15] | Desplanques et al., "ECAPA-TDNN", Interspeech 2020 |
| [16] | Ravanelli et al., "SpeechBrain", arXiv 2021 |
| [17] | Snoek et al., "Early versus late fusion", ACM Multimedia 2005 |
| [18] | Vaswani et al., "Attention is all you need", NeurIPS 2017 |
| [19] | Nagrani et al., "Learnable PINs", ECCV 2018 |
| [20] | Bonomi et al., "Fog computing and IoT", ACM MCC 2012 |
| [21] | Goodfellow et al., Deep Learning, MIT Press 2016 |
| [22] | Loshchilov & Hutter, "AdamW", ICLR 2019 |
| [23] | McFee et al., "librosa", SciPy 2015 |

---

**Học viện Công nghệ Bưu chính Viễn thông – Cơ sở TP.HCM**  
Môn học: Phát triển Ứng dụng IoT · Năm học 2025–2026  

*Mọi thắc mắc vui lòng liên hệ nhóm tác giả qua hệ thống quản lý học vụ của Học viện.*
