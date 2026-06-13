# AIoT Smart Lock Server - Multimodal Authentication

Đây là ứng dụng Backend Server quản lý hệ thống Smart Lock sử dụng công nghệ nhận diện Đa phương thức (Multimodal Authentication): Kết hợp nhận diện Khuôn mặt (Face Recognition) và Giọng nói (Voice Verification/Anti-Spoofing). 

Hệ thống kết nối trực tiếp với ESP32-CAM và module thu âm I2S ở vùng biên (Edge Device) để cấp quyền truy cập đóng/mở cửa hiện đại.

---

## 🚀 Tính năng nổi bật
*   **Late Fusion AI:** Kết hợp điểm tin cậy của 2 luồng Face và Voice lại với nhau, hỗ trợ tinh chỉnh trọng số linh hoạt (ví dụ: Khớp mặt đạt 0.7, khớp giọng 0.4 vẫn được cho vào theo tổng trọng số).
*   **Real-time Streaming:** Hỗ trợ xem trực tiếp camera ESP32-CAM trên Dashboard qua luồng MJPEG.
*   **Dashbord Giám Sát Web (UI/UX Hiện Đại):** Có giao diện Web Control Panel chuyên nghiệp cho phép tinh chỉnh Thanh Trượt ngưỡng Identity, Voice, theo dõi log mở cửa trực tiếp qua SSE (Server-Sent Events).
*   **Tích hợp Firebase:** Chức năng Upload lịch sử lưu hình chụp kẻ lạ mặt lên lưu trữ đám mây (Cloud Storage). 
*   **PyTorch Deep Learning:** Sử dụng các mô hình học sâu hiện đại được xây dựng trực tiếp trên kiến trúc PyTorch để xử lý phân tích và chống giả mạo chính xác.

---

## 🛠 Yêu cầu hệ thống (Prerequisites)
*   **OS:** Windows 10/11, Linux (Ubuntu/Debian), macOS.
*   **Python:** Lớn hơn phiên bản `>= 3.9` (Khuyến nghị 3.10 hoặc 3.13)
*   **Phần cứng AI:** Có thể chạy được trên CPU, nhưng khuyên dùng PC/Server có GPU (Card màn hình rời) để quá trình nhúng đặc trưng khuôn mặt và chạy PyTorch được tăng tốc.
*   **Mạng:** Có thể cấu hình Router Wifi tĩnh cấp chung mạng cho ESP32 và Server.

---

## ⚙️ Hướng dẫn cài đặt (Setup)

### Bước 1: Clone dự án và tạo môi trường
Bạn mở Terminal (Command Prompt hoặc Powershell), di chuyển tới Desktop và thực hiện lệnh:
```bash
git clone <đường-dẫn-repo> AIot_Server
cd AIot_Server

# (Tuỳ chọn nhưng cực kỳ khuyên dùng) Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường (Windows)
venv\Scripts\activate
# (Hoặc MacOS/Linux)
# source venv/bin/activate
```

### Bước 2: Cài đặt thư viện Python phụ thuộc
Bạn cần các thư viện nền tảng lõi. Chạy câu lệnh tổng hợp dưới đây:
```bash
pip install fastapi uvicorn pydantic python-multipart
pip install opencv-python numpy
pip install requests face-recognition librosa torchmetrics lightning speechbrain torch torchvision
pip install firebase-admin
```

> **Lưu ý đối với cài `face-recognition` trên Windows**: Nền tảng dlib core thỉnh thoảng yêu cầu có cài Visual Studio Build Tools (C++) cùng cmake. Để đơn giản hãy tải bản build wheel trên mạng thay vì pip build gốc.

### Bước 3: Chuẩn bị Cấu hình
1. Sửa file `configs/config.json`. Hãy thay `ESP32_IP` bằng ID Wifi hiển thị trên mạch ESP:
```json
{
  "ESP32_IP": "192.168.1.xxx",   
  "DOOR_SECRET": "SECRET_KEY_CUA_BAN"
}
```
2. Mở file `firebase_key.json` *(nếu bạn sử dụng Storage)*, chép nội dung tải trực tiếp lấy từ Google Cloud Console/Firebase Service Account.

### Bước 4: Đưa ảnh mẫu và file AI vào Database
Để nhận diện chính bạn là chủ phòng:
1. Thêm ảnh gốc khuôn mặt rõ nét lưu vào `database/chu_nhan.jpg`.
2. Đọc khoảng một đoạn giọng nói trong file chuẩn WAV lưu vào thư mục định sẵn `pretrained_models/.../owner_voice.wav` theo nội dung logic của file `auth_services.py`.

---

## ▶️ Cách Khởi Chạy Và Sử Cụng (Run)

Khởi động server API FastAPI bằng Uvicorn Python:

```bash
python main.py
```
Hoặc:
```bash
uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

*   **Kiểm tra tình trạng:**
    Terminal sẽ nháy sáng: `INFO: Uvicorn running on http://0.0.0.0:5000 (Press CTRL+C to quit)`

*   **Truy cập Dashboard Quản lý Web:**
    Mở trình duyệt (Google Chrome) nhập:
    🔗 `http://localhost:5000/dashboard`

## 🧠 Sơ đồ Kiến trúc AIoT Flow 
1. Khách đi tới nhấn nút vật lý Enroll trên mạch. ->
2. ESP32 Gửi Request POST ảnh và âm thanh dạng Binary cho API `/api/enroll_edge`
3. Backend Server kích hoạt logic `auth_services.py`.
   - Thread Face Module: Dùng `face_recognition` Extract 128 đặc trưng tính độ Lệch.
   - Thread Voice Module: Dùng `SpeechBrain (ECAPA-TDNN)` Lọc âm thanh qua PCM, chấm Voice Pattern
   - Layer Anti Spoofing: Quét mảng Voice qua mô hình PyTorch chống thu âm giả phát lại `VoiceAntiSpoof`.
4. Logic LateFusion kết luận ngưỡng Mở khoá.
5. Server POST lệnh quay về Firmware thiết bị. ESP32 đá Role điều khiển Relay khoá. Đèn xanh bật chốt cửa được bung ra.

---
*Dự án DPT — PTITHCM.*
