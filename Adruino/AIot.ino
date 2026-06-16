#include <WiFi.h>
#include "esp_camera.h"
#include <driver/i2s.h>
#include "esp_http_server.h"
#include "soc/soc.h"           // Khai báo thêm ở đầu file
#include "soc/rtc_cntl_reg.h"
#include <ESPmDNS.h>

// ================= CẤU HÌNH MẠNG =================
const char* ssid = "HaiDtrai";       
const char* password = "comlangvong";

// ================= CẤU HÌNH PHẦN CỨNG =================
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22
#define FLASH_GPIO_NUM 4

#define I2S_WS 15
#define I2S_SD 13
#define I2S_SCK 14
#define I2S_PORT I2S_NUM_1 

#define SAMPLE_RATE 16000
#define RECORD_TIME 3  
const int audioSize = SAMPLE_RATE * 4 * RECORD_TIME; 
uint8_t* audioBuffer = nullptr;

httpd_handle_t camera_httpd = NULL;
httpd_handle_t stream_httpd = NULL;

#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

//#define BUZZER_PIN 12
bool isDoorOpen = false;
unsigned long doorOpenTime = 0;
// ================= CÁC API CỦA ESP32 =================

// 1. API TRẢ VỀ VIDEO LIVE STREAM (Port 81)
static esp_err_t stream_handler(httpd_req_t *req){
    camera_fb_t * fb = NULL;
    esp_err_t res = ESP_OK;
    size_t _jpg_buf_len = 0;
    uint8_t * _jpg_buf = NULL;
    char part_buf[64];

    res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
    if(res != ESP_OK) return res;

    while(true){
        fb = esp_camera_fb_get();
        if (!fb) {
            Serial.println("❌ Lỗi stream camera");
            res = ESP_FAIL;
        } else {
            _jpg_buf_len = fb->len;
            _jpg_buf = fb->buf;
        }
        if(res == ESP_OK){
            size_t hlen = snprintf((char *)part_buf, 64, _STREAM_PART, _jpg_buf_len);
            res = httpd_resp_send_chunk(req, (const char *)part_buf, hlen);
        }
        if(res == ESP_OK) res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, _jpg_buf_len);
        if(res == ESP_OK) res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
        if(fb) esp_camera_fb_return(fb);
        if(res != ESP_OK) break;
        delay(100); 
    }
    return res;
}

// 2. API CẤP AUDIO CHO SERVER PYTHON (Port 80)
static esp_err_t get_audio_handler(httpd_req_t *req){
    Serial.println("Server Python đang yêu cầu Audio!");
    
    // Nháy đèn báo hiệu cho người dùng biết bắt đầu thu âm
    digitalWrite(FLASH_GPIO_NUM, HIGH); 
    //digitalWrite(BUZZER_PIN, LOW); 
    delay(100);
    digitalWrite(FLASH_GPIO_NUM, LOW); 
    //digitalWrite(BUZZER_PIN, HIGH); 
    delay(100);
    
    size_t bytesRead = 0;
    // Thu âm 3 giây
    i2s_read(I2S_PORT, (char*)audioBuffer, audioSize, &bytesRead, portMAX_DELAY);

    int32_t* samples32 = (int32_t*)audioBuffer;
    int16_t* samples16 = (int16_t*)audioBuffer;
    int numSamples   = bytesRead / 4;

    for (int i = 0; i < numSamples; i++) {
        int32_t amplified = samples32[i] >> 11;
        // Clamp tránh overflow
        if (amplified >  32767) amplified =  32767;
        if (amplified < -32768) amplified = -32768;
        samples16[i] = (int16_t)amplified;
    }
    size_t outBytes = numSamples * 2;
    // Gửi thẳng cục Audio Raw về cho Python
    httpd_resp_set_type(req, "application/octet-stream");
    esp_err_t res = httpd_resp_send(req, (const char *)audioBuffer, outBytes);
    
    Serial.printf("✅ Đã gửi Audio xong!\n", outBytes);
    return res;
}

// 3. API CHỤP ẢNH TĨNH CHO ENROLLMENT (Port 80)
static esp_err_t capture_handler(httpd_req_t *req){
    camera_fb_t * fb = esp_camera_fb_get();
    if (!fb) return ESP_FAIL;
    httpd_resp_set_type(req, "image/jpeg");
    esp_err_t res = httpd_resp_send(req, (const char *)fb->buf, fb->len);
    esp_camera_fb_return(fb);
    return res;
}
static esp_err_t open_door_handler(httpd_req_t *req){
    Serial.println("\n=================================");
    Serial.println("🟢 NHẬN LỆNH TỪ AI: ĐÚNG MẬT KHẨU!");
    Serial.println("ĐÃ MỞ CỬA! (Bắt đầu đếm ngược 5 giây...)");
    Serial.println("=================================\n");
    
    // Kích hoạt cờ trạng thái để hàm loop() bắt đầu đếm giờ
    isDoorOpen = true;         
    doorOpenTime = millis();

    //digitalWrite(BUZZER_PIN, LOW); delay(100);
    //digitalWrite(BUZZER_PIN, HIGH);  delay(100);
    //digitalWrite(BUZZER_PIN, LOW); delay(100);
    //digitalWrite(BUZZER_PIN, HIGH);   
    
    // Báo về cho Python biết là đã nhận lệnh thành công
    httpd_resp_set_type(req, "text/plain");
    esp_err_t res = httpd_resp_send(req, "DOOR_OPENED", HTTPD_RESP_USE_STRLEN);
    return res;
}

// ================= KHỞI TẠO =================
void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
  Serial.begin(115200);
  pinMode(FLASH_GPIO_NUM, OUTPUT);

  //pinMode(BUZZER_PIN, OUTPUT);
  //digitalWrite(BUZZER_PIN, HIGH); 
  WiFi.begin(ssid, password);
  Serial.print("Đang kết nối WiFi");
  while (WiFi.status() != WL_CONNECTED) {
      delay(500);
      Serial.print(".");
  }
  Serial.println("\n✅ WiFi đã kết nối!");
  Serial.print("Địa chỉ IP động được cấp: ");
  Serial.println(WiFi.localIP());

  if (!MDNS.begin("smartlock")) {
      Serial.println("❌ Lỗi khởi tạo mDNS!");
  } else {
      Serial.println("✅ mDNS khởi tạo thành công!");
      Serial.println("Kết nối qua: http://smartlock.local");
  }
  
  // Init Camera
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0; config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM; config.pin_d1 = Y3_GPIO_NUM; config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM; config.pin_d4 = Y6_GPIO_NUM; config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM; config.pin_d7 = Y9_GPIO_NUM; config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM; config.pin_vsync = VSYNC_GPIO_NUM; config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM; config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM; config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000; config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_VGA; config.jpeg_quality = 12; config.fb_count = 2;
  esp_camera_init(&config);
  
  // Init I2S
  const i2s_config_t i2s_config = {
    .mode = i2s_mode_t(I2S_MODE_MASTER | I2S_MODE_RX), 
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT, 
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_I2S | I2S_COMM_FORMAT_I2S_MSB),
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1, 
    .dma_buf_count = 8, 
    .dma_buf_len = 1024,
    .use_apll = false, 
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };
  const i2s_pin_config_t pin_config = { .bck_io_num = I2S_SCK, .ws_io_num = I2S_WS, .data_out_num = -1, .data_in_num = I2S_SD };
  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL); i2s_set_pin(I2S_PORT, &pin_config);
  
  audioBuffer = (uint8_t*)ps_malloc(audioSize);

  // Start Servers
  httpd_config_t config_server = HTTPD_DEFAULT_CONFIG();
  config_server.max_uri_handlers = 16;
  
  httpd_uri_t audio_uri = { .uri = "/get_audio", .method = HTTP_GET, .handler = get_audio_handler, .user_ctx = NULL };
  httpd_uri_t capture_uri = { .uri = "/capture", .method = HTTP_GET, .handler = capture_handler, .user_ctx = NULL };
  httpd_uri_t stream_uri = { .uri = "/stream", .method = HTTP_GET, .handler = stream_handler, .user_ctx = NULL };
  httpd_uri_t open_door_uri = { .uri = "/open_door", .method = HTTP_GET, .handler = open_door_handler, .user_ctx = NULL };
  
  if (httpd_start(&camera_httpd, &config_server) == ESP_OK) {
      httpd_register_uri_handler(camera_httpd, &audio_uri);
      httpd_register_uri_handler(camera_httpd, &capture_uri);
      httpd_register_uri_handler(camera_httpd, &open_door_uri);
  }
  
  config_server.server_port += 1; config_server.ctrl_port += 1;
  if (httpd_start(&stream_httpd, &config_server) == ESP_OK) {
      httpd_register_uri_handler(stream_httpd, &stream_uri);
  }
}

void loop() {
  // HOÀN TOÀN TRỐNG! ESP32 CHỈ NGỒI CHỜ SERVER PYTHON GỌI
  // Tự động kiểm tra: Nếu cửa đang mở và đã qua 5 giây -> Tự động khóa lại
  if (isDoorOpen) {
    if (millis() - doorOpenTime > 5000) { // Sau 5 giây
      isDoorOpen = false;
      Serial.println("🔴 TỰ ĐỘNG KHÓA CỬA!");
      // Kêu 1 tiếng Bíp dài báo hiệu đã sập chốt an toàn
      //digitalWrite(BUZZER_PIN, LOW);
      //delay(500); 
      //digitalWrite(BUZZER_PIN, HIGH);
    }
  }
  delay(10); // Nghỉ 10ms để giải phóng CPU
}