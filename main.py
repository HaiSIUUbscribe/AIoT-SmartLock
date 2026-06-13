#pylint: disable=all
import json
import logging
import asyncio
import cv2
import threading
import time
import requests
import shutil
import os
import face_recognition
import numpy as np
import scipy.io.wavfile as wavio
import firebase_admin
from firebase_admin import credentials, storage

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from contextlib import asynccontextmanager
from fastapi.templating import Jinja2Templates

from auth_services import FaceAuthService, VoiceAuthService, AIFusionService, decode_esp32_audio

# --- ĐỌC CONFIG ---
with open("configs/config.json", "r") as f: CFG = json.load(f)
with open("configs/thresholds.json", "r") as f: THRESHOLDS = json.load(f)
with open("configs/label_map.json", "r") as f: LABELS = json.load(f)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AuthServer")

latest_frame_jpeg = b'' 
current_door_status = {"message": "Đang giám sát...", "color": "#00ff00"}
is_enrolling = False 

def update_status(message: str, color: str):
    global current_door_status
    current_door_status = {"message": message, "color": color}

# --- LUỒNG GIÁM SÁT ---
def camera_surveillance_task():
    global is_enrolling, latest_frame_jpeg, THRESHOLDS
    stream_url = f"http://{CFG['ESP32_IP']}:81/stream"
    audio_url = f"http://{CFG['ESP32_IP']}/get_audio"
    
    while True:
        try:
            if is_enrolling:
                time.sleep(1)
                continue

            res = requests.get(stream_url, stream=True, timeout=5)
            if res.status_code != 200:
                time.sleep(2)
                continue
                
            bytes_data = b''
            frame_count = 0
            last_auth_time = 0 
            
            for chunk in res.iter_content(chunk_size=4096):
                if is_enrolling:
                    res.close()
                    break

                bytes_data += chunk
                a = bytes_data.find(b'\xff\xd8')
                b = bytes_data.find(b'\xff\xd9')
                
                if a != -1 and b != -1:
                    jpg = bytes_data[a:b+2]
                    bytes_data = bytes_data[b+2:] 
                    latest_frame_jpeg = jpg

                    if time.time() - last_auth_time < 10:
                        if time.time() - last_auth_time > 8:
                            update_status("Đang giám sát...", "#00ff00")
                        continue

                    frame_count += 1
                    if frame_count % 15 == 0:
                        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None:
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            face_locations = face_recognition.face_locations(rgb_frame)
                            
                            if face_locations:
                                face_locations = sorted(face_locations, key=lambda x: (x[2]-x[0])*(x[1]-x[3]), reverse=True)
                                best_face_loc = face_locations[0]
                                
                                update_status("Đang kiểm tra khuôn mặt...", "#ffff00")
                                identity_score = face_service.verify_cv2_image(rgb_frame, best_face_loc)
                                
                                if identity_score >= THRESHOLDS["IDENTITY_THRESHOLD"]:
                                    update_status("Đang thu âm...", "#ffff00")
                                    res.close() 
                                    
                                    try:
                                        audio_resp = requests.get(audio_url, timeout=6)
                                        if audio_resp.status_code == 200:
                                            voice_match_score = voice_auth_service.verify(audio_resp.content)
                                            top, right, bottom, left = best_face_loc
                                            face_crop = rgb_frame[top:bottom, left:right]
                                            liveness_score = ai_fusion_service.verify(face_crop, audio_resp.content)
                                            
                                            logger.info(f"Scores -> Face:{identity_score:.2f} | Voice:{voice_match_score:.2f} | Live:{liveness_score:.2f}")
                                            
                                            if voice_match_score >= THRESHOLDS["VOICE_MATCH_THRESHOLD"] and liveness_score >= THRESHOLDS["FUSION_THRESHOLD"]:
                                                update_status(f"MỞ CỬA! (Live: {liveness_score*100:.0f}%)", "#00ff00")
                                                try: requests.get(f"http://{CFG['ESP32_IP']}/open_door", timeout=3)
                                                except: pass
                                            else:
                                                update_status(f"KHÓA CỬA! (V: {voice_match_score:.2f}, L: {liveness_score:.2f})", "#ff0000")
                                    except: 
                                        update_status("Lỗi kết nối Micro ESP32!", "#ff0000")
                                    
                                    last_auth_time = time.time()
                                    break 
                                else:
                                    update_status(f"KHÓA CỬA! Lạ (Face: {identity_score:.2f})", "#ff0000")
                                    last_auth_time = time.time()
        except Exception as e:
            time.sleep(2)

def sync_cloud_model():
    """Kiểm tra và tải Model từ Firebase về Edge Server"""
    model_path = CFG["LATE_FUSION_CKPT"] # Đường dẫn model trong config.json của bạn
    
    if not os.path.exists(model_path):
        logger.info("Không tìm thấy Model cục bộ. Đang tải từ Firebase...")
        try:
            # Chỉ initialize_app nếu nó chưa được khởi tạo
            if not firebase_admin._apps:
                cred = credentials.Certificate("configs/firebase_credentials.json")
                firebase_admin.initialize_app(cred, {
                    'storageBucket': 'aiot-face-voice-c8455.firebasestorage.app'
                })
                
            bucket = storage.bucket()
            blob = bucket.blob("models/late_fusion_V2.ckpt")
            
            # Tải file về máy
            blob.download_to_filename(model_path)
            logger.info("Đã đồng bộ Model từ Firebase thành công!")
        except Exception as e:
            logger.error(f"Lỗi tải từ Firebase: {e}")
            
# --- APP LIFESPAN ---
face_service, voice_auth_service, ai_fusion_service = None, None, None

@asynccontextmanager
async def lifespan(app: FastAPI):
    sync_cloud_model()
    global face_service, voice_auth_service, ai_fusion_service
    face_service = FaceAuthService(CFG["KNOWN_FACE_DIR"])
    voice_auth_service = VoiceAuthService(CFG["VOICE_DIR"], CFG, THRESHOLDS)
    ai_fusion_service = AIFusionService(CFG["LATE_FUSION_CKPT"], CFG, THRESHOLDS, LABELS)
    threading.Thread(target=camera_surveillance_task, daemon=True).start()
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

templates = Jinja2Templates(directory="templates")

# --- API ENDPOINTS ---
@app.get("/video_feed")
async def video_feed(request: Request):
    async def frame_generator():
        global latest_frame_jpeg
        try:
            while True:
                if await request.is_disconnected():
                    break
                    
                if latest_frame_jpeg:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + latest_frame_jpeg + b'\r\n')
                await asyncio.sleep(0.05) 
        except asyncio.CancelledError:
            pass # Trình duyệt chủ động ngắt kết nối
            
    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/status_stream")
async def status_stream(request: Request):
    async def event_generator():
        last_sent_status = None
        try:
            while True:
                if await request.is_disconnected():
                    break
                    
                global current_door_status
                if current_door_status != last_sent_status:
                    yield f"data: {json.dumps(current_door_status)}\n\n"
                    last_sent_status = current_door_status.copy()
                await asyncio.sleep(0.2) 
        except asyncio.CancelledError:
            pass
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/enroll_edge")
async def enroll_from_edge():
    global is_enrolling
    is_enrolling = True 
    try:
        update_status("Đang lấy ảnh...", "#ffff00")
        if os.path.exists(CFG["KNOWN_FACE_DIR"]): shutil.rmtree(CFG["KNOWN_FACE_DIR"])
        os.makedirs(CFG["KNOWN_FACE_DIR"], exist_ok=True)
        
        for i in range(5):
            try:
                resp = requests.get(f"http://{CFG['ESP32_IP']}/capture", timeout=3)
                if resp.status_code == 200:
                    with open(os.path.join(CFG["KNOWN_FACE_DIR"], f"chu_nhan_{i}.jpg"), "wb") as f: f.write(resp.content)
            except: pass
            await asyncio.sleep(0.5) 
            os.makedirs(CFG["VOICE_DIR"], exist_ok=True)
        update_status("Chuẩn bị thu âm...", "#ffff00")
        await asyncio.sleep(1.5)    
        try:
            audio_resp = requests.get(f"http://{CFG['ESP32_IP']}/get_audio", timeout=6)
            if audio_resp.status_code == 200:
                y_float = decode_esp32_audio(audio_resp.content)
                y_int16 = (y_float * 32767).astype(np.int16)
                wavio.write(os.path.join(CFG["VOICE_DIR"], "owner_voice.wav"), CFG["SAMPLE_RATE"], y_int16)
        except: pass

        face_service.reload_database(CFG["KNOWN_FACE_DIR"])
        voice_auth_service.load_owner_voice()
        
        is_enrolling = False
        update_status("Đăng ký xong!", "#00ff00")
        return {"status": "success"}
    except Exception as e:
        is_enrolling = False
        update_status("Lỗi Đăng ký!", "#ff0000")
        raise HTTPException(status_code=500, detail="Lỗi kết nối tới ESP32")

# API Cập nhật Thresholds từ Giao diện Web
@app.post("/api/update_thresholds")
async def update_thresholds(req: Request):
    global THRESHOLDS
    data = await req.json()
    THRESHOLDS.update(data)
    with open("configs/thresholds.json", "w") as f: json.dump(THRESHOLDS, f, indent=4)
    # Cập nhật vào services
    voice_auth_service.thresholds = THRESHOLDS
    ai_fusion_service.thresholds = THRESHOLDS
    return {"status": "success"}

# --- GIAO DIỆN WEB ---
@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    return templates.TemplateResponse(
        request= request,
        name="web.html", 
        context={"request": request, "THRESHOLDS": THRESHOLDS}
    )
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=False, workers=1)