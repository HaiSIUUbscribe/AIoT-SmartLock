#pylint: disable=all
import os
import glob
import logging
import cv2
import numpy as np
import librosa
import face_recognition
import torch
import torchvision.transforms as transforms
from PIL import Image
from scipy.spatial.distance import cosine
from models import MultimodalLateFusion
import torchaudio
if not hasattr(torchaudio, 'set_audio_backend'):
    torchaudio.set_audio_backend = lambda x: None
from speechbrain.pretrained import EncoderClassifier

logger = logging.getLogger("AuthServices")

# Hàm giải mã Audio (Dùng hàm 16-bit mà bạn đã test thành công)
def decode_esp32_audio(raw_audio_bytes: bytes) -> np.ndarray:
    if not raw_audio_bytes: return np.array([], dtype=np.float32)
    usable_len = len(raw_audio_bytes) - (len(raw_audio_bytes) % 2)
    if usable_len <= 0: return np.array([], dtype=np.float32)
    audio_array_16 = np.frombuffer(raw_audio_bytes[:usable_len], dtype=np.int16)
    return audio_array_16.astype(np.float32) / 32768.0 

def audio_has_enough_signal(y: np.ndarray, source: str, thresholds: dict) -> bool:
    if y.size == 0: return False
    abs_y = np.abs(y)
    peak = float(np.max(abs_y))
    rms = float(np.sqrt(np.mean(np.square(y))))
    voiced_ratio = float(np.mean(abs_y >= thresholds["MIN_AUDIO_PEAK"]))
    print(f"DEBUG {source} = peak:{peak:.6f}, rms:{rms:.6f}, voiced:{voiced_ratio:.4f}")
    return (peak >= thresholds["MIN_AUDIO_PEAK"] and 
            rms >= thresholds["MIN_AUDIO_RMS"] and 
            voiced_ratio >= thresholds["MIN_AUDIO_VOICED_RATIO"])

class FaceAuthService:
    def __init__(self, owner_folder_path: str):
        self.known_encodings = []
        self._load_known_faces(owner_folder_path)

    def _load_known_faces(self, folder_path: str):
        if not os.path.exists(folder_path): os.makedirs(folder_path, exist_ok=True)
        for path in glob.glob(os.path.join(folder_path, "*.*")):
            try:
                image = face_recognition.load_image_file(path)
                face_locs = face_recognition.face_locations(image)
                if face_locs:
                    face_locs = sorted(face_locs, key=lambda x: (x[2]-x[0])*(x[1]-x[3]), reverse=True)
                    encodings = face_recognition.face_encodings(image, [face_locs[0]])
                    if encodings: self.known_encodings.append(encodings[0])
            except: pass
        logger.info(f"✅ Đã nạp thành công {len(self.known_encodings)} vector góc mặt.")

    def reload_database(self, folder_path: str):
        self.known_encodings.clear()
        self._load_known_faces(folder_path)

    def verify_cv2_image(self, rgb_image: np.ndarray, face_loc: tuple) -> float:
        if not self.known_encodings: return 0.0
        try:
            encodings = face_recognition.face_encodings(rgb_image, [face_loc])
            if not encodings: return 0.0
            distances = face_recognition.face_distance(self.known_encodings, encodings[0])
            return float(max(0.0, 1.0 - min(distances)))
        except: return 0.0

class VoiceAuthService:
    def __init__(self, voice_dir: str, config: dict, thresholds: dict):
        self.voice_dir = voice_dir
        self.config = config
        self.thresholds = thresholds
        self.owner_embedding = None
        
        print("⏳ Đang khởi tạo AI Giọng nói SpeechBrain (ECAPA-TDNN)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Tải mô hình
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device}
        )
        self.load_owner_voice()

    def load_owner_voice(self):
        os.makedirs(self.voice_dir, exist_ok=True)
        voice_path = os.path.join(self.voice_dir, "owner_voice.wav")
        if os.path.exists(voice_path):
            try:
                y, fs = librosa.load(voice_path, sr=16000)
                
                # Chuyển numpy array thành Tensor cho AI
                signal = torch.from_numpy(y).unsqueeze(0).float().to(self.device)
                
                with torch.no_grad():
                    embeddings = self.classifier.encode_batch(signal)
                    self.owner_embedding = embeddings.squeeze() 
                print("✅ Đã nạp Vector Danh tính của chủ nhân bằng Librosa thành công!")
            except Exception as e: 
                print(f"Lỗi load_owner_voice: {e}")

    def verify(self, raw_audio_bytes: bytes) -> float:
        if self.owner_embedding is None: return 0.0
        try:
            # 1. Giải mã ESP32 PCM
            y = decode_esp32_audio(raw_audio_bytes)
            
            # 2. Bộ lọc DSP (Loại bỏ tiếng ồn/huýt sáo/loa to)
            if not audio_has_enough_signal(y, "verify_voice", self.thresholds): 
                return 0.0
            
            raw_peak = np.max(np.abs(y))
            if raw_peak > 0:
                y = y / raw_peak
                
            peak = np.max(np.abs(y))
            rms = np.sqrt(np.mean(y**2))
            
            if rms > 0.85: return 0.0 #0.10
            #if peak >= 0.99: return 0.0 

            # 3. Ép âm thanh từ ESP32 về đúng chuẩn 16000Hz để AI không bị "điếc"
            if self.config["SAMPLE_RATE"] != 16000:
                y = librosa.resample(y, orig_sr=self.config["SAMPLE_RATE"], target_sr=16000)

            # 4. Giao cho SpeechBrain phân tích
            signal = torch.from_numpy(y).unsqueeze(0).float().to(self.device)
            
            with torch.no_grad():
                current_embedding = self.classifier.encode_batch(signal).squeeze()
                
            # 5. So sánh độ giống nhau
            cos_sim = torch.nn.functional.cosine_similarity(self.owner_embedding, current_embedding, dim=0)
            score = (cos_sim.item() + 1.0) / 2.0 
            
            return float(score)
            
        except Exception as e: 
            print(f"Lỗi Voice verify: {e}")
            return 0.0

class AIFusionService:
    def __init__(self, ckpt_path: str, config: dict, thresholds: dict, label_map: dict):
        self.config = config
        self.thresholds = thresholds
        self.label_map = label_map
        self.model = MultimodalLateFusion.load_from_checkpoint(ckpt_path, map_location='cpu')
        self.model.eval()
        self.face_transforms = transforms.Compose([
            transforms.Resize((112, 112)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])

    # Bổ sung face_loc vào tham số để cắt ảnh trước khi đưa cho AI
    def verify(self, rgb_image: np.ndarray, raw_audio_bytes: bytes, face_loc: tuple = None) -> float:
        try:
            # 1. CẮT ẢNH (Phần code cũ đã đúng)
            if face_loc is not None:
                top, right, bottom, left = face_loc
                h, w = rgb_image.shape[:2]
                margin = int((bottom - top) * 0.2)
                rgb_image = rgb_image[max(0, top-margin):min(h, bottom+margin), 
                                      max(0, left-margin):min(w, right+margin)]
            
            img = Image.fromarray(rgb_image)
            face_tensor = self.face_transforms(img).unsqueeze(0)
            
            y = decode_esp32_audio(raw_audio_bytes)
            if not audio_has_enough_signal(y, "verify_fusion", self.thresholds): return 0.0
            
            # --- BẢN VÁ QUAN TRỌNG NHẤT CHO ÂM THANH ---
            
            # BƯỚC 1: Cắt sạch nhiễu tĩnh điện ESP32 ở khoảng lặng 2 đầu
            y_trimmed, _ = librosa.effects.trim(y, top_db=30)
            if len(y_trimmed) >= self.config["SAMPLE_RATE"] * 0.5:
                y = y_trimmed
            
            # Chuẩn hóa âm lượng
            peak = np.max(np.abs(y))
            if peak >= self.thresholds.get("MIN_AUDIO_PEAK", 0.05): 
                y = y / peak
                
            # BƯỚC 2: KHÔNG DÙNG ZERO-PADDING. Thay bằng lặp lại tín hiệu (Wrap Padding)
            target_len = int(self.config["SAMPLE_RATE"] * self.config["AUDIO_DURATION_SEC"])
            if len(y) < target_len:
                repeats = int(np.ceil(target_len / len(y)))
                audio = np.tile(y, repeats)[:target_len] # Nhân bản giọng nói để lấp đầy 3s
            else:
                audio = y[:target_len]
                
            # Tạo phổ Mel với tín hiệu tinh khiết
            mel = librosa.feature.melspectrogram(y=audio, sr=self.config["SAMPLE_RATE"], n_mels=80, n_fft=512, hop_length=160, win_length=400)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_norm = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
            
            mel_norm = mel_norm[:, :300] if mel_norm.shape[1] > 300 else np.pad(mel_norm, ((0, 0), (0, 300 - mel_norm.shape[1])))
            voice_tensor = torch.tensor(mel_norm).unsqueeze(0).unsqueeze(0).float()
            
            # --------------------------------------------

            with torch.no_grad():
                logits = self.model(face_tensor, voice_tensor)
                probs = torch.softmax(logits, dim=1)
                return float(probs[0][self.label_map["FUSION_REAL_CLASS"]].item())
        except Exception as e: 
            print(f"Fusion Lỗi: {e}")
            return 0.0
