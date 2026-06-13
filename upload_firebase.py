#pylint: disable=all
import firebase_admin
from firebase_admin import credentials, storage
import os

# 1. Khởi tạo kết nối với Firebase
# Thay 'your-bucket-name' bằng cái tên appspot.com bạn copy ở Bước 1
cred = credentials.Certificate("configs/firebase_credentials.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'aiot-face-voice-c8455.firebasestorage.app' 
})

def upload_model():
    bucket = storage.bucket()
    print("⏳ Đang tải model lên Firebase Storage...")
    
    # Khai báo đường dẫn file ở dưới máy (Local) và trên Cloud
    files_to_upload = {
        "late_fusion_V2.ckpt": "models/late_fusion_V2.ckpt",
        "configs/metrics.json": "models/metrics.json",
        "configs/label_map.json": "models/label_map.json"
    }
    
    for local_path, cloud_path in files_to_upload.items():
        if os.path.exists(local_path):
            blob = bucket.blob(cloud_path)
            blob.upload_from_filename(local_path)
            print(f"✅ Đã tải thành công: {local_path} -> gs://.../{cloud_path}")
        else:
            print(f"⚠️ Không tìm thấy file: {local_path}")

if __name__ == "__main__":
    upload_model()