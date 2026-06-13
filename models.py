#pylint:disable=all
import torch
import torch.nn as nn
import torchvision.models as models
import lightning as L
import torchmetrics

# ==========================================
# 1. KIẾN TRÚC GIỌNG NÓI (VOICE ANTI-SPOOF)
# ==========================================
class MaxFeatureMap(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = out_channels
        
    def forward(self, x):
        x1, x2 = x.split(self.out_channels, dim=1)
        return torch.max(x1, x2)

class VoiceAntiSpoof(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 5, padding=2), MaxFeatureMap(32),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 96, 1), MaxFeatureMap(48),
            nn.Conv2d(48, 96, 3, padding=1), MaxFeatureMap(48),
            nn.MaxPool2d(2, 2), nn.BatchNorm2d(48)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(48, 2) # 0: Spoof (Giả mạo), 1: Real (Người thật)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1) # (B, 1, 80, T)
        feat = self.features(x)
        return self.classifier(feat)

# ==========================================
# 2. KIẾN TRÚC KHUÔN MẶT (FACE ENCODER)
# ==========================================
class FaceEncoder(nn.Module):
    def __init__(self, embedding_dim=512, pretrained=False):
        super(FaceEncoder, self).__init__()
        # Lúc inference không cần tải lại pretrained weights của MobileNet
        backbone = models.mobilenet_v3_large(weights=None)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        
        self.projection = nn.Sequential(
            nn.Linear(960, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        embedding = self.projection(x)
        return nn.functional.normalize(embedding, p=2, dim=1)

# ==========================================
# 3. KIẾN TRÚC DUNG HỢP (LATE FUSION)
# ==========================================
class MultimodalLateFusion(L.LightningModule):
    def __init__(self, voice_ckpt_path=None, num_classes=2):
        super().__init__()
        
        # Khởi tạo 2 mạng lõi 
        self.face_encoder = FaceEncoder(pretrained=False)
        self.voice_model = VoiceAntiSpoof()
        
        combined_dim = 512 + 2 
        self.fusion_mlp = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, face_imgs, voice_mels):
        # Forward Pass
        f_face = self.face_encoder(face_imgs)            
        f_voice_logits = self.voice_model(voice_mels)    
            
        combined = torch.cat([f_face, f_voice_logits], dim=1) 
        return self.fusion_mlp(combined)