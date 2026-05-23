import torch
import torch.nn as nn
import torch.nn.functional as F

class CBS(nn.Module):
    """Conv2d + BatchNorm + SiLU"""
    def __init__(self, c_in, c_out, k=1, s=1, p=None, g=1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, p if p is not None else k // 2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    """Bottleneck standard con connessione residua."""
    def __init__(self, c_in, c_out, shortcut=True):
        super().__init__()
        c_ = c_out // 2  # canali nascosti
        self.cv1 = CBS(c_in, c_, k=3, s=1)
        self.cv2 = CBS(c_, c_out, k=3, s=1)
        self.add = shortcut and c_in == c_out

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    """CSP Bottleneck con 2 convoluzioni"""
    def __init__(self, c_in, c_out, n=1, shortcut=False):
        super().__init__()
        self.c = int(c_out * 0.5)  # canali nascosti
        self.cv1 = CBS(c_in, 2 * self.c, 1, 1)
        self.cv2 = CBS((2 + n) * self.c, c_out, 1, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class SPPF(nn.Module):
    """Spatial Pyramid Pooling Fast"""
    def __init__(self, c_in, c_out, k=5):
        super().__init__()
        c_ = c_in // 2
        self.cv1 = CBS(c_in, c_, 1, 1)
        self.cv2 = CBS(c_ * 4, c_out, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), 1))
    

class YOLOv8Backbone(nn.Module):
    def __init__(self, in_channels=3, channels=[64, 128, 256]):
        super().__init__()
    
        # Stem
        self.stem = nn.Sequential(
            CBS(in_channels, 16, 3, 2),  # [B, 16, 320, 320]
            CBS(16, 32, 3, 2),  
            C2f(32, 32, n=1, shortcut=True)
        )
        
        # Stage 1 -> P3 (Alta risoluzione, dettagli fini)
        self.stage1 = nn.Sequential(CBS(32, channels[0], 3, 2), C2f(channels[0], channels[0], n=2, shortcut=True))
        # Stage 2 -> P4 (Risoluzione media)
        self.stage2 = nn.Sequential(CBS(channels[0], channels[1], 3, 2), C2f(channels[1], channels[1], n=2, shortcut=True))
        # Stage 3 -> P5 (Bassa risoluzione, contesto profondo)
        self.stage3 = nn.Sequential(CBS(channels[1], channels[2], 3, 2), C2f(channels[2], channels[2], n=1, shortcut=True), SPPF(channels[2], channels[2]))

    def forward(self, x):
        x = self.stem(x)
        p3 = self.stage1(x)
        p4 = self.stage2(p3)
        p5 = self.stage3(p4)
        return p3, p4, p5  # Restituiamo le 3 scale!

class YOLOv8Neck(nn.Module):
    def __init__(self, channels=[64, 128, 256]):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        # Fusion Top-Down
        self.n1 = C2f(channels[1] + channels[2], channels[1], n=1)
        self.n2 = C2f(channels[0] + channels[1], channels[0], n=1)

    def forward(self, p3, p4, p5):
        # Percorso discendente (Top-Down)
        x = self.n1(torch.cat([self.up(p5), p4], dim=1))
        n3 = self.n2(torch.cat([self.up(x), p3], dim=1))
        return n3  # Feature map arricchite finali

class YOLOv8SegHead(nn.Module):
    def __init__(self, in_channels, num_classes=1, num_masks=32):
        super().__init__()
        # 1. ProtoHead: Genera i prototipi ad alta risoluzione (Scale up x2)
        self.proto_cv1 = CBS(in_channels, in_channels, 3, 1)
        self.proto_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.proto_cv2 = CBS(in_channels, num_masks, 1, 1)

        # 2. CoeffHead: Genera i coefficienti per ogni oggetto
        self.coeff_cv = nn.Conv2d(in_channels, num_masks, 1)
        
        # 3. ClassHead: Prevede classe (Strumento o Sangue)
        self.class_cv = nn.Conv2d(in_channels, num_classes, 1)

    def forward(self, x):
        # Calcolo Prototipi [B, 32, H/4, W/4]
        protos = self.proto_cv2(self.proto_up(self.proto_cv1(x)))
        # Calcolo Coefficienti e Classi [B, 32, H/8, W/8] e [B, num_classes, H/8, W/8]
        coeffs = self.coeff_cv(x)
        classes = self.class_cv(x)
        return classes, coeffs, protos

# --- Modello FINALE ---
class YOLOv8Segmenter(nn.Module):

    def __init__(self, in_channels=3, num_classes=2, num_masks=32):
        super().__init__()
        # Struttura modulare
        self.backbone = YOLOv8Backbone(in_channels=in_channels)
        self.neck = YOLOv8Neck() 
        self.head = YOLOv8SegHead(in_channels=64, num_classes=num_classes, num_masks=num_masks)
    

    def forward(self, x):
        # 1. Estrazione spaziale
        p3, p4, p5 = self.backbone(x)
        
        # 2. Fusione multi-scala
        # In futuro, potrai inserire la ConvLSTM qui, passando le feature temporali!
        n3 = self.neck(p3, p4, p5)
        
        # 3. Generazione Maschere e Bounding Box
        classes, coeffs, protos = self.head(n3)
        return classes, coeffs, protos