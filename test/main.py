
from pathlib import Path
from PIL import Image
import numpy as np

# Crea directory structure
root = Path("data/raw/pig1")
img_dir = root / "imgs"
label_dir = root / "labels" / "labels"

img_dir.mkdir(parents=True, exist_ok=True)
label_dir.mkdir(parents=True, exist_ok=True)

print(f"[*] Creando dati di test in {root}")

# Genera 3 sample: immagini RGB 480x640 e maschere binarie
for i in range(3):
    # Immagine: rumore casuale RGB
    img_array = (np.random.rand(480, 640, 3) * 255).astype('uint8')
    img = Image.fromarray(img_array, mode='RGB')
    img.save(img_dir / f"{i:06d}.png")
    
    # Maschera: binaria (0 o 255), poco sangue (5% di pixel)
    mask_array = (np.random.rand(480, 640) > 0.95).astype('uint8') * 255
    mask = Image.fromarray(mask_array, mode='L')
    mask.save(label_dir / f"{i:06d}_mask.png")
    
    print(f"  [{i}] {i:06d}.png + {i:06d}_mask.png creati")

print("[OK] 3 sample creati. Struttura pronta per il benchmark.")
