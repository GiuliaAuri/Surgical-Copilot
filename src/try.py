import matplotlib.pyplot as plt
from surgical_copilot.HemoDataset import HemoDataset
import torch

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    RandGaussianNoised,       # Per il rumore del sensore
    RandGaussianSmoothd,     # Per l'effetto "sfocato" del fumo
    RandAdjustContrastd,      # Per i riflessi e i cambi di luce
    RandScaleIntensityd,      # Per simulare sovraesposizione
    Resized,
    ToTensord
)
\
def visualize_sample(dataset, idx=0):
    image, mask = dataset[idx]
    
    # Se i dati sono tensori (C, H, W), riportiamoli a (H, W, C) per matplotlib
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()
        mask = mask.squeeze().numpy() # Rimuove la dimensione del canale (1, H, W) -> (H, W)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    ax[0].imshow(image.astype('uint8') if image.max() > 1 else image)
    ax[0].set_title(f"Frame Chirurgico - idx {idx}")
    ax[0].axis('off')
    
    # Usiamo una colormap 'jet' o 'inferno' per evidenziare la maschera (sangue/sorgente)
    ax[1].imshow(mask, cmap='jet')
    ax[1].set_title("Mask (Label)")
    ax[1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Definizione della pipeline di "Stress Test" per il Benchmark
distort_transforms = Compose([
    # 1. Caricamento (se passi i path nel dizionario)
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Resized(keys=["image", "label"], spatial_size=(512, 512)),

    # 2. Simulazione Riflessi / Luci (solo sull'immagine)
    # Aumenta drasticamente l'intensità in zone casuali
    RandScaleIntensityd(keys="image", factors=0.5, prob=0.5),
    RandAdjustContrastd(keys="image", prob=0.5),

    # 3. Simulazione Fumo (Smoothing + Noise)
    # Lo smoothing simula la perdita di dettaglio tipica del fumo denso
    RandGaussianSmoothd(keys="image", sigma_x=(1.0, 4.0), prob=0.3),
    # Il rumore aggiunge la grana del fumo
    RandGaussianNoised(keys="image", prob=0.3, mean=0.2, std=0.1),

    ToTensord(keys=["image", "label"])
])
# Esempio di utilizzo:
dataset = HemoDataset(root_dir="data/raw", transform=distort_transforms)
visualize_sample(dataset, idx=42)