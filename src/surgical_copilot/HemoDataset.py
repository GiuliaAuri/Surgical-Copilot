import random
from pathlib import Path
from collections import defaultdict
import torch

from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    Resized,
    AsDiscreted,
    ToTensord,
)

class HemosetDataSet:
    def __init__(self, root_dir="data/raw", image_size=(512, 512)):
        self.root_dir = Path(root_dir)
        self.image_size = image_size

        if not self.root_dir.exists():
            raise FileNotFoundError(f"La directory {self.root_dir} non esiste.")

        self.patient_data = defaultdict(list)

        image_paths = sorted(list(self.root_dir.rglob("*/imgs/**/*.png")))

        for img_path in image_paths:
            # img_path.relative_to(root_dir) diventa "pig1/imgs/imgs/000000.png"
            # .parts[0] estrae esattamente "pig1"
            patient_id = img_path.relative_to(self.root_dir).parts[0]
            
            frame_name = img_path.stem 

            # Costruiamo il path atteso della label.
            mask_path_png = self.root_dir / patient_id / "labels" / "labels" /f"{frame_name}_mask.png"

            if mask_path_png.exists():
                final_mask_path = mask_path_png
            
            else:
                print(f"[Warning] Maschera mancante per l'immagine {img_path.name}. Skip.")
                continue

            self.patient_data[patient_id].append({
                "image": str(img_path),
                "label": str(final_mask_path)
            })

        if not self.patient_data:
            raise RuntimeError("Nessun dato accoppiato (img/mask) trovato. Verifica la struttura delle cartelle.")

        print(f"[*] Dataset caricato: trovati {len(self.patient_data)} subjects (pigN) distinti.")
        print(f"[*] Totale frame validi: {sum(len(frames) for frames in self.patient_data.values())}")

        self.base_transforms = Compose([
            LoadImaged(keys=["image", "label"], reader="PILReader"),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityd(keys=["image"]),
            AsDiscreted(keys=["label"], threshold=0.5),
            Resized(keys=["image", "label"], spatial_size=self.image_size, mode=("bilinear", "nearest")),
            ToTensord(keys=["image", "label"]),
        ])

    def get_loaders(self, train_split=0.8, cache_rate=1.0, batch_size=4, num_workers=4, train_transforms=None):
        
        patients = sorted(list(self.patient_data.keys()))
        random.seed(42)
        random.shuffle(patients)

        split_idx = int(train_split * len(patients))
        train_patients = patients[:split_idx]
        val_patients = patients[split_idx:]

        # 3. Costruzione delle liste piatte contenenti i dizionari frame/mask
        train_files = []
        for p in train_patients:
            train_files.extend(self.patient_data[p])
            
        val_files = []
        for p in val_patients:
            val_files.extend(self.patient_data[p])

        # 4. Shuffle finale dei frame di training per stabilizzare i gradienti nell'epoca
        random.shuffle(train_files)

        print(f"[*] Split applicato a livello subject: {len(train_patients)} train / {len(val_patients)} val.")

        # Accodamento delle perturbazioni di train se presenti
        train_compose = Compose([self.base_transforms, train_transforms]) if train_transforms else self.base_transforms

        train_ds = CacheDataset(train_files, transform=train_compose, cache_rate=cache_rate)
        val_ds = CacheDataset(val_files, transform=self.base_transforms, cache_rate=cache_rate)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True  # Stabilizza la BatchNorm e le metriche se il dataset non è divisibile per batch_size
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

        return train_loader, val_loader