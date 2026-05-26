import random
from pathlib import Path
from collections import defaultdict
import torch

from sklearn.model_selection import GroupKFold, KFold

from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    Resized,
    AsDiscreted,
    ToTensord,
)

class HemosetDataSet:
    def __init__(self, root_dir="data/raw", image_size=(640, 480), seed=42):
        self.root_dir = Path(root_dir)
        self.image_size = image_size

        self.rng = random.Random(seed)

        if not self.root_dir.exists():
            raise FileNotFoundError(f"La directory {self.root_dir} non esiste.")

        self.patient_data = defaultdict(list)

        image_paths = sorted(list(self.root_dir.rglob("*/imgs/**/*.png")))

        for img_path in image_paths:
            # img_path.relative_to(root_dir) diventa "pig1/imgs/imgs/000000.png"
            # .parts[0] estrae esattamente "pig1"
            patient_id = img_path.relative_to(self.root_dir).parts[0]
            
            frame_name = img_path.stem 

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
            ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),            AsDiscreted(keys=["label"], threshold=0.5),
            Resized(keys=["image", "label"], spatial_size=self.image_size, mode=("bilinear", "nearest")),
            ToTensord(keys=["image", "label"], dtype=torch.float32),
        ])

    def get_loaders(self, fold_idx=0, n_splits=5, cache_rate=1.0, batch_size=4, num_workers=4, train_transforms=None):
        
        patients = sorted(list(self.patient_data.keys()))
        self.rng.shuffle(patients)

        if n_splits > len(patients):
            raise ValueError("n_splits > numero di pig")

        patients = patients.copy()
        self.rng.shuffle(patients)

        kf = KFold(
            n_splits=n_splits,
            shuffle=False
        )
        folds = list(kf.split(patients))

        if fold_idx >= len(folds):
            raise ValueError(f"fold_idx deve essere < {n_splits}")


        train_val_idx, test_idx = folds[fold_idx]

        train_val_patients = [
            patients[i]
            for i in train_val_idx
        ]

        test_patients = [
            patients[i]
            for i in test_idx
        ]

        val_size = max(
            1,
            int(0.25 * len(train_val_patients))
        )

        val_patients = train_val_patients[-val_size:]
        train_patients = train_val_patients[:-val_size]

        print("\n[*] Fold info")
        print(f"Train pigs: {train_patients}")
        print(f"Val pigs:   {val_patients}")
        print(f"Test pigs:  {test_patients}")

        train_files = []
        val_files = []
        test_files = []


        for p in train_patients:
            train_files.extend(self.patient_data[p])

        for p in val_patients:
            val_files.extend(self.patient_data[p])

        for p in test_patients:
            test_files.extend(self.patient_data[p])

        self.rng.shuffle(train_files)

        print(
            f"[*] Samples "
            f"train={len(train_files)} "
            f"val={len(val_files)} "
            f"test={len(test_files)}"
        )

        train_compose = (Compose([self.base_transforms,train_transforms]) if train_transforms  else self.base_transforms)

        train_ds = CacheDataset(train_files, transform=train_compose, cache_rate=cache_rate)
        val_ds = CacheDataset(val_files, transform=self.base_transforms, cache_rate=cache_rate)
        test_ds = CacheDataset(test_files, transform=self.base_transforms, cache_rate=cache_rate)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    
        return train_loader, val_loader, test_loader

    def get_sample(self, split="train", index=None, patient_id=None, transform=True):

        patients = sorted(list(self.patient_data.keys()))
        temp_rng = random.Random(42) 
        temp_rng.shuffle(patients)

        split_idx = int(0.8 * len(patients))
        train_patients = patients[:split_idx]
        val_patients = patients[split_idx:]

        selected_patients = train_patients if split == "train" else val_patients

        if patient_id:
            if patient_id not in selected_patients:
                raise ValueError(f"{patient_id} non è nello split {split}")
            files = self.patient_data[patient_id]
        else:
            files = []
            for p in selected_patients:
                files.extend(self.patient_data[p])

        if index is None:
            sample = random.choice(files)
        else:
            sample = files[index % len(files)]

        if transform:
            sample = self.base_transforms(sample)

        return sample