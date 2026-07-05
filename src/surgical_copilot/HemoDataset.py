import random
from pathlib import Path
from PIL import Image
from collections import defaultdict
import torch
import numpy as np
from collections import defaultdict

from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    Resized,
    AsDiscreted,
    ToTensord,
    NormalizeIntensityd,
    EnsureTyped, 
    MapTransform,
)

class HemosetDataSet:
    def __init__(self, root_dir="data/raw", image_size=(640, 480), seed=42, use_imagenet_norm=False):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.use_imagenet_norm = use_imagenet_norm

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
        
        transforms_list = [
            LoadImaged(keys=["image", "label"], reader="PILReader"),
            
            EnsureChannelFirstd(keys=["image"]),
            EnsureChannelFirstd(keys=["label"]),
            
            ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
            AsDiscreted(keys=["label"], threshold=0.5),
        ]

        if use_imagenet_norm:
            print("[*] Normalizzazione ImageNet ATTIVATA.")
            transforms_list.append(
                NormalizeIntensityd(
                    keys=["image"], 
                    subtrahend=[0.485, 0.456, 0.406], 
                    divisor=[0.229, 0.224, 0.225], 
                    channel_wise=True
                )
            )
               
        transforms_list.extend([    
            Resized(keys=["image", "label"], spatial_size=self.image_size, mode=("bilinear", "nearest")),
            ToTensord(keys=["image", "label"], dtype=torch.float32),
        ])
        self.base_transforms = Compose(transforms_list)
        

    def get_loaders(self, fold_idx=0, n_splits=5, cache_rate=1.0, batch_size=4, num_workers=4, train_transforms=None):
        
        patients = sorted(list(self.patient_data.keys()))

        if n_splits > len(patients):
            raise ValueError("n_splits > numero di pig")
        
        gkf = GroupKFold(n_splits=n_splits)

        folds = list(
                gkf.split(
                    X=patients,
                    y=None,
                    groups=patients
                )
            )

        if fold_idx >= len(folds):
            raise ValueError(f"fold_idx deve essere < {n_splits}")

        train_val_idx, test_idx = folds[fold_idx]
        train_val_patients = [patients[i] for i in train_val_idx]
        test_patients = [patients[i] for i in test_idx]

        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)

        tv_idx, val_idx = next(
            gss.split(train_val_patients, groups=train_val_patients)
        )

        train_patients = [train_val_patients[i] for i in tv_idx]
        val_patients = [train_val_patients[i] for i in val_idx]

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

    def get_sample(self, patient_id=None, index=None, transform=True):
    
        if patient_id:
            if patient_id not in self.patient_data:
                raise ValueError(f"{patient_id} non esiste nel dataset")
            files = self.patient_data[patient_id]
        else:
            files = []
            for p in self.patient_data.keys():
                files.extend(self.patient_data[p])

        if not files:
            raise RuntimeError("Nessun sample disponibile.")

        if index is None:
            sample = random.choice(files)
        else:
            sample = files[index % len(files)]

        if transform:
            sample = self.base_transforms(sample)

        return sample
    
class HemosetEarlyFusion(HemosetDataSet):
    def __init__(self, root_dir="data/raw", image_size=(640, 480), seed=42):
        super().__init__(root_dir, image_size, seed)

        self.patient_samples = defaultdict(list)

        for patient , frames in self.patient_data.items():

            for i in range(len(frames)):

                current = frames[i]

                if i == 0:
                    previous = None
                else:
                    previous = frames[i-1]

                self.patient_samples[patient].append(
                    {
                        "image": current["image"],
                        "label": current["label"],
                        "prev_label": None if previous is None else previous["label"],
                        "is_first_frame": previous is None,
                        "patient_id": patient,  
                    }
                )
        self.base_transforms = Compose([
            LoadImaged(keys=["image", "label"], reader="PILReader"),
            
            EnsureChannelFirstd(keys=["image", "label"]), 
            
            CreatePreviousMaskd(keys=["prev_label"]), 
            
            EnsureChannelFirstd(keys=["prev_label"], channel_dim="no_channel"),
            
            ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
            AsDiscreted(keys=["label", "prev_label"], threshold=0.5),
            
            Resized(
                keys=["image", "label", "prev_label"],
                spatial_size=(self.image_size[0], self.image_size[1]), 
                mode=("bilinear", "nearest", "nearest")
            ),
            
            
            EnsureTyped(keys=["image", "label", "prev_label"]),
            ToTensord(keys=["image", "label", "prev_label"]),
        ])        
        


    def get_loaders(self, fold_idx=0, n_splits=5, cache_rate=1.0, batch_size=4, num_workers=4, train_transforms=None):
        
        patients = sorted(list(self.patient_data.keys()))

        if n_splits > len(patients):
            raise ValueError("n_splits > numero di pig")
        
        gkf = GroupKFold(n_splits=n_splits)

        folds = list(
                gkf.split(
                    X=patients,
                    y=None,
                    groups=patients
                )
            )

        if fold_idx >= len(folds):
            raise ValueError(f"fold_idx deve essere < {n_splits}")

        train_val_idx, test_idx = folds[fold_idx]

        train_val_patients = [patients[i] for i in train_val_idx]

        test_patients = [patients[i] for i in test_idx]

        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)

        tv_idx, val_idx = next(
            gss.split(train_val_patients, groups=train_val_patients)
        )

        train_patients = [train_val_patients[i] for i in tv_idx]
        val_patients = [train_val_patients[i] for i in val_idx]

        print("\n[*] Fold info")
        print(f"Train pigs: {train_patients}")
        print(f"Val pigs:   {val_patients}")
        print(f"Test pigs:  {test_patients}")

        train_files = []
        val_files = []
        test_files = []

        for p in train_patients:
            train_files.extend(self.patient_samples[p])

        for p in val_patients:
            val_files.extend(self.patient_samples[p])

        for p in test_patients:
            test_files.extend(self.patient_samples[p])

        print(
            f"[*] Samples "
            f"train={len(train_files)} "
            f"val={len(val_files)} "
            f"test={len(test_files)}"
        )
        
        #train_compose = (Compose([self.base_transforms,train_transforms]) if train_transforms  else self.base_transforms)

        if train_transforms:
            train_compose = Compose([self.base_transforms, train_transforms])
        else:
            train_compose = self.base_transforms

        train_ds = CacheDataset(train_files, transform=train_compose, cache_rate=cache_rate)
        val_ds = CacheDataset(val_files, transform=self.base_transforms, cache_rate=cache_rate)
        test_ds = CacheDataset(test_files, transform=self.base_transforms, cache_rate=cache_rate)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

        return train_loader, val_loader, test_loader

    def get_sample(self, patient_id=None, index=None, transform=True):

        if patient_id is None:
            patient_id = random.choice(list(self.patient_samples.keys()))

        if patient_id not in self.patient_samples:
            raise ValueError(f"Patient '{patient_id}' non trovato.")

        files = self.patient_samples[patient_id]

        if len(files) == 0:
            raise ValueError(f"Nessun sample disponibile per '{patient_id}'.")

        if index is None:
            index = random.randrange(len(files))
        elif index < 0 or index >= len(files):
            raise IndexError(
                f"Index {index} fuori range [0, {len(files)-1}] per il paziente '{patient_id}'."
            )

        sample = files[index].copy()

        if transform:
            sample = self.base_transforms(sample)

        return sample

class HemosetDataSequences(HemosetDataSet):
    def __init__(self, root_dir="data/raw", image_size=(640, 480), seed=42, sequence_length=5, overlapping=0.75):
        super().__init__(root_dir, image_size, seed)

        self.sequence_length = sequence_length
        self.overlapping = overlapping

        self.frame_transforms = Compose([
            LoadImaged(keys=["image", "label"], reader="PILReader"),
            EnsureChannelFirstd(keys=["image", "label"]),

            ScaleIntensityRanged(
                keys=["image"],
                a_min=0,
                a_max=255,
                b_min=0.0,
                b_max=1.0,
                clip=True
            ),

            AsDiscreted(keys=["label"], threshold=0.5),
            ToTensord(keys=["image", "label"], dtype=torch.float32),
        ])

        self.base_transforms = Compose([
            SequenceTransform(self.frame_transforms)
        ])

    def get_loaders(self, fold_idx=0, n_splits=5, cache_rate=1.0, batch_size=4, num_workers=4, train_transforms=None):
        
        patients = sorted(list(self.patient_data.keys()))

        if n_splits > len(patients):
            raise ValueError("n_splits > numero di pig")
        
        gkf = GroupKFold(n_splits=n_splits)

        folds = list(
                gkf.split(
                    X=patients,
                    y=None,
                    groups=patients
                )
            )

        if fold_idx >= len(folds):
            raise ValueError(f"fold_idx deve essere < {n_splits}")

        train_val_idx, test_idx = folds[fold_idx]

        train_val_patients = [patients[i] for i in train_val_idx]

        test_patients = [patients[i] for i in test_idx]

        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)

        tv_idx, val_idx = next(
            gss.split(train_val_patients, groups=train_val_patients)
        )

        train_patients = [train_val_patients[i] for i in tv_idx]
        val_patients = [train_val_patients[i] for i in val_idx]

        print("\n[*] Fold info")
        print(f"Train pigs: {train_patients}")
        print(f"Val pigs:   {val_patients}")
        print(f"Test pigs:  {test_patients}")

        train_files = self._create_sliding_window(train_patients)
        val_files = self._create_sliding_window(val_patients)
        test_files = self._create_sliding_window(test_patients)

        print(
            f"[*] Sequence Samples "
            f"train={len(train_files)} "
            f"val={len(val_files)} "
            f"test={len(test_files)}"
        )

        train_transforms_video = VideoConsistentWrapper(transforms=train_transforms, keys=["image", "label"], appearance_mode="shared")
        train_compose = (Compose([self.base_transforms,train_transforms_video]) if train_transforms  else self.base_transforms)

        train_ds = CacheDataset(train_files, transform=train_compose, cache_rate=cache_rate)
        val_ds = CacheDataset(val_files, transform=self.base_transforms, cache_rate=cache_rate)
        test_ds = CacheDataset(test_files, transform=self.base_transforms, cache_rate=cache_rate)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

        return train_loader, val_loader, test_loader

    def _create_sliding_window(self, patients_list):

        sequences = []
        
        for p in patients_list:

            # for each patient, get the list of frames
            # guarantee the indipendence of the sequences between patients
            patient_frames = self.patient_data[p]
            seq_len = self.sequence_length

            # sequence are contiguous and partially overlapped
            # thanks to the stride frame in different sequences are correlated
            stride = int(seq_len * (1 - self.overlapping)) if self.overlapping > 0 else seq_len
            
            # Sliding window with stride
            for i in range(0, len(patient_frames) - seq_len + 1, stride):

                # Example of how the sliding window works:
                # sequence_length = 5, stride = 3
                # [0 1 2 3 4]
                #     [2 3 4 5 6]
                #         [4 5 6 7 8]

                window = patient_frames[i : i + seq_len]

                images= [frame["image"] for frame in window]
                labels= [frame["label"] for frame in window]
                
                seq_sample = {
                    "image": images,
                    "label": labels,
                    "sequence_id": f"{p}_{i}",
                    "patient_id": p,
                    "start_idx": i
                }
                sequences.append(seq_sample)

        return sequences
    
    def get_sample(self, patient_id=None, index=None, transform=False):
        if patient_id is not None:
            if patient_id not in self.patient_data:
                raise ValueError(f"{patient_id} non esiste nel dataset")
            patients_to_sample = [patient_id]
        else:
            patients_to_sample = list(self.patient_data.keys())

        sequences = self._create_sliding_window(patients_to_sample)

        if len(sequences) == 0:
            raise RuntimeError("Nessuna sequenza disponibile.")

        if index is None:
            sample = random.choice(sequences)
        else:
            sample = sequences[index % len(sequences)]

        if transform:
            sample = self.base_transforms(sample)

        return sample
    
# -----

from monai.transforms import MapTransform, Compose

class SequenceTransform(MapTransform):
    
    def __init__(self, frame_transform):
        self.frame_transform = frame_transform

    def __call__(self, data):

        images = []
        labels = []

        for img_path, lbl_path in zip(data["image"], data["label"]):

            sample = {
                "image": img_path,
                "label": lbl_path,
            }

            sample = self.frame_transform(sample)

            images.append(sample["image"])
            labels.append(sample["label"])

        data["image"] = torch.stack(images)   # (T,C,H,W)
        data["label"] = torch.stack(labels)   # (T,1,H,W)

        return data

class UnflattenSequenced(MapTransform):
    """
    Reshape the tensors from (S*C, H, W) -> (S, C, H, W).
    """
    def __init__(self, keys, sequence_length, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.seq_len = sequence_length

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d:
                SC, H, W = d[key].shape
                
                C = SC // self.seq_len
                if d[key].ndim != 3:
                    raise ValueError(f"Unexpected shape: {d[key].shape}")
                d[key] = d[key].view(self.seq_len, C, H, W)
                
        return d
    
class CreatePreviousMaskd(MapTransform):
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            prev = d[key]

            if prev is None:
                shape = d["label"].shape
                d[key] = torch.zeros_like(d["label"])

            elif isinstance(prev, str):
                loaded = LoadImaged(
                    keys=[key],
                    reader="PILReader",
                    image_only=False
                )({key: prev})

                d[key] = loaded[key]

            else:
                d[key] = prev

        return d

from monai.transforms import (
    RandSpatialCropd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotated,
    Rand2DElasticd
)


class VideoConsistentWrapper(MapTransform):

    def __init__(self, transforms, keys=["image", "label"], appearance_mode="shared"):
        super().__init__(keys)
        self.transforms = Compose(transforms)
        self.appearance_mode = appearance_mode

    def __call__(self, data):

        d = dict(data)

        images = torch.as_tensor(d["image"])
        masks  = torch.as_tensor(d["label"])

        if images.ndim != 4:
            raise ValueError(f"Expected (T,C,H,W), got {images.shape}")

        T = images.shape[0]

        seed = torch.randint(0, 1_000_000, (1,)).item()

        out_img, out_msk = [], []

        for t in range(T):

            if self.appearance_mode == "shared":
                torch.manual_seed(seed)
            else:
                torch.manual_seed(torch.randint(0, 1_000_000, (1,)).item())

            frame = {
                "image": images[t],
                "label": masks[t]
            }

            frame = self.transforms(frame)

            out_img.append(frame["image"])
            out_msk.append(frame["label"])

        d["image"] = torch.stack(out_img)
        d["label"] = torch.stack(out_msk)

        return d