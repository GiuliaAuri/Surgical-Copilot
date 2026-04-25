from pathlib import Path
from monai.data import CacheDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ToTensord
)

def get_hemo_dataset(root_dir="data/raw", cache_rate=1.0):

    root_dir = Path(root_dir)

    image_paths = sorted(list(root_dir.rglob("*/imgs/*.png")))
    mask_paths = sorted(list(root_dir.rglob("*/labels/*.png")))

    if len(image_paths) == 0:
        raise FileNotFoundError(f"Nessuna immagine trovata in {root_dir}")

    if len(image_paths) != len(mask_paths):
        raise RuntimeError(
            f"Mismatch: {len(image_paths)} immagini vs {len(mask_paths)} maschere."
        )

    data_dicts = [
        {"image": str(img), "label": str(mask)}
        for img, mask in zip(image_paths, mask_paths)
    ]

    transforms = Compose([
        LoadImaged(keys=["image", "label"], reader="PILReader"), #load with PIL to preserve RGB channels
        EnsureChannelFirstd(keys=["image", "label"]), # ensure (C,H,W) shape
        ToTensord(keys=["image", "label"]), # convert to PyTorch tensors
    ])

    dataset = CacheDataset( # caching for speed, load all data in RAM to GPU 
        data=data_dicts,
        transform=transforms,
        cache_rate=cache_rate
    )

    return dataset