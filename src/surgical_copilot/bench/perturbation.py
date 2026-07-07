import torch
import torch.nn.functional as F
from monai.transforms import (
    RandSpatialCropd,
    RandCropByPosNegLabeld,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    RandShiftIntensityd,
    Compose,
    RandRotated,
    RandAffined,
    RandFlipd,
    Rand2DElasticd,
    Rand2DElasticd,
    MapTransform, 
    Lambdad,
)

import random


class VideoRandCropByPosNegLabeld(MapTransform):
    """
    Video version of RandCropByPosNegLabeld.

    A crop location is sampled from a single frame (anchor frame) and then
    applied consistently to every frame of the clip.

    Expected shapes
    ----------------
    image : (T, C, H, W)
    label : (T, 1, H, W)
    """

    def __init__(
        self,
        keys,
        label_key,
        spatial_size,
        pos=1,
        neg=1,
        anchor="center",  # "center" | "random"
        allow_missing_keys=False,
    ):
        super().__init__(keys, allow_missing_keys)

        self.label_key = label_key
        self.crop_h, self.crop_w = spatial_size

        self.pos = pos
        self.neg = neg
        self.anchor = anchor

    def _choose_anchor(self, T):

        if self.anchor == "center":
            return T // 2

        if self.anchor == "random":
            return random.randint(0, T - 1)

        raise ValueError(f"Unknown anchor {self.anchor}")

    def __call__(self, data):

        d = dict(data)

        label_video = d[self.label_key]

        # (T,1,H,W)
        T, _, H, W = label_video.shape

        anchor = self._choose_anchor(T)

        mask = label_video[anchor, 0]

        sample_positive = random.random() < self.pos / (self.pos + self.neg)

        if sample_positive and torch.any(mask > 0):

            coords = torch.nonzero(mask > 0)

        else:

            coords = torch.nonzero(mask == 0)

            if len(coords) == 0:
                coords = torch.nonzero(torch.ones_like(mask))

        idx = random.randint(0, len(coords) - 1)

        cy, cx = coords[idx].tolist()

        y0 = cy - self.crop_h // 2
        x0 = cx - self.crop_w // 2

        y0 = max(0, min(y0, H - self.crop_h))
        x0 = max(0, min(x0, W - self.crop_w))

        y1 = y0 + self.crop_h
        x1 = x0 + self.crop_w

        for key in self.keys:

            d[key] = d[key][..., y0:y1, x0:x1]

        return d


class VideoConsistentWrapper(MapTransform):

    def __init__(self, spatial_transform, frame_transform, keys=None):
        super().__init__(keys or ["current_image", "current_label"])
        self.spatial_transform = spatial_transform
        self.frame_transform = frame_transform

    def __call__(self, data):
        d = dict(data)

        img_key = self.keys[0] 
        lbl_key = self.keys[1]

        images = d[img_key]  
        masks  = d[lbl_key]

        # Controllo dinamico: siamo nel Dataset (4D) o nell'Engine (5D)?
        is_batched = images.ndim == 5
        
        if not is_batched:
            # Aggiungiamo una dimensione batch fittizia per uniformare la logica
            images = images.unsqueeze(0)
            masks = masks.unsqueeze(0)

        B = images.shape[0]
        out_images, out_masks = [], []

        for b in range(B):
            # Estraiamo il video b-esimo: (T, C, H, W)
            video = {"current_image": images[b], "current_label": masks[b]}

            # 1. Transform spaziale (T funge da batch, garantendo consistenza broadcasting)
            seed = torch.randint(0, 1_000_000, (1,)).item()
            torch.manual_seed(seed)
            video = self.spatial_transform(video)

            imgs, msks = video["current_image"], video["current_label"]
            
            frames_img, frames_msk = [], []
            T_current = imgs.shape[0]

            # 2. Transform temporale frame-specifico
            for t in range(T_current):
                frame = {"current_image": imgs[t], "current_label": msks[t]}
                frame = self.frame_transform(frame)
                frames_img.append(frame["current_image"])
                frames_msk.append(frame["current_label"])

            out_images.append(torch.stack(frames_img))
            out_masks.append(torch.stack(frames_msk))

        # Ricomponiamo i tensori
        final_images = torch.stack(out_images)
        final_masks = torch.stack(out_masks)

        # Se il dato in ingresso era 4D (Dataset), rimuoviamo il batch fittizio
        if not is_batched:
            final_images = final_images.squeeze(0)
            final_masks = final_masks.squeeze(0)

        d[img_key] = final_images
        d[lbl_key] = final_masks

        return d

class RandSpecularReflectiond(MapTransform):
    
    def __init__(self, keys, prob=0.1, intensity=0.1, blob_size=16, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.prob = prob
        self.intensity = intensity
        self.blob_size = blob_size

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            if torch.rand(1).item() < self.prob:
                x = d[key]
                
                if x.ndim == 4:
                    B, C, H, W = x.shape
                elif x.ndim == 3:
                    B = 1
                    C, H, W = x.shape
                else:
                    continue

                small_h, small_w = max(1, H // self.blob_size), max(1, W // self.blob_size)
                noise = torch.rand((1, 1, small_h, small_w), device=x.device, dtype=x.dtype)
                
                blobby_noise = F.interpolate(noise, size=(H, W), mode='bicubic', align_corners=False).squeeze(0)
                
                threshold = 1.0 - (self.intensity * 0.5)
                soft_mask = torch.clamp((blobby_noise - threshold) / (1.0 - threshold), 0, 1) ** 3

                color_tint = torch.tensor([1.0, 
                                           1.0 - torch.rand(1).item() * 0.1, 
                                           1.0 - torch.rand(1).item() * 0.2], 
                                          device=x.device).view(3, 1, 1)
                
                reflection_color = x.max() * color_tint
                d[key] = torch.clamp((1.0 - soft_mask) * x + soft_mask * reflection_color, 0, 1)
                
        return d

class RandSurgicalSmoked(MapTransform):
    def __init__(self, keys, prob=0.2, intensity_range=(0.1, 0.4), allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.prob = prob
        self.intensity_range = intensity_range

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            if torch.rand(1).item() < self.prob:
                img = d[key]

                if img.ndim == 4:
                    B, C, H, W = img.shape
                elif img.ndim == 3:
                    B = 1
                    C, H, W = img.shape
                else:
                    continue
                
                smoke_low_res = torch.rand((1, H // 32, W // 32), device=img.device)
                smoke_mask = F.interpolate(smoke_low_res.unsqueeze(0), size=(H, W), mode='bicubic', align_corners=False).squeeze(0)
                
                intensity = torch.empty(1).uniform_(*self.intensity_range).item()
                smoke_mask = (smoke_mask - smoke_mask.min()) / (smoke_mask.max() - smoke_mask.min()) * intensity
                
                d[key] = torch.clamp(img * (1.0 - smoke_mask) + smoke_mask, 0, 1)
        return d

class PerturbationFactory:

    @staticmethod
    def gaussian_noise(p=0.3, std=0.1):
        """Simulate sensor noise by adding Gaussian noise with a specified standard deviation."""
        return RandGaussianNoised(keys="current_image", prob=p, mean=0.0, std=std)

    @staticmethod
    def gaussian_blur(p=0.3, sigma=(0.5, 1.5)):
        """Simulate motion blur or defocus by applying a Gaussian blur with a randomly selected sigma value."""
        return RandGaussianSmoothd(keys="current_image", prob=p, sigma_x=sigma, sigma_y=sigma)

    @staticmethod
    def contrast(p=0.3, gamma=(0.7, 1.5)):
        """Simulate changes in lighting conditions by randomly adjusting the contrast of the image."""
        return RandAdjustContrastd(keys="current_image", prob=p, gamma=gamma)

    @staticmethod
    def intensity_shift(p=0.2, offset=0.1):
        """Simulate changes in lighting conditions by randomly shifting the intensity of the image."""
        return RandShiftIntensityd(keys="current_image", prob=p, offsets=offset)

    @staticmethod
    def surgical_smoke(p=0.2, intensity=(0.1, 0.3)):
        """Simulate surgical smoke by overlaying a semi-transparent noise pattern that mimics the appearance of smoke."""
        return RandSurgicalSmoked(keys="current_image", prob=p, intensity_range=intensity)

    @staticmethod
    def specular(p=0.2, intensity=0.1):
        """Simulate specular reflections by adding bright, localized highlights that mimic the appearance of light reflecting off wet or shiny surfaces."""
        return RandSpecularReflectiond(keys="current_image", prob=p, intensity=intensity)


class Force2DTransformd(MapTransform):
    """
    Trasformatore che normalizza i dati a (C, H, W) prima del crop,
    indipendentemente da quante dimensioni extra (Z o T) siano presenti.
    """
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            x = d[key]
            # Se è (T, C, H, W) o (C, Z, H, W) -> riduciamo a (C, H, W)
            if x.ndim == 4:
                # Prendiamo il frame centrale (T//2) o il primo (0) per normalizzare
                d[key] = x[0] if x.shape[0] < x.shape[1] else x[:, 0, :, :]
            # Se è già (C, H, W) o (1, H, W) non facciamo nulla
        return d

class PerturbationPipelines:

    KEY_MAPS = {
        "none": ["current_image", "current_label"],
        "early_fusion": ["current_image", "current_label", "prev_label"],
        "late_fusion": ["current_image", "current_label"]
    }

    

    @staticmethod
    def get_train_pipeline(mode="none", is_sequential=False):
        dynamic_keys = PerturbationPipelines.KEY_MAPS[mode]

        spatial_list = [
            Force2DTransformd(keys=dynamic_keys),
            RandSpatialCropd(keys=dynamic_keys, roi_size=(320, 320), random_size=False),
            RandCropByPosNegLabeld(
                keys=dynamic_keys,
                label_key="current_label",
                spatial_size=(320, 320),
                pos=2, neg=1, num_samples=2
            ),
            RandFlipd(keys=dynamic_keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=dynamic_keys, prob=0.5, spatial_axis=1),
            RandRotated(
                keys=dynamic_keys, 
                prob=0.3, 
                range_x=0.4, 
                mode=["bilinear"] + ["nearest"] * (len(dynamic_keys) - 1)
            ),
            Rand2DElasticd(
                keys=dynamic_keys, 
                prob=0.2, 
                spacing=(20, 20), 
                magnitude_range=(1, 2), 
                mode=["bilinear"] + ["nearest"] * (len(dynamic_keys) - 1)
            )
        ]

        appearance_list = [
            RandAdjustContrastd(keys=["current_image"], prob=0.5, gamma=(0.5, 1.5)),
            PerturbationFactory.gaussian_noise(),
            PerturbationFactory.gaussian_blur(),
            PerturbationFactory.specular(),
            PerturbationFactory.surgical_smoke(),
            PerturbationFactory.intensity_shift(),
        ]

    
        if not is_sequential:
            return Compose(spatial.transforms + appearance.transforms)

        return (
            Compose(spatial_list),
            Compose(appearance_list),
        )

    @staticmethod
    def get_eval_scenarios(mode="none", is_sequential=False):

        dynamic_keys = PerturbationPipelines.KEY_MAPS[mode]

        base_pipeline = {
            "clean": Compose([]),

            "noise_only": Compose([
                PerturbationFactory.gaussian_noise(p=1.0, std=0.2)
            ]),

            "blur_only": Compose([
                PerturbationFactory.gaussian_blur(p=1.0)
            ]),

            "intensity_shift_only": Compose([
                PerturbationFactory.intensity_shift(p=1.0, offset=0.2)
            ]),

            "smoke_only": Compose([
                PerturbationFactory.surgical_smoke(p=1.0, intensity=(0.2, 0.4))
            ]),

            "contrast_only": Compose([
                PerturbationFactory.contrast(p=1.0, gamma=(1.5, 2.0))
            ]),

            "specular_only": Compose([
                PerturbationFactory.specular(p=1.0, intensity=0.15)
            ]),

            "chirurgical_worst_case": Compose([
                PerturbationFactory.gaussian_noise(p=1.0, std=0.2),
                PerturbationFactory.gaussian_blur(p=1.0),
                PerturbationFactory.contrast(p=1.0, gamma=(1.5, 2.0)),
                PerturbationFactory.specular(p=1.0, intensity=0.15),
                PerturbationFactory.surgical_smoke(p=1.0, intensity=(0.2, 0.4)),
                PerturbationFactory.intensity_shift(p=1.0, offset=0.2)
            ]),
        }

        if not is_sequential:
            return base_pipeline

        return {
            k: VideoConsistentWrapper(
                spatial_transform=v,
                frame_transform=lambda x: x,
            )
            for k, v in base_pipeline.items()
        }