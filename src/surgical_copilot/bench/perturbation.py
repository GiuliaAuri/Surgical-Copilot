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
    MapTransform
)


class RandSpecularReflectiond(MapTransform):
    def __init__(self, keys, prob=0.2, intensity=0.5, sigma_range=(0.05, 0.15)):
        super().__init__(keys)
        self.prob = prob
        self.intensity = intensity
        self.sigma_range = sigma_range

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            if torch.rand(1).item() < self.prob:
                img = d[key]
                C, H, W = img.shape[-3:]
                
                # 1. Genera coordinate casuali per il centro del riflesso
                center_y, center_x = torch.rand(2) * H, torch.rand(2) * W
                
                # 2. Crea una mappa di calore (Gaussiana) ellittica
                y = torch.linspace(0, H - 1, H)
                x = torch.linspace(0, W - 1, W)
                grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
                
                # Variabilità nella forma dell'ellisse
                sigma_y = torch.rand(1).uniform_(*self.sigma_range) * H
                sigma_x = torch.rand(1).uniform_(*self.sigma_range) * W
                
                gaussian = torch.exp(-(((grid_y - center_y)**2 / (2 * sigma_y**2)) + 
                                       ((grid_x - center_x)**2 / (2 * sigma_x**2))))
                
                # 3. Normalizza e applica intensità
                specular_mask = (gaussian / gaussian.max()) * self.intensity
                
                # 4. Applica il riflesso (Additive blending)
                # Usiamo min(1, img + mask) per evitare clipping brutale 
                # e simulare la saturazione del sensore
                d[key] = torch.clamp(img + specular_mask, 0, 1)
        
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
        return RandGaussianNoised(keys="image", prob=p, mean=0.0, std=std)

    @staticmethod
    def gaussian_blur(p=0.3, sigma=(0.5, 1.5)):
        """Simulate motion blur or defocus by applying a Gaussian blur with a randomly selected sigma value."""
        return RandGaussianSmoothd(keys="image", prob=p, sigma_x=sigma, sigma_y=sigma)

    @staticmethod
    def contrast(p=0.3, gamma=(0.7, 1.5)):
        """Simulate changes in lighting conditions by randomly adjusting the contrast of the image."""
        return RandAdjustContrastd(keys="image", prob=p, gamma=gamma)

    @staticmethod
    def intensity_shift(p=0.2, offset=0.1):
        """Simulate changes in lighting conditions by randomly shifting the intensity of the image."""
        return RandShiftIntensityd(keys="image", prob=p, offsets=offset)

    @staticmethod
    def surgical_smoke(p=0.2, intensity=(0.1, 0.3)):
        """Simulate surgical smoke by overlaying a semi-transparent noise pattern that mimics the appearance of smoke."""
        return RandSurgicalSmoked(keys="image", prob=p, intensity_range=intensity)

    @staticmethod
    def specular(p=0.2, intensity=0.1):
        """Simulate specular reflections by adding bright, localized highlights that mimic the appearance of light reflecting off wet or shiny surfaces."""
        return RandSpecularReflectiond(keys="image", prob=p, intensity=intensity)

class PerturbationPipelines:

    @staticmethod
    def get_train_pipeline():
        # same cofiguaration of Hemoset's authors for training
        return Compose([
            #RandSpatialCropd(keys=['image', 'label'], roi_size=(320, 320), random_size=False), 
            RandCropByPosNegLabeld(
                keys=['image', 'label'],
                label_key='label',
                spatial_size=(320, 320),
                pos=1, # Peso per patch con target
                neg=1, # Peso per patch di solo background
                num_samples=1
            ),
            RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.5, 1.5)) 
        ])

    @staticmethod
    def get_eval_scenarios():
        # our expanded version to evalute where model fail and what are the condition
        return {
            "clean": Compose([]),
            "noise_only": Compose([PerturbationFactory.gaussian_noise(p=1.0, std=0.2)]),
            "blur_only": Compose([PerturbationFactory.gaussian_blur(p=1.0)]),
            "intensity_shift_only": Compose([PerturbationFactory.intensity_shift(p=1.0, offset=0.2)]),
            "smoke_only": Compose([PerturbationFactory.surgical_smoke(p=1.0, intensity=(0.2, 0.4))]),
            "contrast_only": Compose([PerturbationFactory.contrast(p=1.0, gamma=(1.5, 2.0))]),
            "specular_only": Compose([PerturbationFactory.specular(p=1.0, intensity=0.15)]),
            "chirurgical_worst_case": Compose([
                PerturbationFactory.gaussian_noise(p=1.0, std=0.2),
                PerturbationFactory.gaussian_blur(p=1.0),
                PerturbationFactory.contrast(p=1.0, gamma=(1.5, 2.0)),
                PerturbationFactory.specular(p=1.0, intensity=0.15),
                PerturbationFactory.surgical_smoke(p=1.0, intensity=(0.2, 0.4)),
                PerturbationFactory.intensity_shift(p=1.0, offset=0.2)
            ])
        }