import torch
from monai.transforms import (
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    RandShiftIntensityd,
    Compose,
    MapTransform
)

class RandSpecularReflectiond(MapTransform):
   
    def __init__(self, keys, prob=0.1, intensity=0.1, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.prob = prob
        self.intensity = intensity

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            if torch.rand(1).item() < self.prob:
                x = d[key]
                mask = (torch.rand_like(x) < self.intensity)
                d[key] = x.clone()
                d[key][mask] = x.max() 
        return d

class PerturbationFactory:

    @staticmethod
    def gaussian_noise(p=0.3, std=0.1):
        return RandGaussianNoised(keys="image", prob=p, mean=0.0, std=std)

    @staticmethod
    def gaussian_blur(p=0.3, sigma=(0.5, 1.5)):
        return RandGaussianSmoothd(keys="image", prob=p, sigma_x=sigma, sigma_y=sigma)

    @staticmethod
    def contrast(p=0.3, gamma=(0.7, 1.5)):
        return RandAdjustContrastd(keys="image", prob=p, gamma=gamma)

    @staticmethod
    def intensity_shift(p=0.2, offset=0.1):
        return RandShiftIntensityd(keys="image", prob=p, offsets=offset)

    @staticmethod
    def specular(p=0.2, intensity=0.1):
        return RandSpecularReflectiond(keys="image", prob=p, intensity=intensity)

class PerturbationPipelines:

    @staticmethod
    def get_train_pipeline():
        return Compose([
            PerturbationFactory.gaussian_noise(p=0.3, std=0.1),
            PerturbationFactory.contrast(p=0.2),
        ])

    @staticmethod
    def get_eval_scenarios():
        return {
            "clean": Compose([]),
            "noise_only": Compose([PerturbationFactory.gaussian_noise(p=1.0, std=0.2)]),
            "blur_only": Compose([PerturbationFactory.gaussian_blur(p=1.0)]),
            "contrast_only": Compose([PerturbationFactory.contrast(p=1.0, gamma=(1.5, 2.0))]),
            "specular_only": Compose([PerturbationFactory.specular(p=1.0, intensity=0.15)]),
            "chirurgical_worst_case": Compose([
                PerturbationFactory.gaussian_noise(p=1.0, std=0.2),
                PerturbationFactory.gaussian_blur(p=1.0),
                PerturbationFactory.contrast(p=1.0, gamma=(1.5, 2.0)),
                PerturbationFactory.specular(p=1.0, intensity=0.15),
            ])
        }