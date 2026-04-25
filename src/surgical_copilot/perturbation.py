import torch
import torch.nn.functional as F


def apply_motion_blur(x, intensity: float):
    if intensity <= 0:
        return x

    k = int(1 + intensity * 7)
    return F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)


def apply_perturbation(x, perturbation: str, intensity: float):

    if perturbation == "none":
        return x

    if perturbation == "gaussian_noise":
        noise = torch.randn_like(x) * intensity * 0.1
        return x + noise

    if perturbation == "gaussian_blur":
        return F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

    if perturbation == "motion_blur":
        return apply_motion_blur(x, intensity)

    if perturbation == "specular":
        mask = (torch.rand_like(x) < intensity * 0.1)
        x = x.clone()
        x[mask] = 1.0
        return x

    raise ValueError(f"Unknown perturbation: {perturbation}")