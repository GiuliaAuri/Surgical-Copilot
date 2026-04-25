import time
import numpy as np
import torch

from tqdm import tqdm
from surgical_copilot.perturbation import apply_perturbation

import time
import numpy as np
import torch

from tqdm import tqdm
from surgical_copilot.perturbation import apply_perturbation


import time
import numpy as np
import torch

from tqdm import tqdm
from surgical_copilot.perturbation import apply_perturbation


class BenchmarkEngine:

    def __init__(self, model, loader, cfg, device):
        self.model = model
        self.loader = loader
        self.cfg = cfg
        self.device = device

        self._print_model_info()

    def _print_model_info(self):

        n_params = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print("\n" + "=" * 60)
        print("MODEL INFO")
        print("=" * 60)
        print(f"Name (cfg):        {getattr(self.cfg.model, 'name', 'unknown')}")
        print(f"Class:             {self.model.__class__.__name__}")
        print(f"Device:            {self.device}")
        print(f"Parameters:        {n_params:,}")
        print(f"Trainable params:  {trainable:,}")
        print("=" * 60 + "\n")

    def run(self):

        results = []

        use_cuda = self.device.type == "cuda"

        total_configs = len(self.cfg.robustness.perturbations) * len(self.cfg.robustness.intensities)
        config_pbar = tqdm(total=total_configs, desc="Benchmark configs")

        for perturb in self.cfg.robustness.perturbations:
            for intensity in self.cfg.robustness.intensities:

                dsc_scores = []
                latencies = []

                batch_pbar = tqdm(self.loader, desc=f"{perturb} | {intensity}", leave=False)

                for batch_idx, batch in enumerate(batch_pbar):

                    x = batch["image"].to(self.device)
                    y = batch["label"].to(self.device)

                    x = apply_perturbation(x, perturb, intensity)

                    
                    if use_cuda:
                        starter = torch.cuda.Event(enable_timing=True)
                        ender = torch.cuda.Event(enable_timing=True)

                        starter.record()

                        with torch.no_grad():
                            out = self.model(x)

                        ender.record()
                        torch.cuda.synchronize()

                        latency = starter.elapsed_time(ender) / 1000.0

                    else:
                        start = time.time()

                        with torch.no_grad():
                            out = self.model(x)

                        latency = time.time() - start

                    latencies.append(latency)

                    if isinstance(out, dict):
                        out = out.get("logits", list(out.values())[0])

                    pred = (torch.sigmoid(out) > 0.5).float()

                    dsc = compute_dsc(pred, y)
                    dsc_scores.append(dsc)

                    fps = 1.0 / latency if latency > 0 else 0.0

                    batch_pbar.set_postfix({
                        "DSC": f"{dsc:.3f}",
                        "FPS": f"{fps:.1f}",
                        "Model": self.model.__class__.__name__
                    })

                results.append({
                    "perturbation": perturb,
                    "intensity": float(intensity),
                    "dice_mean": float(np.mean(dsc_scores)),
                    "dice_std": float(np.std(dsc_scores)),
                    "fps_mean": float(1.0 / np.mean(latencies)),
                    "fps_min": float(1.0 / np.max(latencies)),
                    "model": self.model.__class__.__name__,
                })

                config_pbar.update(1)

        config_pbar.close()

        return results

def compute_dsc(pred, target, eps=1e-6):

    pred = (pred > 0.5).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    return ((2. * intersection + eps) / (union + eps)).item()