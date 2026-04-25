import hydra
import os
import json
import torch
import torch.utils.data

from omegaconf import DictConfig
from hydra.utils import instantiate

from surgical_copilot.bench.BenchmarkEngine import BenchmarkEngine


@hydra.main(
    version_base=None,
    config_path="../../../configs",
    config_name="config"
)
def run_benchmark(cfg: DictConfig):

    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    model = instantiate(cfg.model).to(device)
    dataset = instantiate(cfg.data)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.robustness.batch_size,
        shuffle=False,
        num_workers=cfg.robustness.num_workers
    )

    model.eval()

    engine = BenchmarkEngine(
        model=model,
        loader=loader,
        cfg=cfg,
        device=device
    )

    results = engine.run()

    os.makedirs("metrics", exist_ok=True)

    out_path = f"metrics/{cfg.model._target_.split('.')[-1]}_benchmark.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"[DONE] Saved to {out_path}")


if __name__ == "__main__":
    run_benchmark()