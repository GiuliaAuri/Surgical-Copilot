import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from surgical_copilot.bench.BenchmarkEngine import BenchmarkEngine
from surgical_copilot.HemoDataset import HemosetDataSet
from surgical_copilot.perturbation import PerturbationPipelines

@hydra.main(version_base=None, config_path="../../../configs", config_name="config") 
def benchmarking(cfg: DictConfig):

    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # dataset
    print(f"[*] Loading dataset") 
    dataset = HemosetDataSet(root_dir=cfg.data.root_dir)
    
    train_loader, val_loader = dataset.get_loaders(
        train_split=0.8, 
        cache_rate=1.0,
        batch_size=cfg.trainer.trainer.batch_size,
        num_workers=cfg.trainer.trainer.num_workers,
        train_transforms=PerturbationPipelines.get_train_pipeline()
    )

    # model
    print(f"[*] Loading model: {cfg.model._target_}") 
    model = instantiate(cfg.model).to(device)

    # training components
    optimizer = instantiate(cfg.trainer.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.trainer.scheduler, optimizer=optimizer)
    loss_fn = instantiate(cfg.trainer.loss)
    scaler = instantiate(cfg.trainer.scaler)

    # trainer
    engine = BenchmarkEngine(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        loss_fn=loss_fn, 
        scaler=scaler, 
        cfg=cfg, 
        device=device
    )

    if cfg.logging.wandb_enabled:
        resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(
            project=cfg.logging.project, 
            config=resolved_cfg, 
            name=f"bench_{cfg.model._target_.split('.')[-1]}"
        )

    engine.run()

    if cfg.logging.wandb_enabled:
        wandb.finish()

if __name__ == "__main__":
    benchmarking()