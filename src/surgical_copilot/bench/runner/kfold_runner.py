import torch
import gc
import wandb
from omegaconf import OmegaConf
from hydra.utils import instantiate

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from surgical_copilot.bench.engine.benchmark_engine import BenchmarkEngine
from surgical_copilot.bench.engine.temporal_engine import TemporalBenchmarkEngine
from surgical_copilot.HemoDataset import HemosetDataSet
from surgical_copilot.bench.perturbation import PerturbationPipelines
from surgical_copilot.transfer_weights import load_or_create_temporal_weights


class KFoldRunner:

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    def run(self):

        dataset = HemosetDataSet(
            root_dir=self.cfg.data.root_dir,
            seed=self.cfg.seed,
            image_size=self.cfg.data.img_size
        )

        all_metrics = []

        exp_name = self.cfg.logging.get("exp_tag", "baseline")
        model_key = self.cfg.get("model_key", "unknown_model")

        for fold in range(self.cfg.data.n_folds):

            print(f"\n[Fold {fold+1}/{self.cfg.data.n_folds}]")

            if self.cfg.logging.wandb_enabled:
                wandb.init(
                    project=self.cfg.logging.project,
                    group=exp_name, 
                    name=f"{model_key}_{exp_name}_fold_{fold}",
                    config=OmegaConf.to_container(self.cfg, resolve=True),
                    reinit=True,
                    tags=[exp_name, f"fold_{fold}"]
                )

            model, loaders, engine, optimizer = self._build_fold(dataset, fold)

            try:
                fold_result = engine.run()
                all_metrics.append(fold_result)
            finally:
                if self.cfg.logging.wandb_enabled:
                    wandb.finish()
                self._cleanup(model, engine, loaders, optimizer)

            if self.cfg.logging.wandb_enabled:
                wandb.finish()

            self._cleanup(model, engine, loaders, optimizer)

        return all_metrics

    def _build_fold(self, dataset, fold):

        model_cfg = OmegaConf.to_container(
            self.cfg.model[self.cfg.model_key],
            resolve=True
        )

        is_temporal = model_cfg.pop("is_temporal", False)
        temporal_mode_str = model_cfg.pop("temporal_mode", "none") 
        target_layer = model_cfg.pop("temporal_target_layer", None)

        model = instantiate(model_cfg).to(self.device)

        if is_temporal:
            model = load_or_create_temporal_weights(
                model=model,
                fold_idx=fold,
                device=self.device,
                target_layer_name=target_layer
            )

        batch_size = 1 if is_temporal else self.cfg.trainer.trainer.batch_size

        train_loader, val_loader, test_loader = dataset.get_loaders(
            fold_idx=fold,
            n_splits=self.cfg.data.n_folds,
            batch_size=batch_size,
            num_workers=self.cfg.trainer.trainer.num_workers,
            train_transforms=PerturbationPipelines.get_train_pipeline(),
            temporal_mode=is_temporal
        )

        optimizer = instantiate(self.cfg.trainer.optimizer, params=model.parameters())

        scheduler = self._build_scheduler(optimizer)

        loss_fn = instantiate(self.cfg.trainer.loss)
        scaler = instantiate(self.cfg.trainer.scaler)

        engine_cls = TemporalBenchmarkEngine if is_temporal else BenchmarkEngine

        engine_kwargs = {
            "model": model,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "loss_fn": loss_fn,
            "scaler": scaler,
            "cfg": self.cfg,
            "device": self.device,
            "fold_idx": fold
        }

        if is_temporal:
            engine_kwargs["temporal_mode"] = temporal_mode_str

        engine = engine_cls(**engine_kwargs)

        return model, (train_loader, val_loader, test_loader), engine, optimizer

    def _build_scheduler(self, optimizer):

        warmup = 5
        max_epochs = self.cfg.trainer.trainer.max_epochs

        return SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.01, total_iters=warmup),
                CosineAnnealingLR(optimizer, T_max=max_epochs - warmup)
            ],
            milestones=[warmup]
        )


    def _cleanup(self, model, engine, loaders, optimizer):
        del model, engine, loaders, optimizer
        torch.cuda.empty_cache()
        gc.collect()
