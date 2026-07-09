import torch
import gc
import wandb
from omegaconf import OmegaConf
from hydra.utils import instantiate

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from src.surgical_copilot.bench.engine.benchmark_engine import BenchmarkEngine
from src.surgical_copilot.bench.engine.temporal_engine import TemporalBenchmarkEngine
from src.surgical_copilot.bench.engine.temporal_mode import TemporalMode
from src.surgical_copilot.HemoDataset import HemosetDataSet, HemosetDataSequences, HemosetEarlyFusion
from src.surgical_copilot.bench.perturbation import PerturbationPipelines
from src.surgical_copilot.transfer_weights import load_or_create_temporal_weights
from src.surgical_copilot.utils.repro import set_seed


class KFoldRunner:

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        # Determine the model key and check if it is temporal
        self.model_key = self.cfg.get("model_key", "unknown_model")
        self.model_cfg = self.cfg.model[self.model_key]
        raw = self.model_cfg.temporal_setting.get("temporal_mode", "none")

        self.temporal_mode = TemporalMode(raw)

    def run(self):

        set_seed(self.cfg.seed) # for reproducibility

        # build the dataset based on whether temporal data is required or not
        if self.temporal_mode == TemporalMode.NONE: dataset_cls = HemosetDataSet
        elif self.temporal_mode == TemporalMode.EARLY_FUSION: dataset_cls = HemosetEarlyFusion
        else: dataset_cls = HemosetDataSequences

        print(dataset_cls.__name__)

        dataset_kwargs = {
            "root_dir": self.cfg.data.root_dir,
            "seed": self.cfg.seed,
            "image_size": self.cfg.data.img_size,
        }

        if self.temporal_mode == TemporalMode.LATE_FUSION:
            dataset_kwargs["sequence_length"] = self.cfg.data.sequence_length
            dataset_kwargs["overlapping"] = self.cfg.data.overlapping

        dataset = dataset_cls(**dataset_kwargs)

        all_metrics = {}

        exp_name = self.cfg.logging.get("exp_tag", "baseline")

        print(f"\n{'='*50}\n[Esperimento] Modello: {self.model_key} | Modalità: {self.temporal_mode.value.upper()}\n{'='*50}")
        
        metrics = []

        for fold in range(self.cfg.data.n_folds):

            print(f"\n[Fold {fold+1}/{self.cfg.data.n_folds}]")

            current_exp_name = f"{exp_name}_{self.temporal_mode}"

            if self.cfg.logging.wandb_enabled:
                wandb.init(
                    project=self.cfg.logging.project,
                    group=exp_name, 
                    name=f"{self.model_key}_{current_exp_name}_fold_{fold}",
                    config=OmegaConf.to_container(self.cfg, resolve=True),
                    reinit=True,
                    tags=[exp_name, f"fold_{fold}"]
                )

            model, loaders, engine, optimizer = self._build_fold(dataset, fold)

            try:
                fold_result = engine.run()
                metrics.append(fold_result)
            finally:
                if self.cfg.logging.wandb_enabled:
                    wandb.finish()
                self._cleanup(model, engine, loaders, optimizer)

            if self.cfg.logging.wandb_enabled:
                wandb.finish()

        all_metrics = metrics

        return all_metrics

    def _build_fold(self, dataset, fold):

        model_cfg = OmegaConf.to_container(
            self.cfg.model[self.cfg.model_key],
            resolve=True
        )

        #if self.temporal_mode != TemporalMode.NONE:
        #    target_layer = model_cfg.temporal_setting.get(
        #        "temporal_target_layer",
        #        None
        #    )

        architecture_cfg = self.cfg.model[self.model_key].architecture

        model = instantiate(architecture_cfg).to(self.device)

        if self.temporal_mode != TemporalMode.NONE:
            
            target_layer = self.cfg.model[self.model_key].temporal_setting.get("temporal_target_layer", None)

            if target_layer is not None:
                model = load_or_create_temporal_weights(
                    model=model,
                    fold_idx=fold,
                    device=self.device,
                    target_layer_name=target_layer
                )

        batch_size = self.cfg.trainer.trainer.batch_size

        is_sequential = True if self.temporal_mode == TemporalMode.LATE_FUSION else False

        train_transforms = PerturbationPipelines.get_train_pipeline(mode=self.temporal_mode.value, is_sequential=is_sequential)

        print(f"[Runner] Modalità: {self.temporal_mode.value} | Sequential: {is_sequential}")

        train_loader, val_loader, test_loader = dataset.get_loaders(
            fold_idx=fold,
            n_splits=self.cfg.data.n_folds,
            batch_size=batch_size,
            num_workers=self.cfg.trainer.trainer.num_workers,
            train_transforms=train_transforms,
        )

        optimizer = instantiate(self.cfg.trainer.optimizer, params=model.parameters())

        scheduler = self._build_scheduler(optimizer)

        loss_fn = instantiate(self.cfg.trainer.loss)
        scaler = instantiate(self.cfg.trainer.scaler)

        engine_cls = TemporalBenchmarkEngine if self.temporal_mode != TemporalMode.NONE else BenchmarkEngine

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
            "fold_idx": fold,
            "temporal_mode": self.temporal_mode
        }

        #if self.temporal_mode != TemporalMode.NONE:
        #    engine_kwargs["temporal_mode"] = self.temporal_mode

        engine = engine_cls(**engine_kwargs)

        return model, (train_loader, val_loader, test_loader), engine, optimizer

    def _build_scheduler(self, optimizer):

        cfg = self.cfg.trainer.scheduler

        warmup = cfg.warmup_epochs
        max_epochs = cfg.cosine.t_max

        #warmup = 5
        #max_epochs = self.cfg.trainer.trainer.max_epochs

        return SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=cfg.warmup_start_factor, total_iters=warmup),
                CosineAnnealingLR(optimizer, T_max=max_epochs - warmup)
            ],
            milestones=[warmup]
        )


    def _cleanup(self, model, engine, loaders, optimizer):
        del model, engine, loaders, optimizer
        torch.cuda.empty_cache()
        gc.collect()
