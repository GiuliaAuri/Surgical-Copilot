import time
import numpy as np
import torch
import wandb
from tqdm import tqdm

from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import Activations, AsDiscrete, Compose

from surgical_copilot.bench.perturbation import PerturbationPipelines


class BenchmarkEngine:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        scaler,
        cfg,
        device
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.scaler = scaler

        self.cfg = cfg
        self.device = device

        self.dice_metric = DiceMetric(reduction="mean")
        self.hd95_metric = HausdorffDistanceMetric(percentile=95)

        self.post_pred = Compose([
            Activations(sigmoid=True),
            AsDiscrete(threshold=0.5)
        ])
        self.post_label = Compose([
            AsDiscrete(threshold=0.5)
        ])

        self.history = {
            "train_loss": [],
            "clean_dice": [],
            "fps": []
        }

        self._print_model_info()

    def _train(self):
        self.model.train()
        losses = []

        pbar = tqdm(self.train_loader, desc="Training")

        for batch in pbar:
            x = batch["image"].to(self.device)
            y = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                logits = self.model(x)
                loss = self.loss_fn(logits, y)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            losses.append(loss.item())
            pbar.set_postfix({"loss": loss.item()})

        self.scheduler.step()

        return float(np.mean(losses))

    def eval(self, epoch):
        print("\n[*] Evaluation & Stress Test")

        self.model.eval()

        eval_scenarios = PerturbationPipelines.get_eval_scenarios()

        metrics = {
            "robust_dice": {},
            "robust_hd95": {},
            "fps": 0.0,
            "clean_dice": 0.0,
            "clean_hd95": 0.0
        }

        #  GPU warmup (important for accurate timing)
        if self.device.type == "cuda":
            dummy = torch.randn(1, *next(iter(self.val_loader))["image"].shape[1:]).to(self.device)
            for _ in range(5):
                _ = self.model(dummy)

        with torch.no_grad():
            for scenario_name, pipeline in eval_scenarios.items():

                self.dice_metric.reset()
                self.hd95_metric.reset()

                total_time = 0.0
                total_images = 0

                pbar = tqdm(self.val_loader, desc=f"Eval [{scenario_name}]")

                for batch_idx, batch in enumerate(pbar):

                    batch = pipeline(batch)

                    x = batch["image"].to(self.device)
                    y = batch["label"].to(self.device)

                    batch_size = x.shape[0]

                    # timing
                    if self.device.type == "cuda":
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)

                        start.record()
                        logits = self.model(x)
                        end.record()
                        torch.cuda.synchronize()

                        elapsed = start.elapsed_time(end) / 1000.0
                    else:
                        t0 = time.perf_counter()
                        logits = self.model(x)
                        elapsed = time.perf_counter() - t0

                    total_time += elapsed
                    total_images += batch_size

                    # ---------------- metrics ----------------
                    preds = [self.post_pred(i) for i in logits]
                    labels = [self.post_label(i) for i in y]

                    self.dice_metric(y_pred=preds, y=labels)
                    self.hd95_metric(y_pred=preds, y=labels)

                # ---------------- aggregate ----------------
                dice = self.dice_metric.aggregate().item()
                hd95 = self.hd95_metric.aggregate().item()
                fps = total_images / max(total_time, 1e-8)

                metrics["robust_dice"][scenario_name] = dice
                metrics["robust_hd95"][scenario_name] = hd95

                if scenario_name == "clean":
                    metrics["clean_dice"] = dice
                    metrics["clean_hd95"] = hd95
                    metrics["fps"] = fps

        return metrics


    def run(self):
        epochs = self.cfg.trainer.trainer.max_epochs

        for epoch in range(epochs):

            print(f"\n===== Epoch {epoch+1}/{epochs} =====")

            train_loss = self._train()
            metrics = self.eval(epoch)

            self.history["train_loss"].append(train_loss)
            self.history["clean_dice"].append(metrics["clean_dice"])
            self.history["fps"].append(metrics["fps"])

            print(f"Loss: {train_loss:.4f}")
            print(f"Clean Dice: {metrics['clean_dice']:.4f}")
            print(f"FPS: {metrics['fps']:.2f}")

            for k, v in metrics["robust_dice"].items():
                if k != "clean":
                    drop = (metrics["clean_dice"] - v) / (metrics["clean_dice"] + 1e-8)
                    print(f"{k}: {v:.4f} | drop: {drop*100:.1f}%")

            self._log_wandb(epoch, train_loss, metrics)

    def _log_wandb(self, epoch, train_loss, metrics):

        if wandb.run is None:
            return

        log_dict = {
            "epoch": epoch,
            "train/loss": train_loss,
            "metrics/clean_dice": metrics["clean_dice"],
            "metrics/clean_hd95": metrics["clean_hd95"],
            "system/fps": metrics["fps"]
        }

        for k, v in metrics["robust_dice"].items():
            log_dict[f"robust/dice/{k}"] = v

        for k, v in metrics["robust_hd95"].items():
            log_dict[f"robust/hd95/{k}"] = v

        wandb.log(log_dict)

    
    def _print_model_info(self):
        n_params = sum(p.numel() for p in self.model.parameters())

        print("\n" + "=" * 60)
        print("SURGICAL COPILOT - BENCHMARK ENGINE")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Parameters: {n_params:,}")
        print("=" * 60 + "\n")