import time
import numpy as np
import torch
import wandb
from tqdm import tqdm
from pathlib import Path

from monai.metrics import DiceMetric, HausdorffDistanceMetric, MeanIoU
from monai.transforms import Activations, AsDiscrete, Compose

from surgical_copilot.bench.perturbation import PerturbationPipelines
from surgical_copilot.bench.engine.logger_wandb import WandbLogger


class BenchmarkEngine:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        scheduler,
        loss_fn,
        scaler,
        cfg,
        device,
        fold_idx=0,
        is_temporal=False
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.scaler = scaler

        self.is_temporal = is_temporal

        self.cfg = cfg
        self.device = device
        self.fold_idx = fold_idx

        self.dice_metric = DiceMetric(reduction="mean")
        self.hd95_metric = HausdorffDistanceMetric(percentile=95)
        self.iou = MeanIoU(reduction="mean")

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

        self.logger = WandbLogger()
        
        self.logger._print_model_info(model, device)

    def _prepare_inputs(self, batch):
        x = batch["image"].to(self.device)
        y = batch["label"].to(self.device)
        return x, y

    def _forward_step(self, x):
        return self.model(x)

    def _post_forward_hook(self, logits):
        pass

    def _update_metrics(self, preds, labels):
        self.dice_metric(y_pred=preds, y=labels)
        self.hd95_metric(y_pred=preds, y=labels)
        self.iou(y_pred=preds, y=labels)

    def _train(self):

        self.model.train()
        losses = []

        accumulation_steps = self.cfg.trainer.trainer.get("accumulation_steps", 4)
        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc="Training")

        for i, batch in enumerate(pbar):

            x, y = self._prepare_inputs(batch)

            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                
                logits = self._forward_step(x)
                
                # Gestione Deep Supervision
                main_logits = logits[0] if isinstance(logits, list) else logits

                self._post_forward_hook(main_logits)

                # manage the Deep Supervision configuration
                if isinstance(logits, list):
                    loss = sum(self.loss_fn(l, y) for l in logits) / len(logits)
                else:
                    loss = self.loss_fn(logits, y)

                loss = loss / accumulation_steps

            if self.scaler is not None:

                self.scaler.scale(loss).backward()

                if ((i + 1) % accumulation_steps == 0) or (i + 1 == len(self.train_loader)):

                    # Apply clipping ONLY for temporal models
                    if self.is_temporal:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
            else:

                loss.backward()

                if ((i + 1) % accumulation_steps == 0) or (i + 1 == len(self.train_loader)):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            real_loss = loss.item() * accumulation_steps
            losses.append(real_loss)
            pbar.set_postfix({"loss": real_loss})

        self.scheduler.step()
        return float(np.mean(losses))

    def _validate(self, epoch: int) -> dict:

        print("\n[*] Evaluation & Stress Test")
        self.model.eval()
        clean_pipeline  = PerturbationPipelines.get_eval_scenarios()["clean"]

        metrics = {
            "val_loss": 0.0,
            "inference_fps": 0.0,
            "baseline": {"dice": 0.0, "hd95": 0.0, "iou": 0.0},
            "stress": {}
        }

        # Warmup GPU
        with torch.cuda.amp.autocast(enabled=self.scaler is not None):

            if self.device.type == "cuda":
                dummy = torch.randn(1, *next(iter(self.val_loader))["image"].shape[1:]).to(self.device)
                for _ in range(5):
                    _ = self.model(dummy)

        with torch.inference_mode():
            self.dice_metric.reset()
            self.hd95_metric.reset()
            self.iou.reset()

            total_model_time, total_images = 0.0, 0
            val_losses = []
            logged_visuals = False 

            pbar = tqdm(self.val_loader, desc=f"Eval [Clean]")
            for batch_idx, batch in enumerate(pbar):

                batch = clean_pipeline(batch)

                x, y = self._prepare_inputs(batch)

                # Sincronizzazione per FPS
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
        
                start_batch = time.perf_counter()
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    logits = self._forward_step(x)
                
                main_logits = logits[0] if isinstance(logits, list) else logits

                self._post_forward_hook(main_logits)

                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                
                batch_time = time.perf_counter() - start_batch

                # compute FPS 
                total_model_time += batch_time
                    
                #  Deep Supervision
                main_logits = logits[0] if isinstance(logits, list) else logits

                loss = self.loss_fn(main_logits, y)
                val_losses.append(loss.item())

                # Vectorized Post-processing on batch
                preds = self.post_pred(main_logits)
                labels = self.post_label(y)
                
                self._update_metrics(preds, labels)

                # Log visual results 
                if not logged_visuals:
                    
                    epochs_total = self.cfg.trainer.trainer.max_epochs
                    is_last_epoch = (epoch == epochs_total - 1)
                    
                    if epoch == 0 or (epoch + 1) % 5 == 0 or is_last_epoch:
                        if wandb.run is not None:
                            self.logger.log_qualitative_masks(x, y, preds, "clean", epoch)
                                                    
                    logged_visuals = True

                total_images += x.shape[0]
            
            metrics["inference_fps"] = total_images / max(total_model_time, 1e-8)
            metrics["baseline"]["dice"] = self.dice_metric.aggregate().item()
            metrics["baseline"]["hd95"] = self.hd95_metric.aggregate().item()
            metrics["baseline"]["iou"] = self.iou.aggregate().item()
            metrics["val_loss"] = float(np.mean(val_losses))

        return metrics

    def _test(self):

        self.model.eval()

        eval_scenarios = PerturbationPipelines.get_eval_scenarios()

        metrics = {
            "baseline": {"dice": 0.0, "hd95": 0.0, "iou": 0.0},
            "stress": {}
        }

        with torch.inference_mode():

            for scenario_name, pipeline in eval_scenarios.items():

                self.dice_metric.reset()
                self.hd95_metric.reset()
                self.iou.reset()

                total_model_time = 0.0
                total_images = 0

                pbar = tqdm(self.test_loader, desc=f"TEST [{scenario_name}]")

                for batch in pbar:

                    batch = pipeline(batch)

                    x, y = self._prepare_inputs(batch)

                    if self.device.type == "cuda":
                        torch.cuda.synchronize()

                    start_time = time.perf_counter()

                    logits = self._forward_step(x)

                    if self.device.type == "cuda":
                        torch.cuda.synchronize()

                    batch_time = time.perf_counter() - start_time
                    total_model_time += batch_time
                    total_images += x.shape[0]

                    # deep supervision handling
                    main_logits = logits[0] if isinstance(logits, list) else logits

                    self._post_forward_hook(main_logits)
                    
                    # Post-processing e metriche
                    preds = self.post_pred(main_logits)
                    labels = self.post_label(y)

                    self._update_metrics(preds, labels)

                    scores = {
                        "dice": self.dice_metric.aggregate().item(),
                        "hd95": self.hd95_metric.aggregate().item(),
                        "iou": self.iou.aggregate().item(),
                        "inference_fps": total_images / max(total_model_time, 1e-8)}
                    
                    if scenario_name == "clean":
                        metrics["baseline"] = scores
                        drop_info = "" 
                    else:
                        clean_dice = metrics["baseline"].get("dice", 1e-8)
                        robustness_drop = (clean_dice - scores["dice"]) / (clean_dice + 1e-8)
                        scores["drop"] = robustness_drop
                        metrics["stress"][scenario_name] = scores
                        drop_info = f" | Drop: {robustness_drop * 100:>5.1f}%"

                    print(f"[{scenario_name:<20}] Dice: {scores['dice']:.4f} | HD95: {scores['hd95']:>7.2f}{drop_info}")

        self.logger.log_test_metrics(metrics)

        return metrics

    def run(self):
        epochs = self.cfg.trainer.trainer.max_epochs
        best_fold_metrics = {"dice": 0.0, "hd95": 0.0, "iou": 0.0}
        
        best_path = None

        for epoch in range(epochs):

            print(f"\n===== Epoch {epoch+1}/{epochs} =====")

            # TRAIN PROCESS
            train_loss = self._train()

            # VALIDATION PROCESS
            metrics = self._validate(epoch)

            val_loss = metrics["val_loss"]
            clean_dice = metrics["baseline"]["dice"]
            fps = metrics["inference_fps"]

            self.history["train_loss"].append(train_loss)
            self.history.setdefault("val_loss", []).append(val_loss)
            self.history.setdefault("clean_dice", []).append(clean_dice)
            self.history.setdefault("fps", []).append(fps)

            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Clean Dice: {clean_dice:.4f} | FPS: {fps:.2f}")

            if clean_dice > best_fold_metrics["dice"]:
                best_fold_metrics = metrics["baseline"]
                best_path = self._save_checkpoint(self.fold_idx)
    
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.logger.log_epoch_metrics(epoch, train_loss, current_lr, metrics)

        if best_path is None:
            raise RuntimeError("Training finish without any valid checkpoint.")

        self.model.load_state_dict(torch.load(best_path, map_location=self.device))

        # TEST PROCESS
        test_metrics = self._test()

        print("\n=== TEST RESULTS ON BEST MODEL ===")
        print(f"Baseline | Dice: {test_metrics['baseline']['dice']:.4f} | HD95: {test_metrics['baseline']['hd95']:.4f} | IoU: {test_metrics['baseline']['iou']:.4f}")

        return best_fold_metrics

    def _save_checkpoint(self, fold_idx: int) -> str:

        model_name = self.cfg.model_key 
        
        #base_dir = Path("/work/cvcs2026/DeepLook/results/weights")
        base_dir = Path("/homes/cmininno/cvcs2026/Surgical-Copilot/weights")
        weights_dir = base_dir / model_name
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = weights_dir / f"best_fold{fold_idx}.pth"
        
        temp_path = save_path.with_suffix('.tmp')
        torch.save(self.model.state_dict(), temp_path)
        temp_path.replace(save_path)
        
        return str(save_path)
   