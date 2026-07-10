import time
import numpy as np
import torch
import wandb
from tqdm import tqdm
from pathlib import Path

from src.surgical_copilot.bench.metrics.metrics_manager import MetricManager

from monai.transforms.post.array import Activations, AsDiscrete
from monai.transforms.compose import Compose
from monai.transforms.post.array import KeepLargestConnectedComponent

from src.surgical_copilot.bench.perturbation import PerturbationPipelines
from src.surgical_copilot.bench.engine.logger_wandb import WandbLogger
from src.surgical_copilot.bench.engine.temporal_mode import TemporalMode



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
        temporal_mode=TemporalMode.NONE,
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

        if isinstance(temporal_mode, str):
            temporal_mode = TemporalMode(temporal_mode)

        self.temporal_mode = temporal_mode
        self.is_temporal = is_temporal

        self.cfg = cfg
        self.device = device
        self.fold_idx = fold_idx

        self.accumulation_steps = self.cfg.trainer.trainer.get("accumulation_steps", 4)

        self.patience = self.cfg.trainer.early_stopping.patience
        self.monitor_mode = self.cfg.trainer.early_stopping.mode
        self.best_path = None
        self.best_metrics = None
        self.best_score = -float("inf") if self.monitor_mode == "max" else float("inf")
        self.epochs_without_improvement = 0

        self.metrics = MetricManager(device=self.device)

        self.post_pred = Compose([
            Activations(sigmoid=True),
            AsDiscrete(threshold=0.5),
            KeepLargestConnectedComponent(applied_labels=None) ## !!!!
        ])
        self.post_label = Compose([
            AsDiscrete(threshold=0.5)
        ])

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "clean_dice": [],
            "fps": []
        }

        self.logger = WandbLogger()
        
        self.logger._print_model_info(model, device)

    def _prepare_inputs(self, batch):
            
        x = batch["current_image"].to(self.device)
        y = batch["current_label"].to(self.device)
            
        return x, y
    
    def _get_sequence_info(self, batch):

        is_first_frame = batch["is_first_frame"]
        sequence_id = batch["patient_id"]

        return is_first_frame, sequence_id
    
    def _gpu_warmup(self):

        # Warmup GPU
        if self.device.type == "cuda":

            H, W = self.cfg.data.img_size

            if getattr(self, "temporal_mode", None) == TemporalMode.LATE_FUSION:
                T = self.cfg.data.sequence_length
                dummy_input = torch.randn(1, T, 3, H, W, device=self.device)
            elif getattr(self, "temporal_mode", None) == TemporalMode.EARLY_FUSION:
                dummy_input = torch.randn(1, 4, H, W, device=self.device)
            else:
                dummy_input = torch.randn(1, 3, H, W, device=self.device)
            
            # Scalda solo il modello, non il metodo _prepare_inputs
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                for _ in range(5):
                    self.model(dummy_input)

    def _forward_step(self, x, y):
        logits = self.model(x)

        # Gestione Deep Supervision
        logits = logits[0] if isinstance(logits, list) else logits

        # manage the Deep Supervision configuration
        if isinstance(logits, list):
            loss = sum(self.loss_fn(l, y) for l in logits) / len(logits)
        else:
            loss = self.loss_fn(logits, y)

        return {
            "logits": logits,
            "loss": loss / self.accumulation_steps
        }        

    def _scale_loss(self, i, loss):

        if self.scaler is not None:

            self.scaler.scale(loss).backward()

            if ((i + 1) % self.accumulation_steps == 0) or (i + 1 == len(self.train_loader)):

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
        else:

            loss.backward()

            if ((i + 1) % self.accumulation_steps == 0) or (i + 1 == len(self.train_loader)):
                
                self.optimizer.step()
                self.optimizer.zero_grad()

    def _post_processing(self, logits, y):
        # Vectorized Post-processing on batch
        preds = self.post_pred(logits)
        labels = self.post_label(y)

        return preds, labels

    def _update_metrics(
        self,
        preds,
        labels,
        images,
        is_first_frame,
        sequence_id
    ):

        self.metrics.update_spatial(
            preds,
            labels
        )

        rgb = images[:, :3] if images.shape[1] > 3 else images

        self.metrics.update_temporal(
            preds,
            labels,
            rgb,
            is_first_frame,
            sequence_id
        )

    def _train(self):

        self.model.train()
        losses = []

        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc="Training")

        for i, batch in enumerate(pbar):

            x, y = self._prepare_inputs(batch)

            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                loss = self._forward_step(x, y)["loss"]
                #loss = self._post_forward_hook(logits, y)
            
            self._scale_loss(i, loss)
            
            real_loss = loss.item() * self.accumulation_steps
            losses.append(real_loss)
            pbar.set_postfix({"loss": real_loss})

        self.scheduler.step()
        return float(np.mean(losses))

    def _validate(self, epoch: int) -> dict:

        print("\n[*] Evaluation")
        self.model.eval()
        mode_str = self.temporal_mode.value
        is_sequential = True if self.temporal_mode == TemporalMode.LATE_FUSION else False
        clean_pipeline  = PerturbationPipelines.get_eval_scenarios(mode=mode_str, is_sequential=is_sequential)["clean"]

        metrics = {
            "val_loss": 0.0,
            "inference_fps": 0.0,
            "baseline": {"dice": 0.0, "hd95": 0.0, "iou": 0.0},
            "stress": {}
        }

        self._gpu_warmup()

        with torch.inference_mode():

            self.metrics.reset()

            total_model_time, total_images = 0.0, 0
            val_losses = []
            logged_visuals = False 

            pbar = tqdm(self.val_loader, desc=f"Eval [Clean]")
            for batch_idx, batch in enumerate(pbar):

                batch = clean_pipeline(batch)

                x, y = self._prepare_inputs(batch)
                is_first, seq_id = self._get_sequence_info(batch)

                # Sincronizzazione per FPS
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
        
                start_batch = time.perf_counter()
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    results = self._forward_step(x, y)
                    logits = results["logits"]
                    loss = results["loss"]

                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                
                batch_time = time.perf_counter() - start_batch

                # compute FPS 
                total_model_time += batch_time
                val_losses.append(loss.item() * self.accumulation_steps)
                    
                preds, labels = self._post_processing(logits, y)
                self._update_metrics(preds, labels, x, is_first_frame=is_first, sequence_id=seq_id)

                # Log visual results 
                if not logged_visuals:
                    
                    epochs_total = self.cfg.trainer.trainer.max_epochs
                    is_last_epoch = (epoch == epochs_total - 1)
                    
                    if epoch == 0 or (epoch + 1) % 10 == 0 or is_last_epoch:
                        if wandb.run is not None:
                            self.logger.log_qualitative_masks(x, y, preds, "clean", epoch)
                                                    
                    logged_visuals = True

                total_images += x.shape[0]
            
            metrics["inference_fps"] = total_images / max(total_model_time, 1e-8)
            metrics["baseline"] = self.metrics.compute()
            metrics["val_loss"] = float(np.mean(val_losses))

        return metrics

    def _test(self):

        self.model.eval()

        mode_str = self.temporal_mode.value
        is_sequential = True if self.temporal_mode == TemporalMode.LATE_FUSION else False

        eval_scenarios = PerturbationPipelines.get_eval_scenarios(mode=mode_str, is_sequential=is_sequential)

        metrics = {
            "baseline": {"dice": 0.0, "hd95": 0.0, "iou": 0.0},
            "stress": {}
        }

        # define a fictitious epoch for logging purposes, since we are in the test phase
        test_epoch = self.cfg.trainer.trainer.max_epochs

        self._gpu_warmup()

        with torch.inference_mode():

            for scenario_name, pipeline in eval_scenarios.items():

                self.metrics.reset()

                logged_visuals = False

                total_model_time = 0.0
                total_images = 0

                logged_visuals = False

                pbar = tqdm(self.test_loader, desc=f"TEST [{scenario_name}]")

                for batch in pbar:

                    batch = pipeline(batch)

                    x, y = self._prepare_inputs(batch)
                    first_frame, seq_id = self._get_sequence_info(batch) 

                    if self.device.type == "cuda":
                        torch.cuda.synchronize()

                    start_time = time.perf_counter()

                    logits = self._forward_step(x, y)["logits"]
                    
                    main_logits = logits[0] if isinstance(logits, list) else logits

                    if self.device.type == "cuda":
                        torch.cuda.synchronize()

                    batch_time = time.perf_counter() - start_time
                    total_model_time += batch_time
                    total_images += x.shape[0]
                    
                    # Post-processing e metriche
                    preds, labels = self._post_processing(main_logits, y)
                    
                    self._update_metrics(preds, labels, x, is_first_frame=first_frame, sequence_id=seq_id)

                    if not logged_visuals:
                        self.logger.log_qualitative_masks(
                            images=x, 
                            labels=labels, 
                            preds=preds, 
                            scenario_name=scenario_name, 
                            epoch=test_epoch
                        )

                        logged_visuals = True

                scores = self.metrics.compute()

                scores["inference_fps"] = (
                    total_images /
                    max(total_model_time, 1e-8)
                )
                
                if scenario_name == "clean":
                    metrics["baseline"] = scores
                    drop_info = "" 
                else:
                    clean_dice = metrics["baseline"].get("dice", 1e-8)
                    robustness_drop = (clean_dice - scores["dice"]) / (clean_dice + 1e-8)

                    scores["drop"] = robustness_drop
                    metrics["stress"][scenario_name] = scores
                    scores["drop_percent"] = robustness_drop * 100
                    drop_info = f" | Drop: {robustness_drop * 100:>5.1f}%"

                print(f"[{scenario_name:<20}] Dice: {scores['dice']:.4f} | HD95: {scores['hd95']:>7.2f}{drop_info}")

        self.logger.log_test_metrics(metrics)

        return metrics

    def _early_stopping(self, score, epoch):
        
        if self.monitor_mode == "max":
            improved = score > self.best_score
        else:
            improved = score < self.best_score

        if improved:
            self.best_score = score
            self.epochs_without_improvement = 0
            self.best_fold_metrics = score
            self.best_path = self._save_checkpoint(self.fold_idx)
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            return True

        return False

    def run(self):

        epochs = self.cfg.trainer.trainer.max_epochs
        
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
            self.history["val_loss"].append(val_loss)
            self.history["clean_dice"].append(clean_dice)
            self.history["fps"].append(fps)

            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Clean Dice: {clean_dice:.4f} | FPS: {fps:.2f}")

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.logger.log_epoch_metrics(epoch, train_loss, current_lr, metrics)

            if self._early_stopping(clean_dice, epoch):
                break
            
            #if clean_dice > best_fold_metrics["dice"]:
            #    best_fold_metrics = metrics["baseline"]
            #    best_path = self._save_checkpoint(self.fold_idx)
    
        if self.best_path is None:
            raise RuntimeError("Training finish without any valid checkpoint.")

        self.model.load_state_dict(torch.load(self.best_path, map_location=self.device))

        # TEST PROCESS
        test_metrics = self._test()

        print("\n=== TEST RESULTS ON BEST MODEL ===")
        print(f"Baseline | Dice: {test_metrics['baseline']['dice']:.4f} | HD95: {test_metrics['baseline']['hd95']:.4f} | IoU: {test_metrics['baseline']['iou']:.4f}")

        return test_metrics

    def _save_checkpoint(self, fold_idx: int) -> str:

        model_class_name = self.model.__class__.__name__

        base_dir = Path("/work/cvcs2026/DeepLook/results/weights")
        weights_dir = base_dir / model_class_name
        weights_dir.mkdir(parents=True, exist_ok=True)

        save_path = weights_dir / f"best_fold{fold_idx}.pth"

        temp_path = save_path.with_suffix('.tmp')
        torch.save(self.model.state_dict(), temp_path)
        temp_path.replace(save_path)

        return str(save_path)
   
   