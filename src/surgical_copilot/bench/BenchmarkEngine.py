import time
import numpy as np
import torch
import wandb
from tqdm import tqdm
import json
from pathlib import Path

from monai.metrics import DiceMetric, HausdorffDistanceMetric, MeanIoU
from monai.transforms import Activations, AsDiscrete, Compose

from surgical_copilot.bench.perturbation import PerturbationPipelines


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

        self.cfg = cfg
        self.device = device
        self.fold_idx = fold_idx
        self.is_temporal = is_temporal

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

        self._print_model_info()

    def _train(self):
        self.model.train()
        losses = []

        accumulation_steps = self.cfg.trainer.trainer.get("accumulation_steps", 4)
        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc="Training")
        mask_prev = None # Inizializziamo la memoria all'inizio dell'epoca

        for i, batch in enumerate(pbar):
            x = batch["image"].to(self.device)
            y = batch["label"].to(self.device)

            # --- INIZIO LOGICA TEMPORALE ---
            if self.is_temporal:
                # Recupera il flag (Monai restituisce un tensore booleano per i dati singoli)
                is_first = batch.get("is_first_frame", [False])[0]
                if isinstance(is_first, torch.Tensor):
                    is_first = is_first.item()

                # Se è il primo frame o la memoria è vuota, crea una maschera nera
                if is_first or mask_prev is None:
                    mask_prev = torch.zeros((x.shape[0], 1, x.shape[2], x.shape[3]), device=self.device)

                # Uniamo immagine e maschera precedente (da 3 a 4 canali)
                x_input = torch.cat([x, mask_prev], dim=1)
            else:
                # Baseline classica (3 canali)
                x_input = x
            # --- FINE LOGICA TEMPORALE ---

            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                logits = self.model(x_input)

                # manage the Deep Supervision configuration
                if isinstance(logits, list):
                    loss = sum(self.loss_fn(l, y) for l in logits) / len(logits)
                else:
                    loss = self.loss_fn(logits, y)

                loss = loss / accumulation_steps

            # --- AGGIORNO LA MEMORIA PER IL PROSSIMO CICLO ---
            if self.is_temporal:
                # TEACHER FORCING: Durante il training usiamo la maschera vera (Ground Truth) per insegnare alla rete in modo stabile. 
                mask_prev = y.clone().detach()

            if self.scaler is not None:
                self.scaler.scale(loss).backward()

                if ((i + 1) % accumulation_steps == 0) or (i + 1 == len(self.train_loader)):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
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
                
                # SE È TEMPORALE, INGRANDIAMO IL DUMMY A 4 CANALI
                if self.is_temporal:
                    dummy_mask = torch.zeros(1, 1, dummy.shape[2], dummy.shape[3]).to(self.device)
                    dummy = torch.cat([dummy, dummy_mask], dim=1)
                    
                for _ in range(5):
                    _ = self.model(dummy)
                    

        with torch.no_grad():
            self.dice_metric.reset()
            self.hd95_metric.reset()
            self.iou.reset()

            total_model_time, total_images = 0.0, 0
            val_losses = []
            logged_visuals = False 

            pbar = tqdm(self.val_loader, desc=f"Eval [Clean]")
            for batch_idx, batch in enumerate(pbar):
                batch = clean_pipeline(batch)
                x = batch["image"].to(self.device)
                y = batch["label"].to(self.device)

                if self.is_temporal:
                    is_first = batch.get("is_first_frame", [False])[0]
                    if isinstance(is_first, torch.Tensor):
                        is_first = is_first.item()
                    if is_first or val_mask_prev is None:
                        val_mask_prev = torch.zeros((x.shape[0], 1, x.shape[2], x.shape[3]), device=self.device)
                    
                    x_input = torch.cat([x, val_mask_prev], dim=1)
                else:
                    x_input = x

                # Sincronizzazione per FPS
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
        
                start_batch = time.perf_counter()
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    logits = self.model(x_input)
                
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                
                batch_time = time.perf_counter() - start_batch

                # compute FPS 
                total_model_time += batch_time
                    
                #  Deep Supervision
                main_logits = logits[0] if isinstance(logits, list) else logits

                if self.is_temporal:
                    val_mask_prev = (torch.sigmoid(main_logits.detach()) > 0.5).float()

                loss = self.loss_fn(main_logits, y)
                val_losses.append(loss.item())

                # Vectorized Post-processing on batch
                preds = self.post_pred(main_logits)
                labels = self.post_label(y)
                
                self.dice_metric(y_pred=preds, y=labels)
                self.hd95_metric(y_pred=preds, y=labels)
                self.iou(y_pred=preds, y=labels)

                # Log visual results 
                if not logged_visuals:
                    
                    epochs_total = self.cfg.trainer.trainer.max_epochs
                    is_last_epoch = (epoch == epochs_total - 1)
                    
                    if epoch == 0 or (epoch + 1) % 5 == 0 or is_last_epoch:
                        if wandb.run is not None:
                            self._log_masks_wandb(x, y, preds, "clean", epoch)
                                                    
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

        with torch.no_grad():

            for scenario_name, pipeline in eval_scenarios.items():

                self.dice_metric.reset()
                self.hd95_metric.reset()
                self.iou.reset()

                total_model_time = 0.0
                total_images = 0

                pbar = tqdm(self.test_loader, desc=f"TEST [{scenario_name}]")

                for batch in pbar:

                    batch = pipeline(batch)
                    x = batch["image"].to(self.device)
                    y = batch["label"].to(self.device)

                    if self.is_temporal:
                        is_first = batch.get("is_first_frame", [False])[0]
                        if isinstance(is_first, torch.Tensor):
                            is_first = is_first.item()
                        if is_first or test_mask_prev is None:
                            test_mask_prev = torch.zeros((x.shape[0], 1, x.shape[2], x.shape[3]), device=self.device)
                        x_input = torch.cat([x, test_mask_prev], dim=1)
                    else:
                        x_input = x

                    if self.device.type == "cuda":
                        torch.cuda.synchronize()

                    start_time = time.perf_counter()

                    logits = self.model(x_input)

                    if self.device.type == "cuda":
                        torch.cuda.synchronize()

                    batch_time = time.perf_counter() - start_time
                    total_model_time += batch_time
                    total_images += x.shape[0]

                    # deep supervision handling
                    main_logits = logits[0] if isinstance(logits, list) else logits

                    if self.is_temporal:
                        test_mask_prev = (torch.sigmoid(main_logits.detach()) > 0.5).float()
                    
                    preds = self.post_pred(main_logits)
                    labels = self.post_label(y)

                    self.dice_metric(y_pred=preds, y=labels)
                    self.hd95_metric(y_pred=preds, y=labels)
                    self.iou(y_pred=preds, y=labels)

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
        return metrics

    def run(self):
        epochs = self.cfg.trainer.trainer.max_epochs
        eval_freq = self.cfg.trainer.trainer.get("eval_freq", 5) 
        best_fold_metrics = {"dice": 0.0, "hd95": 0.0, "iou": 0.0}
        #best_dice = 0.0

        for epoch in range(epochs):

            print(f"\n===== Epoch {epoch+1}/{epochs} =====")

            train_loss = self._train()

            is_last_epoch = (epoch == epochs - 1)
            #should_run_stress = (epoch == 0) or ((epoch + 1) % eval_freq == 0) or is_last_epoch

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

            #if should_run_stress:
            #    # compute and print drops for each stress scenario
            #    for scenario, scores in metrics["stress"].items():
            #        drop = (clean_dice - scores["dice"]) / (clean_dice + 1e-8)
            #        print(f"{scenario}: {scores['dice']:.4f} | drop: {drop*100:.1f}%")

            if clean_dice > best_fold_metrics["dice"]:
                best_fold_metrics = metrics["baseline"]
                model_name = self.model.__class__.__name__
                save_path = f"results/best_{model_name}_fold{self.fold_idx}.pth"
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), save_path)
                best_path = save_path

            self._log_wandb(epoch, train_loss, metrics)

        self.model.load_state_dict(torch.load(best_path, map_location=self.device))
        test_metrics = self._test()
        print("\n=== TEST RESULTS ON BEST MODEL ===")
        print(f"Baseline | Dice: {test_metrics['baseline']['dice']:.4f} | HD95: {test_metrics['baseline']['hd95']:.4f} | IoU: {test_metrics['baseline']['iou']:.4f}")

        return best_fold_metrics

    def _log_wandb(self, epoch, train_loss, metrics):
        
        if wandb.run is None:
            return

        log_dict = {
            "epoch": epoch,
            "Loss/Train": train_loss,
            "Loss/Validation": metrics["val_loss"],
            "System/Inference_FPS": metrics["inference_fps"],
            
            "Metric_Dice/Baseline": metrics["baseline"]["dice"],
            "Metric_HD95/Baseline": metrics["baseline"]["hd95"],
            "Metric_IoU/Baseline": metrics["baseline"]["iou"],
        }

    
        for scenario, scores in metrics["stress"].items():
            log_dict[f"Metric_Dice/Stress_{scenario}"] = scores["dice"]
            log_dict[f"Metric_HD95/Stress_{scenario}"] = scores["hd95"]
            log_dict[f"Metric_IoU/Stress_{scenario}"] = scores["iou"]

        wandb.log(log_dict)

    def _log_masks_wandb(self, images: torch.Tensor, labels: torch.Tensor, preds: torch.Tensor, scenario_name: str, epoch: int, max_samples: int = 4):
        wandb_images = []
        n_samples = min(images.shape[0], max_samples)

        for i in range(n_samples):

            img = images[i].detach().cpu().float().numpy()

            # CHW -> HWC
            if img.shape[0] in [1, 3]:
                img = np.transpose(img, (1, 2, 0))

            # Safety clamp
            img = np.clip(img, 0, 1)

            # Convert to uint8
            img = (img * 255).astype(np.uint8)

            gt = labels[i].detach().cpu().numpy().squeeze().astype(np.uint8)
            pr = preds[i].detach().cpu().numpy().squeeze().astype(np.uint8)

            wandb_images.append(
                wandb.Image(
                    img,
                    masks={
                        "predictions": {
                            "mask_data": pr,
                            "class_labels": {1: "Hemorrhage"}
                        },
                        "ground_truth": {
                            "mask_data": gt,
                            "class_labels": {1: "Hemorrhage"}
                        }
                    },
                    caption=f"Eval Sample {i}"
                )
            )

        wandb.log({
            f"Qualitative_Results/{scenario_name}": wandb_images,
            "epoch": epoch
        })

    def _print_model_info(self):
        n_params = sum(p.numel() for p in self.model.parameters())

        print("\n" + "=" * 60)
        print("SURGICAL COPILOT - BENCHMARK ENGINE")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Parameters: {n_params:,}")
        print("=" * 60 + "\n")