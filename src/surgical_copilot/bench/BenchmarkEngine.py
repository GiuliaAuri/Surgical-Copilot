import time
import numpy as np
import torch
import wandb
from tqdm import tqdm

from surgical_copilot.perturbation import PerturbationPipelines

class BenchmarkEngine:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, loss_fn, scaler, cfg, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.scaler = scaler
        self.cfg = cfg
        self.device = device
        
        self._print_model_info()

        self.history = {
            "train_loss": [],
            "clean_dice": [],
            "robust_dice": [],
            "fps": []
        }

    def run(self):
        for epoch in range(self.cfg.trainer.trainer.max_epochs):
            print(f"\n=====================\nEpoch {epoch+1}/{self.cfg.trainer.trainer.max_epochs}")
            
            # 1. Training Phase
            train_loss = self._train_epoch(epoch)
            
            # 2. Validation Phase
            if (epoch + 1) % self.cfg.trainer.trainer.val_interval == 0:
                eval_metrics = self.eval(epoch)
                
                # History Update locale
                self.history["train_loss"].append(train_loss)
                self.history["clean_dice"].append(eval_metrics["clean_dice"])
                self.history["fps"].append(eval_metrics["fps"]["fps"])
                
                # Report a terminale
                print(f"Train Loss:   {train_loss:.4f}")
                print(f"Clean Dice:   {eval_metrics['clean_dice']:.4f}")
                print(f"FPS:          {eval_metrics['fps']['fps']:.2f}")
                
                print("\n--- Robustness Breakdown ---")
                for scenario, score in eval_metrics['robust_dice'].items():
                    if scenario != "clean":
                        print(f"  > {scenario}: {score:.4f}")
                print("=====================\n")

                # 3. Logging su Weights & Biases
                if wandb.run is not None:
                    wandb_dict = {
                        "Epoch": epoch + 1,
                        "Loss/Train": train_loss,
                        "Metrics/Clean_Dice": eval_metrics["clean_dice"],
                        "System/FPS": eval_metrics["fps"]["fps"],
                        "System/Learning_Rate": self.optimizer.param_groups[0]['lr']
                    }
                    
                    for scenario, score in eval_metrics['robust_dice'].items():
                        if scenario != "clean":
                            wandb_dict[f"Robustness/{scenario}"] = score
                            
                    wandb.log(wandb_dict)

    def _train_epoch(self, epoch):
        self.model.train()
        train_loss = []

        pbar = tqdm(self.train_loader, desc=f"Training", leave=False, dynamic_ncols=True)

        for batch in pbar:
            x = batch["image"].to(self.device)
            y = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            with torch.autocast(device_type=self.device.type):
                out = self.model(x)
                loss = self.loss_fn(out, y)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            train_loss.append(loss.item())

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        self.scheduler.step()
        return np.mean(train_loss)

    def eval(self, epoch):
        self.model.eval()
        metrics = {"robust_dice": {}}
        eval_scenarios = PerturbationPipelines.get_eval_scenarios()

        with torch.no_grad():
            for scenario_name, pipeline in eval_scenarios.items():
                scores = []
                
                pbar = tqdm(self.val_loader, desc=f"Eval [{scenario_name}]", leave=False, dynamic_ncols=True)
                
                for batch_idx, batch in enumerate(pbar):
                    batch_pert = pipeline(batch)
                    
                    x = batch_pert["image"].to(self.device)
                    y = batch_pert["label"].to(self.device)

                    out = self.model(x)
                    preds = (torch.sigmoid(out) > 0.5).float()
                    scores.append(self.compute_dsc(preds, y))

                    if scenario_name == "clean" and batch_idx == 0:
                        self._log_masks_to_wandb(x, y, preds, epoch)

                mean_score = np.mean(scores)
                metrics["robust_dice"][scenario_name] = mean_score
                
                if scenario_name == "clean":
                    metrics["clean_dice"] = mean_score

        metrics["fps"] = self.evaluate_fps(self.model, self.val_loader, self.device)
        return metrics

    def _log_masks_to_wandb(self, x, y, preds, epoch):
        if wandb.run is None:
            return

        img_tensor = x[0].cpu().numpy()  
        gt_tensor = y[0].cpu().squeeze().numpy().astype(np.uint8)     
        pred_tensor = preds[0].cpu().squeeze().numpy().astype(np.uint8) 

        if img_tensor.shape[0] == 3:
            img_tensor = np.transpose(img_tensor, (1, 2, 0))
        elif img_tensor.shape[0] == 1:
            img_tensor = img_tensor.squeeze()

        class_labels = {
            0: "Background/Tissue",
            1: "Blood Area"
        }

        wandb_img = wandb.Image(
            img_tensor,
            masks={
                "Ground Truth": {
                    "mask_data": gt_tensor,
                    "class_labels": class_labels
                },
                "Prediction": {
                    "mask_data": pred_tensor,
                    "class_labels": class_labels
                }
            },
            caption=f"Epoch {epoch+1} Validation Sample"
        )

        wandb.log({"Visuals/Segmentation_Masks": wandb_img}, commit=False)

    @staticmethod
    def evaluate_fps(model, loader, device, max_batches=50):
        model.eval()
        use_cuda = device.type == "cuda"
        times = []

        pbar = tqdm(loader, desc="Benchmarking FPS", leave=False, dynamic_ncols=True)

        with torch.no_grad():
            for i, batch in enumerate(pbar):
                if i >= max_batches: break
                x = batch["image"].to(device)

                if i == 5 and use_cuda: 
                    torch.cuda.synchronize()

                if use_cuda:
                    starter = torch.cuda.Event(enable_timing=True)
                    ender = torch.cuda.Event(enable_timing=True)
                    starter.record()
                    _ = model(x)
                    ender.record()
                    torch.cuda.synchronize()
                    elapsed = starter.elapsed_time(ender) / 1000.0
                    times.append(elapsed)
                else:
                    start = time.perf_counter()
                    _ = model(x)
                    end = time.perf_counter()
                    times.append(end - start)

        mean_time = np.mean(times) if times else 0
        fps = 1.0 / mean_time if mean_time > 0 else 0
        return {"mean_latency_s": mean_time, "fps": fps}

    @staticmethod
    def compute_dsc(pred, target, eps=1e-6):
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        return ((2. * intersection + eps) / (union + eps)).item()

    def _print_model_info(self):
        n_params = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print("\n" + "=" * 60)
        print("SURGICAL CO-PILOT - BENCHMARK")
        print("=" * 60)
        print(f"Model Class:       {self.model.__class__.__name__}")
        print(f"Device:            {self.device}")
        print(f"Total Params:      {n_params:,}")
        print(f"Trainable Params:  {trainable:,}")
        print("=" * 60 + "\n")