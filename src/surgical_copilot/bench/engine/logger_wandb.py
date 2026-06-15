import numpy as np
import torch
import wandb


class WandbLogger:
    
    def __init__(self):
        self.is_active = wandb.run is not None

    def _print_model_info(self, model, device):
        n_params = sum(p.numel() for p in model.parameters())

        print("\n" + "=" * 60)
        print("SURGICAL COPILOT - BENCHMARK ENGINE")
        print("=" * 60)
        print(f"Device: {device}")
        print(f"Parameters: {n_params:,}")
        print("=" * 60 + "\n")

    def log_epoch_metrics(self, epoch: int, train_loss: float, lr: float, metrics: dict):
        if not self.is_active:
            return

        log_dict = {
            "epoch": epoch,
            "Loss/Train": train_loss,
            "Loss/Validation": metrics["val_loss"],
            "Optimizer/Learning_Rate": lr,
            "System/Inference_FPS": metrics["inference_fps"],
            
            "Metric_Dice/Baseline": metrics["baseline"]["dice"],
            "Metric_HD95/Baseline": metrics["baseline"]["hd95"],
            "Metric_IoU/Baseline": metrics["baseline"]["iou"],

            "Metric_Temporal_Consistancy/IoU": metrics["baseline"]["consistency"].get("temporal_iou", 0.0),
            "Metric_Temporal_Consistancyr/Dice": metrics["baseline"]["consistency"].get("temporal_dice", 0.0),

            "Metric_Temporal_Interframe/IoU": metrics["baseline"]["interframe"].get("temporal_iou", 0.0),
            "Metric_Temporal_Interframe/Dice": metrics["baseline"]["interframe"].get("temporal_dice", 0.0)
        }

        for scenario, scores in metrics.get("stress", {}).items():
            log_dict[f"Metric_Dice/Stress_{scenario}"] = scores["dice"]
            log_dict[f"Metric_HD95/Stress_{scenario}"] = scores["hd95"]
            log_dict[f"Metric_IoU/Stress_{scenario}"] = scores["iou"]

            if "temporal_iou" in scores:
                log_dict[f"Metric_Temporal_Var_IoU/Stress_{scenario}"] = scores["temporal_iou"]
                log_dict[f"Metric_Temporal_Var_Dice/Stress_{scenario}"] = scores["temporal_dice"]

        wandb.log(log_dict)

    def log_test_metrics(self, metrics: dict):

        if not self.is_active:
            return

        columns = ["Scenario", "Dice", "HD95", "IoU", "Inference_FPS", "Drop (%)"]
        table = wandb.Table(columns=columns)

        table.add_data(
            "baseline (clean)",
            round(metrics["baseline"]["dice"], 4),
            round(metrics["baseline"]["hd95"], 2),
            round(metrics["baseline"]["iou"], 4),
            round(metrics["baseline"].get("inference_fps", 0.0), 2),
            0.0
        )

        for scenario, scores in metrics.get("stress", {}).items():
            drop_val = scores.get("drop_percent", scores.get("drop", 0.0) * 100)
            table.add_data(
                scenario,
                round(scores["dice"], 4),
                round(scores["hd95"], 2),
                round(scores["iou"], 4),
                round(scores.get("inference_fps", 0.0), 2),
                round(drop_val, 2)
            )

        wandb.log({"Test/Performance_Table": table})

    def log_qualitative_masks(self, images: torch.Tensor, labels: torch.Tensor, preds: torch.Tensor, scenario_name: str, epoch: int, max_samples: int = 4):
        if not self.is_active:
            return

        wandb_images = []
        n_samples = min(images.shape[0], max_samples)

        for i in range(n_samples):
            img = images[i].detach().cpu().float().numpy()

            # CHW -> HWC
            if img.shape[0] in [1, 3]:
                img = np.transpose(img, (1, 2, 0))

            img = np.clip(img, 0, 1)
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