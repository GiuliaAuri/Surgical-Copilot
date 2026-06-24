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

            "Metric_Temporal_Var/IoU": metrics["baseline"].get("temporal_iou", 0.0),
            "Metric_Temporal_Var/Dice": metrics["baseline"].get("temporal_dice", 0.0),
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

        test_log_dict = {
            "Test_Baseline/Dice": metrics["baseline"]["dice"],
            "Test_Baseline/HD95": metrics["baseline"]["hd95"],
            "Test_Baseline/IoU": metrics["baseline"]["iou"],
            "Test_System/Inference_FPS": metrics["baseline"].get("inference_fps", 0.0),
        }

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
            
            test_log_dict[f"Test_Stress_Dice/{scenario}"] = scores["dice"]
            test_log_dict[f"Test_Stress_HD95/{scenario}"] = scores["hd95"]
            test_log_dict[f"Test_Stress_IoU/{scenario}"] = scores["iou"]
            test_log_dict[f"Test_Stress_Drop/{scenario}"] = scores.get("drop", drop_val / 100)

            table.add_data(
                scenario,
                round(scores["dice"], 4),
                round(scores["hd95"], 2),
                round(scores["iou"], 4),
                round(scores.get("inference_fps", 0.0), 2),
                round(drop_val, 2)
            )

        test_log_dict["Test/Performance_Table"] = table
        wandb.log(test_log_dict)

    def log_qualitative_masks(self, images: torch.Tensor, labels: torch.Tensor, preds: torch.Tensor, scenario_name: str, epoch: int, max_samples: int = 4):
        if not self.is_active:
            return

        class_labels = {
            0: "Tissue/Background",
            1: "Hemorrhage"
        }

        columns = ["Epoch", "Scenario", "Sample ID", "Segmentation Overlay"]
        qualitative_table = wandb.Table(columns=columns)

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

            wandb_img = wandb.Image(
                img,
                masks={
                    "predictions": {
                        "mask_data": pr,
                        "class_labels": class_labels
                    },
                    "ground_truth": {
                        "mask_data": gt,
                        "class_labels": class_labels
                    }
                }
            )

            qualitative_table.add_data(epoch, scenario_name, f"Frame_{i}", wandb_img)

        wandb.log({f"Qualitative_Analysis/{scenario_name}": qualitative_table})