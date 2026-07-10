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

        baseline = metrics.get("baseline", {})
        log_dict = {
            "epoch": epoch,
            "Loss/Train": train_loss,
            "Loss/Validation": metrics["val_loss"],
            "Optimizer/Learning_Rate": lr,
            "System/Inference_FPS": metrics["inference_fps"],
            
            "Metric_Dice/Baseline": baseline.get("dice", 0.0),
            "Metric_HD95/Baseline": baseline.get("hd95", 0.0),
            "Metric_IoU/Baseline": baseline.get("iou", 0.0),
            "Metric_Temporal_IoU/Baseline": baseline.get("temporal_iou", 0.0),
            "Metric_Temporal_Consistency/Baseline": baseline.get("temporal_consistency", 0.0),
        }

        for scenario, scores in metrics.get("stress", {}).items():
            log_dict[f"Metric_Dice/Stress_{scenario}"] = scores.get("dice", 0.0)
            log_dict[f"Metric_HD95/Stress_{scenario}"] = scores.get("hd95", 0.0)
            log_dict[f"Metric_IoU/Stress_{scenario}"] = scores.get("iou", 0.0)
            log_dict[f"Metric_Temporal_IoU/Stress_{scenario}"] = scores.get("temporal_iou", 0.0)
            log_dict[f"Metric_Temporal_Consistency/Stress_{scenario}"] = scores.get("temporal_consistency", 0.0)
                
        wandb.log(log_dict)

    def log_test_metrics(self, metrics: dict):
        if not self.is_active:
            return

        test_log_dict = {}

        columns = ["Scenario", "Dice", "HD95", "IoU", "Temporal_IoU", "Temporal_Consistency", "Inference_FPS", "Drop (%)"]

        table = wandb.Table(columns=columns)

        def get_row(scenario, scores):
            row = [
                scenario,
                round(scores.get("dice", 0.0), 4),
                round(scores.get("hd95", 0.0), 2),
                round(scores.get("iou", 0.0), 4),
                round(scores.get("temporal_iou", 0.0), 4),
                round(scores.get("temporal_consistency", 0.0), 4),
                round(scores.get("inference_fps", 0.0), 2),
                round(scores.get("drop_percent", scores.get("drop", 0.0) * 100), 2)
            ]
            
            return row

        # Baseline
        baseline = metrics.get("baseline", {})
        table.add_data(*get_row("baseline (clean)", baseline))
        test_log_dict["Test/Baseline_Dice"] = baseline.get("dice", 0.0)
        test_log_dict["Test/Baseline_HD95"] = baseline.get("hd95", 0.0)
        test_log_dict["Test/Baseline_IoU"] = baseline.get("iou", 0.0)
        test_log_dict["Test/Baseline_Temporal_IoU"] = baseline.get("temporal_iou", 0.0)
        test_log_dict["Test/Baseline_Temporal_Consistency"] = baseline.get("temporal_consistency", 0.0)
        test_log_dict["Test/Baseline_Inference_FPS"] = baseline.get("inference_fps", 0.0)

        # Stress Scenarios
        for scenario, scores in metrics.get("stress", {}).items():
            test_log_dict[f"Test_Stress_Dice/{scenario}"] = scores.get("dice", 0.0)
            test_log_dict[f"Test_Stress_HD95/{scenario}"] = scores.get("hd95", 0.0)
            test_log_dict[f"Test_Stress_IoU/{scenario}"] = scores.get("iou", 0.0)
            test_log_dict[f"Test_Stress_Temporal_IoU/{scenario}"] = scores.get("temporal_iou", 0.0)
            test_log_dict[f"Test_Stress_Temporal_Consistency/{scenario}"] = scores.get("temporal_consistency", 0.0)
            
            table.add_data(*get_row(scenario, scores))

        test_log_dict["Test/Performance_Table"] = table
        wandb.log(test_log_dict)

    def log_qualitative_masks(self, images: torch.Tensor, labels: torch.Tensor, preds: torch.Tensor, scenario_name: str, epoch: int, max_samples: int = 4):
        if not self.is_active:
            return

        class_labels = {
            0: "Tissue/Background",
            1: "Hemorrhage"
        }

        if images.ndim == 5:
            images = images.flatten(0, 1)
            labels = labels.flatten(0, 1)
            preds = preds.flatten(0, 1)

        if images.shape[1] > 3:
            images = images[:, :3, ...]

        columns = ["Epoch", "Scenario", "Sample ID", "Segmentation Overlay"]
        qualitative_table = wandb.Table(columns=columns)

        n_samples = min(images.shape[0], max_samples)

        for i in range(n_samples):
            img = images[i].detach().cpu().float().numpy()

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