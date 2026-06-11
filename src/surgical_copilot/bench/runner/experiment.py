import hydra
import pandas as pd
from omegaconf import DictConfig

from surgical_copilot.bench.runner.kfold_runner import KFoldRunner

@hydra.main(version_base=None, config_path="../../../../configs", config_name="config")
def main(cfg: DictConfig):

    runner = KFoldRunner(cfg)
    results = runner.run()

    df, summary = report(results)

def report(all_metrics):
    df = pd.DataFrame(all_metrics)

    summary = {
        "dice_mean": df["dice"].mean(),
        "dice_std": df["dice"].std(),
        "hd95_mean": df["hd95"].mean(),
        "hd95_std": df["hd95"].std(),
        "iou_mean": df["iou"].mean(),
        "iou_std": df["iou"].std(),
    }

    if "temporal_dice" in df.columns and "temporal_iou" in df.columns:
        summary["temporal_dice_mean"] = df["temporal_dice"].mean()
        summary["temporal_dice_std"] = df["temporal_dice"].std()
        summary["temporal_iou_mean"] = df["temporal_iou"].mean()
        summary["temporal_iou_std"] = df["temporal_iou"].std()

    print("\n--- K-FOLD RESULTS (Per Fold) ---")
    print(df.to_string(index=False))

    print("\n--- AGGREGATED SUMMARY ---")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}")

    return df, summary

if __name__ == "__main__":
    main()