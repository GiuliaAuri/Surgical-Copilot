from monai.metrics.meandice import DiceMetric
from monai.metrics.hausdorff_distance import HausdorffDistanceMetric
from monai.metrics.meaniou import MeanIoU
from src.surgical_copilot.bench.metrics.temporal_metrics.temporal_consistency import TemporalConsistencyMetric
from src.surgical_copilot.bench.metrics.temporal_metrics.temporal_iou import TemporalIoU

class MetricManager:

    def __init__(self, device):

        self.dice = DiceMetric(reduction="mean")
        self.iou = MeanIoU(reduction="mean")
        self.hd95 = HausdorffDistanceMetric(percentile=95)

        self.temporal_iou = TemporalIoU(
            threshold=0.5,
            from_logits=False,
            eps=1e-6
        )

        self.temporal_consistency = TemporalConsistencyMetric(
            device=device
        )

        self.current_sequence = None


    def reset(self):

        self.dice.reset()
        self.iou.reset()
        self.hd95.reset()

        self.temporal_iou.reset()
        self.temporal_iou.reset_sequence()

        self.temporal_consistency.reset()

        self.current_sequence = None

    def check_sequence(self, sequence_id):

        if sequence_id != self.current_sequence:

            self.temporal_iou.reset_sequence()
            self.temporal_consistency.reset_sequence() 
            self.current_sequence = sequence_id

    def reset_sequence(self):

        self.temporal_iou.reset_sequence()


    def update_spatial(self, preds, labels):
        self.dice(y_pred=preds, y=labels)
        self.hd95(y_pred=preds, y=labels)
        self.iou(y_pred=preds, y=labels)


    def update_temporal(self, preds, labels, images, is_first_frame, sequence_id):

        self.check_sequence(sequence_id)

        self.temporal_iou(preds=preds, is_first_frame=is_first_frame)

        if images is None:
            return
    
        self.temporal_consistency(preds=preds, labels=labels, images=images)

    def compute(self):
        t_iou_res = self.temporal_iou.aggregate()
        t_const_res = self.temporal_consistency.aggregate()

        temporal_iou_val = t_iou_res.get("temporal_iou", t_iou_res.get("Temporal_IoU", 0.0))
        temporal_consistency_val = t_const_res.get("temporal_consistency", t_const_res.get("Temporal_Consistency", 0.0))

        if isinstance(temporal_iou_val, dict):
            temporal_iou_val = temporal_iou_val.get("mean", 0.0)

        return {
            "dice": self.dice.aggregate().item(),
            "iou": self.iou.aggregate().item(),
            "hd95": self.hd95.aggregate().item(),
            "temporal_iou": temporal_iou_val,
            "temporal_consistency": temporal_consistency_val
        }