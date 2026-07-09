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

        return {
            "dice": self.dice.aggregate().item(),
            "iou": self.iou.aggregate().item(),
            "hd95": self.hd95.aggregate().item(),
            **self.temporal_iou.aggregate(),
            **self.temporal_consistency.aggregate()
        }