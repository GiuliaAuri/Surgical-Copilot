import torch
import numpy as np

from surgical_copilot.bench.engine.benchmark_engine import BenchmarkEngine
from surgical_copilot.bench.engine.temporal_mode import TemporalMode
from surgical_copilot.bench.metrics.temporal_metrics.temporal_consistency import TemporalConsistencyMetric
from surgical_copilot.bench.metrics.temporal_metrics.inter_frame import InterFrameTemporalMetric


class TemporalBenchmarkEngine(BenchmarkEngine):

    def __init__(self, *args, temporal_mode=TemporalMode.NONE, **kwargs):
        super().__init__(*args, **kwargs)

        if isinstance(temporal_mode, str):
            temporal_mode = TemporalMode(temporal_mode)

        self.temporal_mode = temporal_mode

        # memory states
        self.recurrent_state = None
        self.mask_prev = None

        self.temporal_metrics = {
            "consistency": TemporalConsistencyMetric(),
            "interframe": InterFrameTemporalMetric()
        }
    
    def _reset_temporal_state(self):
        self.recurrent_state = None
        self.mask_prev = None

    def _reset_temporal_metrics(self):
        
        for metric in self.temporal_metrics.values():
            metric.reset()
    
    def _reset_all(self):
        self._reset_temporal_state()
        self._reset_temporal_metrics()

    def _prepare_inputs(self, batch):

        x, y = super()._prepare_inputs(batch)

        is_first = batch.get("is_first_frame", [False])[0]

        if isinstance(is_first, torch.Tensor):
            is_first = is_first.item()

        # to avoid temporal state contamination, reset the memory states at the beginning of each new sequence
        if is_first:
            self._reset_temporal_state()

        return x, y
    
    def _forward_step(self, x, y):

        B, T, C, H, W = x.shape
        total_loss = 0.0
        all_logits = []

        for t in range(T):
            x_t = x[:, t] # (B, C, H, W)
            y_t = y[:, t] # (B, 1, H, W)

            # EARLY FUSION
            if self.temporal_mode == TemporalMode.EARLY_FUSION:

                if self.mask_prev is None:
                    self.mask_prev = torch.zeros(
                        (x.shape[0], 1, x.shape[2], x.shape[3]),
                        device=self.device
                    )

                x = torch.cat([x, self.mask_prev], dim=1)

            # LATE FUSION
            elif self.temporal_mode == TemporalMode.LATE_FUSION:
                logits_t, self.recurrent_state = self.model(x_t, self.recurrent_state)

            # Manage Deep Supervision and Loss al tempo t
            if isinstance(logits_t, list):
                step_loss = sum(self.loss_fn(l, y_t) for l in logits_t) / len(logits_t)
                main_logits_t = logits_t[0]
            else:
                step_loss = self.loss_fn(logits_t, y_t)
                main_logits_t = logits_t

            total_loss += step_loss
            all_logits.append(main_logits_t)

            # Update memory for Early Fusion
            if self.temporal_mode == TemporalMode.EARLY_FUSION:
                self.mask_prev = (torch.sigmoid(main_logits_t.detach()) > 0.5).float()
    
        stacked_logits = torch.stack(all_logits, dim=1) 
        avg_loss = (total_loss / T) / self.accumulation_steps

        return avg_loss, stacked_logits    
    
    def _update_metrics(self, preds, labels):
    
        B, T = preds.shape[:2]
        
        # Fuse B and T to parallelize the computation: (B*T, C, H, W), MONAI wants 4D Tensors for metrics
        preds_flat = preds.reshape(B * T, *preds.shape[2:])
        labels_flat = labels.reshape(B * T, *labels.shape[2:])
        super()._update_metrics(preds_flat, labels_flat)

        self.temporal_metrics["consistency"](preds, labels)
        self.temporal_metrics["interframe"](preds)

    def _train(self):
        self._reset_all()
        return super()._train()

    def _validate(self, epoch: int):
        self._reset_all()

        metrics = super()._validate(epoch)
        
        temp = {
            **self.temporal_metrics["consistency"].aggregate(),
            **self.temporal_metrics["interframe"].aggregate()
        }
        metrics["baseline"].update(temp)
        
        return metrics

    def _test(self):
        self._reset_all()
        metrics = super()._test()
        
        temp = {
            **self.temporal_metrics["consistency"].aggregate(),
            **self.temporal_metrics["interframe"].aggregate()
        }
        metrics["baseline"].update(temp)
        
        return metrics
