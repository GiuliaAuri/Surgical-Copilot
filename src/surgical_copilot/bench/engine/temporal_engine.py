import torch
import numpy as np

from surgical_copilot.bench.engine.benchmark_engine import BenchmarkEngine
from surgical_copilot.bench.engine.temporal_mode import TemporalMode


class TemporalBenchmarkEngine(BenchmarkEngine):

    def __init__(self, *args, temporal_mode=TemporalMode.NONE, **kwargs):
        super().__init__(*args, **kwargs)

        # forza Enum
        if isinstance(temporal_mode, str):
            temporal_mode = TemporalMode(temporal_mode)

        self.temporal_mode = temporal_mode

        # memory states
        self.recurrent_state = None
        self.mask_prev = None

        self.temporal_metrics = TemporalVariationMetric()
   
    def _reset_temporal_memory(self):
        self.recurrent_state = None
        self.mask_prev = None
        self.temporal_metrics.reset()

    def _prepare_inputs(self, batch, mask_prev=None):

        x, y = super()._prepare_inputs(batch)

        is_first = batch.get("is_first_frame", [False])[0]

        if isinstance(is_first, torch.Tensor):
            is_first = is_first.item()

        if is_first:
            self._reset_temporal_memory()

        # EARLY FUSION
        if self.temporal_mode == TemporalMode.EARLY_FUSION:

            if self.mask_prev is None:
                self.mask_prev = torch.zeros(
                    (x.shape[0], 1, x.shape[2], x.shape[3]),
                    device=self.device
                )

            x = torch.cat([x, self.mask_prev], dim=1)

        return x, y
    
    def _update_metrics(self, preds, labels):
    
        super()._update_metrics(preds, labels)
        self.temporal_metrics(preds, labels)

    def _forward_step(self, x):

        if self.temporal_mode == TemporalMode.RECURRENT:

            outputs, self.recurrent_state = self.model(
                x,
                self.recurrent_state
            )

            # safe detach for GRU/LSTM
            if isinstance(self.recurrent_state, tuple):
                self.recurrent_state = tuple(s.detach() for s in self.recurrent_state)
            else:
                self.recurrent_state = self.recurrent_state.detach()

        else:
            outputs = self.model(x)

        return outputs
    

    # UPDATE MEMORY (EARLY FUSION)
    def _post_forward_hook(self, logits):
        if self.temporal_mode == TemporalMode.EARLY_FUSION:
            self.mask_prev = (torch.sigmoid(logits.detach()) > 0.5).float()
    
    def _train(self):
        self._reset_temporal_memory()
        return super()._train()

    def _validate(self, epoch: int):
        self._reset_temporal_memory()
        metrics = super()._validate(epoch)
        
        temp_scores = self.temporal_metrics.aggregate()
        metrics["baseline"].update(temp_scores)
        
        return metrics

    def _test(self):
        self._reset_temporal_memory()
        metrics = super()._test()
        
        temp_scores = self.temporal_metrics.aggregate()
        metrics["baseline"].update(temp_scores)
        
        return metrics
    


class TemporalVariationMetric:
    
    def __init__(self, smooth=1e-6):
        self.smooth = smooth
        self.reset()

    def reset(self):
        self.prev_pred = None
        self.prev_label = None
        self.ious = []
        self.dices = []

    def __call__(self, preds: torch.Tensor, labels: torch.Tensor):
        # Binarizzazione
        p_bin = preds > 0.5
        l_bin = labels > 0.5

        if self.prev_pred is not None and self.prev_label is not None:
            # Calcolo dei delta (XOR logico per trovare i pixel cambiati)
            delta_pred = (p_bin ^ self.prev_pred).float()
            delta_label = (l_bin ^ self.prev_label).float()

            # Operatori insiemistici
            inter = torch.sum(delta_pred * delta_label, dim=(1, 2, 3))
            sum_parts = torch.sum(delta_pred, dim=(1, 2, 3)) + torch.sum(delta_label, dim=(1, 2, 3))
            union = sum_parts - inter

            # Calcolo metriche per ogni elemento del batch
            for i, u, s, d_gt in zip(inter, union, sum_parts, torch.sum(delta_label, dim=(1, 2, 3))):
                if u > 0:
                    iou = (i + self.smooth) / (u + self.smooth)
                    dice = (2.0 * i + self.smooth) / (s + self.smooth)
                    
                    self.ious.append(iou.item())
                    self.dices.append(dice.item())
                elif d_gt == 0 and u == 0:
                    # Perfetta consistenza: nessuno dei due è cambiato
                    self.ious.append(1.0)
                    self.dices.append(1.0)

        # Update degli stati per t-1
        self.prev_pred = p_bin.clone().detach()
        self.prev_label = l_bin.clone().detach()

    def aggregate(self):
        return {
            "temporal_iou": float(np.mean(self.ious)) if self.ious else 0.0,
            "temporal_dice": float(np.mean(self.dices)) if self.dices else 0.0
        }