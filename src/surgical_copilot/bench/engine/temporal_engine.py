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

        if is_first:
            self._reset_temporal_state()

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

        self.temporal_metrics["consistency"](preds, labels)
        self.temporal_metrics["interframe"](preds)

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
