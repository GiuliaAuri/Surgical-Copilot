import torch
import numpy as np

from surgical_copilot.bench.engine.benchmark_engine import BenchmarkEngine
from surgical_copilot.bench.engine.temporal_mode import TemporalMode
from surgical_copilot.bench.metrics.temporal_metrics.temporal_consistency import TemporalConsistencyMetric
from surgical_copilot.bench.metrics.temporal_metrics.inter_frame import InterFrameTemporalMetric


class TemporalBenchmarkEngine(BenchmarkEngine):

    def __init__(self, *args, temporal_mode=TemporalMode.EARLY_FUSION, **kwargs):
        super().__init__(*args, **kwargs)

        assert temporal_mode!=TemporalMode.NONE, "TemporalBenchmarkEngine should not be used with temporal_mode=TemporalMode.NONE. For spatial modes, use the appropriate engine."

        if isinstance(temporal_mode, str):
            temporal_mode = TemporalMode(temporal_mode)

        self.temporal_mode = temporal_mode

        # memory states
        self.recurrent_state = None
        self.mask_prev = None

        self.temporal_metrics = {
            "consistency": TemporalConsistencyMetric(device=self.device),
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

        # EARLY_FUSION mode: we expect the input to be a single frame,
        # and we concatenate the previous mask (or a zero mask if it's the first frame) 
        # to the current image. We also reset the temporal state at the beginning of each new sequence.

        if self.temporal_mode == TemporalMode.EARLY_FUSION:

            is_first = batch["is_first_frame"]

            if isinstance(is_first, torch.Tensor):
                is_first = bool(is_first[0].item())

            if is_first:
                self._reset_temporal_state()
                
            image = batch["current_image"].to(self.device)

            if self.model.training:
                prev = batch["prev_label"].to(self.device)
            else:

                if self.mask_prev is None:
                    prev = torch.zeros(
                        (image.size(0),1,image.size(2),image.size(3)),
                        device=self.device
                    ).to(self.device)
                else:
                    prev = self.mask_prev

            label = batch["current_label"].to(self.device)

            x = torch.cat(
                (image, prev),
                dim=1
            ).to(self.device)

            self.last_x = x.clone().detach()  # Store the last input for temporal metrics

            return x, label
        
        
        # LATE_FUSION mode: we expect the input to be a sequence of frames, so we don't concatenate the previous mask, 
        # but we still need to reset the temporal state for each new batch.

        images = batch["image"].to(self.device)  # shape: (B, T, C, H, W)
        labels = batch["label"].to(self.device)  # shape: (B, T, 1, H, W)

        self._reset_temporal_state()

        self.last_x = images.clone().detach()  # Store the last input for temporal metrics

        return images, labels

    def _early_fusion_forward(self, x, y):

        assert x.ndim == 4, "Expected input x to be a 4D tensor of shape (B, C, H, W)"

        # teaching forcing: we use the previous mask from the ground truth during training, and the predicted mask during inference.
        logits = self.model(x)

        if isinstance(logits, list):
            loss = sum(self.loss_fn(l, y) for l in logits) / len(logits)
            logits = logits[0]
        else:
            loss = self.loss_fn(logits, y)

        self.mask_prev = torch.sigmoid(logits.detach())

        return {
            "loss": loss / self.accumulation_steps, 
            "logits": logits
        }

    def _late_fusion_forward(self, x, y):

        assert x.ndim == 5, "Expected input x to be a 5D tensor of shape (B, T, C, H, W)"

        B, T, C, H, W = x.shape
        total_loss = 0.0
        all_logits = []

        for t in range(T):
            x_t = x[:, t] # (B, C, H, W)
            y_t = y[:, t] # (B, 1, H, W)

            logits_t, self.recurrent_state = self.model(x_t, self.recurrent_state)

            if isinstance(logits_t, list):
                step_loss = sum(self.loss_fn(l, y_t) for l in logits_t) / len(logits_t)
                main_logits_t = logits_t[0]
            else:
                step_loss = self.loss_fn(logits_t, y_t)
                main_logits_t = logits_t

            total_loss += step_loss
            all_logits.append(main_logits_t)

        stacked_logits = torch.stack(all_logits, dim=1)
        loss = total_loss / T  / self.accumulation_steps
        return  {
            "loss": loss, 
            "logits": stacked_logits
        }

    def _forward_step(self, x, y):

        if self.temporal_mode == TemporalMode.EARLY_FUSION:
            return self._early_fusion_forward(x, y)
        
        return self._late_fusion_forward(x, y) 
    
    def _scale_loss(self, i, loss):

        if self.scaler is not None:

            self.scaler.scale(loss).backward()

            if ((i + 1) % self.accumulation_steps == 0) or (i + 1 == len(self.train_loader)):

                # Apply clipping to the unscaled gradients to avoid exploding gradients
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
        else:

            loss.backward()

            if ((i + 1) % self.accumulation_steps == 0) or (i + 1 == len(self.train_loader)):
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                self.optimizer.zero_grad()

    def _update_metrics(self, preds, labels):
    
        if self.temporal_mode == TemporalMode.EARLY_FUSION:

            super()._update_metrics(preds, labels)

            self.temporal_metrics["consistency"](preds, labels)
            self.temporal_metrics["interframe"](preds)

            return

        B, T = preds.shape[:2]

        preds_flat = preds.reshape(B * T, *preds.shape[2:])
        labels_flat = labels.reshape(B * T, *labels.shape[2:])

        super()._update_metrics(preds_flat, labels_flat)

    def _update_temporal_metrics(self, preds, labels):
        if hasattr(self, 'last_x'):
            self.temporal_metrics["consistency"](preds, labels, self.last_x)
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
    

