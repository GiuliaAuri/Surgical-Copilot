import torch
import numpy as np

from src.surgical_copilot.bench.engine.benchmark_engine import BenchmarkEngine
from src.surgical_copilot.bench.engine.temporal_mode import TemporalMode
from src.surgical_copilot.bench.metrics.temporal_metrics.temporal_consistency import TemporalConsistencyMetric
from src.surgical_copilot.bench.metrics.temporal_metrics.temporal_iou import TemporalIoU

class TemporalBenchmarkEngine(BenchmarkEngine):

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        scheduler,
        loss_fn,
        scaler,
        cfg,
        device,
        fold_idx=0,
        temporal_mode=TemporalMode.EARLY_FUSION
    ):

        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            scaler=scaler,
            cfg=cfg,
            device=device,
            fold_idx=fold_idx,
            temporal_mode=temporal_mode
        )


        if isinstance(temporal_mode, str):
            temporal_mode = TemporalMode(temporal_mode)

        self.temporal_mode = temporal_mode

        # memory states
        self.recurrent_state = None
        self.mask_prev = None
        self._current_patient = None
        self.last_x = None  # Store the last input for temporal metrics

        self.temporal_metrics = {
            "consistency": TemporalConsistencyMetric(device=self.device),
            "temporal_iou": TemporalIoU(
                    threshold=0.5,
                    from_logits=False,   
                    eps=1e-6
                )
        }

    def _check_patient(self, batch):

        patient = batch["patient_id"][0]

        if patient != self._current_patient:
            self.recurrent_state = None
            self._current_patient = patient
    
    def _reset_temporal_state(self):
        self.recurrent_state = None
        self.mask_prev = None

        self.temporal_metrics["consistency"].reset()
        self.temporal_metrics["temporal_iou"].reset_sequence()

    def _reset_temporal_metrics(self):
        
        for metric in self.temporal_metrics.values():
            metric.reset()
    
    def _reset_all(self):
        self._reset_temporal_state()
        self._reset_temporal_metrics()

    def _prepare_inputs(self, batch):

        #self._check_new_video(batch)

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

        self._check_patient(batch)

        images = batch["current_image"].to(self.device)  # shape: (B, T, C, H, W)
        labels = batch["current_label"].to(self.device)  # shape: (B, T, 1, H, W)

        print("[TemporalBenchmarkEngine] Preparing inputs for LATE_FUSION mode:")
        print("  Images shape:", images.shape)
        print("  Labels shape:", labels.shape)

        #self._reset_temporal_state()

        self.last_x = images.clone().detach()  # Store the last input for temporal metrics

        return images, labels
    
    def _detach_state(self, state):

        if state is None:
            return None

        if isinstance(state, torch.Tensor):
            return state.detach()

        if isinstance(state, tuple):
            return tuple(
                s.detach()
                for s in state
            )

        if isinstance(state, list):
            return [
                s.detach()
                for s in state
            ]

        return state.detach()

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

        assert x.ndim == 5

        logits, self.recurrent_state = self.model(x, self.recurrent_state)
        
        # truncate the recurrent state to the current batch size and sequence length
        self.recurrent_state = self._detach_state(
            self.recurrent_state
        )

        loss = self.loss_fn(logits, y)

        return {
            "loss": loss / self.accumulation_steps,
            "logits": logits
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

            self.temporal_metrics["consistency"](preds, labels, self.last_x)
            self.temporal_metrics["temporal_iou"](preds)
            return

        B, T = preds.shape[:2]

        preds_flat = preds.reshape(B * T, *preds.shape[2:])
        labels_flat = labels.reshape(B * T, *labels.shape[2:])

        super()._update_metrics(preds_flat, labels_flat)

    def _train(self):
        self._reset_all()
        return super()._train()

    def _validate(self, epoch: int):
        self._reset_all()

        metrics = super()._validate(epoch)
        
        temp = {
            **self.temporal_metrics["consistency"].aggregate(),
            **self.temporal_metrics["temporal_iou"].aggregate()
        }
        metrics["baseline"].update(temp)
        
        return metrics

    def _test(self):
        self._reset_all()
        metrics = super()._test()
        
        temp = {
            **self.temporal_metrics["consistency"].aggregate(),
            **self.temporal_metrics["temporal_iou"].aggregate()
        }
        metrics["baseline"].update(temp)
        
        return metrics
    

